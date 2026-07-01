#!/usr/bin/env python3
# Shared JAX machinery for the differentiable WW kinematic fits (lnuqq + 4q):
# differentiable pdf evaluators (-2 log P), on-the-fly BW log-Z, a generic batched
# Levenberg-Marquardt (Nielsen damping + trust-region cap), and a parser that reads
# the fitted DCB priors straight from the C++ headers (so JAX auto-syncs with refits).
#
# Use under:  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
import re, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# ── Gauss-Legendre 24-pt (for the BW phase-space normalization integral) ──────
GL_X = jnp.array([-9.9518721999702131e-01,-9.7472855597130947e-01,-9.3827455200273280e-01,-8.8641552700440096e-01,
 -8.2000198597390295e-01,-7.4012419157855436e-01,-6.4809365193697555e-01,-5.4542147138883956e-01,
 -4.3379350762604513e-01,-3.1504267969616340e-01,-1.9111886747361631e-01,-6.4056892862605630e-02,
 6.4056892862605630e-02,1.9111886747361631e-01,3.1504267969616340e-01,4.3379350762604513e-01,
 5.4542147138883956e-01,6.4809365193697555e-01,7.4012419157855436e-01,8.2000198597390295e-01,
 8.8641552700440096e-01,9.3827455200273280e-01,9.7472855597130947e-01,9.9518721999702131e-01])
GL_W = jnp.array([1.2341229799987091e-02,2.8531388628933743e-02,4.4277438817419551e-02,5.9298584915436742e-02,
 7.3346481411080411e-02,8.6190161531953288e-02,9.7618652104114065e-02,1.0744427011596561e-01,
 1.1550566805372561e-01,1.2167047292780342e-01,1.2583745634682830e-01,1.2793819534675221e-01,
 1.2793819534675221e-01,1.2583745634682830e-01,1.2167047292780342e-01,1.1550566805372561e-01,
 1.0744427011596561e-01,9.7618652104114065e-02,8.6190161531953288e-02,7.3346481411080411e-02,
 5.9298584915436742e-02,4.4277438817419551e-02,2.8531388628933743e-02,1.2341229799987091e-02])

# ── DCB core (stable form; no overflow for nL~200; nan-grad-safe via clamped t) ─
def dcb_unnorm(t, aL, nL, aR, nR):
    BL = nL/aL - aL; BR = nR/aR - aR
    t_left  = jnp.minimum(t, -aL); t_right = jnp.maximum(t, aR)
    left  = jnp.exp(-0.5*aL*aL) * ((nL/aL)/(BL - t_left ))**nL
    right = jnp.exp(-0.5*aR*aR) * ((nR/aR)/(BR + t_right))**nR
    core  = jnp.exp(-0.5*t*t)
    return jnp.where(t < -aL, left, jnp.where(t > aR, right, core))

# ── Evaluators: return -2 log P(x), matching kinfit_inputs/dcb_params.h ───────
# DcbGaussParams: mu,sigma,aL,nL,aR,nR,f_wide,mu_wide,sigma_wide,norm
def dcb_gauss_n2ll(x, p):
    mu,sigma,aL,nL,aR,nR,fw,muw,sw,norm = p
    core = dcb_unnorm((x-mu)/sigma, aL,nL,aR,nR)
    wide = jnp.exp(-0.5*((x-muw)/sw)**2)
    f = (1.0-fw)*core + fw*wide
    return -2.0*(jnp.log(jnp.maximum(f,1e-300)) + jnp.log(norm))

# GaussParams: mu,sigma
def gauss_n2ll(x, p):
    mu,sigma = p[0],p[1]; t=(x-mu)/sigma
    return t*t + 2.0*jnp.log(sigma) + jnp.log(2.0*jnp.pi)

# SpikeDcbGaussParams: f_delta,sigma_res,mu,sigma,aL,nL,aR,nR,f_wide,mu_wide,sigma_wide,norm
def spike_dcb_n2ll(x, p):
    f_delta,sigma_res,mu,sigma,aL,nL,aR,nR,fw,muw,sw,norm = p
    inv = 1.0/sigma_res
    delta = f_delta*inv*jnp.exp(-0.5*x*x*inv*inv)/jnp.sqrt(2.0*jnp.pi)
    core = dcb_unnorm((x-mu)/sigma, aL,nL,aR,nR)
    wide = jnp.exp(-0.5*((x-muw)/sw)**2)
    shape = (1.0-fw)*core + fw*wide
    return -2.0*jnp.log(jnp.maximum(delta + (1.0-f_delta)*shape*norm, 1e-300))

# DcbExpRight3GaussParams: mu,sigma,aL,nL,aR,kR,f_s,mu_s,sigma_s,f_o,mu_o,sigma_o,norm
def dcb_er3g_n2ll(x, p):
    mu,sigma,aL,nL,aR,kR,f_s,mu_s,sigma_s,f_o,mu_o,sigma_o,norm = p
    t = (x-mu)/sigma
    BL = nL/aL - aL; t_left = jnp.minimum(t,-aL)
    left  = jnp.exp(-0.5*aL*aL)*((nL/aL)/(BL - t_left))**nL
    right = jnp.exp(-0.5*aR*aR - kR*(jnp.maximum(t,aR)-aR))
    core  = jnp.where(t < -aL, left, jnp.where(t > aR, right, jnp.exp(-0.5*t*t)))
    sh = jnp.exp(-0.5*((x-mu_s)/sigma_s)**2); ou = jnp.exp(-0.5*((x-mu_o)/sigma_o)**2)
    fc = jnp.maximum(0.0, 1.0-f_s-f_o)
    f = fc*core + f_s*sh + f_o*ou
    return -2.0*(jnp.log(jnp.maximum(f,1e-300)) + jnp.log(norm))

# AsymGauss3GParams: mu,sigma,sigma_L,sigma_R,f_s,mu_s,sigma_s,f_o,mu_o,sigma_o (already normalized)
def asymg3g_n2ll(x, p):
    mu,sigma,sL,sR,f_s,mu_s,sigma_s,f_o,mu_o,sigma_o = p
    dx = x-mu
    s_core = jnp.where(dx < 0.0, sL, sR)
    n_core = jnp.sqrt(2.0/jnp.pi)/(sL+sR)
    core = n_core*jnp.exp(-0.5*(dx/s_core)**2)
    sh = (1.0/(sigma_s*jnp.sqrt(2*jnp.pi)))*jnp.exp(-0.5*((x-mu_s)/sigma_s)**2)
    ou = (1.0/(sigma_o*jnp.sqrt(2*jnp.pi)))*jnp.exp(-0.5*((x-mu_o)/sigma_o)**2)
    fc = jnp.maximum(0.0, 1.0-f_s-f_o)
    f = fc*core + f_s*sh + f_o*ou
    return -2.0*jnp.log(jnp.maximum(f,1e-300))

def log_Z_ontf(m_WW, mW, gW):
    mwgw=mW*gW; mW2=mW*mW; s_ww=m_WW*m_WW
    t_min=jnp.arctan(-mW2/mwgw); t_max=jnp.arctan((s_ww-mW2)/mwgw)
    half_d=0.5*(t_max-t_min); half_s=0.5*(t_max+t_min)
    t=half_d*GL_X+half_s
    m=jnp.sqrt(jnp.maximum(mW2+mwgw*jnp.tan(t),1e-12)); inv=1.0/m
    mh=m[:,None]; ml=m[None,:]; invh=inv[:,None]; invl=inv[None,:]
    lam=(s_ww-(mh+ml)**2)*(s_ww-(mh-ml)**2)
    pos=lam>0.0; lam_s=jnp.where(pos,lam,1.0)
    integ=jnp.where(pos, jnp.sqrt(lam_s)*invh*invl/(4.0*s_ww), 0.0)
    Z=jnp.sum((GL_W[:,None]*GL_W[None,:])*integ)*half_d*half_d
    return jnp.log(jnp.maximum(Z,1e-300))

def y2x(y, p):                 # linear y-rescale around prior mu with width sigma
    return p[0] + p[1]*y

# ── Parse fitted prior constants directly from a C++ header ───────────────────
def load_params(header_path):
    txt = open(header_path).read()
    out = {}
    # matches both `constexpr <Struct> NAME = {...}` and `constexpr std::array<...> NAME = {...}`
    for m in re.finditer(r'constexpr\s+(?:\w+|std::array<[^>]*>)\s+([A-Z0-9_]+)\s*=\s*\{([^}]*)\}', txt):
        name, body = m.group(1), m.group(2)
        try:
            out[name] = np.array([float(v) for v in body.split(',') if v.strip()])
        except ValueError:
            pass     # skip arrays-of-structs aggregates (e.g. *_BINS_160)
    return out

def get_bins(P, base, ecm, nbin=5):
    """Return (bins[nbin, nfields], edges[nbin+1]) for a binned prior base name."""
    bins = np.stack([P[f"{base}_BIN{k}_{ecm}"] for k in range(nbin)], axis=0)
    edges = P[f"{base}_EDGES_{ecm}"]
    return bins, edges

def pick_bin_idx(x, edges):
    """Vectorized bin index (0..nbin-1) for value(s) x given nbin+1 edges (host/numpy)."""
    return np.clip(np.searchsorted(edges[1:-1], x), 0, len(edges)-2)

# ── Generic batched Levenberg-Marquardt (Nielsen damping + trust-region cap) ──
def make_bfgs_solver(chi2, npar, maxiter=500):
    # jax.scipy BFGS (pure-jax quasi-Newton + line search) — robust globalization
    # on the stiff multi-basin surface (reproduces Minuit's mW), and vmap/pmap-able
    # unlike scipy. Returns (y, fun, |grad|, success).
    from jax.scipy.optimize import minimize as jmin
    def fit(D):
        r = jmin(lambda y: chi2(y, D), jnp.zeros(npar), method="BFGS",
                 options={"maxiter": maxiter})
        return r.x, r.fun, jnp.linalg.norm(r.jac), r.success
    return fit

def make_solver_mn(chi2, npar, niter=100, nls=40, ftol=1e-3, c1=1e-4):
    # Modified-Newton + Armijo backtracking line search (globally convergent, the
    # textbook robust recipe; ports cleanly to C++ via a sym-eigen-decomp + a line
    # search loop). Hessian eigenvalues are floored RELATIVE to the largest
    # (w -> max(w, ftol*max|w|)) so the Newton direction is descent AND well-scaled
    # (no flat-direction blow-up), then alpha=1,1/2,... until Armijo decrease.
    grad = jax.grad(chi2, argnums=0); hess = jax.hessian(chi2, argnums=0)
    def fit(D):
        def body(k, carry):
            y, c = carry
            g=grad(y,D); H=hess(y,D)
            w,V = jnp.linalg.eigh(H)
            wf = jnp.maximum(w, ftol*jnp.max(jnp.abs(w)))     # relative PD floor
            d = -(V @ ((V.T @ g)/wf))
            gd = jnp.dot(g, d)                                # <0 (descent)
            def ls(j, st):
                a, found, cb, yb = st
                yt=y+a*d; ct=chi2(yt,D)
                ok = jnp.isfinite(ct) & (ct <= c + c1*a*gd) & (~found)
                cb=jnp.where(ok,ct,cb); yb=jnp.where(ok,yt,yb)
                return (a*0.5, found|ok, cb, yb)
            _,_,c,y = jax.lax.fori_loop(0,nls,ls,(1.0,False,c,y))
            return (y,c)
        y0=jnp.zeros(npar); c0=chi2(y0,D)
        y,c = jax.lax.fori_loop(0,niter,body,(y0,c0))
        g=grad(y,D); H=hess(y,D)
        w,V=jnp.linalg.eigh(H)
        edm=0.5*jnp.dot(g, V @ ((V.T @ g)/jnp.where(w>0,w,jnp.inf)))   # only PD dirs
        return y, c, jnp.linalg.norm(g), edm
    return fit

def make_solver(chi2, npar, niter=150, trust=1.0, LAM0_FAC=1.0):
    grad = jax.grad(chi2, argnums=0); hess = jax.hessian(chi2, argnums=0)
    EYE = jnp.eye(npar)
    def fit(D):
        y0 = jnp.zeros(npar)
        H0 = hess(y0, D)
        # Conservative start: lam0 ~ max|diag(H)| => first steps are gradient-descent-
        # like (stay in the start basin), then lam shrinks toward Newton near the min.
        # Prevents the free-mW Newton runaway to wrong high-mW basins.
        lam0 = LAM0_FAC*jnp.maximum(jnp.max(jnp.abs(jnp.diag(H0))), 1.0)
        def body(k, carry):
            y, lam, nu, c = carry
            g=grad(y,D); H=hess(y,D)
            d = jnp.linalg.solve(H + lam*EYE, -g)
            dn = jnp.linalg.norm(d)
            d = d*jnp.minimum(1.0, trust/jnp.maximum(dn,1e-30))
            yn = y+d; cn = chi2(yn,D)
            pred = 0.5*jnp.dot(d, lam*d - g)
            rho = (c-cn)/jnp.where(jnp.abs(pred)>1e-30, pred, 1e-30)
            ok = jnp.isfinite(cn) & (rho>0.0)
            y=jnp.where(ok,yn,y); c=jnp.where(ok,cn,c)
            lam=jnp.clip(jnp.where(ok, lam*jnp.maximum(1.0/3.0,1.0-(2.0*rho-1.0)**3), lam*nu),1e-12,1e12)
            nu=jnp.where(ok,2.0,nu*2.0)
            return (y,lam,nu,c)
        c0 = chi2(y0,D)
        y,lam,nu,c = jax.lax.fori_loop(0,niter,body,(y0,lam0,2.0,c0))
        g=grad(y,D); H=hess(y,D)
        w,V = jnp.linalg.eigh(H)
        edm = 0.5*jnp.dot(g, V @ ((V.T @ g)/jnp.where(w>0,w,jnp.inf)))   # only PD dirs (match make_solver_mn)
        gn  = jnp.linalg.norm(g)
        return y, c, gn, edm
    return fit
