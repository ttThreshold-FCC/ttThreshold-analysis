#!/usr/bin/env python3
# ── Fully-differentiable WW→4q kinematic fit (JAX prototype) ─────────────────
#
# Faithful port of FCCAnalyses::WWFunctions::kinFit4q's per-event chi2 (= -2 log
# posterior) to JAX, minimized with a BATCHED Levenberg-Marquardt (damped Newton)
# using autodiff gradient + Hessian. vmap over events x 3 jet->W pairings, jit'd.
#
# Goal: 100% convergence "by construction" — damped Newton is deterministic
# monotone descent to a stationary point, so every event yields a finite result
# AND a Hessian-based covariance (no Minuit status-3 failures). The remaining
# questions become accuracy (vs Minuit/gen) and global-optimality (right basin),
# not convergence.
#
# This first version uses the INCLUSIVE (non-binned) 4q jet priors for simplicity
# (the C++ production uses per-jet-p bins; adding binning is a lookup, TODO).
# mW is FREE-FLOATING (rescale 2 GeV, no prior) — the mW-measurement config.
# gW is CONSTRAINED (Gaussian prior), matching production KF_GW_MODE=constrained.
#
# Run:
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/kinfit4q_jax.py <events.npz>
# where the npz is produced by dump_jets_npz.py (reco jets + gen_pairing_true +
# the C++ kinfit branches, for a head-to-head comparison on identical events).

import sys, time
import numpy as np
import jax, jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)   # match Minuit double precision

# ── Constants (ported from WWKinReco.h / WWKinReco4q.h, ecm160) ──────────────
KF_MW_INIT          = 80.419
KF_GW_FIXED         = 2.049
KF_MW_PHYS_SIGMA    = 2.0            # mW y-rescale in free mode
KF_GW_PRIOR_SIGMA   = 0.01 * KF_GW_FIXED   # 0.02049
KF_GW_PRIOR_LOG_NORM = np.log(2.0*np.pi*KF_GW_PRIOR_SIGMA**2)
ECM                 = 160.0
KF_M_LOSS_BARRIER_SIGMA_FRAC = 0.1

# DcbGaussParams field order: mu,sigma,aL,nL,aR,nR,f_wide,mu_wide,sigma_wide,norm
DCBG_JET_P_RESP   = (0.971489,0.022227,0.762052,3.937635,0.996607,7.497136,0.198135,0.951518,0.033038,12.568614861)
DCBG_JET_THETA    = (-0.000053,0.016782,0.869409,200.0,0.864114,200.0,0.093148,-0.000149,0.028912,18.378250313)
DCBG_JET_PHI      = (0.000114,0.020919,0.828841,195.349147,0.838992,71.808335,0.191349,13.658962,0.217992,6.2992986303)
# SpikeDcbGaussParams: f_delta,sigma_res,mu,sigma,aL,nL,aR,nR,f_wide,mu_wide,sigma_wide,norm
SDCBG_ISR_PZ      = (0.090019,0.001000,-0.0,0.022004,1.087860,1.376116,1.092686,1.362837,0.010000,-0.013670,0.664043,7.8631074106)
SDCBG_WW_MLOSS    = (0.054662,0.001000,-0.0,0.045682,1.249671,1.356050,1.249671,1.356050,0.010000,-0.0,1.236735,4.3675846325)
# Gaussian priors (mu,sigma)
BES_M_PRIOR = (0.0,0.05); BES_PZ_PRIOR = (0.0,0.05)
ISR_PX_PRIOR = (0.0,0.10); ISR_PY_PRIOR = (0.0,0.10)
BARRIER_SIGMA = 0.05      # soft m_loss>0 penalty scale (C++ hard wall = 1e-4)
TRUST_RADIUS  = 1.0       # LM step cap in y-units (prior-sigma); stops mW runaway
import os
MW_FREE       = os.environ.get("MW_FREE","1")=="1"   # free mW (measurement) vs prior
MW_PRIOR_SIGMA= float(os.environ.get("MW_PRIOR_SIGMA","0.2"))  # GeV, used if not free
MW_SCALE      = KF_MW_PHYS_SIGMA if MW_FREE else MW_PRIOR_SIGMA

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

# ── Differentiable pdf evaluators (return -2 log P) ──────────────────────────
def _dcb_unnorm(t, aL, nL, aR, nR):
    # Stable form: AL*(BL-t)^-nL == exp(-aL^2/2) * ((nL/aL)/(BL-t))^nL, with the
    # base O(1) so no overflow (the naive separate AL=(nL/aL)^nL overflows for
    # nL~200). Each tail uses an IN-DOMAIN clamped t so the unused jnp.where
    # branch can't produce inf/nan gradients.
    BL = nL/aL - aL; BR = nR/aR - aR
    t_left  = jnp.minimum(t, -aL)            # ensure t <= -aL  -> base in (0,1]
    t_right = jnp.maximum(t,  aR)            # ensure t >=  aR  -> base in (0,1]
    left  = jnp.exp(-0.5*aL*aL) * ((nL/aL)/(BL - t_left))**nL
    right = jnp.exp(-0.5*aR*aR) * ((nR/aR)/(BR + t_right))**nR
    core  = jnp.exp(-0.5*t*t)
    return jnp.where(t < -aL, left, jnp.where(t > aR, right, core))

def dcb_gauss_n2ll(x, p):
    mu,sigma,aL,nL,aR,nR,fw,muw,sw,norm = p
    core = _dcb_unnorm((x-mu)/sigma, aL,nL,aR,nR)
    wide = jnp.exp(-0.5*((x-muw)/sw)**2)
    f = (1.0-fw)*core + fw*wide
    return -2.0*(jnp.log(jnp.maximum(f,1e-300)) + jnp.log(norm))

def gauss_n2ll(x, p):
    mu,sigma = p; t=(x-mu)/sigma
    return t*t + 2.0*jnp.log(sigma) + jnp.log(2.0*jnp.pi)

def spike_dcb_n2ll(x, p):
    f_delta,sigma_res,mu,sigma,aL,nL,aR,nR,fw,muw,sw,norm = p
    inv = 1.0/sigma_res
    delta = f_delta*inv*jnp.exp(-0.5*x*x*inv*inv)/jnp.sqrt(2.0*jnp.pi)
    core = _dcb_unnorm((x-mu)/sigma, aL,nL,aR,nR)
    wide = jnp.exp(-0.5*((x-muw)/sw)**2)
    shape = (1.0-fw)*core + fw*wide
    smooth = (1.0-f_delta)*shape*norm
    return -2.0*jnp.log(jnp.maximum(delta+smooth,1e-300))

def log_Z_ontf(m_WW, mW, gW):
    mwgw = mW*gW; mW2 = mW*mW; s_ww = m_WW*m_WW
    t_min = jnp.arctan(-mW2/mwgw); t_max = jnp.arctan((s_ww-mW2)/mwgw)
    half_d = 0.5*(t_max-t_min); half_s = 0.5*(t_max+t_min)
    t = half_d*GL_X + half_s
    m = jnp.sqrt(jnp.maximum(mW2 + mwgw*jnp.tan(t), 1e-12)); inv = 1.0/m   # (24,)
    mh = m[:,None]; ml = m[None,:]; invh = inv[:,None]; invl = inv[None,:]
    sum2 = (mh+ml)**2; dif2 = (mh-ml)**2
    lam = (s_ww-sum2)*(s_ww-dif2)
    pos = lam > 0.0                                  # nan-grad-safe sqrt(max(lam,0))
    lam_safe = jnp.where(pos, lam, 1.0)
    integ = jnp.where(pos, jnp.sqrt(lam_safe)*invh*invl/(4.0*s_ww), 0.0)
    Z = jnp.sum((GL_W[:,None]*GL_W[None,:])*integ) * half_d*half_d
    return jnp.log(jnp.maximum(Z,1e-300))

def _y2x(y, p):  # p = DcbGauss tuple; uses mu,sigma
    return p[0] + p[1]*y

# ── The per-event chi2 (one pairing). jets in pairing order: Wa=(0,1) Wb=(2,3) ─
# x4 = arrays (4,) of jet p, theta, phi (massless reco jets).
@partial(jax.jit, static_argnums=())
def chi2(y, P, TH, PH):
    mW = KF_MW_INIT + MW_SCALE*y[0]                  # free-floating or prior-rescaled mW
    gW = KF_GW_FIXED + KF_GW_PRIOR_SIGMA*y[1]
    s = _y2x(y[2:6],  DCBG_JET_P_RESP)
    t = _y2x(y[6:10], DCBG_JET_THETA)
    q = _y2x(y[10:14],DCBG_JET_PHI)
    bes_m  = _y2x(y[14], BES_M_PRIOR)
    bes_pz = _y2x(y[15], BES_PZ_PRIOR)

    p = P/s; th = TH - t; ph = PH - q
    sinth = jnp.sin(th)
    px = p*sinth*jnp.cos(ph); py = p*sinth*jnp.sin(ph); pz = p*jnp.cos(th); E = p
    def W(i,j):
        return (E[i]+E[j], px[i]+px[j], py[i]+py[j], pz[i]+pz[j])
    Ea,pxa,pya,pza = W(0,1); Eb,pxb,pyb,pzb = W(2,3)
    ma = jnp.sqrt(jnp.maximum(Ea*Ea-(pxa*pxa+pya*pya+pza*pza),1e-12))
    mb = jnp.sqrt(jnp.maximum(Eb*Eb-(pxb*pxb+pyb*pyb+pzb*pzb),1e-12))
    EW=Ea+Eb; pxW=pxa+pxb; pyW=pya+pyb; pzW=pza+pzb
    s_ww = EW*EW-(pxW*pxW+pyW*pyW+pzW*pzW)
    m_WW = jnp.sqrt(jnp.maximum(s_ww,1e-12))

    # BW x BW x phase-space + log-Z normalization
    mwgw=mW*gW; dh=ma*ma-mW*mW; dl=mb*mb-mW*mW
    bw_h=mwgw/(dh*dh+mwgw*mwgw); bw_l=mwgw/(dl*dl+mwgw*mwgw)
    lam=(s_ww-(ma+mb)**2)*(s_ww-(ma-mb)**2); lam=jnp.sqrt(lam*lam+1e-24)
    bw = -2.0*(jnp.log(bw_h)+jnp.log(bw_l)) + 4.0*jnp.log(jnp.pi) \
         - jnp.log(lam) + 2.0*jnp.log(s_ww) + 2.0*log_Z_ontf(m_WW,mW,gW)

    bes = gauss_n2ll(bes_m,BES_M_PRIOR) + gauss_n2ll(bes_pz,BES_PZ_PRIOR)
    isr_px=-pxW; isr_py=-pyW; isr_pz=bes_pz-pzW
    isr = gauss_n2ll(isr_px,ISR_PX_PRIOR)+gauss_n2ll(isr_py,ISR_PY_PRIOR)+spike_dcb_n2ll(isr_pz,SDCBG_ISR_PZ)

    m_ee=ECM+bes_m; m_loss=m_WW-m_ee
    m_loss_abs=jnp.sqrt(m_loss*m_loss+1e-8)         # smooth |m_loss| (C1 at 0)
    mlt = spike_dcb_n2ll(m_loss_abs,SDCBG_WW_MLOSS)
    # One-sided m_loss>0 penalty. The C++ hard wall (sigma_bar=1e-4) wrecks the
    # Hessian; here it's a SOFT penalty at the data-driven m_loss resolution
    # (~0.05 GeV), kept C1 at 0. Set BARRIER_SIGMA small to recover the hard wall.
    mlt = mlt + jnp.where(m_loss>0.0,(m_loss/BARRIER_SIGMA)**2,0.0)

    scale = jnp.sum(jax.vmap(lambda si: dcb_gauss_n2ll(si,DCBG_JET_P_RESP))(s))
    ang   = jnp.sum(jax.vmap(lambda ti: dcb_gauss_n2ll(ti,DCBG_JET_THETA))(t)) \
          + jnp.sum(jax.vmap(lambda qi: dcb_gauss_n2ll(qi,DCBG_JET_PHI))(q))
    mw_term = 0.0 if MW_FREE else (y[0]*y[0] + jnp.log(2.0*jnp.pi*MW_PRIOR_SIGMA**2))
    gw_term = y[1]*y[1] + KF_GW_PRIOR_LOG_NORM       # constrained gW
    return bw+bes+isr+mlt+scale+ang+mw_term+gw_term

_grad = jax.grad(chi2); _hess = jax.hessian(chi2)

# ── Batched Levenberg-Marquardt (damped Newton) ──────────────────────────────
def make_solver(niter=120):
    # Nielsen adaptive-damping Levenberg-Marquardt on the scalar chi2.
    # step solves (H + lam*I) d = -g; gain ratio rho = actual/predicted reduction
    # adapts lam: good step -> shrink lam (toward Newton), bad step -> grow lam
    # (toward gradient descent) AND reject. (H+lam I) bounds flat-direction steps
    # (no runaway) and indefinite-H steps (always descent at large lam). Robust to
    # the free-mW near-flat direction. lam0 from the Hessian diagonal scale.
    EYE = jnp.eye(16)
    def fit(P, TH, PH):
        H0 = _hess(jnp.zeros(16),P,TH,PH)
        lam0 = 1e-3*jnp.maximum(jnp.max(jnp.abs(jnp.diag(H0))), 1.0)
        def body(k, carry):
            y, lam, nu, c = carry
            g=_grad(y,P,TH,PH); H=_hess(y,P,TH,PH)
            d = jnp.linalg.solve(H + lam*EYE, -g)
            dn = jnp.linalg.norm(d)                     # trust-region cap (y-units):
            d = d * jnp.minimum(1.0, TRUST_RADIUS/jnp.maximum(dn,1e-30))  # bound weakly-curved steps
            yn = y + d; cn = chi2(yn,P,TH,PH)
            pred = 0.5*jnp.dot(d, lam*d - g)            # predicted reduction >0
            rho = (c - cn)/jnp.where(jnp.abs(pred)>1e-30, pred, 1e-30)
            ok = jnp.isfinite(cn) & (rho > 0.0)
            y = jnp.where(ok, yn, y); c = jnp.where(ok, cn, c)
            lam = jnp.where(ok, lam*jnp.maximum(1.0/3.0, 1.0-(2.0*rho-1.0)**3), lam*nu)
            lam = jnp.clip(lam, 1e-12, 1e12)
            nu  = jnp.where(ok, 2.0, nu*2.0)
            return (y, lam, nu, c)
        c0 = chi2(jnp.zeros(16),P,TH,PH)
        y,lam,nu,c = jax.lax.fori_loop(0,niter,body,(jnp.zeros(16),lam0,2.0,c0))
        g=_grad(y,P,TH,PH); H=_hess(y,P,TH,PH)
        w,V = jnp.linalg.eigh(H)
        edm = 0.5*jnp.dot(g, V @ ((V.T @ g)/jnp.maximum(w,1e-12)))
        return y, c, jnp.linalg.norm(g), edm
    return fit

# pairing jet index orders, matching C++ order[3][4]
ORDERS = np.array([[0,1,2,3],[0,2,1,3],[0,3,1,2]])

def main():
    if len(sys.argv)<2:
        print("usage: kinfit4q_jax.py <events.npz>"); sys.exit(1)
    d = np.load(sys.argv[1])
    P  = np.stack([d[f"reco_jet{i}_p"]     for i in (1,2,3,4)],axis=1)  # (N,4)
    TH = np.stack([d[f"reco_jet{i}_theta"] for i in (1,2,3,4)],axis=1)
    PH = np.stack([d[f"reco_jet{i}_phi"]   for i in (1,2,3,4)],axis=1)
    N = P.shape[0]; print(f"[jax kinfit4q] N={N} events, x3 pairings, jax {jax.__version__}")
    solver = make_solver(niter=80)
    vfit = jax.jit(jax.vmap(solver))

    chi2_p=np.full((N,3),np.inf); grad_p=np.full((N,3),np.inf); edm_p=np.full((N,3),np.inf); mW_p=np.zeros((N,3))
    t0=time.time()
    for k,o in enumerate(ORDERS):
        Pp=jnp.array(P[:,o]); THp=jnp.array(TH[:,o]); PHp=jnp.array(PH[:,o])
        y,c,gn,edm = vfit(Pp,THp,PHp); y=np.asarray(y)
        chi2_p[:,k]=np.asarray(c); grad_p[:,k]=np.asarray(gn); edm_p[:,k]=np.asarray(edm)
        mW_p[:,k]=KF_MW_INIT+KF_MW_PHYS_SIGMA*y[:,0]
    dt=time.time()-t0
    best=np.argmin(chi2_p,axis=1)
    bchi2=chi2_p[np.arange(N),best]; bgrad=grad_p[np.arange(N),best]; bedm=edm_p[np.arange(N),best]; bmW=mW_p[np.arange(N),best]

    aedm=np.abs(edm_p)
    print(f"\n=== RESULTS ({dt:.1f}s, {1e3*dt/(3*N):.2f} ms/fit) ===")
    print(f"finite chi2: per-pairing {100*np.mean(np.isfinite(chi2_p)):.1f}% | best {100*np.mean(np.isfinite(bchi2)):.1f}%")
    print(f"|grad| median per-pairing={np.median(grad_p):.2e}  best={np.median(bgrad):.2e}")
    print(f"EDM    median per-pairing={np.median(aedm):.2e}  best={np.median(np.abs(bedm)):.2e}")
    for tol in (1e-2,1e-3,1e-4):
        print(f"  convergence EDM<{tol:.0e}: per-pairing {100*np.mean(aedm<tol):.1f}% | best {100*np.mean(np.abs(bedm)<tol):.1f}%")
    m=np.isfinite(bmW)
    print(f"free mW (all best): mean={np.mean(bmW[m]):.3f} std={np.std(bmW[m]):.3f} GeV  (gen~80.385)")

    if "gen_pairing_true" in d:
        gt=d["gen_pairing_true"]; ok=gt>=0
        acc=100*np.mean(best[ok]==gt[ok]); print(f"pairing accuracy vs gen_pairing_true: {acc:.1f}% (N={np.sum(ok)})")
    if "kinfit4q_valid" in d:
        print(f"[C++ Minuit on same events] valid={100*np.mean(d['kinfit4q_valid']):.1f}%  "
              f"valid_loose={100*np.mean(d.get('kinfit4q_valid_loose',d['kinfit4q_valid'])):.1f}%  "
              f"pairing_correct={100*np.mean(d.get('kinfit4q_pairing_correct',[np.nan])):.1f}%")

if __name__=="__main__":
    main()
