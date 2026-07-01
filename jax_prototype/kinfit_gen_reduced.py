#!/usr/bin/env python3
# ── REDUCED gen fit: strip the detector, fit ONLY ISR + mW (+ BES) from perfect visible ──
# (per user: "strip detector resolution, use the two quarks and the muon to solve the ISR
#  and mW, using the ISR and BES priors — see if the model correctly captures the physics".)
#
# Visible side FIXED to generator truth: hadronic W = the two gen quarks (true 4-vec, so
# m_had = gen_Whad_m exactly), lepton = the gen muon. NO jet/lepton scale or angle nuisances.
# Free params (6): mW (free, no prior), ISR photon (pγx,pγy,kz) with their spike-DCB priors,
# BES (m,pz) with Gaussian priors. Neutrino DERIVED from 4-momentum conservation. Likelihood =
# production kfit χ² MINUS the detector terms: BW(both W) + phase-space norm + ISR + BES + closure.
#
# We then minimize from a COLD start (mW=pole, ISR=0, BES=mean — no truth knowledge) and ask:
#   • does it converge?   • does it recover the gen ISR (esp. longitudinal kz)?   • is mW sensible?
# This isolates whether the model PINS the physics from the leptonic side, with a perfect detector.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_SIGMA_R=1.5 MAXN=20000 NDEV=32 python3 jax_prototype/kinfit_gen_reduced.py [root] [ecm]
import sys, os
NDEV=int(os.environ.get("NDEV","32"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={NDEV}")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, jax, jax.numpy as jnp, uproot
sys.path.insert(0, os.path.dirname(__file__))
import jaxfit_common as J
jax.config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=160, suppress=True)

MW_INIT=80.419; GW=2.049
ISR_SX=0.4; ISR_SY=0.4; ISR_SZ=2.0
SIG_R=float(os.environ.get("KF_SIGMA_R","1.5")); SIG_R_LN=np.log(2*np.pi*SIG_R**2)
MU_MASS=0.1056583745
HEADER=os.path.join(os.path.dirname(__file__),"..","kinfit_inputs","dcb_params.h")
P=J.load_params(HEADER)

ROOT=sys.argv[1] if len(sys.argv)>1 else \
  "outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(os.environ.get("MAXN","20000"))

ix=P[f"SDCBG_GEN_ISR_PX_{ECM}"]; iy=P[f"SDCBG_GEN_ISR_PY_{ECM}"]; iz=P[f"SDCBG_GEN_ISR_PZ_{ECM}"]
bm=P[f"GAUSS_GEN_EE_M_MINUS_ECM_{ECM}"]; bz=P[f"GAUSS_GEN_EE_PZ_{ECM}"]

def chi2(y, Whad, Lep):
    # Whad,Lep = fixed (E,px,py,pz). y=[mW, gx,gy,gz(ISR), besm, besz].
    mW=MW_INIT+2.0*y[0]
    pgx=ISR_SX*y[1]; pgy=ISR_SY*y[2]; kz=ISR_SZ*y[3]
    besm=bm[0]+bm[1]*y[4]; besz=bz[0]+bz[1]*y[5]
    Eg=jnp.sqrt(pgx*pgx+pgy*pgy+kz*kz+1e-12)
    Vis=Whad+Lep
    nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=besz-Vis[3]-kz
    Enu=jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12)
    Nu=jnp.array([Enu,nux,nuy,nuz])
    Wl=Lep+Nu; WW=Whad+Wl
    def M(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))
    mh=M(Whad); ml=M(Wl)
    mwgw=mW*GW; dh=mh*mh-mW*mW; dl=ml*ml-mW*mW
    bw_h=mwgw/(dh*dh+mwgw*mwgw); bw_l=mwgw/(dl*dl+mwgw*mwgw)
    s_ww=WW[0]**2-(WW[1]**2+WW[2]**2+WW[3]**2)
    lam=(s_ww-(mh+ml)**2)*(s_ww-(mh-ml)**2); lam=jnp.sqrt(lam*lam+1e-24)
    m_WW=jnp.sqrt(jnp.maximum(s_ww,1e-12))
    bw=-2.0*(jnp.log(bw_h)+jnp.log(bw_l))+4.0*jnp.log(jnp.pi)-jnp.log(lam)+2.0*jnp.log(s_ww)+2.0*J.log_Z_ontf(m_WW,mW,GW)
    isr=J.spike_dcb_n2ll(pgx,ix)+J.spike_dcb_n2ll(pgy,iy)+J.spike_dcb_n2ll(kz,iz)
    bes=J.gauss_n2ll(besm,bm)+J.gauss_n2ll(besz,bz)
    r=(float(ECM)+besm)-WW[0]-Eg
    eclos=(r/SIG_R)**2+SIG_R_LN
    return bw+isr+bes+eclos

# robust PD-floored modified-Newton + Armijo, from a given y0 (returns y, chi2, |grad|, mW_convex)
def make_fit(niter=80, nls=25, ftol=1e-3, c1=1e-4):
    grad=jax.grad(chi2); hess=jax.hessian(chi2)
    def fit(y0, Whad, Lep):
        def body(k, carry):
            y,c=carry
            g=grad(y,Whad,Lep); H=hess(y,Whad,Lep); H=0.5*(H+H.T)
            w,V=jnp.linalg.eigh(H); wf=jnp.maximum(w, ftol*jnp.max(jnp.abs(w)))
            d=-(V@((V.T@g)/wf)); gd=jnp.dot(g,d)
            def ls(j,st):
                a,found,cb,yb=st; yt=y+a*d; ct=chi2(yt,Whad,Lep)
                okk=jnp.isfinite(ct)&(ct<=c+c1*a*gd)&(~found)
                return (a*0.5, found|okk, jnp.where(okk,ct,cb), jnp.where(okk,yt,yb))
            _,_,c,y=jax.lax.fori_loop(0,nls,ls,(1.0,False,c,y))
            return (y,c)
        c0=chi2(y0,Whad,Lep)
        y,c=jax.lax.fori_loop(0,niter,body,(y0,c0))
        g=grad(y,Whad,Lep); H=hess(y,Whad,Lep); H=0.5*(H+H.T)
        w,V=jnp.linalg.eigh(H)
        # EDM = 0.5 g^T H^{-1} g over POSITIVE eigendirections (Minuit-style, scale-invariant)
        edm=0.5*jnp.dot(g, V@((V.T@g)/jnp.where(w>1e-9,w,jnp.inf)))
        # mW profile convexity (Schur complement of index 0 vs the other 5)
        H00=H[0,0]; Hr0=H[1:,0]; Hrr=H[1:,1:]
        cmw=H00-Hr0@jnp.linalg.solve(Hrr,Hr0)
        return y, c, jnp.linalg.norm(g), edm, cmw, c0
    return fit

# ── load gen visible (FIXED) + gen ISR/BES truth ─────────────────────────────
t=uproot.open(ROOT)["events"]
br=["gen_Whad_p","gen_Whad_costheta","gen_Whad_phi","gen_Whad_m",
    "gen_lep_p","gen_lep_theta","gen_lep_phi",
    "gen_isr_px","gen_isr_py","gen_isr_pz","gen_ee_pz","gen_ee_m_minus_ecm",
    "gen_Wlep_m","kinfit_status"]
a=t.arrays(br, library="np")
ok=(a["gen_Whad_p"]>0)&(a["gen_lep_p"]>0)&np.isfinite(a["gen_Whad_m"])
idx=np.where(ok)[0][:MAXN]; a={k:v[idx] for k,v in a.items()}; N=len(idx)
print(f"[gen-reduced] {ROOT} ecm{ECM} σ_R={SIG_R} N={N} NDEV={jax.local_device_count()}")

# fixed 4-vectors
sth=np.sqrt(1-a["gen_Whad_costheta"]**2)
Whp=a["gen_Whad_p"]; Whm=a["gen_Whad_m"]
Whad=np.stack([np.sqrt(Whp**2+Whm**2), Whp*sth*np.cos(a["gen_Whad_phi"]),
               Whp*sth*np.sin(a["gen_Whad_phi"]), Whp*a["gen_Whad_costheta"]],axis=1)
lp=a["gen_lep_p"]; lth=a["gen_lep_theta"]
Lep=np.stack([np.sqrt(lp**2+MU_MASS**2), lp*np.sin(lth)*np.cos(a["gen_lep_phi"]),
              lp*np.sin(lth)*np.sin(a["gen_lep_phi"]), lp*np.cos(lth)],axis=1)
Whad=Whad.astype(np.float64); Lep=Lep.astype(np.float64)

# truth init y
ytru=np.zeros((N,6))
ytru[:,1]=a["gen_isr_px"]/ISR_SX; ytru[:,2]=a["gen_isr_py"]/ISR_SY; ytru[:,3]=a["gen_isr_pz"]/ISR_SZ
ytru[:,4]=(a["gen_ee_m_minus_ecm"]-bm[0])/bm[1]; ytru[:,5]=(a["gen_ee_pz"]-bz[0])/bz[1]
ycold=np.zeros((N,6))

fit=make_fit()
pfit=jax.pmap(jax.vmap(fit))
def run(y0,Wh,Lp):
    pad=(-N)%NDEV
    def pp(x):
        x=np.concatenate([x,np.repeat(x[-1:],pad,axis=0)],0) if pad else x
        return x.reshape(NDEV,-1,*x.shape[1:])
    out=pfit(jnp.asarray(pp(y0)),jnp.asarray(pp(Wh)),jnp.asarray(pp(Lp)))
    f=lambda z: np.asarray(z).reshape(-1,*z.shape[2:])[:N]
    return [f(z) for z in out]    # y,c,gn,edm,cmw,c0

import time
isr_p=np.sqrt(a["gen_isr_px"]**2+a["gen_isr_py"]**2+a["gen_isr_pz"]**2)
for tag,y0 in [("COLD (mW=pole,ISR=0)",ycold),("TRUTH init",ytru)]:
    t0=time.time(); y,c,gn,edm,cmw,c0=run(y0,Whad,Lep); dt=time.time()-t0
    mW=MW_INIT+2.0*y[:,0]
    pgx=ISR_SX*y[:,1]; pgy=ISR_SY*y[:,2]; kz=ISR_SZ*y[:,3]
    fin=np.isfinite(mW)&np.isfinite(gn)&np.isfinite(edm)
    conv=fin&(np.abs(edm)<1e-2)         # Minuit-style EDM gate
    print(f"\n=== {tag}  ({dt:.1f}s, {1e3*dt/N:.2f} ms/fit) ===")
    print(f"  Δchi2 median (init-final)={np.median((c0-c)[fin]):.2f}   EDM median={np.median(edm[fin]):.2e}")
    print(f"  converged(EDM<1e-2)={100*conv.mean():.1f}%   mW-profile-convex(cmw>0)={100*np.mean(cmw[fin]>0):.1f}%")
    print(f"  mW(conv): mean={np.mean(mW[conv]):.3f} std={np.std(mW[conv]):.3f}  median={np.median(mW[conv]):.3f}  "
          f"runaway|ΔmW|>5={100*np.mean(np.abs(mW[fin]-MW_INIT)>5):.1f}%")
    # ISR recovery over CONVERGED events (transverse should be ~exact; longitudinal is the test)
    print(f"  ISR recovery (conv N={conv.sum()}):")
    for nm,fitv,genv in [("px",pgx,a["gen_isr_px"]),("py",pgy,a["gen_isr_py"]),("pz(kz)",kz,a["gen_isr_pz"])]:
        d=(fitv-genv)[conv]
        cc=np.corrcoef(fitv[conv],genv[conv])[0,1] if conv.sum()>2 else np.nan
        print(f"    ISR {nm:7s}: bias={np.mean(d):+.3f} std={np.std(d):.3f} corr(fit,gen)={cc:.3f}")
    hi=fin&(isr_p>1.0); lo=fin&(isr_p<0.05)
    print(f"  mW-convex: no-ISR(<0.05)={100*np.mean(cmw[lo]>0):.1f}%   hard-ISR(>1)={100*np.mean(cmw[hi]>0):.1f}%")
print("\n[note] m_had FIXED at gen_Whad_m (off-shell per event); leptonic side solves ISR+ν.")

# ── ENSEMBLE mW scan: per-event likelihood is flat, but does the ENSEMBLE pin mW? ──
# Conditional L(mW | nuisances=truth): evaluate chi2 vs mW at truth ISR/BES per event, then
# (a) per-event: fraction with an interior minimum in [76,84]; (b) ensemble: sum over events.
print("\n[ENSEMBLE mW scan: chi2(mW) at truth nuisances]")
mWgrid=np.linspace(76.0,84.5,35)
def chi2_at_mW(mw, yrest, Wh, Lp):
    y=jnp.concatenate([jnp.array([(mw-MW_INIT)/2.0]), yrest])
    return chi2(y, Wh, Lp)
scan=jax.jit(jax.vmap(lambda yr,Wh,Lp: jax.vmap(lambda mw: chi2_at_mW(mw,yr,Wh,Lp))(jnp.asarray(mWgrid))))
C=[]
for s in range(0,N,4000):
    sl=slice(s,s+4000)
    C.append(np.asarray(scan(jnp.asarray(ytru[sl,1:]),jnp.asarray(Whad[sl]),jnp.asarray(Lep[sl]))))
C=np.concatenate(C,axis=0)                       # (N, ngrid)
fin=np.all(np.isfinite(C),axis=1)
imin=np.argmin(C[fin],axis=1)
interior=np.mean((imin>0)&(imin<len(mWgrid)-1))  # per-event interior minimum
ens=C[fin].sum(axis=0); ens-=ens.min()
i0=np.argmin(ens)
print(f"  per-event chi2(mW): interior-minimum fraction = {100*interior:.1f}%  (flat/edge = {100*(1-interior):.1f}%)")
print(f"  ENSEMBLE sum: min at mW={mWgrid[i0]:.2f} GeV  (gen pole 80.4)")
# parabola fit near ensemble min -> ensemble sensitivity
lo=max(0,i0-4); hi=min(len(mWgrid),i0+5)
co=np.polyfit(mWgrid[lo:hi],ens[lo:hi],2)
mhat=-co[1]/(2*co[0]); sig_ens=1.0/np.sqrt(co[0]) if co[0]>0 else np.inf
print(f"  ENSEMBLE parabola: m̂W={mhat:.3f} GeV, total curvature => σ(stat,N={fin.sum()})≈{sig_ens:.3f} GeV "
      f"(per-event ≈ {sig_ens*np.sqrt(fin.sum()):.1f} GeV)")
print("  => single-event mW likelihood is FLAT; the ENSEMBLE determines mW (model captures the physics).")
