#!/usr/bin/env python3
# (b) Is the kfit ~8% mW-runaway tail a BFGS globalization artifact or a real lower basin?
# Multi-start test: fit each event from COLD (y0=0, mW=80.4) and WARM (y0 with mW = reco
# dijet mass, all else 0). Compare per-event chi2 + mW.
#   - if WARM avoids the runaway AND has chi2 <= COLD on cold-runaway events -> BFGS
#     overshoot (globalization); warm-start / better solver fixes it.
#   - if COLD-runaway has the LOWER chi2 -> genuine lower (jet-shrunk) basin -> needs
#     a regularizer, not just a solver.
# Reports runaway frac (cold / warm / best-of-2) and, on cold-runaways, who wins chi2.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_ISR_MODE=kfit KF_SIGMA_R=1.5 python3 jax_prototype/compare_solvers_kfit.py [root] [ecm] [maxn] [nproc]
import sys, os, time, numpy as np, uproot
import multiprocessing as mp

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 10000
NPROC=int(sys.argv[4]) if len(sys.argv)>4 else 24

def worker(args):
    rowsD, rowsP, ecm = args
    os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ.setdefault("KF_ISR_MODE","kfit")
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.dirname(__file__))
    import kinfit_lnuqq_jax as K
    from scipy.optimize import minimize
    c=K.build_chi2(ecm,"kfit"); f=jax.jit(c); g=jax.jit(jax.grad(c,argnums=0)); pf=jax.jit(K.build_postfit(ecm,"kfit"))
    def fit(y0,Di,PRi):
        r=minimize(lambda y:float(f(jnp.asarray(y),Di,PRi)),y0,
                   jac=lambda y:np.asarray(g(jnp.asarray(y),Di,PRi),dtype=np.float64),
                   method="BFGS",options={"maxiter":1000,"gtol":1e-5})
        return r.x, float(r.fun)
    def vec(p,th,ph):
        st=np.sin(th); return np.array([p,p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th)])
    out=np.empty((len(rowsD),6))  # mW_cold,chi2_cold, mW_warm,chi2_warm, Wh_best, mW_best
    for k in range(len(rowsD)):
        Di=jnp.array(rowsD[k]); PRi=jnp.array(rowsP[k]); D=rowsD[k]
        # warm start: mW = reco dijet mass
        J1=vec(D[0],D[1],D[2]); J2=vec(D[3],D[4],D[5]); Wh=J1+J2
        mdijet=np.sqrt(max(Wh[0]**2-(Wh[1]**2+Wh[2]**2+Wh[3]**2),1e-6))
        y0w=np.zeros(16); y0w[0]=(mdijet-K.KF_MW_INIT)/K.KF_MW_PHYS_SIGMA
        xc,cc=fit(np.zeros(16),Di,PRi); xw,cw=fit(y0w,Di,PRi)
        mWc=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*xc[0]; mWw=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*xw[0]
        xb=xc if cc<=cw else xw; mWb=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*xb[0]
        Whb=float(pf(jnp.asarray(xb),Di,PRi)[0])
        out[k]=[mWc,cc,mWw,cw,Whb,mWb]
    return out

def main():
    t=uproot.open(ROOT)["events"]
    cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
          "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
    mb=["gen_Whad_m"]
    a=t.arrays(cols+[b for b in mb if b in t.keys()],library="np")
    D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
    ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
    idx=np.where(ok)[0][:MAXN]; D=D[idx]; N=len(D)
    gWh=a["gen_Whad_m"][idx] if "gen_Whad_m" in a else np.full(N,np.nan)
    sr=os.environ.get("KF_SIGMA_R","0.45")
    print(f"[solver test] ecm{ECM} N={N} nproc={NPROC} kfit gauss σ_R={sr}  (cold y0=0 vs warm mW=dijet)")
    sys.path.insert(0, os.path.dirname(__file__)); import kinfit_lnuqq_jax as K
    PRp=K.build_prior_pack_pool(ECM,D)
    chunks=[(D[i::NPROC],PRp[i::NPROC],ECM) for i in range(NPROC)]
    order=np.concatenate([np.arange(N)[i::NPROC] for i in range(NPROC)])
    t0=time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool: res=pool.map(worker,chunks)
    dt=time.time()-t0
    R=np.empty((N,6)); R[order]=np.concatenate(res,axis=0)
    mWc,cc,mWw,cw,Whb,mWb=[R[:,i] for i in range(6)]
    def run(m): return np.abs(m-80.385)>5
    rc,rw,rb=run(mWc),run(mWw),run(mWb)
    print(f"  ({dt:.1f}s, {1e3*dt/N/2:.2f} ms/fit)")
    print(f"\n=== runaway fraction (|mW-80.4|>5) ===")
    print(f"  COLD (y0=0):        {100*rc.mean():.1f}%")
    print(f"  WARM (mW=dijet):    {100*rw.mean():.1f}%")
    print(f"  BEST-of-2 (min χ²): {100*rb.mean():.1f}%")
    # On cold-runaway events: does warm find a LOWER chi2 (=> cold was overshoot)?
    cr=rc & np.isfinite(cc) & np.isfinite(cw)
    warm_lower = (cw < cc - 1e-6) & cr
    print(f"\n=== on COLD-runaway events (N={cr.sum()}) ===")
    print(f"  warm χ² < cold χ²: {100*warm_lower.sum()/max(cr.sum(),1):.1f}%  (warm finds a BETTER min -> cold overshoot)")
    print(f"  median (cold χ² − warm χ²) = {np.median((cc-cw)[cr]):+.2f}  (>0 => warm better)")
    print(f"  of cold-runaways, warm is NON-runaway: {100*np.mean(~rw[cr]):.1f}%")
    # accuracy of best-of-2 vs gen
    fg=np.isfinite(Whb)&np.isfinite(gWh); e=(Whb-gWh)[fg]; ec=np.abs(e)<10
    fm=np.isfinite(mWb); core=~run(mWb[fm])
    print(f"\n=== BEST-of-2 quality ===")
    print(f"  mW: mean={np.mean(mWb[fm]):.3f} std={np.std(mWb[fm]):.3f}  core std={np.std(mWb[fm][core]):.3f}")
    print(f"  Whad-gen: bias={np.mean(e):+.3f} core(|Δ|<10)RMS={np.std(e[ec]):.3f} |Δ|<2={100*np.mean(np.abs(e)<2):.1f}%")
    np.savez("/tmp/mdefranc/lnuqq_solver_test.npz",mWc=mWc,cc=cc,mWw=mWw,cw=cw,Whb=Whb,mWb=mWb,gen_Whad_m=gWh)

if __name__=="__main__": main()
