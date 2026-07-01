#!/usr/bin/env python3
# Fork C — measure PHYSICS-RECOVERY of the differentiable lnuqq fit vs Minuit at
# scale, using the ROBUST scipy BFGS (analytic jax grad) parallelized across cores
# with multiprocessing (jax.scipy BFGS is too weak on this stiff surface). Also
# times throughput. Main process stays jax-free; workers import jax after spawn.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/recovery_lnuqq_mp.py [root] [ecm] [maxn] [nproc]
import sys, os, time, numpy as np, uproot
import multiprocessing as mp

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 10000
NPROC=int(sys.argv[4]) if len(sys.argv)>4 else 48

DOSWAP=os.environ.get("DOSWAP","1")=="1"
def worker(args):
    rowsD, rowsN, rowsS, ecm = args
    os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.dirname(__file__))
    import kinfit_lnuqq_jax as K
    from scipy.optimize import minimize
    chi2=K.build_chi2(ecm); pf=jax.jit(K.build_postfit(ecm)); gof=jax.jit(K.build_gof(ecm))
    f=jax.jit(chi2); g=jax.jit(jax.grad(chi2,argnums=0))
    START_MW=float(os.environ.get("START_MW","80.419"))   # dip diagnostic: move mW start
    y0=np.zeros(16); y0[0]=(START_MW-80.419)/2.0           # mW=80.419+2*y0[0]
    def fit_one(Di,PRi):
        fun=lambda y: float(f(jnp.asarray(y),Di,PRi))
        jac=lambda y: np.asarray(g(jnp.asarray(y),Di,PRi),dtype=np.float64)
        return minimize(fun, y0.copy(), jac=jac, method="BFGS", options={"maxiter":1000,"gtol":1e-5})
    out=np.empty((len(rowsD),7))   # mW, chi2, Whad, Wlep, WW, swapped, gof
    for k in range(len(rowsD)):
        Di=jnp.array(rowsD[k]); PRn=jnp.array(rowsN[k])
        r=fit_one(Di,PRn); PRw_used=PRn; swapped=0.0
        if DOSWAP:
            PRw=jnp.array(rowsS[k]); rs=fit_one(Di,PRw)
            if np.isfinite(rs.fun) and (not np.isfinite(r.fun) or rs.fun<r.fun):
                r=rs; PRw_used=PRw; swapped=1.0
        yx=jnp.asarray(r.x); m=np.asarray(pf(yx,Di,PRw_used)); gf=float(gof(yx,Di,PRw_used))
        out[k]=[K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*r.x[0], r.fun, m[0], m[1], m[2], swapped, gf]
    return out

def main():
    t=uproot.open(ROOT)["events"]
    cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
          "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
    mb=["kinfit_mW","kinfit_chi2","kinfit_Whad_m","kinfit_Wlep_m","kinfit_WW_m",
        "kinfit_valid","kinfit_valid_loose","gen_Whad_m","gen_WW_m"]
    a=t.arrays(cols+[b for b in mb if b in t.keys()],library="np")
    D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
    ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
    idx=np.where(ok)[0][:MAXN]; D=D[idx]; N=len(D)
    cmw=a["kinfit_mW"][idx]; v=a["kinfit_valid"][idx].astype(bool); vl=a["kinfit_valid_loose"][idx].astype(bool)
    print(f"[recovery MP] ecm{ECM} N={N} nproc={NPROC} binned+swap(DOSWAP={os.environ.get('DOSWAP','1')})")
    sys.path.insert(0, os.path.dirname(__file__))
    import kinfit_lnuqq_jax as K
    PRn, PRs = K.build_prior_packs(ECM, D)                 # (N,93) natural + jet-swapped
    chunks=[(D[i::NPROC], PRn[i::NPROC], PRs[i::NPROC], ECM) for i in range(NPROC)]
    order=np.concatenate([np.arange(N)[i::NPROC] for i in range(NPROC)])
    t0=time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        res=pool.map(worker, chunks)
    dt=time.time()-t0
    R=np.empty((N,7)); R[order]=np.concatenate(res,axis=0)
    mW=R[:,0]; fin=np.isfinite(mW); gofv=R[:,6]
    print(f"  jet1<->jet2 swap chosen for {100*np.mean(R[:,5]):.1f}% of events")
    gf=gofv[np.isfinite(gofv)&(gofv<1e4)]
    print(f"  GoF (mode-ref residual ~chi2): mean={np.mean(gf):.2f}  median={np.median(gf):.2f}  (mean ~ effective ndof)")
    # save everything for plotting
    save={"mW_diff":R[:,0],"chi2_diff":R[:,1],"Whad_diff":R[:,2],"Wlep_diff":R[:,3],"WW_diff":R[:,4],
          "swapped":R[:,5],"gof":R[:,6],"valid":v,"valid_loose":vl}
    for b in mb:
        if b in a: save[b.replace("kinfit_","mn_").replace("gen_","gen_")]=a[b][idx]
    np.savez("/tmp/mdefranc/lnuqq_compare.npz", **save)
    print(f"  saved /tmp/mdefranc/lnuqq_compare.npz")
    d=mW-cmw; gv=fin&np.isfinite(cmw)
    print(f"\n=== RECOVERY ({dt:.1f}s wall, {1e3*dt/N:.2f} ms/fit eff, {N/dt:.0f} fits/s on {NPROC} cores) ===")
    print(f"finite diff-fit: {100*np.mean(fin):.1f}%")
    print(f"diff-fit mW(all finite): mean={np.mean(mW[fin]):.3f} std={np.std(mW[fin]):.3f}")
    print(f"[Minuit] valid={100*np.mean(v):.1f}%  valid_loose={100*np.mean(vl):.1f}%  mW(valid) mean={np.mean(cmw[v]):.3f} std={np.std(cmw[v]):.3f}")
    for thr in (0.25,0.5,1.0,2.0):
        print(f"  recovery |ΔmW|<{thr}: vs Minuit-valid {100*np.mean(np.abs(d[gv&v])<thr):.1f}%   vs Minuit-all {100*np.mean(np.abs(d[gv])<thr):.1f}%")
    print(f"  median |ΔmW| vs valid = {np.median(np.abs(d[gv&v])):.3f} GeV")
    # diff-fit mW resolution on the agreeing core
    core=gv&v&(np.abs(d)<2.0)
    print(f"  diff-fit mW on agreeing core (|Δ|<2): mean={np.mean(mW[core]):.3f} std={np.std(mW[core]):.3f}  (N={core.sum()})")

if __name__=="__main__": main()
