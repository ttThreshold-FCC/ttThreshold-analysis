#!/usr/bin/env python3
# Compare POOLED jet prior (single jet1+jet2 family, NO swap) vs the current
# binned+swap (separate JET1/JET2 families, try both orderings, keep min-chi2).
# Per event, fit 3 ways and compare mW recovery vs Minuit + mutual agreement.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/compare_pool_vs_swap_mp.py [root] [ecm] [maxn] [nproc]
import sys, os, time, numpy as np, uproot
import multiprocessing as mp

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 10000
NPROC=int(sys.argv[4]) if len(sys.argv)>4 else 48

def worker(args):
    rowsD, rowsN, rowsS, rowsP, ecm = args
    os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.dirname(__file__))
    import kinfit_lnuqq_jax as K
    from scipy.optimize import minimize
    chi2=K.build_chi2(ecm); pf=jax.jit(K.build_postfit(ecm))
    f=jax.jit(chi2); g=jax.jit(jax.grad(chi2,argnums=0)); y0=np.zeros(16)
    def fit(Di,PRi):
        fun=lambda y: float(f(jnp.asarray(y),Di,PRi))
        jac=lambda y: np.asarray(g(jnp.asarray(y),Di,PRi),dtype=np.float64)
        r=minimize(fun, y0.copy(), jac=jac, method="BFGS", options={"maxiter":1000,"gtol":1e-5})
        mW=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*r.x[0]; return mW, r.fun, r.x
    # cols: mW_swap, mW_pool, Whad_swap, Whad_pool, chi2_swap, chi2_pool, swapped
    out=np.empty((len(rowsD),7))
    for k in range(len(rowsD)):
        Di=jnp.array(rowsD[k])
        PRnat=jnp.array(rowsN[k]); PRsw=jnp.array(rowsS[k]); PRpl=jnp.array(rowsP[k])
        mWn,cn,xn=fit(Di,PRnat)
        mWs,cs,xs=fit(Di,PRsw)
        swapped=0.0; mW_sw,c_sw,x_sw,PR_sw = mWn,cn,xn,PRnat
        if np.isfinite(cs) and (not np.isfinite(cn) or cs<cn):
            mW_sw,c_sw,x_sw,PR_sw,swapped = mWs,cs,xs,PRsw,1.0
        mWp,cp,xp=fit(Di,PRpl)
        Wh_sw=float(pf(jnp.asarray(x_sw),Di,PR_sw)[0])    # postfit hadronic-W mass
        Wh_pl=float(pf(jnp.asarray(xp),Di,PRpl)[0])
        out[k]=[mW_sw, mWp, Wh_sw, Wh_pl, c_sw, cp, swapped]
    return out

def main():
    t=uproot.open(ROOT)["events"]
    cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
          "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
    mb=["kinfit_valid","kinfit_mW","gen_Whad_m"]
    a=t.arrays(cols+[b for b in mb if b in t.keys()],library="np")
    D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
    ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
    idx=np.where(ok)[0][:MAXN]; D=D[idx]; N=len(D)
    v=a["kinfit_valid"][idx].astype(bool); cmw=a["kinfit_mW"][idx]
    gWh=a["gen_Whad_m"][idx] if "gen_Whad_m" in a else np.full(N,np.nan)
    print(f"[pool-vs-swap] ecm{ECM} N={N} nproc={NPROC}")
    sys.path.insert(0, os.path.dirname(__file__)); import kinfit_lnuqq_jax as K
    PRn,PRs=K.build_prior_packs(ECM,D); PRp=K.build_prior_pack_pool(ECM,D)
    chunks=[(D[i::NPROC],PRn[i::NPROC],PRs[i::NPROC],PRp[i::NPROC],ECM) for i in range(NPROC)]
    order=np.concatenate([np.arange(N)[i::NPROC] for i in range(NPROC)])
    t0=time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        res=pool.map(worker,chunks)
    dt=time.time()-t0
    R=np.empty((N,7)); R[order]=np.concatenate(res,axis=0)
    mW_sw=R[:,0]; mW_pl=R[:,1]; Wh_sw=R[:,2]; Wh_pl=R[:,3]; c_sw=R[:,4]; c_pl=R[:,5]; swapped=R[:,6]
    np.savez("/tmp/mdefranc/lnuqq_pool_vs_swap.npz",
             mW_swap=mW_sw,mW_pool=mW_pl,Whad_swap=Wh_sw,Whad_pool=Wh_pl,
             chi2_swap=c_sw,chi2_pool=c_pl,swapped=swapped,valid=v,mn_mW=cmw,gen_Whad_m=gWh)
    print(f"  saved /tmp/mdefranc/lnuqq_pool_vs_swap.npz  ({dt:.1f}s)")
    fin=np.isfinite(mW_sw)&np.isfinite(mW_pl)&np.isfinite(cmw)
    print(f"\n  swap chosen (binned+swap config): {100*np.mean(swapped):.1f}%")
    print(f"  finite: swap={100*np.mean(np.isfinite(mW_sw)):.1f}%  pool={100*np.mean(np.isfinite(mW_pl)):.1f}%")
    print(f"  mW mean/std: swap={np.mean(mW_sw[fin]):.3f}/{np.std(mW_sw[fin]):.3f}  pool={np.mean(mW_pl[fin]):.3f}/{np.std(mW_pl[fin]):.3f}")
    dpl=mW_pl-cmw; dsw=mW_sw-cmw; gv=fin&v
    print(f"\n=== recovery vs Minuit-valid (N={gv.sum()}) [BIASED: Minuit uses swap] ===")
    for thr in (0.1,0.25,0.5,1.0):
        print(f"  |ΔmW|<{thr}:  swap {100*np.mean(np.abs(dsw[gv])<thr):5.1f}%   pool {100*np.mean(np.abs(dpl[gv])<thr):5.1f}%")
    print(f"  median |ΔmW|:  swap {np.median(np.abs(dsw[gv])):.3f}   pool {np.median(np.abs(dpl[gv])):.3f} GeV")
    # ===== UNBIASED: accuracy vs GEN truth (hadronic W = the jet side) =====
    fg=np.isfinite(Wh_sw)&np.isfinite(Wh_pl)&np.isfinite(gWh)
    eh_sw=Wh_sw-gWh; eh_pl=Wh_pl-gWh
    print(f"\n=== ACCURACY vs GEN (postfit Whad - gen_Whad, N={fg.sum()}) — the decisive test ===")
    print(f"  bias mean:  swap {np.mean(eh_sw[fg]):+.3f}   pool {np.mean(eh_pl[fg]):+.3f} GeV")
    print(f"  RMS:        swap {np.std(eh_sw[fg]):.3f}   pool {np.std(eh_pl[fg]):.3f} GeV")
    print(f"  median|Δ|:  swap {np.median(np.abs(eh_sw[fg])):.3f}   pool {np.median(np.abs(eh_pl[fg])):.3f} GeV")
    for thr in (0.5,1.0,2.0,5.0):
        print(f"  |Whad-gen|<{thr}:  swap {100*np.mean(np.abs(eh_sw[fg])<thr):5.1f}%   pool {100*np.mean(np.abs(eh_pl[fg])<thr):5.1f}%")
    # also on the agreeing core (drop wrong-pairing/ISR tails) to compare resolution
    core=fg&(np.abs(eh_sw)<10)&(np.abs(eh_pl)<10)
    print(f"  core(|Δ|<10) RMS:  swap {np.std(eh_sw[core]):.3f}   pool {np.std(eh_pl[core]):.3f} GeV  (N={core.sum()})")
    dd=mW_pl-mW_sw
    print(f"\n  pool vs swap mutual mW: median|Δ|={np.median(np.abs(dd[fin])):.3f}  |Δ|<0.05={100*np.mean(np.abs(dd[fin])<0.05):.1f}%")

if __name__=="__main__": main()
