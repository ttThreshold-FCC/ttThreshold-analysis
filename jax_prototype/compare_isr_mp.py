#!/usr/bin/env python3
# Compare ISR models on identical events (POOLED jet prior, BFGS). Modes:
#   mloss  = OLD spike+barrier on m_WW-ECM (punishes ISR)
#   photon = collinear-photon energy residual, neutrino still FREE (3 MET params)
#   kfit   = NEWEST: explicit ISR (px_γ,py_γ,k) + neutrino DERIVED from 4-mom
#            conservation (removes redundant nu_pz<->k freedom -> kill runaway tail)
# Reports per mode: GoF p~0 pile-up, free-mW (mean/std + runaway tail), accuracy vs
# gen (postfit Whad - gen_Whad). Goal: kfit collapses BOTH the p~0 tail and the mW
# runaway tail without biasing the core.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   MODES=photon,kfit python3 jax_prototype/compare_isr_mp.py [root] [ecm] [maxn] [nproc]
import sys, os, time, numpy as np, uproot
import multiprocessing as mp

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 10000
NPROC=int(sys.argv[4]) if len(sys.argv)>4 else 48
MODES=os.environ.get("MODES","photon,kfit").split(",")

def worker(args):
    rowsD, rowsP, ecm, modes = args
    os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.dirname(__file__))
    import kinfit_lnuqq_jax as K
    from scipy.optimize import minimize
    fns={}
    for mode in modes:
        c=K.build_chi2(ecm,mode)
        fns[mode]=(jax.jit(c), jax.jit(jax.grad(c,argnums=0)),
                   jax.jit(K.build_gof(ecm,mode)), jax.jit(K.build_postfit(ecm,mode)))
    y0=np.zeros(16)
    def fit(f,g,Di,PRi):
        r=minimize(lambda y:float(f(jnp.asarray(y),Di,PRi)),y0.copy(),
                   jac=lambda y:np.asarray(g(jnp.asarray(y),Di,PRi),dtype=np.float64),
                   method="BFGS",options={"maxiter":1000,"gtol":1e-5})
        return r.x
    out=np.empty((len(rowsD),3*len(modes)))  # per mode: mW, Whad, gof
    for k in range(len(rowsD)):
        Di=jnp.array(rowsD[k]); PRi=jnp.array(rowsP[k]); row=[]
        for mode in modes:
            f,g,gof,pf=fns[mode]; x=fit(f,g,Di,PRi)
            mW=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*x[0]
            Wh=float(pf(jnp.asarray(x),Di,PRi)[0]); gf=float(gof(jnp.asarray(x),Di,PRi))
            row+=[mW,Wh,gf]
        out[k]=row
    return out

def main():
    t=uproot.open(ROOT)["events"]
    cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
          "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
    mb=["kinfit_valid","gen_Whad_m"]
    a=t.arrays(cols+[b for b in mb if b in t.keys()],library="np")
    D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
    ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
    idx=np.where(ok)[0][:MAXN]; D=D[idx]; N=len(D)
    v=a["kinfit_valid"][idx].astype(bool) if "kinfit_valid" in a else np.zeros(N,bool)
    gWh=a["gen_Whad_m"][idx] if "gen_Whad_m" in a else np.full(N,np.nan)
    print(f"[ISR compare] ecm{ECM} N={N} nproc={NPROC}  modes={MODES}  SIGMA_R={os.environ.get('KF_SIGMA_R','0.45')} (pooled prior)")
    sys.path.insert(0, os.path.dirname(__file__)); import kinfit_lnuqq_jax as K
    PRp=K.build_prior_pack_pool(ECM,D)
    chunks=[(D[i::NPROC],PRp[i::NPROC],ECM,MODES) for i in range(NPROC)]
    order=np.concatenate([np.arange(N)[i::NPROC] for i in range(NPROC)])
    t0=time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        res=pool.map(worker,chunks)
    dt=time.time()-t0
    R=np.empty((N,3*len(MODES))); R[order]=np.concatenate(res,axis=0)
    cols_out={}
    for j,mode in enumerate(MODES):
        cols_out[f"mW_{mode}"]=R[:,3*j]; cols_out[f"Whad_{mode}"]=R[:,3*j+1]; cols_out[f"gof_{mode}"]=R[:,3*j+2]
    TAG=os.environ.get("OUT_TAG","")
    np.savez(f"/tmp/mdefranc/lnuqq_isr_compare{TAG}.npz",valid=v,gen_Whad_m=gWh,modes=np.array(MODES),**cols_out)
    print(f"  saved /tmp/mdefranc/lnuqq_isr_compare{TAG}.npz  ({dt:.1f}s, {1e3*dt/N/len(MODES):.2f} ms/fit)")
    from scipy.stats import chi2 as chi2dist
    print("\n=== per-mode summary ===")
    for mode in MODES:
        mW=cols_out[f"mW_{mode}"]; Wh=cols_out[f"Whad_{mode}"]; g=cols_out[f"gof_{mode}"]
        fin=np.isfinite(g)&(g>0)&(g<1e4); gg=g[fin]; kk=max(1,int(round(np.mean(gg)))); pv=chi2dist.sf(gg,kk)
        fm=np.isfinite(mW); mWf=mW[fm]
        run=np.abs(mWf-80.385)>5; core=~run
        fg=np.isfinite(Wh)&np.isfinite(gWh); e=(Wh-gWh)[fg]; ec=np.abs(e)<10
        print(f"\n[{mode}]")
        print(f"  GoF mean={np.mean(gg):.2f} k={kk}  p<0.01={100*np.mean(pv<0.01):.1f}%  p<0.05={100*np.mean(pv<0.05):.1f}%")
        print(f"  free mW: mean={np.mean(mWf):.3f} std={np.std(mWf):.3f}  runaway(|mW-80.4|>5)={100*np.mean(run):.1f}%")
        print(f"           core mW: mean={np.mean(mWf[core]):.3f} median={np.median(mWf[core]):.3f} std={np.std(mWf[core]):.3f}")
        print(f"  Whad-gen: bias={np.mean(e):+.3f} RMS={np.std(e):.3f} med|Δ|={np.median(np.abs(e)):.3f} "
              f"core(|Δ|<10)RMS={np.std(e[ec]):.3f}  |Δ|<2={100*np.mean(np.abs(e)<2):.1f}%")

if __name__=="__main__": main()
