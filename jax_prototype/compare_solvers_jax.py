#!/usr/bin/env python3
# (b) Does a Minuit-like CONSERVATIVE solver avoid the low-mW basin that BFGS escapes into?
# All jax-native + vmap/pmap on IDENTICAL events (kfit, gauss closure). Compares:
#   bfgs = jax.scipy BFGS (line-search quasi-Newton; wanders)
#   lm   = make_solver       (LM, conservative lam0 ~ max|diag H| -> gradient-like early,
#                             stays in start basin; built to stop the free-mW runaway)
#   mn   = make_solver_mn    (modified-Newton + Armijo; PD-floored Hessian)
# y0=0 = jets response-corrected, mW=80.4 (the GOOD basin). Reports runaway frac + accuracy.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_ISR_MODE=kfit KF_SIGMA_R=1.5 NDEV=16 python3 jax_prototype/compare_solvers_jax.py [root] [ecm] [maxn]
import sys, os, time, numpy as np
NDEV=int(os.environ.get("NDEV","16"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={NDEV}")
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("KF_ISR_MODE","kfit")
import uproot, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(__file__))
import kinfit_lnuqq_jax as K, jaxfit_common as J

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 10000

c=K.build_chi2(ECM,"kfit")
def chi2_2(y, DP): return c(y, DP[:12], DP[12:])   # pack (D,PR) -> single data arg for the solvers
pf=K.build_postfit(ECM,"kfit")
def pf_2(y, DP): return pf(y, DP[:12], DP[12:])

SOLVERS={
    "bfgs": J.make_bfgs_solver(chi2_2,16,maxiter=500),
    "lm":   J.make_solver(chi2_2,16,niter=150),
    "mn":   J.make_solver_mn(chi2_2,16,niter=100),
}

def main():
    t=uproot.open(ROOT)["events"]
    cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
          "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
    a=t.arrays(cols+(["gen_Whad_m"] if "gen_Whad_m" in t.keys() else []),library="np")
    D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
    ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
    idx=np.where(ok)[0][:MAXN]; D=D[idx]; N=len(D)
    gWh=a["gen_Whad_m"][idx] if "gen_Whad_m" in a else np.full(N,np.nan)
    PR=K.build_prior_pack_pool(ECM,D)
    DP=np.concatenate([D,PR],axis=1)                     # (N,105)
    print(f"[solver-jax] ecm{ECM} N={N} NDEV={NDEV} kfit gauss σ_R={os.environ.get('KF_SIGMA_R','0.45')}")
    ndev=jax.local_device_count(); pad=(-N)%ndev
    DPp=np.concatenate([DP,np.repeat(DP[-1:],pad,axis=0)],axis=0) if pad else DP
    DPp=DPp.reshape(ndev,-1,105)
    res={}
    for name,fit in SOLVERS.items():
        pfit=jax.pmap(jax.vmap(fit))
        t0=time.time(); out=pfit(jnp.asarray(DPp)); jax.block_until_ready(out[0]); dt=time.time()-t0
        y=np.asarray(out[0]).reshape(-1,16)[:N]; c_=np.asarray(out[1]).reshape(-1)[:N]
        mW=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*y[:,0]
        # postfit Whad
        pfb=jax.pmap(jax.vmap(pf_2))
        yp=np.concatenate([y,np.repeat(y[-1:],pad,axis=0)],axis=0).reshape(ndev,-1,16) if pad else y.reshape(ndev,-1,16)
        Wh=np.asarray(pfb(jnp.asarray(yp),jnp.asarray(DPp))).reshape(-1,3)[:N,0]
        fin=np.isfinite(mW)&np.isfinite(c_)
        run=np.abs(mW-80.385)>5
        fg=np.isfinite(Wh)&np.isfinite(gWh); e=(Wh-gWh)[fg]; ec=np.abs(e)<10
        core=fin&~run
        res[name]=(mW,c_,Wh)
        print(f"\n[{name}]  ({dt:.1f}s, {1e3*dt/N:.2f} ms/fit)  finite={100*fin.mean():.1f}%")
        print(f"  runaway(|mW-80.4|>5)={100*run.mean():.1f}%   mW mean={np.mean(mW[fin]):.3f} std={np.std(mW[fin]):.3f}  core std={np.std(mW[core]):.3f}")
        print(f"  Whad-gen: bias={np.mean(e):+.3f} core(|Δ|<10)RMS={np.std(e[ec]):.3f} |Δ|<2={100*np.mean(np.abs(e)<2):.1f}%")
    # cross-solver: do lm/mn keep the bfgs-runaways in the good basin AND at <= chi2?
    mWb,cb,_=res["bfgs"]; rb=np.abs(mWb-80.385)>5
    for name in ("lm","mn"):
        mWs,cs,_=res[name]; keep=rb&np.isfinite(cs)&np.isfinite(cb)
        print(f"\n  on BFGS-runaways (N={rb.sum()}): {name} non-runaway={100*np.mean(np.abs(mWs[keep]-80.385)<=5):.1f}%  "
              f"median({name} χ² − bfgs χ²)={np.median((cs-cb)[keep]):+.2f} (>0 => {name} stayed at higher χ² = chose good basin)")
    np.savez("/tmp/mdefranc/lnuqq_solver_jax.npz",**{f"mW_{k}":res[k][0] for k in res},
             **{f"chi2_{k}":res[k][1] for k in res},**{f"Wh_{k}":res[k][2] for k in res},gen_Whad_m=gWh)

if __name__=="__main__": main()
