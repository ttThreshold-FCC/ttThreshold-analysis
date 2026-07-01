#!/usr/bin/env python3
# jax-BFGS on kfit (gauss σ_R): full diagnostics + outlier autopsy + postfit plots.
# jax-BFGS had the TIGHTEST core (mW std 0.81) & fewest runaways (2.1%) but a std-88
# catastrophic-outlier tail. Question: is the tight core GENUINE (well-converged, low
# |grad|) or under-convergence? And what are the catastrophic outliers (free-mW line-
# search blow-up vs low-mW basin vs non-converged)?
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_ISR_MODE=kfit KF_SIGMA_R=1.5 NDEV=16 python3 jax_prototype/fit_kfit_bfgs.py [root] [ecm] [maxn]
import sys, os, time, numpy as np
NDEV=int(os.environ.get("NDEV","16"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={NDEV}")
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("KF_ISR_MODE","kfit")
import uproot, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(__file__))
import kinfit_lnuqq_jax as K, jaxfit_common as J

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 10000
SR=os.environ.get("KF_SIGMA_R","0.45")

c=K.build_chi2(ECM,"kfit")
def chi2_2(y,DP): return c(y,DP[:12],DP[12:])
pf=K.build_postfit(ECM,"kfit");
def pf_2(y,DP): return pf(y,DP[:12],DP[12:])
gof=K.build_gof(ECM,"kfit")
def gof_2(y,DP): return gof(y,DP[:12],DP[12:])
fit=J.make_bfgs_solver(chi2_2,16,maxiter=500)

def main():
    t=uproot.open(ROOT)["events"]
    cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
          "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
    extra=[b for b in ["gen_Whad_m","gen_Wlep_m","kinfit_valid"] if b in t.keys()]
    a=t.arrays(cols+extra,library="np")
    D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
    ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
    idx=np.where(ok)[0][:MAXN]; D=D[idx]; N=len(D)
    gWh=a["gen_Whad_m"][idx] if "gen_Whad_m" in a else np.full(N,np.nan)
    gWl=a["gen_Wlep_m"][idx] if "gen_Wlep_m" in a else np.full(N,np.nan)
    PR=K.build_prior_pack_pool(ECM,D); DP=np.concatenate([D,PR],axis=1)
    ndev=jax.local_device_count(); pad=(-N)%ndev
    DPp=np.concatenate([DP,np.repeat(DP[-1:],pad,axis=0)],axis=0) if pad else DP
    DPp=DPp.reshape(ndev,-1,105)
    print(f"[kfit jax-BFGS] ecm{ECM} N={N} NDEV={NDEV} σ_R={SR}")
    pfit=jax.pmap(jax.vmap(fit)); t0=time.time()
    y,cc,gn,succ=pfit(jnp.asarray(DPp)); jax.block_until_ready(y); dt=time.time()-t0
    y=np.asarray(y).reshape(-1,16)[:N]; cc=np.asarray(cc).reshape(-1)[:N]
    gn=np.asarray(gn).reshape(-1)[:N]; succ=np.asarray(succ).reshape(-1)[:N].astype(bool)
    yp=np.concatenate([y,np.repeat(y[-1:],pad,axis=0)],axis=0).reshape(ndev,-1,16) if pad else y.reshape(ndev,-1,16)
    M=np.asarray(jax.pmap(jax.vmap(pf_2))(jnp.asarray(yp),jnp.asarray(DPp))).reshape(-1,3)[:N]
    g=np.asarray(jax.pmap(jax.vmap(gof_2))(jnp.asarray(yp),jnp.asarray(DPp))).reshape(-1)[:N]
    mW=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*y[:,0]; Wh,Wl,WW=M[:,0],M[:,1],M[:,2]
    fin=np.isfinite(mW)&np.isfinite(cc)
    run=np.abs(mW-80.385)>5; extreme=np.abs(mW-80.385)>20; core=fin&~run
    print(f"\n  ({dt:.1f}s)  finite={100*fin.mean():.1f}%  BFGS success={100*succ.mean():.1f}%  median|grad|={np.median(gn[fin]):.2e}")
    print(f"  runaway(>5)={100*run.mean():.2f}%  extreme(>20)={100*extreme.mean():.2f}%")
    print(f"  mW core(|Δ|<5): mean={np.mean(mW[core]):.3f} median={np.median(mW[core]):.3f} std={np.std(mW[core]):.3f}")
    # ---- outlier autopsy ----
    print("\n=== OUTLIER AUTOPSY (runaway |mW-80.4|>5) ===")
    oc=run&fin
    print(f"  N={oc.sum()}  success={100*succ[oc].mean():.1f}% (core success={100*succ[core].mean():.1f}%)")
    print(f"  |grad| median outlier={np.median(gn[oc]):.2e} vs core={np.median(gn[core]):.2e}  (high => not converged)")
    print(f"  chi2 median outlier={np.median(cc[oc]):.1f} vs core={np.median(cc[core]):.1f}")
    print(f"  mW sign: low(<75.4)={100*np.mean(mW[oc]<75.4):.0f}%  high(>85.4)={100*np.mean(mW[oc]>85.4):.0f}%")
    print(f"  |y| max comp median outlier={np.median(np.max(np.abs(y[oc]),axis=1)):.1f} vs core={np.median(np.max(np.abs(y[core]),axis=1)):.2f}")
    # which params blow up on extreme events?
    if extreme.sum()>0:
        names=["mW","gW","s1","s2","sl","px","t1","t2","py","p1","p2","k","tlep","plep","besm","besz"]
        big=np.median(np.abs(y[extreme&fin]),axis=0)
        order=np.argsort(big)[::-1][:5]
        print(f"  extreme(>20) N={extreme.sum()}: largest |y| params: "+", ".join(f"{names[i]}={big[i]:.1f}" for i in order))
    np.savez("/tmp/mdefranc/lnuqq_kfit_bfgs.npz",y=y,mW=mW,chi2=cc,gradnorm=gn,success=succ,
             Wh=Wh,Wl=Wl,WW=WW,gof=g,gen_Whad_m=gWh,gen_Wlep_m=gWl,D=D)
    # ---- postfit plots ----
    def cstd(x,lo,hi): xx=x[(x>=lo)&(x<=hi)]; return np.std(xx),np.mean(xx)
    fig,ax=plt.subplots(2,3,figsize=(17,9))
    s,m=cstd(mW,75,86); ax[0,0].hist(mW[(mW>=75)&(mW<=86)],bins=80,range=(75,86),histtype="step",lw=1.8,color="C0")
    ax[0,0].axvline(80.385,color="grey",ls=":"); ax[0,0].set_title(f"mW core (μ={m:.2f} σ={s:.2f})"); ax[0,0].set_xlabel("mW [GeV]"); ax[0,0].grid(alpha=.3)
    ax[0,1].hist(np.clip(mW,40,120),bins=120,range=(40,120),histtype="step",lw=1.8,color="C0")
    ax[0,1].axvline(80.385,color="grey",ls=":"); ax[0,1].set_yscale("log"); ax[0,1].set_title(f"mW full (runaway {100*run.mean():.1f}%, extreme {100*extreme.mean():.2f}%)"); ax[0,1].set_xlabel("mW [GeV]"); ax[0,1].grid(alpha=.3)
    fgh=np.isfinite(Wh)&np.isfinite(gWh); e=(Wh-gWh)[fgh]
    ax[0,2].hist(e[np.abs(e)<15],bins=120,range=(-15,15),histtype="step",lw=1.8,color="C0",label=f"bias={np.mean(e):+.2f} RMS={np.std(e[np.abs(e)<10]):.2f}")
    ax[0,2].set_title("Whad - gen"); ax[0,2].set_xlabel("GeV"); ax[0,2].legend(fontsize=8); ax[0,2].grid(alpha=.3)
    fgl=np.isfinite(Wl)&np.isfinite(gWl); el=(Wl-gWl)[fgl]
    ax[1,0].hist(el[np.abs(el)<20],bins=120,range=(-20,20),histtype="step",lw=1.8,color="C1",label=f"bias={np.mean(el):+.2f} RMS={np.std(el[np.abs(el)<15]):.2f}")
    ax[1,0].set_title("Wlep - gen"); ax[1,0].set_xlabel("GeV"); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.3)
    ax[1,1].scatter(np.clip(mW[fin],40,120),np.log10(np.maximum(gn[fin],1e-12)),s=2,alpha=.2)
    ax[1,1].set_xlabel("mW [GeV]"); ax[1,1].set_ylabel("log10 |grad|"); ax[1,1].set_title("convergence vs mW (outliers high |grad|?)"); ax[1,1].grid(alpha=.3)
    ax[1,2].scatter(np.clip(mW[fin],40,120),np.clip(cc[fin],0,200),s=2,alpha=.2)
    ax[1,2].set_xlabel("mW [GeV]"); ax[1,2].set_ylabel("chi2"); ax[1,2].set_title("chi2 vs mW"); ax[1,2].grid(alpha=.3)
    fig.suptitle(f"kfit jax-BFGS ecm{ECM} σ_R={SR}: core σ(mW)={s:.2f}, runaway {100*run.mean():.1f}%, success {100*succ.mean():.0f}%, med|grad|={np.median(gn[fin]):.1e}",fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.96]); out="/eos/user/m/mdefranc/www/mW/jax_lnuqq"; os.makedirs(out,exist_ok=True)
    fig.savefig(f"{out}/kfit_bfgs_postfit_ecm{ECM}.png",dpi=110); print(f"\nwrote {out}/kfit_bfgs_postfit_ecm{ECM}.png")

if __name__=="__main__": main()
