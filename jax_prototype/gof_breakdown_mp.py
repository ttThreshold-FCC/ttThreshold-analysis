#!/usr/bin/env python3
# Diagnose the p-value~0 pile-up in the lnuqq GoF: re-fit events (binned+swap, BFGS),
# and for the winning fit record the PER-TERM mode-referenced residuals
# (bw,isr,mloss,bes,gw,scale,ang). Lets us see which contributions drive the
# high-GoF (low-p) tail, and split GoF by m_loss / Minuit-valid / swap.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/gof_breakdown_mp.py [root] [ecm] [maxn] [nproc]
import sys, os, time, numpy as np, uproot
import multiprocessing as mp

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
MAXN=int(sys.argv[3]) if len(sys.argv)>3 else 10000
NPROC=int(sys.argv[4]) if len(sys.argv)>4 else 48

def worker(args):
    rowsD, rowsN, rowsS, ecm = args
    os.environ["XLA_FLAGS"]="--xla_force_host_platform_device_count=1"
    os.environ["OMP_NUM_THREADS"]="1"; os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    sys.path.insert(0, os.path.dirname(__file__))
    import kinfit_lnuqq_jax as K
    from scipy.optimize import minimize
    chi2=K.build_chi2(ecm); pf=jax.jit(K.build_postfit(ecm))
    gof=jax.jit(K.build_gof(ecm)); gterms=jax.jit(K.build_gof_terms(ecm))
    f=jax.jit(chi2); g=jax.jit(jax.grad(chi2,argnums=0))
    y0=np.zeros(16)
    def fit_one(Di,PRi):
        fun=lambda y: float(f(jnp.asarray(y),Di,PRi))
        jac=lambda y: np.asarray(g(jnp.asarray(y),Di,PRi),dtype=np.float64)
        return minimize(fun, y0.copy(), jac=jac, method="BFGS", options={"maxiter":1000,"gtol":1e-5})
    NT=len(K.GOF_TERMS)
    out=np.empty((len(rowsD), 5+NT))   # mW, gof, m_loss, swapped, mh, + NT terms
    for k in range(len(rowsD)):
        Di=jnp.array(rowsD[k]); PRn=jnp.array(rowsN[k])
        r=fit_one(Di,PRn); PRu=PRn; swapped=0.0
        PRw=jnp.array(rowsS[k]); rs=fit_one(Di,PRw)
        if np.isfinite(rs.fun) and (not np.isfinite(r.fun) or rs.fun<r.fun):
            r=rs; PRu=PRw; swapped=1.0
        yx=jnp.asarray(r.x); m=np.asarray(pf(yx,Di,PRu)); gf=float(gof(yx,Di,PRu))
        terms=np.asarray(gterms(yx,Di,PRu),dtype=np.float64)
        mW=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*r.x[0]
        m_loss=m[2]-(ecm+K.KF_MW_PHYS_SIGMA*0.0)   # mWW - ecm (besm~0 at start ref; approx)
        out[k]=[mW, gf, m_loss, swapped, m[0]]+list(terms)
    return out

def main():
    t=uproot.open(ROOT)["events"]
    cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
          "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
    mb=["kinfit_valid","kinfit_valid_loose","kinfit_mW","gen_WW_m"]
    a=t.arrays(cols+[b for b in mb if b in t.keys()],library="np")
    D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
    ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
    idx=np.where(ok)[0][:MAXN]; D=D[idx]; N=len(D)
    v=a["kinfit_valid"][idx].astype(bool) if "kinfit_valid" in a else np.zeros(N,bool)
    print(f"[gof breakdown] ecm{ECM} N={N} nproc={NPROC}")
    sys.path.insert(0, os.path.dirname(__file__)); import kinfit_lnuqq_jax as K
    PRn, PRs = K.build_prior_packs(ECM, D)
    chunks=[(D[i::NPROC], PRn[i::NPROC], PRs[i::NPROC], ECM) for i in range(NPROC)]
    order=np.concatenate([np.arange(N)[i::NPROC] for i in range(NPROC)])
    NT=len(K.GOF_TERMS); W=5+NT
    t0=time.time()
    with mp.get_context("spawn").Pool(NPROC) as pool:
        res=pool.map(worker, chunks)
    dt=time.time()-t0
    R=np.empty((N,W)); R[order]=np.concatenate(res,axis=0)
    mW=R[:,0]; gof=R[:,1]; mloss=R[:,2]; swapped=R[:,3]; mh=R[:,4]; terms=R[:,5:]
    np.savez("/tmp/mdefranc/lnuqq_gof_breakdown.npz",
             mW=mW, gof=gof, mloss=mloss, swapped=swapped, mh=mh, terms=terms,
             term_names=np.array(K.GOF_TERMS), valid=v)
    print(f"  saved /tmp/mdefranc/lnuqq_gof_breakdown.npz  ({dt:.1f}s, {N/dt:.0f} fits/s)")
    fin=np.isfinite(gof)&(gof<1e4)
    g=gof[fin]; k=max(1,int(round(np.mean(g))))
    from scipy.stats import chi2 as chi2dist
    pv=chi2dist.sf(g,k)
    lowp=pv<0.01
    print(f"\n=== GoF tail diagnosis (ndof ref k={k}) ===")
    print(f"  GoF mean={np.mean(g):.2f} median={np.median(g):.2f}  90%={np.percentile(g,90):.1f} 99%={np.percentile(g,99):.1f}")
    print(f"  fraction p<0.01: {100*np.mean(lowp):.1f}%   p<0.05: {100*np.mean(pv<0.05):.1f}%")
    print(f"\n  mean per-term residual  (all finite vs low-p<0.01):")
    tn=K.GOF_TERMS
    Tf=terms[fin]
    for j,name in enumerate(tn):
        print(f"   {name:7s}  all={np.mean(Tf[:,j]):7.2f}   low-p={np.mean(Tf[lowp][:,j]):8.2f}   "
              f"frac-of-GoF(low-p)={100*np.sum(Tf[lowp][:,j])/np.sum(g[lowp]):5.1f}%")
    # which single term is the dominant contributor per low-p event
    dom=np.argmax(Tf[lowp],axis=1)
    print(f"\n  dominant term among low-p (p<0.01) events:")
    for j,name in enumerate(tn):
        frac=100*np.mean(dom==j)
        if frac>0: print(f"   {name:7s}: {frac:5.1f}% of low-p events")
    # correlation of GoF with m_loss magnitude
    print(f"\n  median |m_loss|: all={np.median(np.abs(mloss[fin])):.2f}  low-p={np.median(np.abs(mloss[fin][lowp])):.2f} GeV")
    print(f"  Minuit-valid fraction: all={100*np.mean(v[fin]):.1f}%  low-p={100*np.mean(v[fin][lowp]):.1f}%")
    print(f"  swap fraction:         all={100*np.mean(swapped[fin]):.1f}%  low-p={100*np.mean(swapped[fin][lowp]):.1f}%")

if __name__=="__main__": main()
