#!/usr/bin/env python3
# ── HARD-CONSTRAINT fit on RECO (the real convergence test) ───────────────────
# On reco the jets ARE smeared, so jet-scale nuisances are legitimate (unlike gen quarks). Keep
# them; drop the soft ISR/BES freedom and impose HARD 3-momentum conservation at (√s,0,0,0):
# the neutrino is fully determined by the (scaled) visible system. Free params (4): mW + jet
# scales s1,s2 + lepton scale sl (angles fixed at reco — small resolution). The hadronic & leptonic
# W masses follow from the scales; mW is pinned by the double BW + log_Z; the scales by their DCB
# priors. No soft ISR direction ⇒ should converge ~100%. Compare convergence + m̂W distribution to
# the production SOFT free-mW fit stored in the tree (kinfit_mW / kinfit_valid, ~75% valid).
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   MAXN=40000 NDEV=32 python3 jax_prototype/kinfit_reco_hard.py [root] [ecm]
import sys, os
NDEV=int(os.environ.get("NDEV","32"))
os.environ.setdefault("XLA_FLAGS",f"--xla_force_host_platform_device_count={NDEV}"); os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, jax, jax.numpy as jnp, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); import jaxfit_common as J, kinfit_lnuqq_jax as K
jax.config.update("jax_enable_x64", True); np.set_printoptions(linewidth=160, suppress=True)
MW_INIT=80.419; GW=2.049; MU=0.1056583745; POLE=80.385
ROOT=sys.argv[1] if len(sys.argv)>1 else \
  "outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160; MAXN=int(os.environ.get("MAXN","40000"))
NUMODE=os.environ.get("NUMODE","energy")   # "mom"=ν from 3-momentum (energy free) ; "energy"=ν from energy conservation
OUT="/eos/user/m/mdefranc/www/mW/kinfit_correlations"

t=uproot.open(ROOT)["events"]
RECO=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
      "reco_lep_p","reco_lep_theta","reco_lep_phi"]
a=t.arrays(RECO+["kinfit_mW","kinfit_valid","kinfit_status"], library="np")
D=np.stack([a[c] for c in RECO],1).astype(np.float64)
ok=np.all(np.isfinite(D),1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)
idx=np.where(ok)[0];
if MAXN>0: idx=idx[:MAXN]
D=D[idx]; a={k:v[idx] for k,v in a.items()}; N=len(idx)
PR=K.build_prior_pack_pool(ECM, D)                  # pooled jet + lep p-resp priors at reco kin
pj1=PR[:,0:10]; pj2=PR[:,30:40]; pl=PR[:,60:73]     # DcbGauss(10) jets, DcbExpRight3Gauss(13) lep
mu_j1=pj1[:,0]; mu_j2=pj2[:,0]; mu_l=pl[:,0]

def vec(p,th,ph,m=0.0):
    st=jnp.sin(th); return jnp.array([jnp.sqrt(p*p+m*m),p*st*jnp.cos(ph),p*st*jnp.sin(ph),p*jnp.cos(th)])
def Mass(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))
def chi2(y, Dd, pr1,pr2,prl):
    j1p,j1t,j1f,j2p,j2t,j2f,lp,lt,lf=[Dd[i] for i in range(9)]
    mW=MW_INIT+2.0*y[0]; s1=y[1]; s2=y[2]; sl=y[3]
    bad=(s1<=0.05)|(s2<=0.05)|(sl<=0.05)
    s1=jnp.maximum(s1,0.05); s2=jnp.maximum(s2,0.05); sl=jnp.maximum(sl,0.05)
    J1=vec(j1p/s1,j1t,j1f); J2=vec(j2p/s2,j2t,j2f); L=vec(lp/sl,lt,lf,MU)
    Vis=J1+J2+L; nux=-Vis[1]; nuy=-Vis[2]
    pen=0.0
    if NUMODE=="energy":
        # ν from ENERGY conservation: E_ν = √s − E_vis ; ν_z from massless condition (±, pick
        # root closest to −Vis_z). Energy constraint couples E_ν to the jet scale => pins the
        # common visible-energy scale (breaks the jet-scale↔mW flat direction).
        Enu=ECM-Vis[0]; disc=Enu*Enu-(nux*nux+nuy*nuy)
        r=jnp.sqrt(jnp.maximum(disc,1e-12)); nuz=jnp.where(-Vis[3]>=0.0,r,-r)
        pen=jnp.where(disc<0.0,(disc/1.0)**2,0.0)+jnp.where(Enu<0.0,(Enu)**2*100.0,0.0)
    else:
        nuz=-Vis[3]; Enu=jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12)
    Nu=jnp.array([Enu,nux,nuy,nuz]); Wl=L+Nu; Wh=J1+J2; WW=Vis+Nu
    mh=Mass(Wh); ml=Mass(Wl); mwgw=mW*GW; dh=mh*mh-mW*mW; dl=ml*ml-mW*mW
    bwh=mwgw/(dh*dh+mwgw*mwgw); bwl=mwgw/(dl*dl+mwgw*mwgw)
    s_ww=WW[0]**2-(WW[1]**2+WW[2]**2+WW[3]**2); m_WW=jnp.sqrt(jnp.maximum(s_ww,1e-12))
    bw=-2.0*(jnp.log(bwh)+jnp.log(bwl))+2.0*J.log_Z_ontf(m_WW,mW,GW)
    sp=J.dcb_gauss_n2ll(s1,pr1)+J.dcb_gauss_n2ll(s2,pr2)+J.dcb_er3g_n2ll(sl,prl)
    return bw+sp+pen+jnp.where(bad,1e3,0.0)

def make_fit(niter=70,nls=25,ftol=1e-3,c1=1e-4):
    g=jax.grad(chi2); h=jax.hessian(chi2)
    def fit(y0,Dd,p1,p2,pl_):
        def body(k,carry):
            y,c=carry; gg=g(y,Dd,p1,p2,pl_); H=h(y,Dd,p1,p2,pl_); H=0.5*(H+H.T)
            w,V=jnp.linalg.eigh(H); wf=jnp.maximum(w,ftol*jnp.max(jnp.abs(w)))
            d=-(V@((V.T@gg)/wf)); gd=jnp.dot(gg,d)
            def ls(j,st):
                aa,f,cb,yb=st; yt=y+aa*d; ct=chi2(yt,Dd,p1,p2,pl_)
                okk=jnp.isfinite(ct)&(ct<=c+c1*aa*gd)&(~f)
                return (aa*0.5,f|okk,jnp.where(okk,ct,cb),jnp.where(okk,yt,yb))
            _,_,c,y=jax.lax.fori_loop(0,nls,ls,(1.0,False,c,y)); return (y,c)
        y,c=jax.lax.fori_loop(0,niter,body,(y0,chi2(y0,Dd,p1,p2,pl_)))
        gg=g(y,Dd,p1,p2,pl_); H=h(y,Dd,p1,p2,pl_); H=0.5*(H+H.T); w,V=jnp.linalg.eigh(H)
        edm=0.5*jnp.dot(gg,V@((V.T@gg)/jnp.where(w>1e-9,w,jnp.inf)))
        H00=H[0,0]; Hr0=H[1:,0]; cmw=H00-Hr0@jnp.linalg.solve(H[1:,1:],Hr0)
        return y,c,jnp.linalg.norm(gg),edm,cmw
    return jax.pmap(jax.vmap(fit))
fit=make_fit()
y0=np.zeros((N,4)); y0[:,1]=mu_j1; y0[:,2]=mu_j2; y0[:,3]=mu_l
pad=(-N)%NDEV
def pp(x):
    x=np.concatenate([x,np.repeat(x[-1:],pad,0)],0) if pad else x; return x.reshape(NDEV,-1,*x.shape[1:])
import time; t0=time.time()
out=fit(jnp.asarray(pp(y0)),jnp.asarray(pp(D)),jnp.asarray(pp(pj1)),jnp.asarray(pp(pj2)),jnp.asarray(pp(pl)))
f=lambda z: np.asarray(z).reshape(-1,*z.shape[2:])[:N]; y,c,gn,edm,cmw=[f(z) for z in out]; dt=time.time()-t0
mW=MW_INIT+2.0*y[:,0]; fin=np.isfinite(mW)&np.isfinite(edm)
conv=fin&(np.abs(edm)<1e-2)
print(f"[reco-hard] NUMODE={NUMODE} N={N} ecm{ECM} ({dt:.1f}s, {1e3*dt/N:.2f} ms/fit)")
print(f"  HARD-CONSTRAINT: convergence(EDM<1e-2)={100*conv.mean():.2f}%  mW-convex={100*np.mean(cmw[fin]>0):.1f}%")
print(f"    m̂W(conv): mean={mW[conv].mean():.3f} median={np.median(mW[conv]):.3f} std={mW[conv].std():.3f} bias={np.median(mW[conv])-POLE:+.3f}")
vs=a["kinfit_valid"].astype(bool); cm=a["kinfit_mW"]
print(f"  PRODUCTION SOFT (tree): valid={100*vs.mean():.2f}%  mW(valid) mean={cm[vs].mean():.3f} median={np.median(cm[vs]):.3f} std={cm[vs].std():.3f}")

fig,ax=plt.subplots(1,2,figsize=(14,5.2))
ax[0].hist(cm[vs&np.isfinite(cm)],bins=100,range=(74,90),histtype="step",lw=2,color="C3",
           label=f"SOFT free-mW (valid={100*vs.mean():.0f}%)\nmed={np.median(cm[vs]):.2f} std={cm[vs].std():.2f}")
ax[0].hist(mW[conv],bins=100,range=(74,90),histtype="step",lw=2,color="C0",
           label=f"HARD constraint (conv={100*conv.mean():.0f}%)\nmed={np.median(mW[conv]):.2f} std={mW[conv].std():.2f}")
ax[0].axvline(POLE,color="k",ls="--",lw=1,label=f"pole {POLE}")
ax[0].set_xlabel(r"per-event $\hat m_W$ [GeV]"); ax[0].set_ylabel("events"); ax[0].set_xlim(74,90)
ax[0].set_title(f"RECO ℓνqq: hard vs soft (ecm{ECM}, N={N})"); ax[0].legend(fontsize=9)
# zoom core
ax[1].hist(cm[vs&np.isfinite(cm)],bins=80,range=(76,86),density=True,histtype="step",lw=2,color="C3",label="SOFT (valid)")
ax[1].hist(mW[conv],bins=80,range=(76,86),density=True,histtype="step",lw=2,color="C0",label="HARD")
ax[1].axvline(POLE,color="k",ls="--",lw=1); ax[1].set_xlabel(r"$\hat m_W$ [GeV]"); ax[1].set_ylabel("normalized")
ax[1].set_title("core (normalized)"); ax[1].legend(fontsize=9); ax[1].set_xlim(76,86)
plt.tight_layout(); png=f"{OUT}/mW_reco_hard_{NUMODE}_ecm{ECM}.png"; plt.savefig(png,dpi=110)
print(f"\n[plot] {png}")
