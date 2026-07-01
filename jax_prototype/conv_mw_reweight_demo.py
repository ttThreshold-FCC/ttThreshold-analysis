#!/usr/bin/env python3
# ── VISUAL: how a FIXED MC sample is morphed to ANY trial mW by reweighting ────
# The MC is generated at one mW0. We never re-simulate. For a trial mW each event gets weight
# w_i(mW)=P_true(gen_i;mW)/P_true(gen_i;mW0) (the gen lineshape ratio — detector response cancels).
# Panels:
#  (A) the weight w vs gen m_qq for three trial mW — it tilts the sample toward the new peak;
#  (B) the gen m_qq distribution, nominal vs reweighted — the BW peak slides to the trial mW;
#  (C) the RECO m_qq TEMPLATE (what the fit compares to data) sliding with mW, same fixed MC.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_reweight_demo.py
import sys, os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"jax_prototype"); import jaxfit_common as J
GW=2.085; ECM=160
GLX=np.asarray(J.GL_X); GLW=np.asarray(J.GL_W)
def log_Z(m,mW,gW=GW):
    m=np.atleast_1d(np.asarray(m,float)); mwgw=mW*gW; mW2=mW*mW; s=m*m
    tmn=np.arctan(-mW2/mwgw); tmx=np.arctan((s-mW2)/mwgw); hd=0.5*(tmx-tmn); hs=0.5*(tmx+tmn)
    t=hd[:,None]*GLX[None,:]+hs[:,None]; mm=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1/mm
    mh=mm[:,:,None]; ml=mm[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=s[:,None,None]
    lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2); ig=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il/(4*sE),0)
    W=GLW[:,None]*GLW[None,:]; return np.log(np.maximum(np.sum(W[None]*ig,axis=(1,2))*hd*hd,1e-300))
def bw(m,mW,gW=GW): d=m*m-mW*mW; mwgw=mW*gW; return mwgw/(d*d+mwgw*mwgw)

F=f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
t=uproot.open(F)["events"]; a=t.arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_Whad_m","reco_Wlep_m"],library="np")
ok=np.isfinite(a["gen_Whad_m"])&np.isfinite(a["reco_Whad_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"])
qg=a["gen_Whad_m"][ok]; lg=a["gen_Wlep_m"][ok]; wg=a["gen_WW_m"][ok]; qr=a["reco_Whad_m"][ok]; lr=a["reco_Wlep_m"][ok]
N=len(qg)
# grid true-level model (marginalized over the measured s' spectrum), for the per-event weights
TLO,THI,NT=28.,96.,137; tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")
cnt,edg=np.histogram(wg,bins=60,range=(ECM-45,ECM+1.5)); SN=0.5*(edg[:-1]+edg[1:]); SW=cnt.astype(float); m=SW>0; SN=SN[m]; SW=SW[m]
from scipy.interpolate import RegularGridInterpolator
def Ptrue_interp(mW):
    BW=bw(tg,mW)[:,None]*bw(tg,mW)[None,:]; u=np.zeros((NT,NT)); lz=log_Z(SN,mW)
    for k,(sp,wk) in enumerate(zip(SN,SW)):
        s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2); PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0); u+=wk/np.exp(lz[k])*BW*PS
    return RegularGridInterpolator((tg,tg),u/(u.sum()*dt*dt),bounds_error=False,fill_value=1e-300)
# reference mW0 = gen ensemble best fit (weights = 1 here)
mwscan=np.linspace(79,81.5,41); P0=[Ptrue_interp(m) for m in mwscan]
Lg=np.array([-2*np.log(np.maximum(p(np.stack([qg,lg],1)),1e-300)) for p in P0]).T
mW0=mwscan[int(np.argmin(Lg.sum(0)))]
f0=Ptrue_interp(mW0); p0=np.maximum(f0(np.stack([qg,lg],1)),1e-300)
TRIALS=[(78.5,"C0"),(mW0,"C2"),(82.0,"C3")]                   # wide spread so the morph is visible
print(f"mW0(ref, weights=1)={mW0:.3f}; trial mW shown: {[t[0] for t in TRIALS]}")

fig,ax=plt.subplots(1,3,figsize=(19,5.4))
qbins=np.arange(40,92,1.5); qc=0.5*(qbins[:-1]+qbins[1:])
# data reco (nominal, fixed) for panel C
datc,_=np.histogram(qr,bins=qbins,density=True)
for mW,col in TRIALS:
    f=Ptrue_interp(mW); w=np.maximum(f(np.stack([qg,lg],1)),1e-300)/p0
    w=np.minimum(w,np.percentile(w,99.5))
    # (A) weight vs gen m_qq (bin-averaged)
    wb=np.array([w[(qg>=qbins[i])&(qg<qbins[i+1])].mean() if ((qg>=qbins[i])&(qg<qbins[i+1])).sum()>20 else np.nan for i in range(len(qbins)-1)])
    ax[0].plot(qc,wb,"-",color=col,lw=2,label=f"trial mW={mW:.2f}")
    # (B) gen m_qq reweighted
    gh,_=np.histogram(qg,bins=qbins,weights=w,density=True)
    ax[1].step(qc,gh,where="mid",color=col,lw=2,label=f"mW={mW:.2f}")
    ax[1].axvline(mW,color=col,ls=":",lw=1)
    # (C) reco m_qq template reweighted (the actual fit input)
    rh,_=np.histogram(qr,bins=qbins,weights=w,density=True)
    ax[2].step(qc,rh,where="mid",color=col,lw=2,label=f"template @ mW={mW:.2f}")
ax[0].axhline(1,color="k",ls="--",lw=1); ax[0].set_xlabel("gen m_qq [GeV]"); ax[0].set_ylabel("event weight w(mW)")
ax[0].set_title("(A) reweighting tilts the FIXED sample"); ax[0].legend(); ax[0].set_ylim(0,2.2)
# nominal gen (unweighted) for reference
g0,_=np.histogram(qg,bins=qbins,density=True); ax[1].step(qc,g0,where="mid",color="k",lw=1.3,ls="--",label="nominal (no reweight)")
ax[1].set_xlabel("gen m_qq [GeV]"); ax[1].set_ylabel("normalized"); ax[1].set_title("(B) gen peak slides to trial mW"); ax[1].legend()
ax[2].step(qc,datc,where="mid",color="k",lw=2.5,label="DATA (reco, fixed)")
ax[2].set_xlabel("reco m_qq [GeV]"); ax[2].set_ylabel("normalized")
ax[2].set_title("(C) reco TEMPLATE slides; fit picks the mW matching data"); ax[2].legend()
plt.tight_layout(); png="/eos/user/m/mdefranc/www/mW/conv_mw/reweight_demo_ecm160.png"; plt.savefig(png,dpi=110)
print(f"[plot] {png}")
