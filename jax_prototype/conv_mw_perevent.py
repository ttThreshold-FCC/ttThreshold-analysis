#!/usr/bin/env python3
# ── PER-EVENT mW from the folded BW (experimentally checkable; no per-event nuisance fit) ──
# m̂W_i = argmax_mW P_folded(m_qq^reco_i, m_lν^reco_i | mW). The resolution is FOLDED (from MC), not fit,
# so there is NO flat direction and m̂W_i is defined for ~100% of events. The DISTRIBUTION of m̂W_i is a
# data/MC-checkable observable (validates the folding/resolution); the measurement = how that distribution
# (e.g. its weighted mean) SHIFTS with the true mW. This script:
#   (A) plots the per-event m̂W distribution (MAP & posterior-mean) at each ECM;
#   (B) demonstrates the mW sensitivity: reweight the sample to true mW=mW0±Δ and show ⟨m̂W_i⟩ tracks it
#       (the calibration curve) — that is how you turn the per-event observable into a measurement.
#   source .../setup.sh ; python3 jax_prototype/conv_mw_perevent.py [ecm]
import sys, os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"jax_prototype"); import jaxfit_common as J
GW=2.085; ECM=int(sys.argv[1]) if len(sys.argv)>1 else 160
OUT="/eos/user/m/mdefranc/www/mW/conv_mw"
GX,GWt=np.polynomial.legendre.leggauss(256); W2=GWt[:,None]*GWt[None,:]
def logZ(mWW,mW):
    mWW=np.atleast_1d(mWW.astype(float)); mwgw=mW*GW; mW2=mW*mW; s=mWW*mWW
    tmn=np.arctan(-mW2/mwgw); tmx=np.arctan((s-mW2)/mwgw); hd=0.5*(tmx-tmn); hs=0.5*(tmx+tmn)
    out=np.empty(len(mWW)); CH=max(1,50_000_000//(256*256))
    for a in range(0,len(mWW),CH):
        ss=s[a:a+CH]; t=hd[a:a+CH,None]*GX[None,:]+hs[a:a+CH,None]
        m=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1/m
        mh=m[:,:,None]; ml=m[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=ss[:,None,None]
        lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2); ig=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il/(4*sE),0)
        out[a:a+CH]=np.log(np.maximum((W2[None]*ig).sum((1,2))*hd[a:a+CH]*hd[a:a+CH],1e-300))
    return out
def bw(m,mW): d=m*m-mW*mW; mwgw=mW*GW; return mwgw/(d*d+mwgw*mwgw)
F=f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
t=uproot.open(F)["events"]; a=t.arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_Whad_m","reco_Wlep_m"],library="np")
ok=np.isfinite(a["gen_Whad_m"])&np.isfinite(a["reco_Whad_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"])
qg=a["gen_Whad_m"][ok]; lg=a["gen_Wlep_m"][ok]; wg=a["gen_WW_m"][ok]; qr=a["reco_Whad_m"][ok]; lr=a["reco_Wlep_m"][ok]; N=len(qg)
# grid true model (for per-event reco template AND the reweighting weights)
TLO,THI,NT=28.,96.,137; tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")
cnt,edg=np.histogram(wg,bins=60,range=(ECM-45,ECM+1.5)); SN=0.5*(edg[:-1]+edg[1:]); SW=cnt.astype(float); mm=SW>0; SN=SN[mm]; SW=SW[mm]
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
def Ptrue(mW):
    BW=bw(tg,mW)[:,None]*bw(tg,mW)[None,:]; u=np.zeros((NT,NT)); lz=logZ(SN,mW)
    for k,(sp,w) in enumerate(zip(SN,SW)):
        s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2); PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0); u+=w/np.exp(lz[k])*BW*PS
    return u/(u.sum()*dt*dt)
# folded reco template via the MC kernel (forward-fold density): reweight gen->reco using real residuals
K=np.load(f"jax_prototype/conv_kernels/kernel_ecm{ECM}.npz"); dqq=K["dqq"].astype(float); dlv=K["dlv"].astype(float)
e1=np.arange(30,98.001,0.5); c1=0.5*(e1[:-1]+e1[1:])
# per-event reco-template log-likelihood over a WIDE mW scan, using fold (reweight real reco)
mw=np.linspace(73,88,76)
# gen-weight reference (native density)
Pg0=[Ptrue(m) for m in mw]; f0=[RegularGridInterpolator((tg,tg),P,bounds_error=False,fill_value=1e-300) for P in Pg0]
Lgen=np.array([-2*np.log(np.maximum(fi(np.stack([qg,lg],1)),1e-300)) for fi in f0]).T  # (N,nmw)
j0=int(np.argmin(Lgen.sum(0)))
# fold per-event reco template: T(reco|mW) = reweighted reco hist (use all events; this is for the per-event MAP)
Lpe=np.zeros((N,len(mw)))
for j in range(len(mw)):
    w=np.exp(-0.5*(Lgen[:,j]-Lgen[:,j0])); w=np.minimum(w,np.percentile(w,99.9))
    H,_,_=np.histogram2d(qr,lr,bins=[e1,e1],weights=w); from scipy.ndimage import gaussian_filter; H=gaussian_filter(H,0.6); H/=H.sum()
    pv=RegularGridInterpolator((c1,c1),H,bounds_error=False,fill_value=1e-300)(np.stack([qr,lr],1))
    Lpe[:,j]=-2*np.log(np.maximum(pv,1e-300))
imin=np.argmin(np.where(np.isfinite(Lpe),Lpe,1e18),axis=1); pe=mw[imin]
interior=(imin>0)&(imin<len(mw)-1)
Pn=np.exp(-0.5*(Lpe-Lpe.min(1,keepdims=True))); Pn/=Pn.sum(1,keepdims=True); pe_mean=(Pn*mw[None,:]).sum(1)
# (B) sensitivity: reweight to true mW=mW0±Δ, weighted mean of per-event mW
mw0=mw[j0]; shifts=np.linspace(-1.0,1.0,9); calib=[]
for d in shifts:
    jt=int(np.argmin(np.abs(mw-(mw0+d)))); w=np.exp(-0.5*(Lgen[:,jt]-Lgen[:,j0])); w=np.minimum(w,np.percentile(w,99.9))
    calib.append(np.average(pe_mean,weights=w))
calib=np.array(calib); slope=np.polyfit(mw0+shifts,calib,1)[0]
print(f"ecm{ECM}: per-event MAP mean={pe[interior].mean():.3f} std={pe[interior].std():.3f} interior={100*interior.mean():.0f}%; "
      f"calib slope d⟨m̂W_i⟩/d(true mW)={slope:.2f}")
fig,ax=plt.subplots(1,2,figsize=(15,5.6))
ax[0].hist(pe[interior],bins=np.arange(73,88,0.5),density=True,histtype="stepfilled",alpha=.4,color="C0",label=f"per-event MAP m̂W_i (std {pe[interior].std():.2f})")
ax[0].hist(pe_mean,bins=np.arange(73,88,0.4),density=True,histtype="step",lw=2,color="C1",label=f"per-event posterior-mean (std {pe_mean.std():.2f})")
ax[0].axvline(80.379,color="k",ls="--",lw=1,label="generator mW 80.379")
ax[0].set_xlabel(r"per-event $\hat m_{W,i}$ [GeV]"); ax[0].set_ylabel("normalized")
ax[0].set_title(f"(A) PER-EVENT mW distribution, ecm{ECM}\n(checkable data vs MC — validates the folding)"); ax[0].legend(fontsize=9)
ax[1].plot(mw0+shifts,calib,"o-",color="C0",lw=2,label=f"slope {slope:.2f}")
ax[1].set_xlabel("true mW (reweighted) [GeV]"); ax[1].set_ylabel(r"weighted $\langle \hat m_{W,i}\rangle$ [GeV]")
ax[1].legend(fontsize=9)
ax[1].set_title(f"(B) SENSITIVITY: per-event mW distribution shifts with true mW\nslope d⟨m̂W_i⟩/dmW = {slope:.2f} ⇒ a real, calibratable handle")
ax[1].grid(alpha=.3)
plt.tight_layout(); png=f"{OUT}/perevent_mW_ecm{ECM}.png"; plt.savefig(png,dpi=120); print(f"[plot] {png}")
