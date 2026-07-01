#!/usr/bin/env python3
# ── Is the ecm157 hadronic-W low-mass bump under-prediction a √s' BINNING artifact, or genuine 4f? ───
# build_Ptrue folds BW×BW×PS over the empirical gen √s' spectrum, histogrammed into NBINS (default 60).
# The off-shell bump sits at m_qq ≈ √s'−mW, so it TRACKS √s'; a coarse √s' histogram quantizes/smears it.
# We sweep NBINS (60 → 960 = the unbinned limit) and watch whether the model m_qq marginal converges ONTO
# gen.  If the low-mass excess (frac m_qq<76) → 0 as NBINS grows, the under-prediction was pure binning; if it
# PLATEAUS at a finite value, the residual is genuine 4f / non-double-resonant content beyond CC03.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_radiator_binning.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
GW=2.049; MWREF=80.419; STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"
ECMS=[157,160]; NBINS_LIST=[60,120,240,480,960]

LOGZ_N=256; _GX,_GWl=np.polynomial.legendre.leggauss(LOGZ_N); _W2=_GWl[:,None]*_GWl[None,:]
def log_Z(m_WW,mW,gW=GW):
    m_WW=np.atleast_1d(np.asarray(m_WW,float)); mwgw=mW*gW; mW2=mW*mW
    s=m_WW**2; t_min=np.arctan(-mW2/mwgw); t_max=np.arctan((s-mW2)/mwgw)
    hd=0.5*(t_max-t_min); hs=0.5*(t_max+t_min); t=hd[:,None]*_GX[None,:]+hs[:,None]
    m=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1/m
    dd=m*m-mW2; wm=gW*m*m/mW; rf=(m*m/mW2)*(dd*dd+mwgw*mwgw)/(dd*dd+wm*wm)
    mh=m[:,:,None]; ml=m[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=s[:,None,None]
    rh=rf[:,:,None]; rl=rf[:,None,:]
    lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2)
    integ=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il*rh*rl/(4*sE),0.0)
    Z=np.sum(_W2[None]*integ,axis=(1,2))*hd*hd; return np.log(np.maximum(Z,1e-300))
def bw_run(m,mW,gW=GW): d=m*m-mW*mW; wm=gW*m*m/mW; return wm/(d*d+wm*wm)
TLO,THI,NT=28.0,96.0,137
tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")
def radiator_obs(mWW_g,ecm,nbins):
    cnt,edg=np.histogram(mWW_g,bins=nbins,range=(max(TLO,ecm-45),ecm+1.5))
    sn=0.5*(edg[:-1]+edg[1:]); w=cnt.astype(float); m=w>0; return sn[m],w[m]
def model_mqq(mW,rad):
    sn,sw=rad; bwh=bw_run(tg,mW); BWout=bwh[:,None]*bwh[None,:]
    u=np.zeros((NT,NT)); logZ=log_Z(sn,mW)                       # per-√s'-node normalization (verbatim build_Ptrue)
    for k,(sp,wk) in enumerate(zip(sn,sw)):
        s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0.0)
        u+=(wk/np.exp(logZ[k]))*BWout*PS
    Z=u.sum()*dt*dt; Pmh=(u/np.maximum(Z,1e-300)).sum(1)*dt; return Pmh

plt.rcParams.update({"font.size":11})
fig,AX=plt.subplots(1,3,figsize=(20,6))
cmap=plt.cm.viridis(np.linspace(0.15,0.9,len(NBINS_LIST)))
conv={}
print(f"{'ECM':>4} {'NBINS':>6} {'binw[GeV]':>9} {'model frac<76':>13} {'gen frac<76':>12} {'excess(gen-mod)%':>16}")
for col,ecm in enumerate(ECMS):
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=(np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq,mlv,mWW=a["gen_Whad_m"][ok],a["gen_Wlep_m"][ok],a["gen_WW_m"][ok]
    gen_lt=np.mean(mqq<76.0)
    ax=AX[col]
    h,e=np.histogram(mqq,bins=90,range=(60,95),density=True); c=0.5*(e[:-1]+e[1:])
    ax.step(c,h,where="mid",color="k",lw=1.8,label="gen m_qq",zorder=10)
    excesses=[]
    for i,nb in enumerate(NBINS_LIST):
        Pmh=model_mqq(MWREF,radiator_obs(mWW,ecm,nb))
        mlt=np.trapz(Pmh[tg<76.0],tg[tg<76.0]); excesses.append(100*(gen_lt-mlt))
        binw=((ecm+1.5)-max(TLO,ecm-45))/nb            # √s' histogram bin width [GeV]
        ax.plot(tg,Pmh,color=cmap[i],lw=1.6,label=f"NBINS={nb} (Δ√s'={binw:.2f}, exc {100*(gen_lt-mlt):.1f}%)")
        print(f"{ecm:>4} {nb:>6} {binw:9.3f} {mlt:13.4f} {gen_lt:12.4f} {100*(gen_lt-mlt):16.2f}")
    conv[ecm]=excesses
    ax.axvline(MWREF,color="0.5",ls=":",lw=1.0); ax.axvspan(60,76,color="0.93")
    ax.set_xlim(66,90); ax.set_xlabel("m_qq [GeV]"); ax.set_ylabel("normalized")
    ax.set_title(f"ecm{ecm}: model m_qq vs √s'-binning (gray = low-mass bump region)"); ax.legend(fontsize=8.5)
# convergence panel
ax=AX[2]
for ecm in ECMS:
    ax.plot(NBINS_LIST,conv[ecm],"o-",lw=1.8,label=f"ecm{ecm}")
ax.axhline(0,color="k",lw=0.8); ax.set_xscale("log"); ax.set_xlabel("NBINS (√s' histogram)")
ax.set_ylabel("low-mass excess (gen−model) frac m_qq<76 [%]")
ax.set_title("does the bump under-prediction → 0 as binning refines?\n(→0 = binning artifact; plateau = genuine 4f)")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
fig.suptitle("Test (1): √s' radiator binning vs the ecm157 hadronic-W low-mass bump",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95]); png=f"{EOSW}/radiator_binning_day8.png"; fig.savefig(png,dpi=125,bbox_inches="tight")
print(f"\n[plot] {png}\n[link] https://mdefranc.web.cern.ch/mW/conv_mw/radiator_binning_day8.png")
