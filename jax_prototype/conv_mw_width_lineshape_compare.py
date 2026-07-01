#!/usr/bin/env python3
# ── At the TRUE generator mW=80.419, does the gen W lineshape match FIXED- or RUNNING-width BW×PS? ───
# WHIZARD uses a CONSTANT (fixed) width propagator (ground-truth: wd_tl=2.049 flat).  Yet our gen self-fit
# closes only with a RUNNING-width template below threshold (fixed → −200 MeV).  Decisive test: overlay the gen
# m_qq / m_lν on BOTH model lineshapes evaluated at the FIXED true mass mW=80.419 (no fitting), with pulls +
# χ².  Whichever the gen follows is the better lineshape model; if gen is running-LIKE despite the constant-width
# propagator, the off-shell shape is sculpted by 4f phase-space/production, not the propagator width.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_width_lineshape_compare.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
GW=2.049; MWREF=80.419; TWO_MW=2*MWREF; ECMS=[157,160,163]
STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"

LOGZ_N=256; _GX,_GWl=np.polynomial.legendre.leggauss(LOGZ_N); _W2=_GWl[:,None]*_GWl[None,:]
def log_Z(m_WW,mW,run,gW=GW):
    m_WW=np.atleast_1d(np.asarray(m_WW,float)); mwgw=mW*gW; mW2=mW*mW
    s=m_WW**2; t_min=np.arctan(-mW2/mwgw); t_max=np.arctan((s-mW2)/mwgw)
    hd=0.5*(t_max-t_min); hs=0.5*(t_max+t_min); t=hd[:,None]*_GX[None,:]+hs[:,None]
    m=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1/m
    if run: dd=m*m-mW2; wm=gW*m*m/mW; rf=(m*m/mW2)*(dd*dd+mwgw*mwgw)/(dd*dd+wm*wm)
    else:   rf=np.ones_like(m)
    mh=m[:,:,None]; ml=m[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=s[:,None,None]
    rh=rf[:,:,None]; rl=rf[:,None,:]
    lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2)
    integ=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il*rh*rl/(4*sE),0.0)
    Z=np.sum(_W2[None]*integ,axis=(1,2))*hd*hd; return np.log(np.maximum(Z,1e-300))
def bw(m,mW,run,gW=GW):
    d=m*m-mW*mW
    if run: wm=gW*m*m/mW; return wm/(d*d+wm*wm)
    return (mW*gW)/(d*d+(mW*gW)**2)
TLO,THI,NT=28.0,96.0,137
tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")
def radiator_obs(mWW_g,ecm):
    cnt,edg=np.histogram(mWW_g,bins=120,range=(max(TLO,ecm-45),ecm+1.5))
    sn=0.5*(edg[:-1]+edg[1:]); w=cnt.astype(float); m=w>0; return sn[m],w[m]
def build_marg(mW,rad,run):
    sn,sw=rad; bwh=bw(tg,mW,run); BWout=bwh[:,None]*bwh[None,:]
    u=np.zeros((NT,NT)); logZ=log_Z(sn,mW,run)
    for k,(sp,wk) in enumerate(zip(sn,sw)):
        s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0.0)
        u+=(wk/np.exp(logZ[k]))*BWout*PS
    Z=u.sum()*dt*dt; u=u/np.maximum(Z,1e-300); return u.sum(1)*dt, u.sum(0)*dt   # m_qq, m_lν marginals

plt.rcParams.update({"font.size":11})
fig=plt.figure(figsize=(20,11)); gs=GridSpec(4,3,height_ratios=[3,1,3,1],hspace=0.06,wspace=0.2)
print(f"{'ECM':>4} {'thr':>6} {'quantity':6} {'χ²/dof FIXED':>13} {'χ²/dof RUN':>11}   (lineshape match at true mW=80.419)")
def chi2(h,cnt,mdl,bw_):
    n=cnt.sum(); sig2=np.maximum(cnt,1)/(n*bw_)**2; dof=np.sum(cnt>0)
    return np.sum((h-mdl)**2/sig2)/dof
for r,ecm in enumerate(ECMS):
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=(np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq,mlv,mWW=a["gen_Whad_m"][ok],a["gen_Wlep_m"][ok],a["gen_WW_m"][ok]
    rad=radiator_obs(mWW,ecm)
    mqq_fix,mlv_fix=build_marg(MWREF,rad,run=False)
    mqq_run,mlv_run=build_marg(MWREF,rad,run=True)
    thr="below" if ecm<TWO_MW else "above"
    axU=fig.add_subplot(gs[0,r]); axL=fig.add_subplot(gs[1,r],sharex=axU)
    h,e=np.histogram(mqq,bins=90,range=(60,95),density=True); c=0.5*(e[:-1]+e[1:]); cnt,_=np.histogram(mqq,bins=90,range=(60,95)); bw_=e[1]-e[0]
    mf=np.interp(c,tg,mqq_fix); mr=np.interp(c,tg,mqq_run)
    axU.step(c,h,where="mid",color="k",lw=1.6,label="gen m_qq",zorder=9)
    axU.plot(tg,mqq_fix,color="C3",lw=2.2,label=f"FIXED-width BW×PS (χ²/dof {chi2(h,cnt,mf,bw_):.1f})")
    axU.plot(tg,mqq_run,color="C0",lw=2.2,ls="--",label=f"RUNNING-width BW×PS (χ²/dof {chi2(h,cnt,mr,bw_):.1f})")
    axU.axvline(MWREF,color="0.6",ls=":",lw=1.0); axU.set_xlim(66,90); axU.tick_params(labelbottom=False)
    axU.set_title(f"ecm{ecm} ({thr} threshold): m_qq @ true mW"); axU.legend(fontsize=8.5)
    if r==0: axU.set_ylabel("normalized")
    sig=np.sqrt(np.maximum(cnt,1))/(cnt.sum()*bw_)
    axL.axhspan(-2,2,color="0.9"); axL.axhline(0,color="0.4",lw=0.7)
    axL.step(c,(h-mf)/sig,where="mid",color="C3",lw=1.0); axL.step(c,(h-mr)/sig,where="mid",color="C0",lw=1.0,ls="--")
    axL.set_xlim(66,90); axL.set_ylim(-8,8); axL.set_xlabel("m_qq [GeV]")
    if r==0: axL.set_ylabel("pull/σ")
    print(f"{ecm:>4} {thr:>6} {'m_qq':6} {chi2(h,cnt,mf,bw_):13.1f} {chi2(h,cnt,mr,bw_):11.1f}")
fig.suptitle("Which width model matches the gen W lineshape AT THE TRUE mW=80.419?  (WHIZARD propagator is CONSTANT-width)\n"
             "red=fixed  blue-dashed=running  — below threshold gen is running-LIKE; above, fixed should win",fontsize=12)
png=f"{EOSW}/width_lineshape_compare_day8.png"; fig.savefig(png,dpi=120,bbox_inches="tight")
print(f"\n[plot] {png}\n[link] https://mdefranc.web.cern.ch/mW/conv_mw/width_lineshape_compare_day8.png")
