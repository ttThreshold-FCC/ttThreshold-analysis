#!/usr/bin/env python3
# ── The ecm157 residual in gen_validation.png: gen W lineshape vs the double-BW×PS model ─────────────
# In gen_validation col 1/2 the model = build_Ptrue marginal = BW(running)×PS folded over the EMPIRICAL gen
# √s' (radiator_obs).  So this residual is NOT an ISR/√s' effect (the true √s' is already used) — it is the
# intrinsic mismatch between our CC03 double-BW×phase-space lineshape and the true WHIZARD 4-fermion lineshape.
# We quantify it with a pull panel (gen−model)/√gen under each lineshape, per ECM.  Expectation: a LOW-MASS
# excess in gen m_qq that the model under-predicts, WORST at ecm157 (furthest below 2mW=160.84 ⇒ hadronic W
# forced most off-shell) — the same off-shell-lineshape physics that drives the +38 MeV gen_pe bias at 163.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_lineshape_residual.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
GW=2.049; MWREF=80.419; ECMS=[157,160,163]
STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"

# ── running-width BW×PS lineshape model (verbatim machinery from conv_mw_gen_validation.py) ──────────
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
def radiator_obs(mWW_g,ecm):
    cnt,edg=np.histogram(mWW_g,bins=60,range=(max(TLO,ecm-45),ecm+1.5))
    sn=0.5*(edg[:-1]+edg[1:]); w=cnt.astype(float); m=w>0; return sn[m],w[m]
def build_Ptrue(mW,rad):
    sn,sw=rad; bwh=bw_run(tg,mW); BWout=bwh[:,None]*bwh[None,:]
    u=np.zeros((NT,NT)); logZ=log_Z(sn,mW)
    for k,(sp,wk) in enumerate(zip(sn,sw)):
        s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0.0)
        u+=(wk/np.exp(logZ[k]))*BWout*PS
    Z=u.sum()*dt*dt; return u/np.maximum(Z,1e-300)

# ── load gen, build model marginals ─────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size":11})
fig=plt.figure(figsize=(20,7.5))
gs=GridSpec(2,3,height_ratios=[3,1],hspace=0.05,wspace=0.2)
print(f"{'ECM':>4} {'quantity':8} {'gen<m>':>8} {'model<m>':>9} {'gen peak':>9} "
      f"{'lowtail gen':>11} {'lowtail mod':>11} {'excess%':>8}   (lowtail = frac m<76 GeV)")
for r,ecm in enumerate(ECMS):
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(
        ["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=(np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&
        (a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq,mlv,mWW=a["gen_Whad_m"][ok],a["gen_Wlep_m"][ok],a["gen_WW_m"][ok]
    P=build_Ptrue(MWREF,radiator_obs(mWW,ecm)); Pmh=P.sum(1)*dt; Pml=P.sum(0)*dt
    col=r
    axU=fig.add_subplot(gs[0, col]); axL=fig.add_subplot(gs[1, col],sharex=axU)
    # hadronic
    h,e=np.histogram(mqq,bins=90,range=(60,95),density=True); c=0.5*(e[:-1]+e[1:]); cnt,_=np.histogram(mqq,bins=90,range=(60,95))
    mdl=np.interp(c,tg,Pmh)
    axU.step(c,h,where="mid",color="k",lw=1.6,label=f"gen m_qq")
    axU.plot(tg,Pmh,color="C3",lw=2.4,label="model BW×PS (running)")
    axU.axvline(MWREF,color="0.5",ls=":",lw=1.1)
    axU.set_xlim(66,92); axU.set_title(f"ecm{ecm}: hadronic-W lineshape"); axU.legend(fontsize=9); axU.tick_params(labelbottom=False)
    # pull = (gen-model)/sigma_gen with binwidth norm
    bw_=e[1]-e[0]; n=cnt.sum(); sig=np.sqrt(np.maximum(cnt,1))/(n*bw_)
    pull=(h-mdl)/sig
    axL.axhspan(-2,2,color="0.85"); axL.axhline(0,color="0.4",lw=0.8)
    axL.step(c,pull,where="mid",color="C3",lw=1.2); axL.set_xlim(66,92); axL.set_ylim(-8,8)
    axL.set_xlabel("m_qq [GeV]");
    if col==0: axL.set_ylabel("pull (gen−mod)/σ"); axU.set_ylabel("normalized")
    # numbers
    lt_g=np.mean(mqq<76.0); lt_m=np.trapz(Pmh[tg<76.0],tg[tg<76.0])
    print(f"{ecm:>4} {'m_qq':8} {mqq.mean():8.3f} {np.trapz(tg*Pmh,tg):9.3f} {c[np.argmax(h)]:9.2f} "
          f"{lt_g:11.4f} {lt_m:11.4f} {100*(lt_g-lt_m):8.2f}")
fig.suptitle("ecm157 residual = gen W lineshape vs CC03 double-BW×PS (off-shell low-mass excess, worst below threshold)\n"
             "model uses the EMPIRICAL gen √s' ⇒ this is a 4f-lineshape mismatch, NOT an ISR effect",fontsize=13)
png=f"{EOSW}/lineshape_residual_day8.png"; fig.savefig(png,dpi=125,bbox_inches="tight")
print(f"\n[plot] {png}")
print(f"[link] https://mdefranc.web.cern.ch/mW/conv_mw/lineshape_residual_day8.png")
