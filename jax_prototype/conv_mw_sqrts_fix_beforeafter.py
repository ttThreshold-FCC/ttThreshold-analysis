#!/usr/bin/env python3
# ── BEFORE/AFTER the lumi_density above-ECM radiator-clip fix: model √s' vs gen, per ECM ─────────────
# One picture for the DAY7 headline: the model √s' spectrum was a SPIKE at ECM ("RMS 0.44, peaked above
# ECM") purely because the ISR radiator was evaluated for √s'>ECM (x clipped to the (1-x)^(β-1) endpoint
# singularity) → spurious plateau above ECM. Zeroing the radiator for √s'>=ECM makes it a peaked-with-tail
# distribution that approximately matches gen (mean to ~0.05 GeV).  BEFORE = buggy, AFTER = fixed.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_sqrts_fix_beforeafter.py
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import uproot

GW=2.085; MW_REF=80.379; ALPHA=1.0/137.035999; ME=0.000510999
SIG_BES_ECM={157:0.116,160:0.119,163:0.121}
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
LOGZ_N=256; _GX,_GWl=np.polynomial.legendre.leggauss(LOGZ_N); _W2=_GWl[:,None]*_GWl[None,:]
def log_Z(m_WW,mW,gW=GW):
    m_WW=np.atleast_1d(np.asarray(m_WW,float)); mwgw=mW*gW; mW2=mW*mW
    s=m_WW**2; t0=np.arctan(-mW2/mwgw); t1=np.arctan((s-mW2)/mwgw)
    hd=0.5*(t1-t0); hs=0.5*(t1+t0); t=hd[:,None]*_GX[None,:]+hs[:,None]
    m=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1/m
    mh=m[:,:,None]; ml=m[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=s[:,None,None]
    lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2)
    integ=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il/(4*sE),0.0)
    return np.log(np.maximum(np.sum(_W2[None]*integ,axis=(1,2))*hd*hd,1e-300))

def lumi(grid,ecm,sig,buggy):
    s=ecm*ecm; beta=(2*ALPHA/np.pi)*(np.log(s/ME**2)-1.0)
    fine=np.linspace(ecm-60.0,ecm+3.0,8000); x=np.clip((fine/ecm)**2,1e-12,1-1e-12)
    D=beta*(1-x)**(beta-1)*(1+0.75*beta)-0.5*beta*(1+x)
    p=np.maximum(D,0.0)*(2*fine/s)
    if not buggy: p=np.where(fine<ecm,p,0.0)                  # AFTER: radiator 0 above ECM
    if sig>0:
        dxg=fine[1]-fine[0]; n=int(6*sig/dxg); kx=np.arange(-n,n+1)*dxg
        ker=np.exp(-0.5*(kx/sig)**2); ker/=ker.sum(); p=np.convolve(p,ker,mode="same")
    p=np.maximum(p,0.0); p/=np.trapz(p,fine); return np.interp(grid,fine,p,left=0,right=0)
def model(grid,ecm,sig,buggy):
    p=lumi(grid,ecm,sig,buggy)*np.exp(log_Z(grid,MW_REF)); return p/np.trapz(p,grid)
def mr(c,d): d=d/np.trapz(d,c); mu=np.trapz(c*d,c); return mu,np.sqrt(np.trapz((c-mu)**2*d,c))

ECMS=[157,160,163]; plt.rcParams.update({"font.size":12})
fig,AX=plt.subplots(1,3,figsize=(19,5.2))
for j,ECM in enumerate(ECMS):
    a=uproot.open(f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root")["events"].arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=np.isfinite(a["gen_WW_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]); sqs=a["gen_WW_m"][ok]
    sig=SIG_BES_ECM[ECM]; lo,hi=ECM-14,ECM+2
    edges=np.arange(lo,hi+0.2,0.2); cen=0.5*(edges[:-1]+edges[1:])
    Pg,_=np.histogram(sqs,bins=edges,density=True); Pg=Pg/np.trapz(Pg,cen)
    mB=model(cen,ECM,sig,True); mA=model(cen,ECM,sig,False)
    gm,gr=mr(cen,Pg); bm,br=mr(cen,mB); am,ar=mr(cen,mA)
    ax=AX[j]
    ax.step(cen,Pg,where="mid",color="k",lw=1.8,label=f"gen √s'  (mean {gm:.2f}, rms {gr:.2f})")
    ax.plot(cen,mB,color="C3",lw=2.2,ls="--",label=f"BEFORE (bug)  (mean {bm:.2f}, rms {br:.2f})")
    ax.plot(cen,mA,color="C2",lw=2.6,label=f"AFTER (fix)  (mean {am:.2f}, rms {ar:.2f})")
    ax.axvline(ECM,color="0.6",ls=":",lw=1); ax.text(ECM,2.5," ECM",color="0.4",fontsize=9,rotation=90,va="top")
    ax.set_yscale("log"); ax.set_ylim(2e-3,5); ax.set_xlim(lo,hi)
    ax.set_title(f"ecm{ECM}:  √s' = m_lνuqq"); ax.set_xlabel("√s' [GeV]"); ax.legend(fontsize=9.5,loc="upper left")
AX[0].set_ylabel("normalized density")
fig.suptitle("√s' luminosity model BEFORE vs AFTER the above-ECM radiator-clip fix  (model = SF⊗BES × Z_CC03)",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96]); png=f"{EOSW}/sqrts_fix_beforeafter_day7.png"; fig.savefig(png,dpi=130); print(f"[plot] {png}")
