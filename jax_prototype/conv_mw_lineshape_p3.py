#!/usr/bin/env python3
# ── The principled lineshape at p=3 vs gen: const-width propagator × m³ W→qq decay factor ────────────
# Overlay gen m_qq / m_lν on three models AT THE TRUE mW=80.419 (no fitting), with pull panels:
#   fixed p0  : |const-width prop|²            (WHIZARD propagator, NO decay factor)  — too narrow
#   running   : running-width BW (DAY7)        (= implicit m² numerator)
#   p3        : m³ × |const-width prop|²        (WHIZARD propagator × physical W→qq decay factor) ← principled
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_lineshape_p3.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
GW=2.049; MWREF=80.419; TWO_MW=2*MWREF; ECMS=[157,160,163]
STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"
TLO,THI,NT=40.0,98.0,201
tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")

def Dfix(m): d=m*m-MWREF*MWREF; return d*d+(MWREF*GW)**2
def Drun(m): d=m*m-MWREF*MWREF; wm=GW*m*m/MWREF; return d*d+wm*wm
FORMS={"fixed p0  (|prop|², no decay)":(lambda m:1.0/Dfix(m),"C3","-"),
       "running BW  (DAY7)"           :(lambda m:(GW*m*m/MWREF)/Drun(m),"C1","-"),
       "p3 = m³·|const-width prop|²  (principled)":(lambda m:m**3/Dfix(m),"C2","-")}

def radiator_obs(mWW_g,ecm):
    cnt,edg=np.histogram(mWW_g,bins=120,range=(max(TLO,ecm-45),ecm+1.5))
    sn=0.5*(edg[:-1]+edg[1:]); w=cnt.astype(float); m=w>0; return sn[m],w[m]
def build_marg(f,rad):
    sn,sw=rad; fh=f(tg); Fout=fh[:,None]*fh[None,:]; u=np.zeros((NT,NT))
    for sp,wk in zip(sn,sw):
        s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2); PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0.0)
        Zk=(Fout*PS).sum()*dt*dt; u+=(wk/max(Zk,1e-300))*Fout*PS
    Z=u.sum()*dt*dt; uu=u/max(Z,1e-300); return uu.sum(1)*dt, uu.sum(0)*dt   # m_qq, m_lν marginals

plt.rcParams.update({"font.size":11})
fig=plt.figure(figsize=(20,13)); gs=GridSpec(4,3,height_ratios=[3,1,3,1],hspace=0.07,wspace=0.18)
print(f"{'ECM':>4} {'quantity':6}  "+ "  ".join(f"{n.split('(')[0].strip()[:9]:>9}" for n in FORMS)+"   (χ²/dof @ true mW)")
for r,ecm in enumerate(ECMS):
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=(np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq,mlv,mWW=a["gen_Whad_m"][ok],a["gen_Wlep_m"][ok],a["gen_WW_m"][ok]
    rad=radiator_obs(mWW,ecm); margs={n:build_marg(f,rad) for n,(f,_,_) in FORMS.items()}
    thr="below" if ecm<TWO_MW else "above"
    for qi,(qname,data,idx) in enumerate([("m_qq (hadronic W)",mqq,0),("m_lν (leptonic W)",mlv,1)]):
        axU=fig.add_subplot(gs[2*qi,r]); axL=fig.add_subplot(gs[2*qi+1,r],sharex=axU)
        h,e=np.histogram(data,bins=90,range=(60,95),density=True); c=0.5*(e[:-1]+e[1:]); cnt,_=np.histogram(data,bins=90,range=(60,95)); bwd=e[1]-e[0]
        sig=np.sqrt(np.maximum(cnt,1))/(cnt.sum()*bwd); dof=np.sum(cnt>0)
        axU.step(c,h,where="mid",color="k",lw=1.7,label="gen",zorder=9)
        axL.axhspan(-2,2,color="0.9"); axL.axhline(0,color="0.4",lw=0.7)
        chis={}
        for n,(f,col,ls) in FORMS.items():
            mdl=np.interp(c,tg,margs[n][idx]); chi=np.sum((h-mdl)**2/sig**2)/dof; chis[n]=chi
            axU.plot(tg,margs[n][idx],color=col,ls=ls,lw=2.0,label=f"{n.split('(')[0].strip()}  χ²/dof {chi:.1f}")
            axL.step(c,(h-mdl)/sig,where="mid",color=col,lw=1.0)
        axU.axvline(MWREF,color="0.6",ls=":",lw=0.9); axU.set_xlim(66,90); axU.tick_params(labelbottom=False)
        axU.set_title(f"ecm{ecm} ({thr} thr): {qname}",fontsize=11); axU.legend(fontsize=7.8,loc="upper left")
        axL.set_xlim(66,90); axL.set_ylim(-8,8); axL.set_xlabel(f"{qname.split()[0]} [GeV]")
        if r==0: axU.set_ylabel("normalized"); axL.set_ylabel("pull/σ")
        print(f"{ecm:>4} {['m_qq','m_lv'][qi]:6}  "+"  ".join(f"{chis[n]:9.1f}" for n in FORMS))
fig.suptitle("Principled W lineshape p=3 (const-width propagator × m³ W→qq decay factor) vs gen, at true mW=80.419\n"
             "green (p3) tracks gen far better than red (no decay) and beats running (orange) — same WHIZARD const-width scheme",fontsize=12)
png=f"{EOSW}/lineshape_p3_day8.png"; fig.savefig(png,dpi=118,bbox_inches="tight")
print(f"\n[plot] {png}\n[link] https://mdefranc.web.cern.ch/mW/conv_mw/lineshape_p3_day8.png")
