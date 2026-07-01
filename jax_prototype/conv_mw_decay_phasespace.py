#!/usr/bin/env python3
# ── The principled W lineshape: WHIZARD const-width propagator × explicit W→qq decay factor ──────────
# WHIZARD propagator is CONSTANT-width: |prop|² = 1/((m²−M²)²+(M·Γ)²).  But the OBSERVED m_qq density =
# |prop|² × (m-dependent W→qq decay / 4f factor).  Our running-width BW [m²/((m²−M²)²+(Γm²/M)²)] fits the gen
# 3-16× better than fixed precisely because its m² numerator mimics that decay factor.  Here we DECOMPOSE:
# scan a numerator power  m^p  over BOTH denominators (FIXED D=(m²−M²)²+(MΓ)² ; RUNNING D=(m²−M²)²+(Γm²/M)²),
# and find which f(m)=m^p/D reproduces the gen m_qq lineshape at the TRUE mW=80.419.  Isolates:
#   • is it the NUMERATOR (decay phase-space power p) or the DENOMINATOR (width scheme) that matters?
#   • can a FIXED-denominator (= WHIZARD's actual scheme) + decay factor m^p match the gen as well as running?
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_decay_phasespace.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
GW=2.049; MWREF=80.419; TWO_MW=2*MWREF; ECMS=[157,160,163]
STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"
TLO,THI,NT=28.0,96.0,221
tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")

def Dfix(m): d=m*m-MWREF*MWREF; return d*d+(MWREF*GW)**2
def Drun(m): d=m*m-MWREF*MWREF; wm=GW*m*m/MWREF; return d*d+wm*wm
# candidate lineshapes f(m) = m^p / D ; p is the W→qq decay-phase-space power on the const-width propagator
def make_f(denom, p):
    D = Dfix if denom=="fix" else Drun
    def f(m): return (m**p)/D(m)
    return f
FORMS = {
    "fix p0  (|prop|² only)" : make_f("fix",0),
    "fix p1  (×m decay)"     : make_f("fix",1),
    "fix p2  (×m² decay)"    : make_f("fix",2),
    "fix p3  (×m³ decay)"    : make_f("fix",3),
    "run p2  (=running BW)"  : make_f("run",2),
    "run p0"                 : make_f("run",0),
}

def radiator_obs(mWW_g,ecm):
    cnt,edg=np.histogram(mWW_g,bins=120,range=(max(TLO,ecm-45),ecm+1.5))
    sn=0.5*(edg[:-1]+edg[1:]); w=cnt.astype(float); m=w>0; return sn[m],w[m]
def build_marg(f,rad):
    sn,sw=rad; fh=f(tg); Fout=fh[:,None]*fh[None,:]
    u=np.zeros((NT,NT))
    for sp,wk in zip(sn,sw):
        s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2); PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0.0)
        Zk=(Fout*PS).sum()*dt*dt                          # per-√s'-node normalization (direct grid)
        u+=(wk/max(Zk,1e-300))*Fout*PS
    Z=u.sum()*dt*dt; return (u/max(Z,1e-300)).sum(1)*dt   # normalized m_qq marginal

plt.rcParams.update({"font.size":11})
fig,AX=plt.subplots(1,3,figsize=(20,6))
cols=plt.cm.tab10(np.linspace(0,1,10))
print(f"{'ECM':>4} {'thr':>6}  "+ "  ".join(f"{k.split('(')[0].strip():>9}" for k in FORMS) + "   (χ²/dof of gen m_qq @ true mW)")
RES={}
for r,ecm in enumerate(ECMS):
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=(np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq,mlv,mWW=a["gen_Whad_m"][ok],a["gen_Wlep_m"][ok],a["gen_WW_m"][ok]
    rad=radiator_obs(mWW,ecm)
    h,e=np.histogram(mqq,bins=90,range=(60,95),density=True); c=0.5*(e[:-1]+e[1:]); cnt,_=np.histogram(mqq,bins=90,range=(60,95)); bwd=e[1]-e[0]
    sig=np.sqrt(np.maximum(cnt,1))/(cnt.sum()*bwd); dof=np.sum(cnt>0)
    thr="below" if ecm<TWO_MW else "above"
    ax=AX[r]; ax.step(c,h,where="mid",color="k",lw=1.8,label="gen m_qq",zorder=9)
    row={}
    for i,(name,f) in enumerate(FORMS.items()):
        marg=build_marg(f,rad); mdl=np.interp(c,tg,marg); chi2=np.sum((h-mdl)**2/sig**2)/dof; row[name]=chi2
        if "p1" in name or "p2  (=run" in name or "p2  (×m²" in name or "p0  (|prop" in name:
            ax.plot(tg,marg,color=cols[i],lw=1.6,label=f"{name.split('(')[0].strip()}: χ²/dof {chi2:.1f}")
    RES[(ecm,thr)]=row
    ax.axvline(MWREF,color="0.6",ls=":",lw=1.0); ax.set_xlim(66,90); ax.set_xlabel("m_qq [GeV]")
    if r==0: ax.set_ylabel("normalized")
    ax.set_title(f"ecm{ecm} ({thr} thr): decay-factor scan @ true mW"); ax.legend(fontsize=8)
    print(f"{ecm:>4} {thr:>6}  " + "  ".join(f"{row[k]:9.1f}" for k in FORMS))
fig.suptitle("Principled W lineshape = const-width propagator × m^p decay factor.  Which p (and denominator) matches gen?\n"
             "(if fix-p2 ≈ run-p2, the DECAY NUMERATOR — not the running denominator — is what matters; keeps WHIZARD's const-width)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94]); png=f"{EOSW}/decay_phasespace_day8.png"; fig.savefig(png,dpi=120,bbox_inches="tight")
print(f"\n[plot] {png}\n[link] https://mdefranc.web.cern.ch/mW/conv_mw/decay_phasespace_day8.png")
