#!/usr/bin/env python3
# ── VISUAL: where P(gen|mW) comes from — it's the true-level LINESHAPE (a physics model) ──
# P(gen=(m_qq,m_lν) | mW) = BW(m_qq;mW)·BW(m_lν;mW) · PhaseSpace(√s',m_qq,m_lν) / Z,  marginalized over √s'.
#   ingredient 1: each W is a relativistic Breit–Wigner resonance peaking at mW (width ΓW) — carries the mW info
#   ingredient 2: phase space — the two W's share the available √s' (m_qq+m_lν ≤ √s'), pushing masses below mW
#   ingredient 3: ISR — √s' is reduced from √s; average over its spectrum.  Z normalizes it to a pdf.
# Panels: (A) the BW peak moves with mW; (B) BW vs BW×phasespace×ISR (the full marginal); (C) it matches gen data.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_lineshape_demo.py
import sys, numpy as np, uproot
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
t=uproot.open(F)["events"]; a=t.arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
ok=np.isfinite(a["gen_Whad_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"])
qg=a["gen_Whad_m"][ok]; lg=a["gen_Wlep_m"][ok]; wg=a["gen_WW_m"][ok]
TLO,THI,NT=28.,98.,141; tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")
cnt,edg=np.histogram(wg,bins=60,range=(ECM-45,ECM+1.5)); SN=0.5*(edg[:-1]+edg[1:]); SW=cnt.astype(float); m=SW>0; SN=SN[m]; SW=SW[m]
def Ptrue_marg(mW, with_ps=True, with_isr=True):
    """m_qq marginal of P(gen|mW). with_ps=False ⇒ naked BW (no phase space); with_isr=False ⇒ √s'=√s only."""
    BW=bw(tg,mW)[:,None]*bw(tg,mW)[None,:]; u=np.zeros((NT,NT))
    nodes=list(zip(SN,SW)) if with_isr else [(float(ECM),1.0)]
    lz=log_Z(np.array([n[0] for n in nodes]),mW)
    for k,(sp,wk) in enumerate(nodes):
        if with_ps:
            s=sp*sp; lam=(s-(TH+TL)**2)*(s-(TH-TL)**2); PS=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s,0)
        else:
            PS=1.0
        u+=(wk/np.exp(lz[k]))*BW*PS
    u/= (u.sum()*dt*dt)
    return u.sum(1)*dt                       # integrate out m_lν ⇒ m_qq marginal density

fig,ax=plt.subplots(1,3,figsize=(19,5.4))
# (A) the BW peak alone, for three mW (normalized over the grid)
for mW,col in [(78.5,"C0"),(80.25,"C2"),(82.0,"C3")]:
    y=bw(tg,mW); y=y/(y.sum()*dt); ax[0].plot(tg,y,color=col,lw=2,label=f"BW, mW={mW:.2f}")
    ax[0].axvline(mW,color=col,ls=":",lw=1)
ax[0].set_xlim(70,90); ax[0].set_xlabel("W mass m [GeV]"); ax[0].set_ylabel("BW(m; mW) [norm]")
ax[0].set_title("(A) ingredient 1: each W is a Breit–Wigner\npeak AT mW — this carries the mW info"); ax[0].legend()
# (B) build-up: naked BW → ×phase space → ×phase space×ISR  (all m_qq marginals at mW0=80.25)
mW0=80.25
ax[1].plot(tg, bw(tg,mW0)/(bw(tg,mW0).sum()*dt), "C2", lw=2, label="naked BW(m_qq)")
ax[1].plot(tg, Ptrue_marg(mW0, with_ps=True, with_isr=False), "C1", lw=2, label="× phase space (√s'=√s)")
ax[1].plot(tg, Ptrue_marg(mW0, with_ps=True, with_isr=True),  "C4", lw=2.4, label="× phase space × ISR  =  P(gen|mW)")
ax[1].axvline(mW0,color="k",ls=":",lw=1,label=f"mW={mW0}")
ax[1].set_xlim(50,90); ax[1].set_xlabel("gen m_qq [GeV]"); ax[1].set_ylabel("normalized")
ax[1].set_title("(B) ×phase space (two W's share √s')\npushes the mass BELOW the peak"); ax[1].legend()
# (C) the full model marginal vs the ACTUAL gen masses (validation that P(gen|mW) is right)
qb=np.arange(40,95,1.5); qc=0.5*(qb[:-1]+qb[1:]); h,_=np.histogram(qg,bins=qb,density=True)
ax[2].step(qc,h,where="mid",color="k",lw=2.5,label="actual gen m_qq (MC truth)")
ax[2].plot(tg, Ptrue_marg(mW0), "C4", lw=2.4, label=f"P(gen|mW={mW0}) model")
ax[2].set_xlim(40,95); ax[2].set_xlabel("gen m_qq [GeV]"); ax[2].set_ylabel("normalized")
ax[2].set_title("(C) the model = the true-mass distribution\n(this is what makes reweighting valid)"); ax[2].legend()
plt.tight_layout(); png="/eos/user/m/mdefranc/www/mW/conv_mw/lineshape_demo_ecm160.png"; plt.savefig(png,dpi=110)
print(f"[plot] {png}")
