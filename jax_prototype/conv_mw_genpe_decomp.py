#!/usr/bin/env python3
# ── WHERE does the +38 MeV gen_pe bias at ecm163 come from?  (DAY8) ──────────────────────────────────
# gen_pe = the ensemble per-event self-fit of the GEN (m_qq,m_lν,√s') with the running-width double-BW×PS
# model.  At the WHIZARD generator mass mW=80.419: gen_pe−mW = −1.0/−2.6/+38.2 MeV @157/160/163.  157&160
# are BELOW 2mW=160.838 (W's forced off-shell, phase-space squeezed); 163 is ABOVE (W's can be on-shell).
# We localize the +38 by toggling the model ingredients (all on the IDENTICAL gen events), each with its own
# consistent normalization Z(mW):
#   full   : −2lnBW(m_qq) −2lnBW(m_lν) −2lnPS + 2lnZ2D   (PS=√λ/s ; Z2D=∫∫BW·BW·PS)   ← reproduces +38
#   noPS   : −2lnBW(m_qq) −2lnBW(m_lν)        + 2lnZ2D0  (no phase space ; Z2D0=(∫BW)² )
#   had1d  : −2lnBW(m_qq)                     + 2lnZ1D   (bare 1-D hadronic-W lineshape)
#   lep1d  :              −2lnBW(m_lν)         + 2lnZ1D
# If 'noPS' kills the +38 ⇒ the phase-space/threshold weighting is the culprit; if 'had1d'/'lep1d' already
# show +38 ⇒ the gen W lineshape itself peaks higher at 163 (a real above-threshold on-shell effect).
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_genpe_decomp.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
GW=2.049; MWREF=80.419; ECMS=[157,160,163]
NMW=161; mw_scan=np.linspace(79.7,81.0,NMW)    # parab-interpolated argmin ⇒ robust to NMW
STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"

# ── running-width BW + normalizations ───────────────────────────────────────────────────────────────
def bw_run(m,mW,gW=GW): d=m*m-mW*mW; wm=gW*m*m/mW; return wm/(d*d+wm*wm)
LOGZ_N=144; _GX,_GWl=np.polynomial.legendre.leggauss(LOGZ_N); _W2=_GWl[:,None]*_GWl[None,:]
def log_Z2D(m_WW,mW,gW=GW):                                  # ∫∫ BW·BW·PS  (running width, exact reweight)
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
_mfine=np.linspace(40.0,100.0,4000)                          # 1-D mass grid for the no-PS normalizations
def logZ1D(mW): return np.log(np.trapz(bw_run(_mfine,mW),_mfine))        # ∫BW dm
def logZ2D0(mW): return 2.0*logZ1D(mW)                                   # (∫BW)²

def parab_min(xs,ys):
    j=int(np.argmin(ys));
    if j<=0 or j>=len(xs)-1: return xs[j]
    x0,x1,x2=xs[j-1],xs[j],xs[j+1]; y0,y1,y2=ys[j-1],ys[j],ys[j+1]
    d=(x0-x1)*(x0-x2)*(x1-x2)
    if d==0: return x1
    A=(x2*(y1-y0)+x1*(y0-y2)+x0*(y2-y1))/d
    B=(x2*x2*(y0-y1)+x1*x1*(y2-y0)+x0*x0*(y1-y2))/d
    return x1 if A<=0 else -B/(2*A)

def gen_pe(mqq,mlv,mWW,mode):
    """ensemble -2lnL self-fit; returns argmin mW and the summed -2lnL curve."""
    s=mWW*mWW; lam=(s-(mqq+mlv)**2)*(s-(mqq-mlv)**2); bad=lam<=0; lamc=np.where(bad,1.0,lam)
    wn=np.linspace(mWW.min()-0.5,mWW.max()+0.5,200)
    tot=np.zeros(NMW)
    for j,mW in enumerate(mw_scan):
        lbh=np.log(np.maximum(bw_run(mqq,mW),1e-300)); lbl=np.log(np.maximum(bw_run(mlv,mW),1e-300))
        if mode=="full":
            lz=np.interp(mWW,wn,log_Z2D(wn,mW))
            tv=-2*lbh-2*lbl-np.log(lamc)+2*np.log(s)+2*lz
            tv=np.where(bad,1e6,tv)
        elif mode=="noPS":
            tv=-2*lbh-2*lbl+2*logZ2D0(mW)
        elif mode=="had1d":
            tv=-2*lbh+2*logZ1D(mW)
        elif mode=="lep1d":
            tv=-2*lbl+2*logZ1D(mW)
        tot[j]=tv.sum()
    return parab_min(mw_scan,tot),tot

# ── load gen, run toggles ─────────────────────────────────────────────────────────────────────────
DAT={}
print(f"{'ECM':>4} {'<m_qq>':>8} {'<m_lν>':>8} {'<√s'+chr(39)+'>':>8} {'frac<2mW':>9}  "
      f"{'full':>8} {'noPS':>8} {'had1d':>8} {'lep1d':>8}   (gen_pe−80.419, MeV)")
RES={}
for ecm in ECMS:
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(
        ["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=(np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&
        (a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq,mlv,mWW=a["gen_Whad_m"][ok],a["gen_Wlep_m"][ok],a["gen_WW_m"][ok]
    DAT[ecm]=(mqq,mlv,mWW)
    r={m:gen_pe(mqq,mlv,mWW,m)[0] for m in ["full","noPS","had1d","lep1d"]}
    RES[ecm]=r
    f2=100*np.mean(mWW<2*MWREF)
    print(f"{ecm:>4} {mqq.mean():8.3f} {mlv.mean():8.3f} {mWW.mean():8.3f} {f2:8.1f}%  "
          + "  ".join(f"{1000*(r[m]-MWREF):+7.1f}" for m in ["full","noPS","had1d","lep1d"]))

# ── PLOT ──────────────────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size":11})
fig,AX=plt.subplots(1,3,figsize=(20,6))
cols={157:"C0",160:"C1",163:"C2"}
# (1) gen m_qq lineshapes overlaid — does 163 peak higher?
ax=AX[0]
for ecm in ECMS:
    mqq,mlv,mWW=DAT[ecm]; h,e=np.histogram(mqq,bins=120,range=(60,95),density=True); c=0.5*(e[:-1]+e[1:])
    pk=c[np.argmax(h)]
    ax.step(c,h,where="mid",color=cols[ecm],lw=1.8,label=f"ecm{ecm}  <m_qq>={mqq.mean():.2f}  peak≈{pk:.1f}")
ax.axvline(MWREF,color="k",ls=":",lw=1.3,label=f"gen mW={MWREF}")
ax.set_xlim(68,92); ax.set_xlabel("gen m_qq (hadronic W) [GeV]"); ax.set_ylabel("normalized")
ax.set_title("gen hadronic-W lineshape vs ECM\n(below-thr 157/160 squeezed low; above-thr 163 on-shell)"); ax.legend(fontsize=9)
# (2) gen m_lν lineshapes
ax=AX[1]
for ecm in ECMS:
    mqq,mlv,mWW=DAT[ecm]; h,e=np.histogram(mlv,bins=120,range=(60,95),density=True); c=0.5*(e[:-1]+e[1:])
    pk=c[np.argmax(h)]
    ax.step(c,h,where="mid",color=cols[ecm],lw=1.8,label=f"ecm{ecm}  <m_lν>={mlv.mean():.2f}  peak≈{pk:.1f}")
ax.axvline(MWREF,color="k",ls=":",lw=1.3,label=f"gen mW={MWREF}")
ax.set_xlim(68,92); ax.set_xlabel("gen m_lν (leptonic W) [GeV]"); ax.set_ylabel("normalized")
ax.set_title("gen leptonic-W lineshape vs ECM"); ax.legend(fontsize=9)
# (3) gen_pe−mW bar chart per toggle
ax=AX[2]; modes=["full","noPS","had1d","lep1d"]; xb=np.arange(len(modes)); w=0.25
for i,ecm in enumerate(ECMS):
    vals=[1000*(RES[ecm][m]-MWREF) for m in modes]
    bars=ax.bar(xb+(i-1)*w,vals,w,color=cols[ecm],label=f"ecm{ecm}")
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+(1 if v>=0 else -3),f"{v:+.0f}",ha="center",fontsize=7.5)
ax.axhline(0,color="k",lw=0.8); ax.set_xticks(xb); ax.set_xticklabels(modes)
ax.set_ylabel("gen_pe − 80.419 [MeV]"); ax.set_title("self-fit bias by model ingredient\n(which toggle removes the +38 @163?)"); ax.legend(fontsize=9)
fig.suptitle("Localizing the +38 MeV gen_pe bias at ecm163 (running-width double-BW self-fit of WHIZARD gen)",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.95]); png=f"{EOSW}/genpe_decomp_day8.png"; fig.savefig(png,dpi=125,bbox_inches="tight")
print(f"\n[plot] {png}")
print(f"[link] https://mdefranc.web.cern.ch/mW/conv_mw/genpe_decomp_day8.png")
