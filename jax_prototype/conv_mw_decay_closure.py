#!/usr/bin/env python3
# ── Does the principled lineshape (const-width propagator × m^p decay factor) CLOSE the gen self-fit? ─
# Decay-phasespace scan showed fix-p2 ≈ running BW in lineshape χ² (the m² numerator = W→qq decay factor,
# NOT the running denominator).  THE decisive test: per-event gen self-fit gen_pe for each form, all 3 ECMs.
# Baselines to beat:  fixed (p0) = −202/−209/−24 ;  running = −1/−3/+38.  A model that CLOSES UNIFORMLY
# (small, ECM-stable gen_pe−mW) with WHIZARD's actual CONSTANT-width propagator is the principled template.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_decay_closure.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
GW=2.049; MWREF=80.419; TWO_MW=2*MWREF; ECMS=[157,160,163]
STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"
NMW=81; mw_scan=np.linspace(79.6,81.0,NMW)
TLO,THI,NT=40.0,98.0,161
tg=np.linspace(TLO,THI,NT); dt=tg[1]-tg[0]; TH,TL=np.meshgrid(tg,tg,indexing="ij")

# lineshape forms f(m;mW) = m^p / D(m;mW) ; fixed denom = WHIZARD const-width propagator
def D_of(m,mW,denom):
    d=m*m-mW*mW
    if denom=="fix": return d*d+(mW*GW)**2
    wm=GW*m*m/mW; return d*d+wm*wm
FORMS={"fix p0":("fix",0),"fix p2":("fix",2),"fix p3":("fix",3),"run p2":("run",2)}

def logf(m,mW,denom,p): return p*np.log(np.maximum(m,1e-9)) - np.log(D_of(m,mW,denom))     # ln f

def gen_pe(mqq,mlv,mWW,denom,p):
    # tabulate per-(√s',mW) normalization  Z=∫∫ f(m_h)f(m_l) PS  on a √s' grid, interp per event
    slo,shi=mWW.min()-0.3,mWW.max()+0.3; ns=70; sgrid=np.linspace(slo,shi,ns)
    s2=sgrid*sgrid
    # precompute PS(√s' node) on the (NT,NT) grid
    PS=np.empty((ns,NT,NT))
    for k in range(ns):
        lam=(s2[k]-(TH+TL)**2)*(s2[k]-(TH-TL)**2); PS[k]=np.where(lam>0,np.sqrt(np.maximum(lam,0))/s2[k],0.0)
    s=mWW*mWW; lam_ev=(s-(mqq+mlv)**2)*(s-(mqq-mlv)**2); bad=lam_ev<=0
    tot=np.zeros(NMW)
    for j,mW in enumerate(mw_scan):
        fg=np.exp(p*np.log(tg)-np.log(D_of(tg,mW,denom))); Fout=fg[:,None]*fg[None,:]
        logZ=np.log(np.maximum((Fout[None]*PS).sum((1,2))*dt*dt,1e-300))    # (ns,)
        lz_ev=np.interp(mWW,sgrid,logZ)
        lfh=logf(mqq,mW,denom,p); lfl=logf(mlv,mW,denom,p)
        ps_ev=np.where(bad,1.0,np.sqrt(np.maximum(lam_ev,1.0))/s)
        tv=-2*lfh-2*lfl-2*np.log(ps_ev)+2*lz_ev; tv=np.where(bad,1e6,tv)
        tot[j]=tv.sum()
    # parabolic min
    i=int(np.argmin(tot))
    if 0<i<NMW-1:
        x0,x1,x2=mw_scan[i-1:i+2]; y0,y1,y2=tot[i-1:i+2]
        d=(x0-x1)*(x0-x2)*(x1-x2); A=(x2*(y1-y0)+x1*(y0-y2)+x0*(y2-y1))/d
        B=(x2*x2*(y0-y1)+x1*x1*(y2-y0)+x0*x0*(y1-y2))/d
        return -B/(2*A) if A>0 else x1
    return mw_scan[i]

print(f"{'ECM':>4} {'thr':>6}  " + "  ".join(f"{k:>8}" for k in FORMS) + "    (gen_pe − 80.419, MeV)")
RES={}
for ecm in ECMS:
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"],library="np")
    ok=(np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&(a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq,mlv,mWW=a["gen_Whad_m"][ok],a["gen_Wlep_m"][ok],a["gen_WW_m"][ok]
    thr="below" if ecm<TWO_MW else "above"
    row={name:1000*(gen_pe(mqq,mlv,mWW,den,p)-MWREF) for name,(den,p) in FORMS.items()}
    RES[ecm]=row
    print(f"{ecm:>4} {thr:>6}  " + "  ".join(f"{row[k]:+8.1f}" for k in FORMS))

# plot: gen_pe-mW vs ECM per form
plt.rcParams.update({"font.size":12})
fig,ax=plt.subplots(figsize=(10,6.5)); ecmA=np.array(ECMS)
mk={"fix p0":("s","C3","fixed-width (p0): WHIZARD propagator, NO decay factor"),
    "fix p2":("o","C2","const-width × m² decay (principled)"),
    "fix p3":("D","C4","const-width × m³ decay"),
    "run p2":("^","C0","running-width BW (DAY7)")}
for name in FORMS:
    m,c,lab=mk[name]; ax.plot(ecmA,[RES[e][name] for e in ECMS],m+"-",color=c,ms=9,lw=2,label=lab)
ax.axhline(0,color="k",ls="--",lw=1.3,label="perfect closure (gen mW=80.419)")
ax.axhspan(-5,5,color="0.9"); ax.axvline(TWO_MW,color="0.5",ls=":",lw=1.2); ax.text(TWO_MW+0.05,-180,"2mW=160.84",fontsize=9,color="0.4")
ax.set_xticks(ECMS); ax.set_xlabel("ECM [GeV]"); ax.set_ylabel("gen_pe − 80.419 [MeV]")
ax.set_title("Does the const-width × decay-factor model CLOSE uniformly?\n(WHIZARD propagator is const-width; m² decay factor = the physical W→qq phase space)")
ax.legend(fontsize=9,loc="lower right"); ax.grid(alpha=0.3)
png=f"{EOSW}/decay_closure_day8.png"; fig.savefig(png,dpi=130,bbox_inches="tight")
print(f"\n[plot] {png}\n[link] https://mdefranc.web.cern.ch/mW/conv_mw/decay_closure_day8.png")
