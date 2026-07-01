#!/usr/bin/env python3
# Final results figure for the forward-folding (MEM-by-reweighting) 1-D mW fit.
# Reads the /tmp/F_*.log matrix (fold 2d/qq/lv, gen_marg, identity per ECM) + the binning scan.
#   source .../setup.sh ; python3 jax_prototype/conv_mw_final_plot.py
import re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/eos/user/m/mdefranc/www/mW/conv_mw"
def grab(f):
    try: s=open(f).read()
    except: return None
    m=re.search(r"m̂W = ([\d.]+) GeV   σ_mW\(curv\) = ([\d.]+)",s)
    return (float(m.group(1)),float(m.group(2))) if m else None
ECMS=[157,160,163]
gm=[grab(f"/tmp/F_gm_{E}.log") for E in ECMS]
f2=[grab(f"/tmp/F_2d_{E}.log") for E in ECMS]
fq=[grab(f"/tmp/F_qq_{E}.log") for E in ECMS]
fl=[grab(f"/tmp/F_lv_{E}.log") for E in ECMS]
idt=[grab(f"/tmp/F_id_{E}.log") for E in ECMS]
# binning systematic band @160 (sensible configs bin<=0.5, sm<=0.6)
sysv=[]
for B in ["0.25","0.5"]:
    for S in ["0.0","0.6"]:
        g=grab(f"/tmp/sys_b{B}_s{S}.log")
        if g: sysv.append(g[0])
sysband=(max(sysv)-min(sysv)) if sysv else 0.0

fig,ax=plt.subplots(1,2,figsize=(15,5.6)); x=np.arange(3)
# Panel 1: closure
ax[0].plot(x,[g[0] for g in gm],"ks",ms=12,label="gen-level target (gen_marg)",zorder=3)
ax[0].errorbar(x,[v[0] for v in f2],yerr=[v[1] for v in f2],fmt="o",ms=10,color="C0",capsize=5,
               label="forward-fold reco (k-fold, mW-only, ~100% events)",zorder=4)
ax[0].plot(x,[v[0] for v in idt],"D",ms=7,color="C1",alpha=.8,label="identity test (estimator floor)",zorder=3)
for i,E in enumerate(ECMS):
    ax[0].annotate(f"{1000*(f2[i][0]-gm[i][0]):+.0f} MeV",(x[i],f2[i][0]),textcoords="offset points",xytext=(10,-14),fontsize=9,color="C0")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"ecm{E}" for E in ECMS]); ax[0].set_ylabel(r"$\hat m_W$ [GeV]")
ax[0].set_title(f"Closure: fold reco vs gen-level (+3/−24/−21 MeV); σ_mW≈7–10 MeV\nbinning syst @160 ≈ ±{1000*sysband/2:.0f} MeV (sensible configs)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
# Panel 2: precision, 2D vs 1D + independent floor
w=0.25
sq=np.array([v[1] for v in fq])*1000; sl=np.array([v[1] for v in fl])*1000; s2=np.array([v[1] for v in f2])*1000
floor=1000/np.sqrt(1/np.array([v[1] for v in fq])**2+1/np.array([v[1] for v in fl])**2)
ax[1].bar(x-w,sq,w,color="C2",label="m_qq only")
ax[1].bar(x,sl,w,color="C3",label="m_lν only")
ax[1].bar(x+w,s2,w,color="C0",label="2D joint (m_qq,m_lν)")
ax[1].plot(x+w,floor,"k_",ms=22,mew=2,label="independent-comb. floor")
for i in range(3):
    ax[1].annotate(f"−{100*(1-s2[i]/sq[i]):.0f}%",(x[i]+w,s2[i]),textcoords="offset points",xytext=(0,3),ha="center",fontsize=9,color="C0")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"ecm{E}" for E in ECMS]); ax[1].set_ylabel(r"$\sigma_{m_W}$ [MeV]")
ax[1].set_title("Precision: 2D ≈ independent combination of the two masses\n(~40% tighter than m_qq-only; score-corr≈0 ⇒ near-independent channels)")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3,axis="y")
plt.tight_layout(); png=f"{OUT}/fold_final_results.png"; plt.savefig(png,dpi=120)
print(f"[plot] {png}")
print(f"binning syst band (sensible) = {1000*sysband:.0f} MeV  -> ±{1000*sysband/2:.0f}")
