#!/usr/bin/env python3
# Visualise WHY the lnuqq GoF has a p-value~0 pile-up, from lnuqq_gof_breakdown.npz.
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2dist

d=np.load("/tmp/mdefranc/lnuqq_gof_breakdown.npz", allow_pickle=True)
gof=d["gof"]; terms=d["terms"]; tn=list(d["term_names"]); mloss=d["mloss"]; mh=d["mh"]
swapped=d["swapped"]; v=d["valid"]
fin=np.isfinite(gof)&(gof<1e4)
g=gof[fin]; T=terms[fin]; ml=mloss[fin]; mhh=mh[fin]; vv=v[fin]; sw=swapped[fin]
k=max(1,int(round(np.mean(g)))); pv=chi2dist.sf(g,k); lowp=pv<0.01
N=len(g)

fig,ax=plt.subplots(2,3,figsize=(17,9))
# (0,0) GoF dist vs chi2(k) on log-y -> shows the heavy tail
ax[0,0].hist(g,bins=80,range=(0,80),density=True,histtype="step",lw=1.8,color="C0",
             label=f"GoF (mean={np.mean(g):.1f}, med={np.median(g):.1f})")
xx=np.linspace(0.1,80,400); ax[0,0].plot(xx,chi2dist.pdf(xx,k),"r--",lw=1.3,label=f"χ²(k={k}) ref")
ax[0,0].set_yscale("log"); ax[0,0].set_xlabel("GoF [≈χ²]"); ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)
ax[0,0].set_title(f"GoF has heavier RIGHT TAIL than χ²(k={k}) → p~0 pile-up")
# (0,1) p-value dist
ax[0,1].hist(pv,bins=25,range=(0,1),histtype="step",lw=1.8,color="C2")
ax[0,1].axhline(N/25,color="k",ls=":",lw=1)
ax[0,1].set_xlabel(f"p-value [χ²(k={k})]"); ax[0,1].grid(alpha=.3)
ax[0,1].set_title(f"p<0.01: {100*np.mean(lowp):.1f}%   p<0.05: {100*np.mean(pv<0.05):.1f}%")
# (0,2) mean per-term: all vs low-p
x=np.arange(len(tn)); wln=.38
ax[0,2].bar(x-wln/2,[np.mean(T[:,j]) for j in range(len(tn))],wln,label="all events",color="C0")
ax[0,2].bar(x+wln/2,[np.mean(T[lowp][:,j]) for j in range(len(tn))],wln,label="low-p (<0.01)",color="C3")
ax[0,2].set_xticks(x); ax[0,2].set_xticklabels(tn,rotation=30); ax[0,2].legend(fontsize=9); ax[0,2].grid(alpha=.3)
ax[0,2].set_title("mean per-term residual: which terms blow up")
# (1,0) per-term contribution distributions (where does GoF mass live)
for j,name in enumerate(tn):
    a=T[:,j]; a=a[np.isfinite(a)&(a<60)]
    ax[1,0].hist(a,bins=60,range=(0,30),histtype="step",lw=1.5,label=f"{name} (μ={np.mean(T[:,j]):.1f})")
ax[1,0].set_yscale("log"); ax[1,0].set_xlabel("per-term residual"); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.3)
ax[1,0].set_title("per-term residual distributions")
# (1,1) dominant term among low-p events (pie/bar)
dom=np.argmax(T[lowp],axis=1)
fr=[100*np.mean(dom==j) for j in range(len(tn))]
ax[1,1].bar(x,fr,color="C3"); ax[1,1].set_xticks(x); ax[1,1].set_xticklabels(tn,rotation=30)
ax[1,1].set_ylabel("% of low-p events"); ax[1,1].grid(alpha=.3)
ax[1,1].set_title("dominant (largest) term per low-p event")
# (1,2) GoF vs |m_loss| (radiation) 2D
sel=np.isfinite(ml)
ax[1,2].hist2d(np.clip(np.abs(ml[sel]),0,20),np.clip(g[sel],0,60),bins=60,cmap="viridis",cmin=1)
ax[1,2].set_xlabel("|m_WW - ECM|  (radiation proxy) [GeV]"); ax[1,2].set_ylabel("GoF")
ax[1,2].set_title("high GoF tracks radiation (m_loss)")
fig.suptitle(f"lnuqq ecm160 GoF p~0 diagnosis  N={N}  |  low-p median|m_loss|={np.median(np.abs(ml[lowp])):.1f} vs all={np.median(np.abs(ml)):.1f} GeV  |  "
             f"Minuit-valid: low-p={100*np.mean(vv[lowp]):.0f}% vs all={100*np.mean(vv):.0f}%", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97])
out="/eos/user/m/mdefranc/www/mW/jax_lnuqq"; os.makedirs(out,exist_ok=True)
fig.savefig(f"{out}/gof_breakdown_ecm160.png",dpi=110)
print("wrote",f"{out}/gof_breakdown_ecm160.png")
