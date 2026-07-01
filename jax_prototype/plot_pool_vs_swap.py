#!/usr/bin/env python3
# Pooled jet prior (no swap) vs binned+swap, from lnuqq_pool_vs_swap.npz.
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d=np.load("/tmp/mdefranc/lnuqq_pool_vs_swap.npz")
mWs=d["mW_swap"]; mWp=d["mW_pool"]; cmw=d["mn_mW"]; v=d["valid"].astype(bool); swapped=d["swapped"]
Whs=d["Whad_swap"]; Whp=d["Whad_pool"]; gWh=d["gen_Whad_m"]
fin=np.isfinite(mWs)&np.isfinite(mWp)&np.isfinite(cmw); gv=fin&v
dsw=mWs-cmw; dpl=mWp-cmw; dd=mWp-mWs
fg=np.isfinite(Whs)&np.isfinite(Whp)&np.isfinite(gWh); ehs=Whs-gWh; ehp=Whp-gWh

fig,ax=plt.subplots(2,3,figsize=(17,9))
# mW overlay
for arr,l,c in [(cmw[v],"Minuit (valid)","k"),(mWs[fin],"JAX swap","C0"),(mWp[fin],"JAX pool (no swap)","C3")]:
    a=arr[(arr>=60)&(arr<=100)]
    ax[0,0].hist(a,bins=80,range=(60,100),histtype="step",lw=1.8,color=c,label=f"{l} (μ={np.mean(a):.2f},σ={np.std(a):.2f})")
ax[0,0].axvline(80.385,color="grey",ls=":",lw=1); ax[0,0].set_xlabel("mW [GeV]"); ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)
ax[0,0].set_title("mW: pool vs swap vs Minuit")
# ΔmW vs Minuit overlay
ax[0,1].hist(dsw[gv],bins=120,range=(-3,3),histtype="step",lw=1.8,color="C0",label=f"swap (med|Δ|={np.median(np.abs(dsw[gv])):.3f})")
ax[0,1].hist(dpl[gv],bins=120,range=(-3,3),histtype="step",lw=1.8,color="C3",label=f"pool (med|Δ|={np.median(np.abs(dpl[gv])):.3f})")
ax[0,1].set_xlabel("mW(JAX) - mW(Minuit) [GeV]"); ax[0,1].legend(fontsize=9); ax[0,1].grid(alpha=.3)
ax[0,1].set_title("recovery vs Minuit-valid")
# pool vs swap direct
ax[0,2].hist(dd[fin],bins=140,range=(-3,3),histtype="step",lw=1.8,color="C2")
ax[0,2].set_xlabel("mW(pool) - mW(swap) [GeV]"); ax[0,2].grid(alpha=.3)
ax[0,2].set_title(f"pool vs swap: med|Δ|={np.median(np.abs(dd[fin])):.3f}, |Δ|<0.05={100*np.mean(np.abs(dd[fin])<0.05):.0f}%")
# recovery bars
thrs=[0.1,0.25,0.5,1.0]; x=np.arange(len(thrs)); w=.38
rs=[100*np.mean(np.abs(dsw[gv])<t) for t in thrs]; rp=[100*np.mean(np.abs(dpl[gv])<t) for t in thrs]
ax[1,0].bar(x-w/2,rs,w,label="swap",color="C0"); ax[1,0].bar(x+w/2,rp,w,label="pool",color="C3")
ax[1,0].set_xticks(x); ax[1,0].set_xticklabels([f"|Δ|<{t}" for t in thrs]); ax[1,0].set_ylabel("% of valid")
ax[1,0].legend(); ax[1,0].grid(alpha=.3); ax[1,0].set_title("recovery vs Minuit-valid")
# 2D pool vs swap
ax[1,1].hist2d(mWs[fin],mWp[fin],bins=80,range=[[60,100],[60,100]],cmap="viridis",cmin=1)
ax[1,1].plot([60,100],[60,100],"r--",lw=1); ax[1,1].set_xlabel("mW swap"); ax[1,1].set_ylabel("mW pool")
ax[1,1].set_title("pool vs swap 2D")
# DECISIVE: accuracy vs GEN truth (Whad - gen_Whad), swap vs pool
ax[1,2].hist(ehs[fg],bins=120,range=(-15,15),histtype="step",lw=1.8,color="C0",
             label=f"swap (RMS={np.std(ehs[fg]):.2f}, bias={np.mean(ehs[fg]):+.2f})")
ax[1,2].hist(ehp[fg],bins=120,range=(-15,15),histtype="step",lw=1.8,color="C3",
             label=f"pool (RMS={np.std(ehp[fg]):.2f}, bias={np.mean(ehp[fg]):+.2f})")
ax[1,2].set_xlabel("postfit Whad - gen_Whad [GeV]"); ax[1,2].legend(fontsize=8); ax[1,2].grid(alpha=.3)
ax[1,2].set_title("ACCURACY vs GEN (unbiased: no Minuit) — swap≈pool")
fig.suptitle(f"lnuqq ecm160  pooled p-binned jet prior (NO swap) vs binned+swap  |  swap chosen {100*np.mean(swapped):.0f}%  |  "
             f"vs Minuit |Δ|<0.25: swap {100*np.mean(np.abs(dsw[gv])<0.25):.1f}%/pool {100*np.mean(np.abs(dpl[gv])<0.25):.1f}% (biased)  |  "
             f"vs GEN RMS: swap {np.std(ehs[fg]):.2f}/pool {np.std(ehp[fg]):.2f} (decisive)", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97])
out="/eos/user/m/mdefranc/www/mW/jax_lnuqq"; os.makedirs(out,exist_ok=True)
fig.savefig(f"{out}/pool_vs_swap_ecm160.png",dpi=110)
print("wrote",f"{out}/pool_vs_swap_ecm160.png")
