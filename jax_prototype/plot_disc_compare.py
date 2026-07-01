#!/usr/bin/env python3
# Discriminant comparison: pairing efficiency vs pairing-confusion bias (gap-from-true), aggregated over
# the seed×bin grid with error bars. Reads conv_mw_4q_ecm160_grid_*.json.
import json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD="jax_prototype/anatomy_results"; EOSW="/eos/user/m/mdefranc/www/mW/conv_mw_4q"
files=sorted(glob.glob(f"{RD}/conv_mw_4q_ecm160_grid_s*_b*.json"))
modes=["true","bw","pll","pllnops","foldpick","kinfit"]
lab={"true":"gen-true","bw":"bare pole BW","pll":"ISR-conv (√λ on)","pllnops":"lineshape BW (no √λ)","foldpick":"forward-fold template","kinfit":"kinfit"}
col={"true":"C2","bw":"C0","pll":"C4","pllnops":"C1","foldpick":"C3","kinfit":"C5"}
clos={m:[] for m in modes}; gap={m:[] for m in modes}; eff={m:[] for m in modes}
for f in files:
    R={r["name"]:r for r in json.load(open(f))["results"]}
    if "true" not in R: continue
    ct=1000*R["true"]["closure"]
    for m in modes:
        if m in R: clos[m].append(1000*R[m]["closure"]); gap[m].append(1000*R[m]["closure"]-ct); eff[m].append(100*R[m]["eff"])

fig,ax=plt.subplots(1,2,figsize=(14,5.4))
# (a) efficiency vs gap-from-true scatter (the money plot: up-left = high eff, low bias)
for m in modes:
    if m=="true": continue
    e=np.mean(eff[m]); g=np.mean(gap[m]); ge=np.std(gap[m])/np.sqrt(len(gap[m]))
    ax[0].errorbar(e,g,yerr=ge,fmt='o',ms=10,color=col[m],capsize=4)
    ax[0].annotate(lab[m],(e,g),textcoords="offset points",xytext=(8,6),fontsize=10,color=col[m])
ax[0].axhline(0,color="C2",ls="--",lw=1.2,label="gen-true (closes)")
ax[0].set_xlabel("jet→W pairing efficiency [%]"); ax[0].set_ylabel("pairing-confusion bias  closure−true [MeV]")
ax[0].set_title("Pairing efficiency vs confusion bias (seed×bin grid, error=SE)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
# (b) closure spread per mode (box-ish: mean±std over grid)
xs=np.arange(len(modes))
for i,m in enumerate(modes):
    c=np.array(clos[m]); ax[1].errorbar(i,c.mean(),yerr=c.std(),fmt='s',ms=9,color=col[m],capsize=5)
    ax[1].scatter(np.full(len(c),i)+np.random.default_rng(1).normal(0,.04,len(c)),c,s=10,color=col[m],alpha=.35)
ax[1].axhline(0,color="k",lw=.7)
ax[1].set_xticks(xs); ax[1].set_xticklabels([lab[m] for m in modes],rotation=20,ha="right",fontsize=9)
ax[1].set_ylabel("closure reco_fit−gen_pe [MeV]"); ax[1].set_title("Closure per pairing (mean±std over grid; dots = grid points)"); ax[1].grid(axis="y",alpha=.3)
fig.tight_layout(); fig.savefig(f"{EOSW}/disc_compare_ecm160.png",dpi=110); plt.close(fig)
print("saved",f"{EOSW}/disc_compare_ecm160.png")
for m in modes:
    print(f"  {m:9s} eff={np.mean(eff[m]):5.1f}%  closure={np.mean(clos[m]):+6.1f}±{np.std(clos[m]):4.1f}  gap={np.mean(gap[m]):+6.1f}±{np.std(gap[m]):4.1f} (N={len(gap[m])})")
