#!/usr/bin/env python3
# Compare two ISR modes (default photon vs kfit) from lnuqq_isr_compare.npz.
# Reads d["modes"]=[A,B] and the per-mode arrays mW_<m>, Whad_<m>, gof_<m>.
import numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import chi2 as chi2dist

TAG=sys.argv[1] if len(sys.argv)>1 else ""
d=np.load(f"/tmp/mdefranc/lnuqq_isr_compare{TAG}.npz")
modes=[str(m) for m in d["modes"]]; A,B=modes[0],modes[1]
gWh=d["gen_Whad_m"]
def get(m): return d[f"mW_{m}"],d[f"Whad_{m}"],d[f"gof_{m}"]
mWa,Wha,ga=get(A); mWb,Whb,gb=get(B)
def pv(g):
    fin=np.isfinite(g)&(g>0)&(g<1e4); gg=g[fin]; k=max(1,int(round(np.mean(gg))))
    return chi2dist.sf(gg,k), k, gg
pva,ka,gga=pv(ga); pvb,kb,ggb=pv(gb)
LA=f"{A}"; LB=f"{B}"; cA="C3"; cB="C0"

fig,ax=plt.subplots(2,3,figsize=(17,9))
ax[0,0].hist(pva,bins=25,range=(0,1),histtype="step",lw=1.8,color=cA,label=f"{LA} (p<0.01={100*np.mean(pva<0.01):.1f}%)")
ax[0,0].hist(pvb,bins=25,range=(0,1),histtype="step",lw=1.8,color=cB,label=f"{LB} (p<0.01={100*np.mean(pvb<0.01):.1f}%)")
ax[0,0].axhline(len(pvb)/25,color="k",ls=":",lw=1); ax[0,0].set_xlabel("p-value"); ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)
ax[0,0].set_title("p-value")
ax[0,1].hist(gga,bins=80,range=(0,80),density=True,histtype="step",lw=1.8,color=cA,label=f"{LA} (mean={np.mean(gga):.1f})")
ax[0,1].hist(ggb,bins=80,range=(0,80),density=True,histtype="step",lw=1.8,color=cB,label=f"{LB} (mean={np.mean(ggb):.1f})")
ax[0,1].set_yscale("log"); ax[0,1].set_xlabel("GoF [≈χ²]"); ax[0,1].legend(fontsize=8); ax[0,1].grid(alpha=.3)
ax[0,1].set_title("GoF distribution")
# free mW full range (show runaway tail)
for arr,l,c in [(mWa,LA,cA),(mWb,LB,cB)]:
    a=arr[np.isfinite(arr)]
    run=100*np.mean(np.abs(a-80.385)>5)
    a=a[(a>=60)&(a<=110)]
    ax[0,2].hist(a,bins=100,range=(60,110),histtype="step",lw=1.8,color=c,label=f"{l} (μ={np.mean(a):.2f},σ={np.std(a):.2f},tail={run:.1f}%)")
ax[0,2].axvline(80.385,color="grey",ls=":",lw=1); ax[0,2].set_yscale("log"); ax[0,2].set_xlabel("mW [GeV]"); ax[0,2].legend(fontsize=8); ax[0,2].grid(alpha=.3)
ax[0,2].set_title("free mW (log y: runaway tail)")
# accuracy vs gen
fg=np.isfinite(Wha)&np.isfinite(Whb)&np.isfinite(gWh); ea=(Wha-gWh)[fg]; eb=(Whb-gWh)[fg]
ax[1,0].hist(ea,bins=120,range=(-15,15),histtype="step",lw=1.8,color=cA,label=f"{LA} (bias={np.mean(ea):+.2f},RMS={np.std(ea):.2f})")
ax[1,0].hist(eb,bins=120,range=(-15,15),histtype="step",lw=1.8,color=cB,label=f"{LB} (bias={np.mean(eb):+.2f},RMS={np.std(eb):.2f})")
ax[1,0].set_xlabel("postfit Whad - gen_Whad [GeV]"); ax[1,0].legend(fontsize=8); ax[1,0].grid(alpha=.3)
ax[1,0].set_title("accuracy vs GEN")
# per-event mW shift
ax[1,1].hist((mWb-mWa)[np.isfinite(mWa)&np.isfinite(mWb)],bins=120,range=(-5,5),histtype="step",lw=1.8,color="C2")
ax[1,1].set_xlabel(f"mW({LB}) - mW({LA}) [GeV]"); ax[1,1].grid(alpha=.3); ax[1,1].set_title("per-event mW shift")
# mW core comparison (zoom)
for arr,l,c in [(mWa,LA,cA),(mWb,LB,cB)]:
    a=arr[np.isfinite(arr)]; a=a[(a>=75)&(a<=86)]
    ax[1,2].hist(a,bins=80,range=(75,86),histtype="step",lw=1.8,color=c,label=f"{l} core σ={np.std(a):.2f}")
ax[1,2].axvline(80.385,color="grey",ls=":",lw=1); ax[1,2].set_xlabel("mW [GeV]"); ax[1,2].legend(fontsize=8); ax[1,2].grid(alpha=.3)
ax[1,2].set_title("mW core (zoom)")
fig.suptitle(f"lnuqq ecm160  ISR: {LA} vs {LB}  |  p<0.01: {100*np.mean(pva<0.01):.1f}%→{100*np.mean(pvb<0.01):.1f}%  |  "
             f"mW runaway: {100*np.mean(np.abs(mWa[np.isfinite(mWa)]-80.385)>5):.1f}%→{100*np.mean(np.abs(mWb[np.isfinite(mWb)]-80.385)>5):.1f}%  |  "
             f"Whad-gen RMS: {np.std(ea):.2f}→{np.std(eb):.2f}", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.97])
out="/eos/user/m/mdefranc/www/mW/jax_lnuqq"; os.makedirs(out,exist_ok=True)
fig.savefig(f"{out}/isr_compare_ecm160.png",dpi=110)
print("wrote",f"{out}/isr_compare_ecm160.png")
