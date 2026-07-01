#!/usr/bin/env python3
# Overlay distributions: differentiable JAX fit (scipy-BFGS) vs Minuit vs gen,
# from /tmp/mdefranc/lnuqq_compare.npz (produced by recovery_lnuqq_mp.py).
import numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from scipy.stats import chi2 as chi2dist
d=np.load("/tmp/mdefranc/lnuqq_compare.npz")
mWd=d["mW_diff"]; v=d["valid"].astype(bool); vl=d["valid_loose"].astype(bool)
mWm=d["mn_mW"]; Whd=d["Whad_diff"]; Whm=d["mn_Whad_m"]; WWd=d["WW_diff"]; WWm=d["mn_WW_m"]
gWh=d.get("gen_Whad_m"); gWW=d.get("gen_WW_m"); chid=d["chi2_diff"]; chim=d["mn_chi2"]
gof=d.get("gof"); swapped=d.get("swapped")
fin=np.isfinite(mWd)
N=len(mWd)
def H(ax,arrs,rng,bins=80,xl=""):
    for a,l in arrs:
        a=a[np.isfinite(a)]; a=a[(a>=rng[0])&(a<=rng[1])]
        ax.hist(a,bins=bins,range=rng,histtype="step",lw=1.8,label=f"{l} (μ={np.mean(a):.2f},σ={np.std(a):.2f})")
    ax.set_xlabel(xl); ax.legend(fontsize=7); ax.grid(alpha=.3)

fig,ax=plt.subplots(2,4,figsize=(21,9))
# mW overlay
H(ax[0,0],[(mWm[v],"Minuit (valid)"),(mWd[fin],"JAX diff (all)"),(mWd[fin&~v],"JAX diff (Minuit-INVALID)")],
  (60,100),xl="kinfit mW [GeV]"); ax[0,0].axvline(80.385,color="k",ls=":",lw=1); ax[0,0].set_title("mW: JAX recovers events Minuit rejects")
# Whad vs gen
arrs=[(Whm[v],"Minuit (valid)"),(Whd[fin],"JAX diff (all)")]
if gWh is not None: arrs=[(gWh,"gen")]+arrs
H(ax[0,1],arrs,(40,120),xl="hadronic W mass [GeV]"); ax[0,1].set_title("Whad: fit vs gen truth")
# WW vs gen
arrs=[(WWm[v],"Minuit (valid)"),(WWd[fin],"JAX diff (all)")]
if gWW is not None: arrs=[(gWW,"gen")]+arrs
H(ax[0,2],arrs,(140,175),xl="WW mass [GeV]"); ax[0,2].set_title("m_WW: fit vs gen truth")
# delta mW
g=fin&v&np.isfinite(mWm); dd=(mWd-mWm)[g]
ax[1,0].hist(dd,bins=120,range=(-5,5),histtype="step",lw=1.8,color="C3")
ax[1,0].set_xlabel("mW(JAX) - mW(Minuit) [GeV]"); ax[1,0].grid(alpha=.3)
ax[1,0].set_title(f"per-event agreement (valid): median|Δ|={np.median(np.abs(dd)):.3f},  |Δ|<1={100*np.mean(np.abs(dd)<1):.1f}%")
# 2D
ax[1,1].hist2d(mWm[g],mWd[g],bins=80,range=[[60,100],[60,100]],cmap="viridis",cmin=1)
ax[1,1].plot([60,100],[60,100],"r--",lw=1); ax[1,1].set_xlabel("mW Minuit"); ax[1,1].set_ylabel("mW JAX diff"); ax[1,1].set_title("2D mW correlation (valid)")
# chi2
H(ax[1,2],[(chim[v],"Minuit (valid)"),(chid[fin],"JAX diff (all)")],
  (np.nanpercentile(np.concatenate([chim[v],chid[fin]]),1),np.nanpercentile(np.concatenate([chim[v],chid[fin]]),99)),
  xl="fit chi2 (-2logL)"); ax[1,2].set_title("chi2")

# GoF (mode-referenced residual ~ chi2)
if gof is not None:
    gg=gof[fin&np.isfinite(gof)&(gof>0)&(gof<200)]
    k=max(1,int(round(np.mean(gg))))
    ax[0,3].hist(gg,bins=80,range=(0,80),density=True,histtype="step",lw=1.8,color="C0",
                 label=f"GoF (mean={np.mean(gg):.1f}, med={np.median(gg):.1f})")
    xx=np.linspace(0.1,80,400); ax[0,3].plot(xx,chi2dist.pdf(xx,k),"r--",lw=1.3,label=f"χ²(k={k}) ref")
    ax[0,3].set_xlabel("GoF = Σ(term−mode)  [≈χ²]"); ax[0,3].legend(fontsize=7); ax[0,3].grid(alpha=.3)
    ax[0,3].set_title("goodness-of-fit from −2logL")
    # p-value distribution (should be ~flat if calibrated)
    pv=chi2dist.sf(gg,k)
    ax[1,3].hist(pv,bins=25,range=(0,1),histtype="step",lw=1.8,color="C2")
    ax[1,3].axhline(len(pv)/25,color="k",ls=":",lw=1); ax[1,3].set_xlabel(f"p-value [χ²(k={k})]")
    ax[1,3].grid(alpha=.3); ax[1,3].set_title("p-value (flat = calibrated)")
rec1=100*np.mean(np.abs(dd)<1.0)
swp=f"  swap={100*np.mean(swapped):.0f}%" if swapped is not None else ""
fig.suptitle(f"lnuqq ecm160  N={N}  |  Minuit valid={100*np.mean(v):.1f}% (loose {100*np.mean(vl):.1f}%)  |  "
             f"JAX diff (binned+swap) finite=100%, recovers Minuit mW(|Δ|<1)={rec1:.1f}%{swp}", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.97])
out="/eos/user/m/mdefranc/www/mW/jax_lnuqq"; os.makedirs(out,exist_ok=True)
fig.savefig(f"{out}/compare_ecm160.png",dpi=110)
print("wrote",f"{out}/compare_ecm160.png")
