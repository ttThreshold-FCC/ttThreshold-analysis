#!/usr/bin/env python3
# ── DAY6 HEADLINE: the DAY5 "−25 MeV structural deep-ISR floor at ecm163" is largely a parab_min ──────
# coarse-mW-scan-grid artifact.  The exact estimator locates each fit's minimum with a 5-point parabola
# over the mW scan (NMW points on [79,81.5]).  At NMW=61 the window is ~167 MeV wide; on the asymmetric
# near-threshold 163 profile the parabola mislocates the minimum by ~−20 MeV.  Converging NMW collapses
# the floor −25→−5.4 (gen_pe target invariant) and shrinks the 157/160/163 closure scatter 26→~7 MeV.
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RD = os.path.join(os.path.dirname(__file__), "anatomy_results"); EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw"
def g(fn):
    p=f"{RD}/{fn}.json"
    if not os.path.exists(p): return None
    d=json.load(open(p)); return dict(floor=1000*(d["decomp"]["exact"]["identity"]-d["gen_pe"]),
                                      closure=1000*d["decomp"]["exact"]["closure"], genpe=d["gen_pe"])
# 163 NMW convergence
NMW=[61,121,241,481,961]
fns=["anatomy_ecm163_d6_base","anatomy_ecm163_d6n_163_nmw121","anatomy_ecm163_d6g_base",
     "anatomy_ecm163_d6n_163_nmw481","anatomy_ecm163_d6n_163_nmw961"]
pts=[(n,g(f)) for n,f in zip(NMW,fns)]; pts=[(n,r) for n,r in pts if r]
kde=g("anatomy_ecm163_d6n_163_kde_nmw481")

fig,ax=plt.subplots(1,2,figsize=(13,5))
# (A) floor & closure vs NMW for 163
xs=[2500/(n-1) for n,_ in pts]
ax[0].plot(xs,[r["floor"] for _,r in pts],"o-",color="C3",label="floor = id_fold(gen)−gen_pe")
ax[0].plot(xs,[r["closure"] for _,r in pts],"s--",color="C0",label="closure = fold(reco)−gen_pe")
if kde: ax[0].plot(2500/480,kde["floor"],"D",color="C2",ms=9,label="KDE0.25 floor (unbinned, NMW481)")
conv=np.mean([r["floor"] for n,r in pts if n>=121])
ax[0].axhline(conv,color="grey",ls=":",lw=1); ax[0].annotate(f"converged floor ≈ {conv:.1f} MeV",(xs[-1],conv),fontsize=9,va="bottom")
ax[0].annotate("DAY5 used NMW=61\n(41.7 MeV) → −25",(xs[0],pts[0][1]["floor"]),fontsize=8.5,color="C3",ha="left",va="top")
ax[0].set_xscale("log"); ax[0].invert_xaxis(); ax[0].set_xlabel("mW-scan grid spacing [MeV]  (NMW: 61→961)")
ax[0].set_ylabel("MeV"); ax[0].set_title("ecm163: the −25 floor is a parab_min grid artifact\n→ converges to ≈−5 MeV"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
# (B) closure 157/160/163 coarse(61) vs fine(481)
ecms=[157,160,163]; co=[]; fi=[]
for e in ecms:
    r61=g(f"anatomy_ecm{e}_d6n_{e}_nmw61") if e!=163 else g("anatomy_ecm163_d6_base")
    r48=g(f"anatomy_ecm{e}_d6n_{e}_nmw481") if e!=163 else g("anatomy_ecm163_d6n_163_nmw481")
    co.append(r61["closure"]); fi.append(r48["closure"])
x=np.arange(3); w=0.35
ax[1].bar(x-w/2,co,w,color="C7",label="NMW=61 (coarse, DAY5)")
ax[1].bar(x+w/2,fi,w,color="C2",label="NMW=481 (converged)")
ax[1].axhline(0,color="k",lw=0.6)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"ecm{e}" for e in ecms]); ax[1].set_ylabel("closure [MeV]")
ax[1].set_title(f"closure scatter collapses: span {max(co)-min(co):.0f}→{max(fi)-min(fi):.0f} MeV"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3,axis="y")
for i,(c_,f_) in enumerate(zip(co,fi)):
    ax[1].annotate(f"{c_:.0f}",(i-w/2,c_),ha="center",va="top" if c_<0 else "bottom",fontsize=8)
    ax[1].annotate(f"{f_:.0f}",(i+w/2,f_),ha="center",va="top" if f_<0 else "bottom",fontsize=8)
fig.suptitle("DAY6: the ecm163 'deep-ISR floor' was mostly a coarse-grid (parab_min) artifact — exact, hist0.25, full N",fontsize=11.5)
fig.tight_layout(); png=f"{EOSW}/nmw_convergence_day6.png"; fig.savefig(png,dpi=120); print(f"[plot] {png}")
print(f"163 converged floor (NMW>=121) = {conv:.2f} MeV;  KDE0.25@481 floor = {kde['floor'] if kde else 'NA'}")
print(f"closure span: NMW61 {min(co):.1f}..{max(co):.1f} ({max(co)-min(co):.1f}) -> NMW481 {min(fi):.1f}..{max(fi):.1f} ({max(fi)-min(fi):.1f})")
