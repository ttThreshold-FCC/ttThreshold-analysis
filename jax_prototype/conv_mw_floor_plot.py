#!/usr/bin/env python3
# ── DAY5 headline: the residual closure floor is NOT a binning artifact — it is a per-scheme STRUCTURAL
#    bias, complementary between grid and exact, driven by the √s'-vs-2mW regime.
#    (left)   exact-scheme closure vs KDE bandwidth h, all 3 ECMs  → 157/160 nullable, 163 stuck at −27
#    (mid)    grid-scheme  closure vs KDE bandwidth h, all 3 ECMs  → 157/163 OK, 160 stuck at −12
#    (right)  163 EXACT closure vs gen-√s' cut                      → collapses exactly at 2mW=160.76
#             ⇒ the 163 exact floor is the sub-threshold ISR tail (drop ≠ clip), not binning.
#  Reads anatomy_results/anatomy_ecm{E}_{full_,kb_,g015,cs_}*.json (conv_mw_closure_anatomy.py = the
#  canonical fit, instrumented).   source LCG_106 ; python3 jax_prototype/conv_mw_floor_plot.py
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RDIR = os.path.join(os.path.dirname(__file__), "anatomy_results")
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW, exist_ok=True)
ECMS = [157, 160, 163]; COL = {157: "C0", 160: "C1", 163: "C3"}
TWO_MW = 2*80.379

def kde_curve(ecm, scheme):
    """h -> closure[MeV] for the given scheme, from all KDE JSONs (no clip), deduped by h."""
    pts = {}
    for f in glob.glob(os.path.join(RDIR, f"anatomy_ecm{ecm}_*kde*.json")) + \
             glob.glob(os.path.join(RDIR, f"anatomy_ecm{ecm}_g015.json")):
        d = json.load(open(f))
        if d.get("est") != "kde" or d.get("fold_clip", 100) < 100: continue
        if scheme not in d["decomp"]: continue
        pts[round(d["kde_h"], 3)] = (1000*d["decomp"][scheme]["closure"], 1000*d["decomp"][scheme]["floor"])
    h = sorted(pts); return np.array(h), np.array([pts[x][0] for x in h]), np.array([pts[x][1] for x in h])

def hist_anchor(ecm, scheme, fbin):
    for f in glob.glob(os.path.join(RDIR, f"anatomy_ecm{ecm}_full_hist{int(fbin*100):03d}.json")):
        d = json.load(open(f))
        if scheme in d["decomp"]: return 1000*d["decomp"][scheme]["closure"]
    return None

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
for s, (axi, scheme) in enumerate(zip(ax[:2], ["exact", "grid"])):
    axi.axhspan(-5, 5, color="green", alpha=0.10); axi.axhline(0, color="k", lw=0.8)
    for ecm in ECMS:
        h, clos, floor = kde_curve(ecm, scheme)
        if len(h) == 0: continue
        axi.plot(h, clos, "o-", color=COL[ecm], label=f"ecm{ecm}")
        a25 = hist_anchor(ecm, scheme, 0.25)
        if a25 is not None: axi.plot([0.43], [a25], "*", color=COL[ecm], ms=12)
    axi.set_xlabel("KDE bandwidth h [GeV]"); axi.set_ylabel("closure (fold reco − gen target) [MeV]")
    axi.set_title(f"{scheme} scheme  (★ = hist bin0.25 @h≈0.43)"); axi.grid(alpha=0.25); axi.legend(fontsize=9)

# right: 163 exact closure vs √s' cut (the money panel)
cs = []
for f in glob.glob(os.path.join(RDIR, "anatomy_ecm163_cs_*.json")):
    d = json.load(open(f)); cut = float(f.split("_cs_")[1].replace(".json", ""))
    cs.append((cut, 1000*d["decomp"]["exact"]["closure"], 1000*d["decomp"]["exact"]["floor"], d["N"]))
cs.sort()
ax[2].axhspan(-5, 5, color="green", alpha=0.10); ax[2].axhline(0, color="k", lw=0.8)
if cs:
    cut = np.array([c[0] for c in cs])
    ax[2].plot(cut, [c[1] for c in cs], "s-", color="C3", label="closure")
    ax[2].plot(cut, [c[2] for c in cs], "o--", color="C3", alpha=0.6, label="floor (identity)")
    # the no-cut full-sample baseline
    base = json.load(open(os.path.join(RDIR, "anatomy_ecm163_full_hist025.json")))
    ax[2].axhline(1000*base["decomp"]["exact"]["closure"], color="gray", ls=":", label="no cut (−27)")
ax[2].axvline(TWO_MW, color="purple", ls="-", lw=1.2, label="2mW=160.76")
ax[2].set_xlabel("gen √s' lower cut [GeV]"); ax[2].set_ylabel("163 exact closure [MeV]")
ax[2].set_title("163: closure collapses when the sub-threshold tail is dropped"); ax[2].grid(alpha=0.25); ax[2].legend(fontsize=8)

plt.suptitle("DAY5: KDE kills the binning artifact, exposing a per-scheme STRUCTURAL closure floor "
             "(grid↔exact complementary across the 2mW threshold)", y=1.00, fontsize=11)
plt.tight_layout(); png = f"{EOSW}/closure_floor_day5.png"; plt.savefig(png, dpi=120, bbox_inches="tight")
print(f"[plot] {png}")
