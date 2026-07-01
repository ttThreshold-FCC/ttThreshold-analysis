#!/usr/bin/env python3
# Test + plots for the standalone BW jet->W pairing tool (WWFunctions/BWPairing.h).
#   - correct-pairing fraction on WW (all / matched-to-quarks / unmatched)
#   - posterior-probability calibration (does prob_best track the true rate?)
#   - WW-vs-ZZ discrimination from the best-pairing gof (W hypothesis)
# Usage: python3 plot_bw_pairing.py [WW.root] [ZZ.root] [outdir]
import os, sys
import numpy as np
import ROOT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WW_FN  = sys.argv[1] if len(sys.argv) > 1 else "outputs/treemaker/4q/bwpairing/had_ww/p8_ee_WW_ecm160.root"
ZZ_FN  = sys.argv[2] if len(sys.argv) > 2 else "outputs/treemaker/4q/bwpairing/had_zz/p8_ee_ZZ_ecm160.root"
OUTDIR = sys.argv[3] if len(sys.argv) > 3 else "/eos/user/m/mdefranc/www/mW/bwpairing"
os.makedirs(OUTDIR, exist_ok=True)


def ensure_index_php(outdir):
    """Drop the CERN web-gallery index.php into the publish dir (so the plots are
    browsable). Reuse the existing convention from a parent www directory."""
    import shutil
    dst = os.path.join(outdir, "index.php")
    if os.path.exists(dst):
        return
    for cand in (os.path.join(os.path.dirname(outdir.rstrip("/")), "index.php"),
                 "/eos/user/m/mdefranc/www/mW/index.php"):
        if os.path.exists(cand):
            shutil.copy(cand, dst)
            print("wrote:", dst)
            return


ensure_index_php(OUTDIR)


def load(fn, cols):
    rdf = ROOT.RDataFrame("events", fn)
    a = rdf.AsNumpy(cols)
    return {c: np.asarray(a[c]) for c in cols}


ww_cols = ["gen_pairing_true", "bwpair_pairing", "bwpair_correct",
           "bwpair_gof_best", "bwpair_prob_best", "bwpair_dgof",
           "jet1_matched_q_dR", "jet2_matched_q_dR", "jet3_matched_q_dR", "jet4_matched_q_dR"]
WW = load(WW_FN, ww_cols)
ZZ = load(ZZ_FN, ["bwpair_gof_best", "bwpair_prob_best", "bwpair_dgof"])

pt   = WW["gen_pairing_true"].astype(int)
corr = WW["bwpair_correct"].astype(float)
good = pt >= 0                                   # gen 4q splits 2-2
dR   = np.stack([WW["jet%d_matched_q_dR" % i] for i in (1, 2, 3, 4)], 1)
matched   = good & (dR.max(1) < 0.1)             # "passes step1 matching"
unmatched = good & ~matched

print("=" * 64)
print(f"WW events: {len(pt)}   gen 2-2 split (good): {good.mean():.3f}")
print(f"   passes step1 matching (dR<0.1, all 4 jets): {matched.mean():.3f} of all,  "
      f"{(matched.sum()/max(1,good.sum())):.3f} of good")
print("-" * 64)
print("CORRECT-PAIRING FRACTION (BW tool, no fit):")
print(f"   all good events : {corr[good].mean():.3f}   (chance = 0.333)")
print(f"   matched         : {corr[matched].mean():.3f}")
print(f"   unmatched       : {corr[unmatched].mean():.3f}")
print("-" * 64)

# ── Truth reliability: gen_pairing_true is only trustworthy where the jets
#    actually match the quarks. Show how (a) the event fraction and (b) the
#    correct-pairing rate behave as the max jet-quark dR cut is tightened. ──
dmax = dR.max(1)
print("CORRECT-PAIRING vs max jet-quark dR cut (truth reliability):")
print(f"   {'dRmax<':>8} {'frac_evts':>10} {'correct':>9}")
for cut in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 1e9]:
    m = good & (dmax < cut)
    tag = "all" if cut > 1e8 else f"{cut:.2f}"
    print(f"   {tag:>8} {m.mean():>10.3f} {corr[m].mean():>9.3f}")
# correct rate in dR SHELLS (not cumulative) — where does truth become noise?
print("   --- in dRmax shells ---")
sh = [0, 0.05, 0.1, 0.2, 0.4, 0.8, 1e9]
for lo, hi in zip(sh[:-1], sh[1:]):
    m = good & (dmax >= lo) & (dmax < hi)
    if m.sum() < 50:
        continue
    print(f"   {lo:.2f}-{hi if hi<1e8 else float('inf'):.2f}: frac={m.mean():.3f} correct={corr[m].mean():.3f} (n={int(m.sum())})")
print("-" * 64)

# ── Probability calibration: among events with prob_best in a bin, the actual
#    correct rate should track the mean prob_best (perfect = diagonal). ──
pb = WW["bwpair_prob_best"]
edges = np.linspace(1.0 / 3.0, 1.0, 11)
cx, cy, cn = [], [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = good & (pb >= lo) & (pb < hi)
    if m.sum() < 20:
        continue
    cx.append(pb[m].mean()); cy.append(corr[m].mean()); cn.append(int(m.sum()))
print("CALIBRATION (predicted prob_best vs actual correct rate):")
for x, y, nb in zip(cx, cy, cn):
    print(f"   prob~{x:.2f}  actual={y:.2f}  (n={nb})")
print("=" * 64)

# ── Plot 1: best-pairing gof — WW matched / unmatched / ZZ ──
gw_m = WW["bwpair_gof_best"][matched]
gw_u = WW["bwpair_gof_best"][unmatched]
gz   = ZZ["bwpair_gof_best"]
bins = np.linspace(0, 40, 61)
plt.figure(figsize=(7, 5))
plt.hist(gw_m, bins=bins, density=True, histtype="step", lw=2,
         label=f"WW matched (n={len(gw_m)}, med={np.median(gw_m):.1f})")
plt.hist(gw_u, bins=bins, density=True, histtype="step", lw=2,
         label=f"WW unmatched (n={len(gw_u)}, med={np.median(gw_u):.1f})")
plt.hist(gz, bins=bins, density=True, histtype="step", lw=2,
         label=f"ZZ (n={len(gz)}, med={np.median(gz):.1f})")
plt.xlabel("best-pairing gof  $-2\\log[\\mathrm{BW}_a\\,\\mathrm{BW}_b]$ (pole-ref)")
plt.ylabel("normalized")
plt.title("BW pairing goodness-of-fit: WW vs ZZ (W hypothesis)")
plt.legend(); plt.tight_layout()
p1 = os.path.join(OUTDIR, "bwpair_gof_WW_vs_ZZ.png")
plt.savefig(p1, dpi=120); plt.close()

# ── Plot 2: ROC-like — keep-WW-matched efficiency vs reject-ZZ, scanning gof<cut ──
cuts = np.linspace(0, 40, 200)
eff_ww = np.array([(gw_m < c).mean() for c in cuts])
eff_zz = np.array([(gz < c).mean() for c in cuts])
plt.figure(figsize=(6, 6))
plt.plot(eff_zz, eff_ww, lw=2)
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
plt.xlabel("ZZ acceptance  (gof < cut)")
plt.ylabel("WW-matched acceptance  (gof < cut)")
plt.title("WW-matched vs ZZ separation by best-pairing gof")
plt.grid(alpha=0.3); plt.tight_layout()
p2 = os.path.join(OUTDIR, "bwpair_roc_WW_vs_ZZ.png")
plt.savefig(p2, dpi=120); plt.close()

# ── Plot 3: calibration ──
plt.figure(figsize=(6, 6))
plt.plot([1/3, 1], [1/3, 1], "k--", lw=1, alpha=0.5, label="perfect")
plt.plot(cx, cy, "o-", lw=2, label="BW tool")
plt.xlabel("predicted prob_best"); plt.ylabel("actual correct fraction")
plt.title("Pairing posterior calibration (WW, good events)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
p3 = os.path.join(OUTDIR, "bwpair_prob_calibration.png")
plt.savefig(p3, dpi=120); plt.close()

print("wrote:", p1); print("wrote:", p2); print("wrote:", p3)
