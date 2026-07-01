#!/usr/bin/env python3
"""WW → 4q kinematic-fit validation.

Headline stage-1 metrics:
  (1) jet→W PAIRING EFFICIENCY — how often the lowest-χ² partition matches the
      gen-truth partition (overall, among valid fits, and vs χ² separation /
      matching quality cuts);
  (2) χ²/ndf SHAPE — should be well-behaved and peak near 1 for the true pairing,
      with the true partition's χ² systematically below the combinatoric ones.

Reads outputs/treemaker/4q/step2/had_<tag>/p8_ee_WW_ecm160.root, writes PNGs to
/eos/user/m/mdefranc/www/mW/kinfit_4q/ (override via WW_PLOT_DIR).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

TAG      = os.environ.get("WW_TAG", "4q_v1")
INFILE   = os.environ.get("WW_INFILE",
    f"outputs/treemaker/4q/step2/had_{TAG}/p8_ee_WW_ecm160.root")
PLOT_DIR = os.environ.get("WW_PLOT_DIR", "/eos/user/m/mdefranc/www/mW/kinfit_4q")
os.makedirs(PLOT_DIR, exist_ok=True)

BR = [
    "kinfit4q_pairing", "gen_pairing_true", "kinfit4q_pairing_correct",
    "kinfit4q_chi2_p0", "kinfit4q_chi2_p1", "kinfit4q_chi2_p2",
    "kinfit4q_dchi2", "kinfit4q_n_pairings_valid",
    "kinfit4q_chi2", "kinfit4q_chi2_ndof",
    "kinfit4q_valid", "kinfit4q_valid_loose", "kinfit4q_status",
    "kinfit4q_mW", "kinfit4q_Wa_m", "kinfit4q_Wb_m", "kinfit4q_WW_m",
    "jet1_matched_q_dR", "jet2_matched_q_dR", "jet3_matched_q_dR", "jet4_matched_q_dR",
]


def main():
    print(f"Reading {INFILE}")
    d = uproot.open(INFILE)["events"].arrays(BR, library="np")
    n = d["kinfit4q_pairing"].size
    valid = d["kinfit4q_valid"].astype(bool)
    ptrue = d["gen_pairing_true"]
    correct = d["kinfit4q_pairing_correct"].astype(bool)
    has_truth = ptrue >= 0
    chi2 = np.stack([d["kinfit4q_chi2_p0"], d["kinfit4q_chi2_p1"], d["kinfit4q_chi2_p2"]], axis=1)
    max_dr = np.max(np.stack([d[f"jet{j}_matched_q_dR"] for j in (1, 2, 3, 4)], axis=1), axis=1)

    # ── headline numbers ────────────────────────────────────────────────────
    eff_all   = correct[has_truth].mean()
    eff_valid = correct[has_truth & valid].mean() if (has_truth & valid).sum() else float("nan")
    # χ²-only winner (argmin, ignoring validity preference) for a clean metric.
    argmin = np.argmin(chi2, axis=1)
    eff_argmin = (argmin[has_truth] == ptrue[has_truth]).mean()
    # matching-clean subset (all 4 jets dR<0.1) — truth label most reliable here.
    clean = has_truth & (max_dr < 0.1)
    eff_clean = correct[clean].mean() if clean.sum() else float("nan")
    print(f"  N = {n}  valid_frac = {valid.mean():.3f}  has_truth = {has_truth.mean():.3f}")
    print(f"  PAIRING EFFICIENCY: all={eff_all:.3f}  valid={eff_valid:.3f}  "
          f"argmin-χ²={eff_argmin:.3f}  clean(dR<0.1)={eff_clean:.3f}")
    print(f"  random-guess baseline = 0.333")

    # ── (1) pairing efficiency vs cuts ──────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    # vs χ²/ndf cut (keep events with chi2_ndof < x)
    cn = d["kinfit4q_chi2_ndof"]
    xs = np.linspace(np.nanpercentile(cn[valid & has_truth], 5),
                     np.nanpercentile(cn[valid & has_truth], 95), 30)
    eff_c, frac_c = [], []
    sel0 = valid & has_truth
    for x in xs:
        s = sel0 & (cn < x)
        eff_c.append(correct[s].mean() if s.sum() else np.nan)
        frac_c.append(s.sum() / sel0.sum())
    ax[0].plot(xs, eff_c, "o-", color="navy", label="pairing eff")
    ax[0].plot(xs, frac_c, "s--", color="gray", label="kept frac")
    ax[0].axhline(1/3, color="crimson", ls=":", label="random (1/3)")
    ax[0].set_xlabel("χ²/ndf cut (keep <)"); ax[0].set_ylabel("fraction")
    ax[0].set_title("efficiency vs χ²/ndf cut"); ax[0].legend(); ax[0].set_ylim(0, 1.05)

    # vs Δχ² (separation between best and 2nd-best pairing)
    dchi2 = d["kinfit4q_dchi2"]
    xs2 = np.linspace(0, np.nanpercentile(dchi2[valid & has_truth & (dchi2 > 0)], 90), 30)
    eff_d, frac_d = [], []
    for x in xs2:
        s = sel0 & (dchi2 > x)
        eff_d.append(correct[s].mean() if s.sum() else np.nan)
        frac_d.append(s.sum() / sel0.sum())
    ax[1].plot(xs2, eff_d, "o-", color="darkgreen", label="pairing eff")
    ax[1].plot(xs2, frac_d, "s--", color="gray", label="kept frac")
    ax[1].axhline(1/3, color="crimson", ls=":", label="random (1/3)")
    ax[1].set_xlabel("Δχ² cut (keep >)"); ax[1].set_ylabel("fraction")
    ax[1].set_title("efficiency vs Δχ² separation"); ax[1].legend(); ax[1].set_ylim(0, 1.05)

    # vs max jet-quark dR (matching quality)
    xs3 = np.linspace(0.02, 0.4, 30)
    eff_r, frac_r = [], []
    for x in xs3:
        s = sel0 & (max_dr < x)
        eff_r.append(correct[s].mean() if s.sum() else np.nan)
        frac_r.append(s.sum() / sel0.sum())
    ax[2].plot(xs3, eff_r, "o-", color="purple", label="pairing eff")
    ax[2].plot(xs3, frac_r, "s--", color="gray", label="kept frac")
    ax[2].axhline(1/3, color="crimson", ls=":", label="random (1/3)")
    ax[2].set_xlabel("max jet-quark dR cut (keep <)"); ax[2].set_ylabel("fraction")
    ax[2].set_title("efficiency vs matching quality"); ax[2].legend(); ax[2].set_ylim(0, 1.05)
    fig.suptitle(f"WW→4q pairing efficiency  [{TAG}]   "
                 f"overall={eff_all:.3f}  valid={eff_valid:.3f}  clean={eff_clean:.3f}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/pairing_efficiency.png", dpi=140)
    plt.close(fig)

    # ── (2) χ²/ndf shape + true-vs-wrong pairing χ² ─────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    cnv = cn[valid & has_truth]
    ax[0].hist(cnv[np.isfinite(cnv)], bins=80, range=(0, np.nanpercentile(cnv, 99)),
               color="steelblue", alpha=0.8)
    ax[0].axvline(1.0, color="crimson", ls="--", label="χ²/ndf = 1")
    ax[0].set_xlabel("kinfit4q_chi2_ndof (winner)"); ax[0].set_ylabel("events")
    ax[0].set_title(f"χ²/ndf shape (valid)\nmedian={np.nanmedian(cnv):.2f}"); ax[0].legend()

    # χ² of TRUE pairing vs the best WRONG pairing (per event, valid+truth+clean)
    s = valid & clean
    idx = np.arange(n)
    chi2_true = chi2[idx, np.where(has_truth, ptrue, 0)]
    wrong = chi2.copy()
    wrong[idx, np.where(has_truth, ptrue, 0)] = np.inf
    chi2_wrong_best = np.min(wrong, axis=1)
    lo = np.nanpercentile(chi2_true[s], 1)
    hi = np.nanpercentile(chi2_wrong_best[s], 99)
    bins = np.linspace(lo, hi, 80)
    ax[1].hist(chi2_true[s], bins=bins, color="darkgreen", alpha=0.6, label="true pairing")
    ax[1].hist(chi2_wrong_best[s], bins=bins, color="orange", alpha=0.6, label="best wrong pairing")
    ax[1].set_xlabel("χ²"); ax[1].set_ylabel("events")
    ax[1].set_title("true vs best-wrong pairing χ² (clean)"); ax[1].legend()

    # per-event Δ = χ²(best wrong) − χ²(true): >0 means true pairing wins
    delta = chi2_wrong_best[s] - chi2_true[s]
    frac_true_lower = (delta > 0).mean()
    dd = delta[np.isfinite(delta)]
    rng = np.nanpercentile(np.abs(dd), 98)
    ax[2].hist(dd, bins=80, range=(-rng, rng), color="slateblue", alpha=0.8)
    ax[2].axvline(0, color="crimson", ls="--")
    ax[2].set_xlabel("χ²(best wrong) − χ²(true)")
    ax[2].set_title(f"true-pairing wins {100*frac_true_lower:.1f}% (clean)")
    fig.suptitle(f"WW→4q χ² discriminant  [{TAG}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/chi2_shape.png", dpi=140)
    plt.close(fig)

    # ── (3) post-fit mass by-products ───────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, br, ttl, rng in [
        (ax[0], "kinfit4q_mW",   "kinfit mW (tight prior)", (80.0, 80.9)),
        (ax[1], "kinfit4q_Wa_m", "post-fit W_a mass",       (60, 100)),
        (ax[2], "kinfit4q_WW_m", "post-fit WW mass",        (140, 175)),
    ]:
        v = d[br][valid]
        v = v[np.isfinite(v)]
        a.hist(v, bins=80, range=rng, color="teal", alpha=0.8)
        a.set_xlabel(br); a.set_title(f"{ttl}\nmed={np.median(v):.3f}")
    # overlay Wb on the Wa panel
    vb = d["kinfit4q_Wb_m"][valid]; vb = vb[np.isfinite(vb)]
    ax[1].hist(vb, bins=80, range=(60, 100), color="firebrick", alpha=0.4, label="W_b")
    ax[1].legend()
    fig.suptitle(f"WW→4q post-fit masses (valid)  [{TAG}]", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/postfit_masses.png", dpi=140)
    plt.close(fig)

    # index.php for the EOS web area (if the helper exists)
    src_php = "index.php"
    if os.path.exists(src_php):
        import shutil
        shutil.copy(src_php, f"{PLOT_DIR}/index.php")
    print(f"Plots → {PLOT_DIR}/")


if __name__ == "__main__":
    main()
