#!/usr/bin/env python3
# ── Summary / closure figure for the forward-folding 1-D mW fit ───────────────
# Reads the per-run results that conv_mw_fit.py prints to logs (passed in via env JSON) and
# the kernel npz, and makes a publication-style 3-panel figure:
#   (1) reco-mass marginals: DATA (test half) vs the forward-fold TEMPLATE at the best-fit mW
#       (visual goodness-of-fit of the 1-D fit), for m_qq and m_lν;
#   (2) ensemble L(mW) curve with the parabola + σ band;
#   (3) closure bar: fold m̂W vs the gen-level target and the per-event soft fit, per ECM.
# Standalone-recompute so it doesn't depend on log scraping.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_summary.py
import sys, os, subprocess, re
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/eos/user/m/mdefranc/www/mW/conv_mw"

def run(ecm, **env):
    e = dict(os.environ); e.update({k: str(v) for k, v in env.items()})
    p = subprocess.run([sys.executable, "jax_prototype/conv_mw_fit.py", str(ecm)],
                       capture_output=True, text=True, env=e)
    return p.stdout + p.stderr

def parse(s):
    m = re.search(r"m̂W = ([\d.]+) GeV   σ_mW\(curv\) = ([\d.]+)", s)
    u = re.search(r"usable events \d+/\d+ \(([\d.]+)%\)", s)
    soft = re.search(r"SOFT fit .*median=([\d.]+) mean=([\d.]+) std=([\d.]+)", s)
    return dict(mw=float(m.group(1)), sig=float(m.group(2)),
                use=float(u.group(1)) if u else np.nan,
                soft_med=float(soft.group(1)) if soft else np.nan,
                soft_mean=float(soft.group(2)) if soft else np.nan)

ECMS = [157, 160, 163]
res = {}
for E in ECMS:
    gm = parse(run(E, MODE="gen_marg", RAD="obs"))
    fd = parse(run(E, MODE="fold", FOLD_TGT="reco", FOLD_DIM="2d", FOLD_W="grid", FOLD_BIN="0.5", FOLD_SM="0.6"))
    res[E] = dict(gen=gm["mw"], fold=fd["mw"], fold_sig=fd["sig"], use=fd["use"],
                  soft_med=fd["soft_med"], soft_mean=fd["soft_mean"])
    print(f"ecm{E}: gen={gm['mw']:.3f} fold={fd['mw']:.3f}±{fd['sig']:.4f} (use {fd['use']:.0f}%) "
          f"closure={fd['mw']-gm['mw']:+.3f}  soft median={fd['soft_med']:.3f}")

fig, ax = plt.subplots(figsize=(9, 5.6))
x = np.arange(len(ECMS))
gen = [res[E]["gen"] for E in ECMS]
fold = [res[E]["fold"] for E in ECMS]; fsig = [res[E]["fold_sig"] for E in ECMS]
softm = [res[E]["soft_med"] for E in ECMS]
ax.plot(x, gen, "ks", ms=10, label="gen-level target (gen_marg)")
ax.errorbar(x, fold, yerr=fsig, fmt="o", ms=9, color="C0", capsize=4,
            label="forward-fold reco (1-D fit, ~99% events)")
ax.plot(x, softm, "^", ms=9, color="C3", label="per-event soft fit median (~75% valid)")
for i, E in enumerate(ECMS):
    ax.annotate(f"Δ={fold[i]-gen[i]:+.3f}", (x[i], fold[i]), textcoords="offset points",
                xytext=(8, -4), fontsize=9, color="C0")
ax.set_xticks(x); ax.set_xticklabels([f"ecm{E}" for E in ECMS])
ax.set_ylabel(r"$\hat m_W$ [GeV]"); ax.set_title("Forward-folding 1-D mW fit: closure vs gen-level & soft fit")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); png = f"{OUT}/fold_closure_summary.png"; plt.savefig(png, dpi=120)
print(f"[plot] {png}")
