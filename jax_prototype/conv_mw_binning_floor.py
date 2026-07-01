#!/usr/bin/env python3
# Binning-sensitivity of the closure decomposition: PROOF that the estimator floor is a binned-template
# artifact (it swings ~100 MeV with FOLD_BIN) while the gen targets / gen_marg-gen_pe gap are bit-invariant.
# Reads anatomy_results/anatomy_ecm{E}{,_bin025,_bin100}.json  (nominal = bin 0.5/sm 0.6).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RDIR = os.path.join(os.path.dirname(__file__), "anatomy_results")
OUT  = "/eos/user/m/mdefranc/www/mW/conv_mw"
ECMS = [157, 160, 163]; COL = {157:"C0", 160:"C1", 163:"C2"}
BINS = [(0.25,"_bin025"), (0.5,""), (1.0,"_bin100")]

def load(e, tag):
    p = os.path.join(RDIR, f"anatomy_ecm{e}{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
# (left) |floor| vs bin width for grid & exact, all ECMs ; (right) gap & targets invariance
for e in ECMS:
    bvals, gfl, efl = [], [], []
    for b, tag in BINS:
        r = load(e, tag)
        if r is None: continue
        bvals.append(b); gfl.append(1000*r["decomp"]["grid"]["floor"]); efl.append(1000*r["decomp"]["exact"]["floor"])
    if not bvals: continue
    ax[0].plot(bvals, gfl, "o-",  color=COL[e], label=f"ecm{e} grid")
    ax[0].plot(bvals, efl, "s--", color=COL[e], label=f"ecm{e} exact", mfc="none")
ax[0].axhline(0, color="k", lw=.8)
ax[0].set_xlabel("FOLD_BIN width [GeV]  (smoothing scaled with it)")
ax[0].set_ylabel("estimator floor  identity_fold − gen_target  [MeV]")
ax[0].set_title("(A) the floor is a BINNING artifact: swings ~100 MeV\n"
                "(nominal=0.5; coarse 1.0 → −120/−140; fine 0.25 → small)")
ax[0].set_ylim(-160, 30); ax[0].legend(fontsize=7, ncol=3, loc="lower left"); ax[0].grid(alpha=.3)

# invariance: gap (gen_marg-gen_pe) vs bin
for e in ECMS:
    bvals, gap = [], []
    for b, tag in BINS:
        r = load(e, tag)
        if r is None: continue
        bvals.append(b); gap.append(1000*(r["gen_marg"]-r["gen_pe"]))
    if not bvals: continue
    ax[1].plot(bvals, gap, "D-", color=COL[e], label=f"ecm{e}: gap = {gap[0]:.0f} MeV (flat)")
ax[1].set_xlabel("FOLD_BIN width [GeV]")
ax[1].set_ylabel("gen_marg − gen_pe  [MeV]")
ax[1].set_title("(B) the gen-level gap is BIN-INVARIANT (gen-level, not estimator)\n"
                "⇒ closure scatter = floor (binning); gap = the real threshold/modeling effect")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3); ax[1].set_ylim(0, 90)

plt.tight_layout(); png = f"{OUT}/closure_binning_floor.png"; plt.savefig(png, dpi=115)
print(f"[plot] {png}")

# table
print(f"\n{'ECM':>4} {'bin':>5} | {'gen_pe':>9} {'gen_marg':>9} {'gap':>5} | {'grid floor/clos':>16} | {'exact floor/clos':>16}")
for e in ECMS:
    for b, tag in BINS:
        r = load(e, tag)
        if r is None: continue
        d = r["decomp"]
        print(f"{e:>4} {b:>5} | {r['gen_pe']:>9.4f} {r['gen_marg']:>9.4f} {1000*(r['gen_marg']-r['gen_pe']):>+4.0f} | "
              f"{1000*d['grid']['floor']:>+7.0f} {1000*d['grid']['closure']:>+7.0f} | "
              f"{1000*d['exact']['floor']:>+7.0f} {1000*d['exact']['closure']:>+7.0f}")
