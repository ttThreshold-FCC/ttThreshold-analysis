#!/usr/bin/env python3
# DAY12 NEXT-1 deliverable plot: analytic template-LR pairing discriminant — efficiency + mW pairing-confusion
# bias vs √s, for pllnops (incumbent) / tmplLR (mass-only) / tmplLRa (mass+angle).
# Numbers from conv_mw_4q.py full-stat runs on the had_ff*_genqk trees (3 fold/LR seeds each; see lrcmp/lrseed logs).
# Bias = reco_fit(mode) − reco_fit(true-pairing) = the pure pairing-confusion mW shift (detector floor cancels).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh ; python3 jax_prototype/plot_pairing_tmpl_lr.py
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)
ECM = np.array([160, 240, 365])
# pairing efficiency [%] (deterministic, full-stat)
eff = {"pllnops": [79.5, 72.7, 80.8], "tmplLR": [79.9, 82.6, 90.1], "tmplLRa": [83.0, 83.3, 91.1]}
# pairing-confusion bias Δbias [MeV], per seed (default 7/2024, 21/777, 99/5150)
bias_seeds = {
    "pllnops": {160: [-6.1, -5.5, -5.1], 240: [ 6.1,  9.6,  9.1], 365: [ 1.7,  3.3,  3.0]},
    "tmplLR":  {160: [-11.2, -12.0, -8.6], 240: [-0.1, -0.5,  1.8], 365: [-0.2, -0.3,  0.9]},
    "tmplLRa": {160: [-3.5, -2.1, -2.8], 240: [-0.3, -0.7,  1.6], 365: [-0.5, -1.7, -0.4]},
}
COL = {"pllnops": "C3", "tmplLR": "C0", "tmplLRa": "C2"}
LAB = {"pllnops": "pllnops (analytic BW — incumbent)", "tmplLR": "tmplLR (mass template-LR)",
       "tmplLRa": "tmplLRa (mass+angle template-LR)"}
fig, (axE, axB) = plt.subplots(1, 2, figsize=(13.2, 5.4))

for m in ("pllnops", "tmplLR", "tmplLRa"):
    axE.plot(ECM, eff[m], "o-", color=COL[m], ms=9, lw=1.8, label=LAB[m])
for i, e in enumerate(ECM):
    axE.annotate(f"+{eff['tmplLR'][i]-eff['pllnops'][i]:.0f}pp", (e, eff["tmplLR"][i]),
                 textcoords="offset points", xytext=(6, 8), fontsize=8.5, color="C0")
axE.set_xlabel("√s [GeV]"); axE.set_ylabel("jet→W pairing efficiency [%]")
axE.set_title("(a) Pairing efficiency — analytic template-LR recovers the GBM's +10pp @240/365")
axE.set_xticks(ECM); axE.set_ylim(68, 95); axE.grid(alpha=.3); axE.legend(fontsize=9, loc="lower right")

for m in ("pllnops", "tmplLR", "tmplLRa"):
    mu = np.array([np.mean(bias_seeds[m][e]) for e in ECM])
    sd = np.array([np.std(bias_seeds[m][e]) for e in ECM])
    dx = {"pllnops": -3, "tmplLR": 0, "tmplLRa": 3}[m]
    axB.errorbar(ECM+dx, mu, yerr=sd, fmt="o-", color=COL[m], ms=9, lw=1.8, capsize=4, label=LAB[m])
axB.axhline(0, color="grey", lw=1.0, ls=":")
axB.axhspan(-2, 2, color="grey", alpha=0.12)
axB.annotate("240: +8.3 → +0.4 MeV\n(mass-only suffices)", (240, 8.3), textcoords="offset points",
             xytext=(8, -2), fontsize=8.5, color="C3")
axB.annotate("160 (thr): mass-LR HURTS\n(peak-mimic fakes); angle fixes it", (160, -10.6),
             textcoords="offset points", xytext=(8, -4), fontsize=8.5, color="C0")
axB.set_xlabel("√s [GeV]"); axB.set_ylabel("pairing-confusion mW bias  reco_fit − reco_fit(true)  [MeV]")
axB.set_title("(b) mW pairing-confusion bias (3 seeds; detector floor cancels)")
axB.set_xticks(ECM); axB.grid(alpha=.3); axB.legend(fontsize=9, loc="upper left")

plt.tight_layout()
p = os.path.join(EOSW, "pairing_tmplLR_bias.png"); plt.savefig(p, dpi=120)
print(f"[plot] {p}")
