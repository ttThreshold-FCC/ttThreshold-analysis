#!/usr/bin/env python3
# DAY13 NEXT-1 deliverable plot: the SMOOTH / analytic+ISR phase-space-MC pairing discriminant vs the adopted
# histogram template-LR (tmplLRa).  Efficiency + mW pairing-confusion bias vs √s, 5 seeds.
#   pllnops    = analytic-BW incumbent
#   tmplLRa    = histogram template-LR, mass+within-W angle  (ADOPTED, DAY12)
#   tmplLRsma  = SMOOTH phase-space-MC template-LR, mass+angle (P_true=BW⊗BW⊗√λ ISR-folded, P_wrong=ps-MC; dijet-smear)
#   tmplLRsmac = tmplLRsma + ISR-conditioning on the per-event reco 4-jet mass
# Numbers: conv_mw_4q.py full-stat on had_ff*_genqk, 5 seeds (FOLD/LR/SM = 2024+s / 7+s / 11+s, s=0..4),
# 4-D mass+angle template at SM_MBIN_A=2.0 GeV / SM_ABIN=6 deg (= incumbent LR_BIN).
# Bias = reco_fit(mode) − reco_fit(true-pairing) = the pure pairing-confusion mW shift (detector floor cancels).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh ; python3 jax_prototype/plot_pairing_smooth.py
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)
ECM = np.array([160, 240, 365])
eff = {"pllnops":[79.5,72.7,80.8], "tmplLRa":[83.0,83.4,91.1], "tmplLRsm":[76.4,81.3,90.2],
       "tmplLRsma":[81.9,81.6,90.2], "tmplLRsmac":[81.7,81.2,90.0]}
bias_seeds = {
 "pllnops":   {160:[-6.1,-6.9,-4.8,-1.1,-6.6], 240:[6.1,9.2,9.3,7.9,7.8],  365:[1.7,1.8,0.8,1.0,1.4]},
 "tmplLRa":   {160:[-3.5,-7.9,-5.9,-0.6,-3.3], 240:[-0.3,2.3,3.0,1.1,-0.0],365:[-0.4,-0.8,-1.7,-1.5,0.1]},
 "tmplLRsm":  {160:[-5.6,-8.8,-4.6,-3.1,-6.6], 240:[1.8,3.4,5.2,3.4,1.3],  365:[0.5,0.1,-0.5,-1.1,-0.0]},
 "tmplLRsma": {160:[-3.6,-7.8,-3.0,-2.6,-3.3], 240:[1.0,2.8,4.0,1.1,0.0],  365:[0.3,-0.3,-0.8,-1.4,-0.3]},
 "tmplLRsmac":{160:[-4.1,-5.1,-2.3,-0.6,-4.0], 240:[0.3,2.8,4.2,2.6,0.4],  365:[-0.1,-0.2,-0.3,-0.9,0.2]},
}
COL = {"pllnops":"C3","tmplLRa":"C2","tmplLRsm":"#999999","tmplLRsma":"C0","tmplLRsmac":"C4"}
LAB = {"pllnops":"pllnops (analytic BW — incumbent)", "tmplLRa":"tmplLRa (histogram template-LR — adopted)",
       "tmplLRsm":"tmplLRsm (smooth, mass-ONLY)", "tmplLRsma":"tmplLRsma (SMOOTH ps-MC, mass+angle)",
       "tmplLRsmac":"tmplLRsmac (+ ISR-conditioned)"}
ORDER = ["pllnops","tmplLRa","tmplLRsm","tmplLRsma","tmplLRsmac"]
fig, (axE, axB) = plt.subplots(1, 2, figsize=(13.8, 5.5))

for m in ORDER:
    axE.plot(ECM, eff[m], "s--" if m=="tmplLRsm" else "o-", color=COL[m], ms=8, lw=1.7, label=LAB[m])
axE.set_xlabel("√s [GeV]"); axE.set_ylabel("jet→W pairing efficiency [%]")
axE.set_title("(a) Pairing efficiency — smooth+angle within ~1–2pp of the histogram")
axE.set_xticks(ECM); axE.set_ylim(70, 94); axE.grid(alpha=.3); axE.legend(fontsize=8.0, loc="lower right")

for m in ORDER:
    mu = np.array([np.mean(bias_seeds[m][e]) for e in ECM]); sd = np.array([np.std(bias_seeds[m][e]) for e in ECM])
    dx = {"pllnops":-5,"tmplLRa":-2.5,"tmplLRsm":0,"tmplLRsma":2.5,"tmplLRsmac":5}[m]
    fmt = "s--" if m=="tmplLRsm" else "o-"
    axB.errorbar(ECM+dx, mu, yerr=sd, fmt=fmt, color=COL[m], ms=7, lw=1.6, capsize=3.5, label=LAB[m])
axB.axhline(0, color="grey", lw=1.0, ls=":"); axB.axhspan(-2, 2, color="grey", alpha=0.12)
axB.annotate("smooth mass+angle (blue) reproduces\nthe histogram (green) at all √s.\nmass-ONLY (grey dashed) is biased\n@160/240 — the angle is essential.",
             (243, 5.0), fontsize=8.0, color="#444444")
axB.set_xlabel("√s [GeV]"); axB.set_ylabel("pairing-confusion mW bias  reco_fit − reco_fit(true)  [MeV]")
axB.set_title("(b) mW pairing-confusion bias (5 seeds; detector floor cancels)")
axB.set_xticks(ECM); axB.grid(alpha=.3); axB.legend(fontsize=8.0, loc="upper left")

plt.tight_layout()
p = os.path.join(EOSW, "pairing_smooth_bias.png"); plt.savefig(p, dpi=120); print(f"[plot] {p}")
