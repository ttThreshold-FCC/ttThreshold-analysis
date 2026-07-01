#!/usr/bin/env python3
# DAY12 C4(i,ii): does the full 4-jet kinematics measure sqrt(s')/ISR better than the naive estimators?
# (true pairing, ecm160). Shows: naive M_4jet / Sum-pz are energy-dominated and WORSE than constant; a GBM on
# the full angular+mass kinematics recovers sqrt(s')/ISR to BELOW their natural spread (the angles carry it).
# This is an INFORMATION PROBE (the real model would be analytic, not ML).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh ; python3 jax_prototype/diag_fullkin_sqrtsp.py
import uproot, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor as GBR
from sklearn.model_selection import train_test_split
f = "outputs/treemaker/4q/step2_ff/had_ff160_genqk/p8_ee_WW_ecm160.root"   # or had_ff160_nod45 (gen_pairing_true based)
need = [f"reco_jet{i}_{k}" for i in (1, 2, 3, 4) for k in ("p", "theta", "phi")] + ["gen_isr_pz", "gen_WW_m", "gen_pairing_true"]
a = uproot.open(f)["events"].arrays(need, library="np")
def comp(i):
    p = a[f"reco_jet{i}_p"]; th = a[f"reco_jet{i}_theta"]; ph = a[f"reco_jet{i}_phi"]; st = np.sin(th)
    return np.stack([p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th), p], 1)
J = [comp(i) for i in (1, 2, 3, 4)]; S = sum(J)
M4 = np.sqrt(np.maximum(S[:, 3]**2 - S[:, 0]**2 - S[:, 1]**2 - S[:, 2]**2, 0)); Spz = S[:, 2]
PO = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
pid = np.clip(a["gen_pairing_true"].astype(int), 0, 2)
W1 = np.zeros((len(M4), 4)); W2 = np.zeros((len(M4), 4))
for k, ((a1, a2), (b1, b2)) in enumerate(PO):
    m = pid == k; W1[m] = J[a1][m]+J[a2][m]; W2[m] = J[b1][m]+J[b2][m]
def mass(v): return np.sqrt(np.maximum(v[:, 3]**2 - v[:, 0]**2 - v[:, 1]**2 - v[:, 2]**2, 0))
def pmag(v): return np.sqrt((v[:, :3]**2).sum(1))
mhi = np.maximum(mass(W1), mass(W2)); mlo = np.minimum(mass(W1), mass(W2))
cos12 = (W1[:, :3]*W2[:, :3]).sum(1)/(pmag(W1)*pmag(W2)+1e-9)
jc = np.stack([J[i][:, 2]/(J[i][:, 3]+1e-9) for i in range(4)], 1)
ok = np.isfinite(M4) & np.isfinite(a["gen_WW_m"]) & (mhi > 0) & (mlo > 0)
feats = np.column_stack([M4, Spz, S[:, 3], mhi, mlo, cos12, W1[:, 2]/(pmag(W1)+1e-9), W2[:, 2]/(pmag(W2)+1e-9),
                         jc[:, 0], jc[:, 1], jc[:, 2], jc[:, 3], pmag(W1), pmag(W2)])[ok]
rms = lambda x: float(np.std(x))
for target, base in [("gen_WW_m", M4), ("gen_isr_pz", Spz)]:
    y = a[target][ok]; b = base[ok]
    Xtr, Xte, ytr, yte, btr, bte = train_test_split(feats, y, b, test_size=0.4, random_state=1)
    pred = GBR(max_iter=300, max_depth=4, learning_rate=0.06).fit(Xtr, ytr).predict(Xte)
    print(f"{target}: naive resid RMS={rms(yte-bte):.2f} -> full-kin GBM RMS={rms(yte-pred):.2f} "
          f"(improve {100*(1-rms(yte-pred)/rms(yte-bte)):+.0f}%) | true spread={rms(yte):.2f}")
