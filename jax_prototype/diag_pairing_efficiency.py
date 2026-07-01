#!/usr/bin/env python3
# DAY12 C4(iii–v): WW->4q pairing efficiency, gen AND reco, pllnops vs multivariate (mass / mass+angle / angle-only).
# The DECISIVE pairing measurement (supersedes the |Dmedian|/sigma "separation power" metric, which was misleading).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# python3 jax_prototype/diag_pairing_efficiency.py [gen|reco]   (run from the repo root; needs the had_ff*_genqk trees)
import sys, uproot, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.model_selection import train_test_split
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "reco"
TREES = {160: "had_ff160_genqk", 240: "had_ff240_genqk", 365: "had_ff365_genqk"}
GW, MWREF = 2.085, 80.4
def bw(m): d = m*m - MWREF*MWREF; mg = MWREF*GW; return mg/(d*d + mg*mg)
def mass(v): return np.sqrt(np.maximum(v[:, 3]**2 - v[:, 0]**2 - v[:, 1]**2 - v[:, 2]**2, 0))
def ang(u, v):
    c = (u[:, :3]*v[:, :3]).sum(1)/(np.linalg.norm(u[:, :3], axis=1)*np.linalg.norm(v[:, :3], axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def load(tag, ecm):
    F = f"outputs/treemaker/4q/step2_ff/{tag}/p8_ee_WW_ecm{ecm}.root"
    if LEVEL == "gen":
        # W-grouped gen quarks: partition 0 = TRUE = (q0 q1)(q2 q3)
        a = uproot.open(F)["events"].arrays([f"gen_qW{i}_{c}" for i in range(4) for c in ("px", "py", "pz", "e")], library="np")
        Q = {i: np.stack([a[f"gen_qW{i}_px"], a[f"gen_qW{i}_py"], a[f"gen_qW{i}_pz"], a[f"gen_qW{i}_e"]], 1) for i in range(4)}
        PO = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        true = np.zeros(len(Q[0]), int)                       # true is always partition 0 in this construction
    else:
        br = [f"reco_jet{i}_{c}" for i in (1, 2, 3, 4) for c in ("p", "theta", "phi")] + ["gen_pairing_true"]
        a = uproot.open(F)["events"].arrays(br, library="np")
        def jv(i):
            p = a[f"reco_jet{i}_p"]; th = a[f"reco_jet{i}_theta"]; ph = a[f"reco_jet{i}_phi"]; st = np.sin(th)
            return np.stack([p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th), p], 1)
        Q = {i: jv(i+1) for i in range(4)}                    # 0..3 -> jets 1..4
        PO = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]   # matches gen_pairing_true convention
        true = a["gen_pairing_true"].astype(int)
    M, perpart = [], []
    for (a1, a2), (b1, b2) in PO:
        mA = mass(Q[a1]+Q[a2]); mB = mass(Q[b1]+Q[b2]); tA = ang(Q[a1], Q[a2]); tB = ang(Q[b1], Q[b2])
        mhi = np.maximum(mA, mB); mlo = np.minimum(mA, mB); thi = np.maximum(tA, tB); tlo = np.minimum(tA, tB)
        sc = (-2*np.log(np.maximum(bw(mA), 1e-300)) - 2*np.log(np.maximum(bw(mB), 1e-300))
              - 2*3*(np.log(np.maximum(mA, 1e-9)) + np.log(np.maximum(mB, 1e-9))))
        M.append(sc); perpart.append(np.stack([mhi, mlo, thi, tlo], 1))
    M = np.stack(M, 1); ok = (true >= 0) & (true < 3)
    return M[ok], [pp[ok] for pp in perpart], true[ok]

def eff_gbm(perpart, true, cols):
    N = len(true); idx = np.arange(N); itr, ite = train_test_split(idx, test_size=0.5, random_state=3)
    Xtr = np.concatenate([perpart[k][itr][:, cols] for k in range(3)], 0)
    ytr = np.concatenate([(true[itr] == k).astype(int) for k in range(3)], 0)
    g = GBC(max_iter=200, max_depth=4, learning_rate=0.08).fit(Xtr, ytr)
    sc = np.stack([g.predict_proba(perpart[k][ite][:, cols])[:, 1] for k in range(3)], 1)
    return np.mean(np.argmax(sc, 1) == true[ite])

print(f"LEVEL={LEVEL}")
print(f"{'sqrt(s)':>7} {'pllnops':>9} {'GBM mass':>9} {'mass+ang':>9} {'ang-only':>9}")
for e in (160, 240, 365):
    M, pp, true = load(TREES[e], e)
    print(f"{e:>7} {100*np.mean(np.argmin(M,1)==true):>8.1f}% {100*eff_gbm(pp,true,[0,1]):>8.1f}% "
          f"{100*eff_gbm(pp,true,[0,1,2,3]):>8.1f}% {100*eff_gbm(pp,true,[2,3]):>8.1f}%   (N={len(true)})")
