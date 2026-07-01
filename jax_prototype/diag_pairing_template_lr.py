#!/usr/bin/env python3
# DAY12 NEXT-1: ANALYTIC / TEMPLATE pairing discriminant — the deliverable version of C4(v).
# pllnops (analytic BW pick) leaves ~10pp at 240/365; the GBM showed the info EXISTS in the reco masses.
# Here we build the NON-ML, generative analog: a 2-D TEMPLATE LIKELIHOOD RATIO on the sorted dijet masses,
#   LR_p = P_true(m_hi^p, m_lo^p) / P_wrong(m_hi^p, m_lo^p),  pick argmax_p LR_p,
# where P_true = density of the TRUE partition's reco masses (MC truth gen_pairing_true) and P_wrong =
# density of the two WRONG partitions' reco masses.  Train/test-HONEST via K folds (templates built on the
# complementary folds, applied to the held-out fold).  Shape-only (each density self-normalised).
# Optional within-W angle adds a small threshold-only refinement (mass+angle = 4-D template, sparser).
# GBM kept only as the upper-bound info probe for cross-check.
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# python3 jax_prototype/diag_pairing_template_lr.py [reco|gen]   (repo root; needs had_ff*_genqk trees)
import sys, os, uproot, numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from sklearn.ensemble import HistGradientBoostingClassifier as GBC
from sklearn.model_selection import train_test_split
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "reco"
TREES = {160: "had_ff160_genqk", 240: "had_ff240_genqk", 365: "had_ff365_genqk"}
GW, MWREF = 2.085, 80.4
LR_K   = int(os.environ.get("LR_K", "4"))
LR_BIN = float(os.environ.get("LR_BIN", "1.0"))
LR_SM  = float(os.environ.get("LR_SM", "1.0"))
M_LO   = float(os.environ.get("M_LO", "20.0"))     # mass-template range [GeV] — MUST cover the wrong-pairing
M_HI   = float(os.environ.get("M_HI", "340.0"))    # high-mass tail (>98 for 74%/94% of wrong pairs @240/365)
A_BIN  = float(os.environ.get("A_BIN", "6.0"))     # angle template bin [deg]
A_SM   = float(os.environ.get("A_SM", "1.0"))
SEED   = int(os.environ.get("LR_SEED", "7"))
def bw(m): d = m*m - MWREF*MWREF; mg = MWREF*GW; return mg/(d*d + mg*mg)
def mass(v): return np.sqrt(np.maximum(v[:, 3]**2 - v[:, 0]**2 - v[:, 1]**2 - v[:, 2]**2, 0))
def ang(u, v):
    c = (u[:, :3]*v[:, :3]).sum(1)/(np.linalg.norm(u[:, :3], axis=1)*np.linalg.norm(v[:, :3], axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def load(tag, ecm):
    F = f"outputs/treemaker/4q/step2_ff/{tag}/p8_ee_WW_ecm{ecm}.root"
    if LEVEL == "gen":
        a = uproot.open(F)["events"].arrays([f"gen_qW{i}_{c}" for i in range(4) for c in ("px","py","pz","e")], library="np")
        Q = {i: np.stack([a[f"gen_qW{i}_px"], a[f"gen_qW{i}_py"], a[f"gen_qW{i}_pz"], a[f"gen_qW{i}_e"]], 1) for i in range(4)}
        true = np.zeros(len(Q[0]), int)
    else:
        br = [f"reco_jet{i}_{c}" for i in (1,2,3,4) for c in ("p","theta","phi")] + ["gen_pairing_true"]
        a = uproot.open(F)["events"].arrays(br, library="np")
        def jv(i):
            p = a[f"reco_jet{i}_p"]; th = a[f"reco_jet{i}_theta"]; ph = a[f"reco_jet{i}_phi"]; st = np.sin(th)
            return np.stack([p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th), p], 1)
        Q = {i: jv(i+1) for i in range(4)}
        true = a["gen_pairing_true"].astype(int)
    PO = [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]   # partition 0 = TRUE in the gen construction
    M, hi3, lo3, thi3, tlo3 = [], [], [], [], []
    for (a1, a2), (b1, b2) in PO:
        mA = mass(Q[a1]+Q[a2]); mB = mass(Q[b1]+Q[b2]); tA = ang(Q[a1], Q[a2]); tB = ang(Q[b1], Q[b2])
        sc = (-2*np.log(np.maximum(bw(mA), 1e-300)) - 2*np.log(np.maximum(bw(mB), 1e-300))
              - 2*3*(np.log(np.maximum(mA, 1e-9)) + np.log(np.maximum(mB, 1e-9))))
        M.append(sc)
        hi3.append(np.maximum(mA, mB)); lo3.append(np.minimum(mA, mB))
        thi3.append(np.maximum(tA, tB)); tlo3.append(np.minimum(tA, tB))
    M = np.stack(M, 1); hi3 = np.stack(hi3, 1); lo3 = np.stack(lo3, 1)
    thi3 = np.stack(thi3, 1); tlo3 = np.stack(tlo3, 1)
    ok = (true >= 0) & (true < 3)
    return M[ok], hi3[ok], lo3[ok], thi3[ok], tlo3[ok], true[ok]

def _dens(cols_train, edges_list):
    """smoothed N-D self-normalised density interpolator over the given feature columns."""
    H, _ = np.histogramdd(cols_train, bins=edges_list)
    if LR_SM > 0: H = gaussian_filter(H, sigma=LR_SM)
    H = np.maximum(H/np.maximum(H.sum(), 1e-300), 1e-300)
    cen = [0.5*(e[:-1]+e[1:]) for e in edges_list]
    return RegularGridInterpolator(tuple(cen), H, bounds_error=False, fill_value=1e-300)

def eff_tmpl_lr(hi3, lo3, thi3, tlo3, true, use_angle=False):
    """analytic template-LR pairing efficiency, train/test-honest K-fold."""
    N = len(true); rng = np.random.default_rng(SEED); fold = rng.integers(0, LR_K, N)
    me = np.arange(M_LO, M_HI+1e-6, LR_BIN)                 # mass edges
    ae = np.arange(0.0, 180.0+1e-6, A_BIN)                  # angle edges
    edges = [me, me] + ([ae, ae] if use_angle else [])
    # Clip evaluation features into the grid interior so an out-of-range partition gets the
    # nearest in-grid density instead of the fill_value tie (both Pt,Pw -> 1e-300 => logLR=0,
    # which would let an out-of-range wrong partition beat a correctly-scored true one).
    cen_lo = np.array([0.5*(e[0]+e[1])   for e in edges])
    cen_hi = np.array([0.5*(e[-2]+e[-1]) for e in edges])
    def feats(p):                                          # feature columns for partition p (all events)
        f = [hi3[:, p], lo3[:, p]]
        if use_angle: f += [thi3[:, p], tlo3[:, p]]
        return np.clip(np.stack(f, 1), cen_lo, cen_hi)
    logLR = np.full((N, 3), -np.inf)
    for k in range(LR_K):
        tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]
        tt = true[tr]
        # true-partition features of training events
        ftrue = [hi3[tr, tt], lo3[tr, tt]] + ([thi3[tr, tt], tlo3[tr, tt]] if use_angle else [])
        ftrue = np.stack(ftrue, 1)
        # wrong-partition features (the two p != true) of training events
        wm = np.ones((len(tr), 3), bool); wm[np.arange(len(tr)), tt] = False
        def gather(arr): return arr[tr][wm]
        fwrong = [gather(hi3), gather(lo3)] + ([gather(thi3), gather(tlo3)] if use_angle else [])
        fwrong = np.stack(fwrong, 1)
        Pt = _dens(ftrue, edges); Pw = _dens(fwrong, edges)
        for p in range(3):
            fp = feats(p)[te]
            logLR[te, p] = np.log(Pt(fp)) - np.log(Pw(fp))
    return np.mean(np.argmax(logLR, 1) == true)

def eff_gbm(hi3, lo3, thi3, tlo3, true, cols):
    perpart = [np.stack([hi3[:, p], lo3[:, p], thi3[:, p], tlo3[:, p]], 1) for p in range(3)]
    N = len(true); idx = np.arange(N); itr, ite = train_test_split(idx, test_size=0.5, random_state=3)
    Xtr = np.concatenate([perpart[k][itr][:, cols] for k in range(3)], 0)
    ytr = np.concatenate([(true[itr] == k).astype(int) for k in range(3)], 0)
    g = GBC(max_iter=200, max_depth=4, learning_rate=0.08).fit(Xtr, ytr)
    sc = np.stack([g.predict_proba(perpart[k][ite][:, cols])[:, 1] for k in range(3)], 1)
    return np.mean(np.argmax(sc, 1) == true[ite])

print(f"LEVEL={LEVEL}  LR_K={LR_K} LR_BIN={LR_BIN} LR_SM={LR_SM} (angle bin={A_BIN})")
print(f"{'sqrt(s)':>7} {'pllnops':>9} {'tmplLR':>9} {'tmplLR+ang':>11} {'GBMmass':>9} {'GBMm+ang':>9}")
for e in (160, 240, 365):
    M, hi3, lo3, thi3, tlo3, true = load(TREES[e], e)
    pll  = 100*np.mean(np.argmin(M, 1) == true)
    lr   = 100*eff_tmpl_lr(hi3, lo3, thi3, tlo3, true, use_angle=False)
    lra  = 100*eff_tmpl_lr(hi3, lo3, thi3, tlo3, true, use_angle=True)
    gbm  = 100*eff_gbm(hi3, lo3, thi3, tlo3, true, [0, 1])
    gbma = 100*eff_gbm(hi3, lo3, thi3, tlo3, true, [0, 1, 2, 3])
    print(f"{e:>7} {pll:>8.1f}% {lr:>8.1f}% {lra:>10.1f}% {gbm:>8.1f}% {gbma:>8.1f}%   (N={len(true)})")
