#!/usr/bin/env python3
# DAY13: cure the log-LR=0 spike (4-D mass+angle template sparsity). Compare, train/test-honest K-fold:
#   mass2d     = 2-D (m_hi,m_lo) LR                          [= tmplLR]
#   joint4d    = 4-D (m_hi,m_lo,th_hi,th_lo) LR              [= tmplLRa, current — has the spike]
#   factor     = logLR(m_hi,m_lo) + logLR(th_hi,th_lo)       [two well-populated 2-D templates, naive-Bayes blocks]
#   backoff    = joint4d per event if all 3 partitions are 4-D-determined, else mass2d (scale-consistent)
# Reports efficiency + the undetermined (logLR==0) fractions, at 160/240/365.
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh ; python3 jax_prototype/diag_pairing_backoff.py
import os, numpy as np, uproot
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
TREES = {160: "had_ff160_genqk", 240: "had_ff240_genqk", 365: "had_ff365_genqk"}
LR_K = 4; M_LO, M_HI, MBIN = 20.0, 340.0, 2.0; ABIN = 6.0; SM = 1.0; SEED = 7
def mass(v): return np.sqrt(np.maximum(v[:, 3]**2 - v[:, 0]**2 - v[:, 1]**2 - v[:, 2]**2, 0))
def ang(u, v):
    c = (u[:, :3]*v[:, :3]).sum(1)/(np.linalg.norm(u[:, :3], axis=1)*np.linalg.norm(v[:, :3], axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
def load(ecm):
    F = f"outputs/treemaker/4q/step2_ff/{TREES[ecm]}/p8_ee_WW_ecm{ecm}.root"
    br = [f"reco_jet{i}_{c}" for i in (1, 2, 3, 4) for c in ("p", "theta", "phi")] + ["gen_pairing_true"]
    a = uproot.open(F)["events"].arrays(br, library="np")
    def jv(i):
        p = a[f"reco_jet{i}_p"]; th = a[f"reco_jet{i}_theta"]; ph = a[f"reco_jet{i}_phi"]; st = np.sin(th)
        return np.stack([p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th), p], 1)
    Q = {i: jv(i+1) for i in range(4)}; true = a["gen_pairing_true"].astype(int); ok = (true >= 0) & (true < 3)
    PO = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
    mhi, mlo, ahi, alo = [], [], [], []
    for (x1, x2), (y1, y2) in PO:
        mA = mass(Q[x1]+Q[x2]); mB = mass(Q[y1]+Q[y2]); tA = ang(Q[x1], Q[x2]); tB = ang(Q[y1], Q[y2]); A = mA >= mB
        mhi.append(np.where(A, mA, mB)); mlo.append(np.where(A, mB, mA)); ahi.append(np.where(A, tA, tB)); alo.append(np.where(A, tB, tA))
    P = lambda L: np.stack(L, 1)[ok]
    return P(mhi), P(mlo), P(ahi), P(alo), true[ok]

def dens(X, edges):
    H, _ = np.histogramdd(X, bins=edges); H = gaussian_filter(H, SM)
    H = np.maximum(H/np.maximum(H.sum(), 1e-300), 1e-300)
    return RegularGridInterpolator(tuple(0.5*(e[:-1]+e[1:]) for e in edges), H, bounds_error=False, fill_value=1e-300)

LAMS = (0.1, 0.25, 0.5)
def scores(ecm):
    mhi, mlo, ahi, alo, true = load(ecm); N = len(true)
    me = np.arange(M_LO, M_HI+1e-6, MBIN); ae = np.arange(0, 180.01, ABIN)
    rng = np.random.default_rng(SEED); fold = rng.integers(0, LR_K, N)
    LR = {k: np.zeros((N, 3)) for k in ("mass2d", "angle2d", "joint4d")}
    # per-partition densities P_true / P_wrong for each block (for the density-mix backoff)
    D = {k: {cl: np.zeros((N, 3)) for cl in ("t", "w")} for k in ("mass2d", "angle2d", "joint4d")}
    feats = {"mass2d": ([mhi, mlo], [me, me]), "angle2d": ([ahi, alo], [ae, ae]),
             "joint4d": ([mhi, mlo, ahi, alo], [me, me, ae, ae])}
    for k, (cols, edges) in feats.items():
        for kf in range(LR_K):
            tri = np.where(fold != kf)[0]; tei = np.where(fold == kf)[0]; tt = true[tri]
            Xt = np.stack([c[tri, tt] for c in cols], 1)
            wm = np.ones((len(tri), 3), bool); wm[np.arange(len(tri)), tt] = False
            Xw = np.stack([c[tri][wm] for c in cols], 1)
            Pt = dens(Xt, edges); Pw = dens(Xw, edges)
            for p in range(3):
                Xp = np.stack([c[tei, p] for c in cols], 1)
                vt = Pt(Xp); vw = Pw(Xp)
                LR[k][tei, p] = np.log(vt) - np.log(vw)
                D[k]["t"][tei, p] = vt; D[k]["w"][tei, p] = vw
    # density-mix: P_class = (1-lam) P_4d + lam P_mass*P_angle  → kills empty-cell spike, keeps the 4-D correlation
    mixLR = {}
    for lam in LAMS:
        Pt = (1-lam)*D["joint4d"]["t"] + lam*D["mass2d"]["t"]*D["angle2d"]["t"]
        Pw = (1-lam)*D["joint4d"]["w"] + lam*D["mass2d"]["w"]*D["angle2d"]["w"]
        mixLR[lam] = np.log(np.maximum(Pt, 1e-300)) - np.log(np.maximum(Pw, 1e-300))
    return LR, mixLR, true

def eff(sc, true): return 100*np.mean(np.argmax(sc, 1) == true)
def wrong0(sc, true):
    wm = np.ones(sc.shape, bool); wm[np.arange(len(true)), true] = False
    return 100*np.mean(np.abs(sc[wm]) < 1e-9)

hdr = f"{'sqrt(s)':>7} | {'mass2d':>7} {'joint4d':>8} {'factor':>7}"
hdr += "".join(f"  mix.{lam:g}" for lam in LAMS) + f" | wrong@0 j4d/mix.{LAMS[0]:g}"
print(hdr)
for ecm in (160, 240, 365):
    LR, mixLR, true = scores(ecm)
    m2, j4 = LR["mass2d"], LR["joint4d"]
    fac = LR["mass2d"] + LR["angle2d"]
    row = f"{ecm:>7} | {eff(m2,true):6.1f}% {eff(j4,true):7.1f}% {eff(fac,true):6.1f}%"
    row += "".join(f" {eff(mixLR[lam],true):6.1f}%" for lam in LAMS)
    row += f" | {wrong0(j4,true):4.1f}% / {wrong0(mixLR[LAMS[0]],true):.1f}%"
    print(row)
