#!/usr/bin/env python3
# DAY13: visualise HOW the mass+angle template-LR pairing discriminant works.
# Fig 1 (pairing_rightwrong_dists.png): right (true) vs wrong pairing distributions in m_hi, m_lo and the
#   within-W jet opening angle, one row per √s — shows the mass tail separates at 240/365, the angle at 160.
# Fig 2 (pairing_massangle_mechanism.png): the (dijet mass, opening angle) plane real-W vs fake pairs @160/240,
#   + the per-partition log-LR score (true vs wrong) for mass-only vs mass+angle @160 — shows the angle sharpens
#   the separation precisely where the masses overlap (threshold).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh ; python3 jax_prototype/plot_pairing_mechanism.py
import os, numpy as np, uproot
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)
TREES = {160: "had_ff160_genqk", 240: "had_ff240_genqk", 365: "had_ff365_genqk"}
PO = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]   # jets 0..3, partition 0 unused (reco uses gen_pairing_true)
GREEN, RED = "#2e9e2e", "#d23b3b"

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
    Q = {i: jv(i+1) for i in range(4)}
    true = a["gen_pairing_true"].astype(int); ok = (true >= 0) & (true < 3)
    mhi, mlo, ahi, alo = [], [], [], []   # per-partition, angle MASS-MATCHED (ahi = opening angle of the hi-mass pair)
    for (x1, x2), (y1, y2) in PO:
        mA = mass(Q[x1]+Q[x2]); mB = mass(Q[y1]+Q[y2]); tA = ang(Q[x1], Q[x2]); tB = ang(Q[y1], Q[y2])
        Ahi = mA >= mB
        mhi.append(np.where(Ahi, mA, mB)); mlo.append(np.where(Ahi, mB, mA))
        ahi.append(np.where(Ahi, tA, tB)); alo.append(np.where(Ahi, tB, tA))
    pack = lambda L: np.stack(L, 1)[ok]
    return pack(mhi), pack(mlo), pack(ahi), pack(alo), true[ok]

def split_tw(arr3, true):
    """arr3 (N,3) -> (true-partition values (N,), wrong-partition values (2N,))."""
    N = len(true); r = np.arange(N)
    t = arr3[r, true]
    wm = np.ones((N, 3), bool); wm[r, true] = False
    return t, arr3[wm]

# ════════════════════════ FIG 1 — right vs wrong distributions ════════════════════════
fig, axs = plt.subplots(3, 3, figsize=(13.5, 10.2))
COLS = [("m$_{hi}$  (higher dijet mass) [GeV]", "mhi", (30, 200)),
        ("m$_{lo}$  (lower dijet mass) [GeV]",  "mlo", (20, 130)),
        ("within-W jet opening angle [deg]",     "ang", (0, 180))]
DATA = {}
for ie, ecm in enumerate((160, 240, 365)):
    mhi, mlo, ahi, alo, true = load(ecm); DATA[ecm] = (mhi, mlo, ahi, alo, true)
    tmhi, wmhi = split_tw(mhi, true); tmlo, wmlo = split_tw(mlo, true)
    # pooled within-W opening angle: true = both pairs of the true partition; wrong = both pairs of the 2 wrong ones
    tang = np.concatenate([ahi[np.arange(len(true)), true], alo[np.arange(len(true)), true]])
    wm = np.ones((len(true), 3), bool); wm[np.arange(len(true)), true] = False
    wang = np.concatenate([ahi[wm], alo[wm]])
    series = {"mhi": (tmhi, wmhi), "mlo": (tmlo, wmlo), "ang": (tang, wang)}
    for ic, (xlab, key, rng) in enumerate(COLS):
        ax = axs[ie, ic]; t, w = series[key]; bins = np.linspace(*rng, 70)
        ax.hist(t, bins=bins, density=True, histtype="stepfilled", color=GREEN, alpha=0.45, label="right (true) pairing")
        ax.hist(w, bins=bins, density=True, histtype="step", color=RED, lw=1.8, label="wrong pairing")
        if ic == 0: ax.set_ylabel(f"√s = {ecm} GeV\n\nnormalised", fontsize=11)
        if ie == 2: ax.set_xlabel(xlab, fontsize=10)
        if ie == 0: ax.set_title(xlab.split("  ")[0] if "  " in xlab else xlab, fontsize=11)
        if key == "mhi": ax.axvline(98, color="k", ls=":", lw=1.0); ax.text(101, ax.get_ylim()[1]*0.82, "98 (old clip)", fontsize=7.5, rotation=90, va="top")
        ax.grid(alpha=.25)
        if ie == 0 and ic == 2: ax.legend(fontsize=9, loc="upper left")
axs[0, 0].set_title("m$_{hi}$ — higher dijet mass", fontsize=11)
axs[0, 1].set_title("m$_{lo}$ — lower dijet mass", fontsize=11)
axs[0, 2].set_title("within-W jet opening angle", fontsize=11)
fig.suptitle("WW→4q jet→W pairing: right (true) vs wrong distributions — the template-LR inputs", fontsize=13, y=0.997)
plt.tight_layout(rect=[0, 0, 1, 0.985])
p1 = os.path.join(EOSW, "pairing_rightwrong_dists.png"); plt.savefig(p1, dpi=115); print(f"[plot] {p1}")
plt.close(fig)

# ════════════════════════ FIG 2 — the (mass, angle) plane + the LR score ════════════════════════
def contour2d(ax, x, y, xr, yr, color, nb=80, levels=4):
    H, xe, ye = np.histogram2d(x, y, bins=[np.linspace(*xr, nb), np.linspace(*yr, nb)])
    H = gaussian_filter(H, 1.3); H /= H.max()
    xc = 0.5*(xe[:-1]+xe[1:]); yc = 0.5*(ye[:-1]+ye[1:])
    ax.contour(xc, yc, H.T, levels=np.linspace(0.08, 0.9, levels), colors=color, linewidths=1.3)

# log-LR per-partition score (train/test split) for mass-only vs mass+angle, @160
def lr_scores(ecm, use_angle):
    mhi, mlo, ahi, alo, true = DATA[ecm]; N = len(true)
    rng = np.random.default_rng(3); tr = rng.random(N) < 0.5; te = ~tr
    me = np.arange(20, 340.01, 2.0); ae = np.arange(0, 180.01, 6.0)
    edges = [me, me] + ([ae, ae] if use_angle else [])
    def feat(rows, p):
        f = [mhi[rows, p], mlo[rows, p]]
        if use_angle: f += [ahi[rows, p], alo[rows, p]]
        return np.stack(f, 1)
    def feat_true(rows):
        f = [mhi[rows, true[rows]], mlo[rows, true[rows]]]
        if use_angle: f += [ahi[rows, true[rows]], alo[rows, true[rows]]]
        return np.stack(f, 1)
    def feat_wrong(rows):
        wm = np.ones((len(rows), 3), bool); wm[np.arange(len(rows)), true[rows]] = False
        f = [mhi[rows][wm], mlo[rows][wm]]
        if use_angle: f += [ahi[rows][wm], alo[rows][wm]]
        return np.stack(f, 1)
    def dens(X):
        H, _ = np.histogramdd(X, bins=edges); H = gaussian_filter(H, 1.0)
        H = np.maximum(H/np.maximum(H.sum(), 1e-300), 1e-300)
        return RegularGridInterpolator(tuple(0.5*(e[:-1]+e[1:]) for e in edges), H, bounds_error=False, fill_value=1e-300)
    tri = np.where(tr)[0]; tei = np.where(te)[0]
    Pt = dens(feat_true(tri)); Pw = dens(feat_wrong(tri))
    sc = np.stack([np.log(Pt(feat(tei, p))) - np.log(Pw(feat(tei, p))) for p in range(3)], 1)
    s_true = sc[np.arange(len(tei)), true[tei]]
    wmask = np.ones((len(tei), 3), bool); wmask[np.arange(len(tei)), true[tei]] = False
    s_wrong = sc[wmask]
    eff = np.mean(np.argmax(sc, 1) == true[tei])
    return s_true, s_wrong, eff

fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.2))
for ax, ecm, mr in [(axs[0], 160, (40, 160)), (axs[1], 240, (40, 220))]:
    mhi, mlo, ahi, alo, true = DATA[ecm]; N = len(true); r = np.arange(N)
    wm = np.ones((N, 3), bool); wm[r, true] = False
    # pool ALL dijet pairs: real-W pairs = both pairs of the true partition; fake pairs = pairs of the wrong partitions
    real_m = np.concatenate([mhi[r, true], mlo[r, true]]); real_a = np.concatenate([ahi[r, true], alo[r, true]])
    fake_m = np.concatenate([mhi[wm], mlo[wm]]);            fake_a = np.concatenate([ahi[wm], alo[wm]])
    contour2d(ax, fake_m, fake_a, mr, (0, 180), RED)
    contour2d(ax, real_m, real_a, mr, (0, 180), GREEN)
    ax.axvspan(70, 90, color="grey", alpha=0.10)
    ax.set_xlabel("dijet mass [GeV]"); ax.set_ylabel("jet opening angle [deg]")
    ax.set_title(f"(a/b) √s={ecm}: real-W (green) vs fake (red) pairs")
    ax.plot([], [], color=GREEN, lw=2, label="real-W pair"); ax.plot([], [], color=RED, lw=2, label="fake pair")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=.25)
# panel 3: log-LR separation @160, mass-only vs mass+angle
ax = axs[2]; bins = np.linspace(-12, 12, 70)
st_m, sw_m, eff_m = lr_scores(160, False); st_a, sw_a, eff_a = lr_scores(160, True)
ax.hist(st_a, bins=bins, density=True, histtype="stepfilled", color=GREEN, alpha=0.45, label=f"true part. (mass+angle)")
ax.hist(sw_a, bins=bins, density=True, histtype="stepfilled", color=RED, alpha=0.35, label=f"wrong part. (mass+angle)")
ax.hist(st_m, bins=bins, density=True, histtype="step", color=GREEN, lw=1.6, ls="--", label="true part. (mass-only)")
ax.hist(sw_m, bins=bins, density=True, histtype="step", color=RED, lw=1.6, ls="--", label="wrong part. (mass-only)")
ax.axvline(0, color="k", lw=0.8, ls=":")
ax.set_xlabel("per-partition log-LR  =  log P$_{true}$ − log P$_{wrong}$")
ax.set_ylabel("normalised"); ax.set_title(f"(c) √s=160 discriminant: angle sharpens it\n"
              f"ε = {100*eff_m:.1f}% (mass) → {100*eff_a:.1f}% (mass+angle)")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.25)
plt.tight_layout()
p2 = os.path.join(EOSW, "pairing_massangle_mechanism.png"); plt.savefig(p2, dpi=115); print(f"[plot] {p2}")
