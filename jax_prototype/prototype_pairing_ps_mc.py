#!/usr/bin/env python3
# DAY13 explore: can we compute the pairing template densities ANALYTICALLY / SMOOTHLY incl. ISR?
# The discriminant is P_true/P_wrong.
#   P_true(m1,m2) = BW(m1)*BW(m2)*sqrt(lambda(s',m1,m2))   (the conv_mw lineshape; ISR via the s' spectrum) — analytic.
#   P_wrong       = combinatorial (jet from W1 + jet from W2) — NOT analytic, but fixed by the SAME physics:
#                   WW phase space at s' + two isotropic-ish W decays.  A wrong-pairing INVARIANT mass is
#                   rotation-invariant ⇒ independent of the WW production angle ⇒ a tiny phase-space MC suffices.
# This script GENERATES both from the model (sampling s' data-drivenly from gen_WW_m, so ISR is included) and
# validates them against the gen_qW truth in the data.  If the wrong-mass spectrum matches, the analytic/MC
# route is viable; the only remaining step for reco is the gen->reco resolution convolution.
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# python3 jax_prototype/prototype_pairing_ps_mc.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)
TREES = {160: "had_ff160_genqk", 240: "had_ff240_genqk", 365: "had_ff365_genqk"}
GW, MW, DECAY_P = 2.085, 80.4, 3.0
GREEN, BLUE = "#2e9e2e", "#1f6fc0"

def bw_line(m):                                   # per-W lineshape density ~ BW_fixed(m) * m^p
    mwgw = MW*GW; d = m*m - MW*MW
    return (mwgw/(d*d + mwgw*mwgw)) * m**DECAY_P
def sample_m(n, rng, lo=45.0, hi=125.0, ng=6000):
    mg = np.linspace(lo, hi, ng); cdf = np.cumsum(bw_line(mg)); cdf /= cdf[-1]
    return np.interp(rng.random(n), cdf, mg)
def mass(*vs):
    s = sum(vs); return np.sqrt(np.maximum(s[:, 3]**2 - s[:, 0]**2 - s[:, 1]**2 - s[:, 2]**2, 0))

def ps_mc(sqrts, rng):
    """phase-space MC of WW->4 partons at the given per-event sqrt(s') (ISR baked in). Returns weighted
    true and wrong dijet masses. W1 fixed along +z, W2 along -z (production angle irrelevant for inv. masses)."""
    n = len(sqrts); s = sqrts**2
    m1 = sample_m(n, rng); m2 = sample_m(n, rng)
    lam = (s - (m1+m2)**2) * (s - (m1-m2)**2)
    w = np.where((m1+m2 < sqrts) & (lam > 0), np.sqrt(np.maximum(lam, 0)), 0.0)   # WW phase-space weight
    E1 = (s + m1**2 - m2**2)/(2*sqrts); E2 = sqrts - E1
    pst = np.sqrt(np.maximum(E1**2 - m1**2, 0))
    b1 = pst/np.maximum(E1, 1e-9); g1 = E1/np.maximum(m1, 1e-9)
    b2 = pst/np.maximum(E2, 1e-9); g2 = E2/np.maximum(m2, 1e-9)
    def parton(mw, g, b, cosA, phi, zs):          # one decay parton, boosted along zs*z
        sA = np.sqrt(np.maximum(1-cosA**2, 0)); e = mw/2.0; bz = zs*b
        px = e*sA*np.cos(phi); py = e*sA*np.sin(phi); pz = e*cosA
        E = g*(e + bz*pz); Pz = g*(pz + bz*e)
        return np.stack([px, py, Pz, E], 1)
    c1 = 2*rng.random(n)-1; p1 = 2*np.pi*rng.random(n)   # W1 decay (isotropic)
    c2 = 2*rng.random(n)-1; p2 = 2*np.pi*rng.random(n)   # W2 decay
    a1 = parton(m1, g1, b1,  c1,        p1,        +1); a2 = parton(m1, g1, b1, -c1, p1+np.pi, +1)
    b1v = parton(m2, g2, b2,  c2,       p2,        -1); b2v = parton(m2, g2, b2, -c2, p2+np.pi, -1)
    # wrong pairings: (a1 b1)(a2 b2) and (a1 b2)(a2 b1)
    wA1 = mass(a1, b1v); wA2 = mass(a2, b2v); wB1 = mass(a1, b2v); wB2 = mass(a2, b1v)
    whi = np.concatenate([np.maximum(wA1, wA2), np.maximum(wB1, wB2)])
    wlo = np.concatenate([np.minimum(wA1, wA2), np.minimum(wB1, wB2)])
    return dict(mtrue=np.concatenate([m1, m2]), wtrue=np.concatenate([w, w]),
                whi=whi, wlo=wlo, ww=np.concatenate([w, w]))

def load_gen(ecm):
    F = f"outputs/treemaker/4q/step2_ff/{TREES[ecm]}/p8_ee_WW_ecm{ecm}.root"
    br = ["gen_W1_m", "gen_W2_m", "gen_WW_m"] + [f"gen_qW{i}_{c}" for i in range(4) for c in ("px","py","pz","e")]
    a = uproot.open(F)["events"].arrays(br, library="np")
    Q = {i: np.stack([a[f"gen_qW{i}_px"], a[f"gen_qW{i}_py"], a[f"gen_qW{i}_pz"], a[f"gen_qW{i}_e"]], 1) for i in range(4)}
    # wrong pairings vs the true (q0q1)(q2q3): (q0q2)(q1q3) and (q0q3)(q1q2)
    wA1 = mass(Q[0], Q[2]); wA2 = mass(Q[1], Q[3]); wB1 = mass(Q[0], Q[3]); wB2 = mass(Q[1], Q[2])
    whi = np.concatenate([np.maximum(wA1, wA2), np.maximum(wB1, wB2)])
    wlo = np.concatenate([np.minimum(wA1, wA2), np.minimum(wB1, wB2)])
    return dict(mtrue=np.concatenate([a["gen_W1_m"], a["gen_W2_m"]]), sqrts=a["gen_WW_m"], whi=whi, wlo=wlo)

rng = np.random.default_rng(1)
fig, axs = plt.subplots(3, 3, figsize=(14, 10.5))
COLS = [("true W mass [GeV]", "mtrue", "wtrue", (55, 105)),
        ("wrong-pairing m$_{hi}$ [GeV]", "whi", "ww", (20, 320)),
        ("wrong-pairing m$_{lo}$ [GeV]", "wlo", "ww", (10, 140))]
for ie, ecm in enumerate((160, 240, 365)):
    d = load_gen(ecm)
    mc = ps_mc(d["sqrts"], rng)                              # sqrt(s') sampled = data gen_WW_m (ISR included)
    series = {"mtrue": (d["mtrue"], None, mc["mtrue"], mc["wtrue"]),
              "whi":   (d["whi"], None, mc["whi"], mc["ww"]),
              "wlo":   (d["wlo"], None, mc["wlo"], mc["ww"])}
    for ic, (xlab, key, wkey, rng_x) in enumerate(COLS):
        ax = axs[ie, ic]; dat, _, mcv, mcw = series[key]; bins = np.linspace(*rng_x, 80)
        ax.hist(dat, bins=bins, density=True, histtype="stepfilled", color=GREEN, alpha=0.40, label="data (gen_qW truth)")
        ax.hist(mcv, bins=bins, weights=mcw, density=True, histtype="step", color=BLUE, lw=1.9, label="phase-space MC (analytic+ISR)")
        if ic == 0: ax.set_ylabel(f"√s = {ecm} GeV\nnormalised", fontsize=11)
        if ie == 0: ax.set_title(xlab, fontsize=11)
        if ie == 2: ax.set_xlabel(xlab, fontsize=10)
        ax.grid(alpha=.25)
        if ie == 0 and ic == 0: ax.legend(fontsize=8.5, loc="upper left")
fig.suptitle("Analytic/phase-space-MC pairing densities (ISR via data s′ spectrum) vs gen_qW truth", fontsize=13, y=0.998)
plt.tight_layout(rect=[0, 0, 1, 0.985])
p = os.path.join(EOSW, "pairing_ps_mc_validation.png"); plt.savefig(p, dpi=115); print(f"[plot] {p}")
# quick numeric agreement: KS-like compare of medians/quantiles for the wrong m_hi
for ecm in (160, 240, 365):
    d = load_gen(ecm); mc = ps_mc(d["sqrts"], rng)
    def q(x, w, p):
        i = np.argsort(x); xs = x[i]; cw = np.cumsum(w[i] if w is not None else np.ones_like(xs)); cw = cw/cw[-1]
        return np.interp(p, cw, xs)
    dq = [np.percentile(d["whi"], p) for p in (25, 50, 75)]
    mq = [q(mc["whi"], mc["ww"], p/100) for p in (25, 50, 75)]
    print(f"ecm{ecm} wrong m_hi quartiles  data {np.round(dq,1)}  MC {np.round(mq,1)}")
