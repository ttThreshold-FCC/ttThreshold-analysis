#!/usr/bin/env python3
"""
Fit DCB (or DCB + Gaussian mixture) to resolution/diff branches.

v2 improvements over v1:
  - MAD-based robust sigma init (not dragged by long tails)
  - Mode-based mu init (histogram peak, not median)
  - Per-branch clip/bin overrides for asymmetric distributions
  - Two-component DCB+Gaussian model for jet resolution variables
    that show a narrow core + secondary shoulder

Outputs (per ECM)
  outputs/response/plots/ecm<N>/<branch>.{png,pdf}   diagnostic plot per branch
  kinfit_inputs/dcb_params.h                         C++ header consumed by WWKinReco.h
  kinfit_inputs/dcb_results_ecm<N>.json              numerical fit results (for diagnostics)
"""

import os, sys, json, math, warnings
# Pin BLAS thread pools to 1 BEFORE numpy is imported. The binned-prior pass
# spawns a 24-process inner×outer pool; without this each worker would also
# spawn (#cores) BLAS threads → severe oversubscription.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# CLI args (= branch names to fit) routed through WW_FIT_ONLY so worker
# subprocesses inherit them.
if len(sys.argv) > 1:
    os.environ["WW_FIT_ONLY"] = ",".join(sys.argv[1:])
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation
from scipy.integrate import quad as _quad
from scipy.special import erf as _sp_erf
from eos_publish import publish

_SQRT2   = math.sqrt(2.0)
_LOG_MAX = math.log(np.finfo(np.float64).max)   # max safe float64 exponent ≈ 709.78
_LOG_MIN = -_LOG_MAX

# Output prefix for diagnostic plots is configurable via FIT_OUT_PREFIX so
# multiple fit_resolutions invocations can run in parallel.
OUT_PREFIX = os.environ.get("FIT_OUT_PREFIX", "outputs/response")
PLOTS_DIR  = f"{OUT_PREFIX}/plots"

# Channel switch. WW_CHANNEL=4q reconfigures the script for the fully-hadronic
# WW→4q pooled-jet resolution fit (p8_ee_WW_ecm160, single ECM): different input
# tree, 4-jet dR cut, pooled-over-4-jets jet response, and a PARAMS-ONLY header
# (kinfit_inputs_4q/dcb_params_4q.h) that reuses the struct defs / evaluators
# from the ℓνqq dcb_params.h (included via WWKinReco4q.h → WWKinReco.h).
WW_CHANNEL = os.environ.get("WW_CHANNEL", "lnuqq").strip()
IS_4Q      = (WW_CHANNEL == "4q")

# Kinfit inputs (dcb_params.h + dcb_results_*.json) live OUTSIDE the regularly
# cleaned outputs/ tree — they're consumed by the kinfit at compile/run time
# alongside logz_table.bin, see WWFunctions/WWKinReco.h.
FUNC_DIR   = os.environ.get("KINFIT_INPUT_DIR",
                            "kinfit_inputs_4q" if IS_4Q else "kinfit_inputs")
os.makedirs(FUNC_DIR, exist_ok=True)

if IS_4Q:
    ECM_LIST    = [160]
    INFILE_TMPL = os.environ.get("WW_INFILE_TMPL",
        "outputs/treemaker/4q/step1/had_4q_v1/p8_ee_WW_ecm{ecm}.root")
else:
    ECM_LIST    = [157, 160, 163]
    INFILE_TMPL = "outputs/treemaker/lnuqq/step1/semihad/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
NBINS_DEF   = 100
CLIP_DEF  = (0.5, 99.5)

# Jet/quark matching dR cut: only events with both jets matched to quarks
# within DR_MAX feed the resolution-prior fits. Looser cuts admit
# matching-failure tails into the priors and degrade kinfit convergence
# (verified non-monotonically at dR ∈ {0.1, 0.15, 0.2}). Override via
# FIT_DR_MAX env var (use a large value like 999 to effectively disable).
DR_MAX = float(os.environ.get("FIT_DR_MAX", "0.1"))
# 4q requires all four jets matched within DR_MAX (AND over DR_BRANCHES).
DR_BRANCHES = (("jet1_matched_q_dR", "jet2_matched_q_dR",
                "jet3_matched_q_dR", "jet4_matched_q_dR")
               if IS_4Q else ("jet1_matched_q_dR", "jet2_matched_q_dR"))

# Reco-kinematics branches loaded alongside resolutions to drive per-bin priors
# (and masked by the same dR cut). Required by BIN_CONFIG.
BIN_VAR_BRANCHES = (("reco_jet1_p", "reco_jet2_p", "reco_jet3_p", "reco_jet4_p",
                     "reco_jet1_costheta", "reco_jet2_costheta",
                     "reco_jet3_costheta", "reco_jet4_costheta")
                    if IS_4Q else
                    ("reco_jet1_p", "reco_jet2_p", "reco_lep_p",
                     "reco_jet1_costheta", "reco_jet2_costheta"))

# Equal-occupancy bin count per binned branch.
N_BINS_PRIOR = 5

# Per-branch binning configuration: map resolution branch → binning variable(s).
# A 1-tuple names a single reco-kinematics branch; a 2-tuple is concat'd to
# match the pooled jet1+jet2 resolution branches (jet_*_resol).  *_costheta
# values are folded via np.abs(...) before binning to fold forward/backward.
# Jet/lep resolutions follow the resol_vs_kin study (see project memory):
#   p_resp / θ resolution → bin on object p
#   φ resolution          → jets on |cosθ| (1/sinθ); lep on p (track scaling)
if IS_4Q:
    # 4q: bin ONLY the pooled-over-4-jets responses, with the binning variable
    # pooled over the same 4 jets (concat order jet1..jet4 matches the pooled
    # response branch). Response & θ bin on jet p; φ on |cosθ| — like ℓνqq.
    BIN_CONFIG = {
        "jet_p_resp_4q":      ("reco_jet1_p", "reco_jet2_p", "reco_jet3_p", "reco_jet4_p"),
        "jet_theta_resol_4q": ("reco_jet1_p", "reco_jet2_p", "reco_jet3_p", "reco_jet4_p"),
        "jet_phi_resol_4q":   ("reco_jet1_costheta", "reco_jet2_costheta",
                               "reco_jet3_costheta", "reco_jet4_costheta"),
    }
else:
    BIN_CONFIG = {
        "jet1_p_resp":      ("reco_jet1_p",),
        "jet2_p_resp":      ("reco_jet2_p",),
        "jet_p_resp":       ("reco_jet1_p", "reco_jet2_p"),
        "jet1_theta_resol": ("reco_jet1_p",),
        "jet2_theta_resol": ("reco_jet2_p",),
        "jet_theta_resol":  ("reco_jet1_p", "reco_jet2_p"),
        "jet1_phi_resol":   ("reco_jet1_costheta",),
        "jet2_phi_resol":   ("reco_jet2_costheta",),
        "jet_phi_resol":    ("reco_jet1_costheta", "reco_jet2_costheta"),
        "lep_p_resp":       ("reco_lep_p",),
        "lep_theta_resol":  ("reco_lep_p",),
        "lep_phi_resol":    ("reco_lep_p",),
    }

# Per-bin jet-pool exclusion for the 4q p-binned jet priors. From the per-bin
# per-jet comparison (jet_binned_priors_by_jet.png): the leading jet (jet1) has
# an anomalously low response when it lands in a low-p bin (mismeasured-hard
# population), and the softest jet (jet4) has ~no statistics in high-p bins. So
# exclude those (jet, bin) combos from the pooled binned FIT. The bin EDGES are
# still computed on the full 4-jet pooled distribution (kept consistent with the
# kinfit's pick_bin), only the per-bin training subset is restricted.
# Keyed: branch -> {1-based jet index: set of 0-based bin indices to EXCLUDE}.
# Applies only to the p-binned branches (φ is binned in |cosθ|, where this p-based
# rule doesn't map). N_BINS_PRIOR=5, so low = {0,1,2}, high = {3,4}.
POOL_BIN_EXCLUDE_4Q = {
    "jet_p_resp_4q":      {1: {0, 1, 2}, 4: {3, 4}},
    "jet_theta_resol_4q": {1: {0, 1, 2}, 4: {3, 4}},
}

# ── Per-branch configuration overrides ─────────────────────────────────────
BRANCH_CONFIG = {
    # Jet p response: detector core + wide-radiation tail → DCB+G.
    # f_wide_max=0.5 enforces "core = bulk" — otherwise some bins land in a
    # degenerate basin where the wide-Gauss takes the bulk and dcb-core tracks
    # an outlier population, faking a second peak in the kinfit pull.
    "jet1_p_resp":          {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet2_p_resp":          {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet_p_resp":           {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet1_theta_resol":     {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet2_theta_resol":     {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet_theta_resol":      {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet1_phi_resol":       {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet2_phi_resol":       {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet_phi_resol":        {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    # WW→4q: jets 3/4 (same detector model as jets 1/2) + the pooled-over-4-jets
    # responses (the only ones the 4q kinfit actually consumes). Conservative
    # single jet prior; see treemaker_common.cluster_jets_4q / WWKinReco4q.h.
    "jet3_p_resp":          {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet4_p_resp":          {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet_p_resp_4q":        {"clip": (0.2, 99.8),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet3_theta_resol":     {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet4_theta_resol":     {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet_theta_resol_4q":   {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet3_phi_resol":       {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet4_phi_resol":       {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "jet_phi_resol_4q":     {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    # Lepton p response: narrow detector core + heavy power-law left tail (FSR)
    # + sharp exponential right cutoff at 1 (kinematic ceiling). dcber3g —
    # mirror-image of expleft2g; physically motivated and fits ~2× better.
    "lep_p_resp":           {"clip": (0.1, 99.9),  "nbins": 300, "model": "dcber3g"},
    # Lepton angular resolutions: tight detector core + wide-angle FSR tails → DCB+G.
    "lep_theta_resol":      {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    "lep_phi_resol":        {"clip": (0.5, 99.5),  "nbins": 150, "model": "dcb2g", "f_wide_max": 0.5},
    # MET: asymmetric core + shoulder + outlier Gaussians (asymgauss3g, smooth
    # everywhere). Two-Gauss variant left χ²/ndf ~25 — single wide-Gauss can't
    # span both the shoulder and the far tail.
    "met_p_resp":           {"clip": (0.5, 99.5),  "nbins": 150, "model": "asymgauss3g"},
    "met_theta_resol":      {"clip": (0.5, 99.5),  "nbins": 150, "model": "asymgauss3g"},
    "met_phi_resol":        {"clip": (0.5, 99.5),  "nbins": 150, "model": "asymgauss3g"},
    # Gen-level total momenta: ~85% of events have collinear (or no) ISR → spike at 0
    # narrower than any reasonable bin; the rest form a smooth ISR tail.
    # dcb2g (narrow DCB core + wide Gaussian) handles the unresolved spike + smooth
    # tail; dcbgb's box plateau does not match the data and gives chi²/ndf in the
    # hundreds for px/py.
    "gen_WW_px":           {"clip": (0.1, 99.9),  "nbins": 1500, "model": "dcb2g",
                             "zoom_xlim": (-0.5, 0.5)},
    "gen_WW_py":           {"clip": (0.1, 99.9),  "nbins": 1500, "model": "dcb2g",
                             "zoom_xlim": (-0.5, 0.5)},
    "gen_WW_pz":           {"clip": (0.1, 99.9),  "nbins": 1500, "model": "dcb2g",
                             "zoom_xlim": (-3.0, 3.0)},
    # Gen WW invariant mass minus ECM. Peak just below 0 (ISR), hard boundary at 0.
    "gen_WW_m_minus_ecm": {"clip": (0.5, 100.0), "nbins": 150, "model": "dcber2g"},
    # m(WW)−m(ee): bounded ≤0 by physics. Fit symmetrized data with
    # spike_dcb2g; kinfit adds a barrier on x>0 (see WWKinReco.h).
    "gen_WW_m_minus_m_ee": {"clip": (0.5, 99.5), "nbins": 200, "model": "spike_dcb2g",
                             "symmetrize": True,
                             "delta_threshold": 0.001, "sig_res": 0.001,
                             "mu_fix": 0.0,
                             "f_wide_max": 0.45,
                             "zoom_xlim": (-0.5, 0.5),
                             "log_y": True},
    # m(e+e-) − ECM at depth=1 in the e± chain (post-BES, pre-ISR). Symmetric
    # Gaussian smearing of the beam energies — single Gauss is sufficient.
    "gen_ee_m_minus_ecm":  {"clip": (0.1, 99.9),  "nbins": 100, "model": "gauss"},
    "gen_ee_pz":           {"clip": (0.1, 99.9),  "nbins": 100, "model": "gauss"},
    # ISR 4-momentum (depth=1 − depth=2 e+e-). True delta-at-0 spike (~40-43% of
    # events have no ISR, sub-bin precision) + smooth ISR tail (σ ≈ 0.3-2 GeV).
    # `spike_dcb2g`: f_delta · G(0, σ_res) + (1−f_delta) · dcb2g(x). f_delta is
    # the fraction with |x|<delta_threshold (no-ISR), σ_res smooths the delta
    # for kinfit numerical stability, and dcb2g is fitted on the |x|>threshold
    # subset.
    # mu_fix=0: ISR p{x,y,z} are symmetric around 0 by construction; pin the
    # body μ_c so the unconstrained fit can't drift to ±10 MeV.
    "gen_isr_px":         {"clip": (0.5, 99.5),  "nbins": 200,  "model": "spike_dcb2g",
                             "delta_threshold": 0.001, "sig_res": 0.001,
                             "mu_fix": 0.0,
                             "zoom_xlim": (-2.0, 2.0)},
    "gen_isr_py":         {"clip": (0.5, 99.5),  "nbins": 200,  "model": "spike_dcb2g",
                             "delta_threshold": 0.001, "sig_res": 0.001,
                             "mu_fix": 0.0,
                             "zoom_xlim": (-2.0, 2.0)},
    # f_wide_max=0.45: at ecm160 the unconstrained fit lands at f_wide=0.82
    # with σ_wide(47 MeV) < σ_core(61 MeV) — wide-Gauss/dcb-core roles swap,
    # σ_core blows up by 7× vs the other ECMs, and the kinfit's effective
    # ISR-pz prior shape becomes ECM-inconsistent. Cap f_wide so the dcb_core
    # is always the dominant component (px/py already have f_wide≈0.01,
    # unaffected by the cap).
    "gen_isr_pz":         {"clip": (0.5, 99.5),  "nbins": 200,  "model": "spike_dcb2g",
                             "f_wide_max": 0.45,
                             "delta_threshold": 0.001, "sig_res": 0.001,
                             "mu_fix": 0.0,
                             "zoom_xlim": (-10.0, 10.0)},
}

# Virtual combined branches: concatenate the per-jet distributions into a single
# pooled distribution for the jet fits. ℓνqq pools jet1+jet2; 4q pools all four
# jets into one conservative response (the pooling code handles N-tuples).
if IS_4Q:
    COMBINED_BRANCHES = {
        "jet_p_resp_4q":      ("jet1_p_resp", "jet2_p_resp", "jet3_p_resp", "jet4_p_resp"),
        "jet_theta_resol_4q": ("jet1_theta_resol", "jet2_theta_resol",
                               "jet3_theta_resol", "jet4_theta_resol"),
        "jet_phi_resol_4q":   ("jet1_phi_resol", "jet2_phi_resol",
                               "jet3_phi_resol", "jet4_phi_resol"),
    }
else:
    COMBINED_BRANCHES = {
        "jet_p_resp":      ("jet1_p_resp",     "jet2_p_resp"),
        "jet_theta_resol": ("jet1_theta_resol", "jet2_theta_resol"),
        "jet_phi_resol":   ("jet1_phi_resol",   "jet2_phi_resol"),
    }

# Branches used by the kinematic fit (also the full set of step1 outputs we fit).
if IS_4Q:
    # WW→4q on p8: per-jet branches (loaded so the dR mask + pooling can run;
    # each is also fit & emitted, harmless) → pooled jet_*_4q. Plus the only two
    # NON-degenerate WW-system priors in p8: longitudinal ISR (gen_isr_pz) and
    # the mass loss (gen_WW_m_minus_m_ee). p8 has negligible BES
    # (gen_ee_* ≈ 0) and purely-collinear ISR (gen_isr_px/py ≈ 0 to machine
    # precision), so those are dropped: no BES nuisances, and transverse balance
    # (WW pT ≈ 0) is enforced by a hardcoded tight Gaussian in WWKinReco4q.h.
    KINFIT_BRANCHES = [
        "jet1_p_resp", "jet2_p_resp", "jet3_p_resp", "jet4_p_resp",
        "jet1_theta_resol", "jet2_theta_resol", "jet3_theta_resol", "jet4_theta_resol",
        "jet1_phi_resol", "jet2_phi_resol", "jet3_phi_resol", "jet4_phi_resol",
        "gen_isr_pz",
        "gen_WW_m_minus_m_ee",
    ]
else:
    KINFIT_BRANCHES = [
        "jet1_p_resp", "jet2_p_resp", "lep_p_resp", "met_p_resp",
        "jet1_theta_resol", "jet2_theta_resol", "jet1_phi_resol", "jet2_phi_resol",
        "lep_theta_resol",  "lep_phi_resol",
        "met_theta_resol",  "met_phi_resol",
        "gen_WW_px", "gen_WW_py", "gen_WW_pz",
        "gen_WW_m_minus_ecm",
        "gen_ee_m_minus_ecm",    # BES proxy: m(post-BES e+e-) − ECM, single-Gauss fit
        "gen_ee_pz",             # BES asymmetry δE+ − δE−, single-Gauss fit
        "gen_WW_m_minus_m_ee",   # pure ISR mass-loss (BES variance removed)
        # ISR 3-momentum: (depth=1 e+e-) − (depth=2 e+e-). Same delta-at-0 shape as
        # gen_WW_*: ~40% no-ISR spike + smooth tail → spike_dcb2g.
        "gen_isr_px", "gen_isr_py", "gen_isr_pz",
    ]

# ── Model functions ──────────────────────────────────────────────────────────

def _dcb_core(t, aL, nL, aR, nR):
    """Unnormalized DCB shape in reduced variable t = (x-mu)/sigma.
    Log-space computation avoids overflow when nL/nR are large; np.where
    evaluates all branches eagerly so inactive branches must not overflow.
    """
    aL, nL, aR, nR = abs(aL), abs(nL), abs(aR), abs(nR)
    BL = nL / aL - aL
    BR = nR / aR - aR
    log_AL = nL * np.log(nL / aL) - 0.5 * aL * aL
    log_AR = nR * np.log(nR / aR) - 0.5 * aR * aR
    return np.where(
        t < -aL,
        np.exp(np.minimum(np.maximum(log_AL - nL * np.log(np.maximum(BL - t, 1e-10)), _LOG_MIN), _LOG_MAX)),
        np.where(
            t > aR,
            np.exp(np.minimum(np.maximum(log_AR - nR * np.log(np.maximum(BR + t, 1e-10)), _LOG_MIN), _LOG_MAX)),
            np.exp(-0.5 * t * t)
        )
    )


def dcb(x, N, mu, sigma, aL, nL, aR, nR):
    return N * _dcb_core((x - mu) / sigma, aL, nL, aR, nR)


def dcb_gauss(x, N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, mu_w, sigma_w):
    """Narrow DCB core + broad Gaussian component with independent centres."""
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


def _dcb_expleft_core(t, aL, kL, aR, nR):
    """DCB variant: exponential left tail (rate kL) + Gaussian core + power-law right tail.
    Left tail: exp(-0.5*aL^2 + kL*(aL+t)) for t < -aL.  kL > 0 gives faster decay than Gaussian.
    np.minimum caps the exponent at 0 in inactive branches to prevent overflow.
    """
    kL, aL, aR, nR = abs(kL), abs(aL), abs(aR), abs(nR)
    BR = nR / aR - aR
    log_AR = nR * np.log(nR / aR) - 0.5 * aR * aR
    return np.where(
        t < -aL,
        np.exp(-0.5 * aL * aL + np.minimum(kL * (aL + t), 0.0)),
        np.where(
            t > aR,
            np.exp(np.minimum(np.maximum(log_AR - nR * np.log(np.maximum(BR + t, 1e-10)), _LOG_MIN), _LOG_MAX)),
            np.exp(-0.5 * t * t)
        )
    )


def dcb_expleft_gauss(x, N, mu_c, sigma_c, aL, kL, aR, nR, f_wide, mu_w, sigma_w):
    """Exp-left DCB core + broad Gaussian component."""
    core = _dcb_expleft_core((x - mu_c) / sigma_c, aL, kL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


def _dcb_expright_core(t, aL, nL, aR, kR):
    """Power-law left tail + Gaussian core + exponential right tail.
    Left:  standard DCB power-law for t < -aL.
    Right: exp(-0.5*aR^2 - kR*(t-aR)) for t > aR (kR > 0 = fast right decay).
    Physically: heavy ISR tail on left, sharp kinematic cutoff on right.
    """
    aL, nL, aR, kR = abs(aL), abs(nL), abs(aR), abs(kR)
    BL = nL / aL - aL
    log_AL = nL * np.log(nL / aL) - 0.5 * aL * aL
    return np.where(
        t < -aL,
        np.exp(np.minimum(np.maximum(log_AL - nL * np.log(np.maximum(BL - t, 1e-10)), _LOG_MIN), _LOG_MAX)),
        np.where(
            t > aR,
            np.exp(np.minimum(-0.5 * aR * aR - kR * (t - aR), _LOG_MAX)),
            np.exp(-0.5 * t * t)
        )
    )


def dcb_expright_gauss(x, N, mu_c, sigma_c, aL, nL, aR, kR, f_wide, mu_w, sigma_w):
    """Power-law-left DCB core + broad Gaussian. kR: exponential right-tail decay rate."""
    core = _dcb_expright_core((x - mu_c) / sigma_c, aL, nL, aR, kR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


def dcb_expright_3gauss(x, N, mu_c, sigma_c, aL, nL, aR, kR,
                        f_s, mu_s, sigma_s, f_o, mu_o, sigma_o):
    """3-component lep_p_resp model: dcber core + shoulder Gaussian + outlier Gaussian.
    Core captures tracker resolution + soft FSR; shoulder Gaussian (~0.94, σ~0.03) the
    intermediate hard-FSR plateau; outlier Gaussian the deep-radiative tail. Sums to
    N when integrated over x. f_core = 1 - f_s - f_o (clamped ≥ 0)."""
    core      = _dcb_expright_core((x - mu_c) / sigma_c, aL, nL, aR, kR)
    shoulder  = np.exp(-0.5 * ((x - mu_s) / sigma_s) ** 2)
    outlier   = np.exp(-0.5 * ((x - mu_o) / sigma_o) ** 2)
    f_core    = max(0.0, 1.0 - abs(f_s) - abs(f_o))
    return N * (f_core * core + abs(f_s) * shoulder + abs(f_o) * outlier)


def gauss(x, N, mu, sigma):
    """Single Gaussian, area-normalised to N: ∫ gauss dx = N."""
    sigma = abs(sigma)
    return N * np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def fit_gauss(centers, counts, mu0, sig0):
    """Single-Gaussian fit. Returns (popt=(N,mu,sigma), pcov, fit_ok, chi2)."""
    sigma_y = np.sqrt(np.maximum(counts, 1.0))
    bw = float(centers[1] - centers[0]) if len(centers) > 1 else 1.0
    p0 = [float(counts.sum()) * bw, mu0, sig0]
    try:
        popt, pcov = curve_fit(gauss, centers, counts, p0=p0, sigma=sigma_y,
                               absolute_sigma=False, maxfev=20000)
        pred = gauss(centers, *popt)
        chi2 = float(np.sum((counts - pred) ** 2 / np.maximum(sigma_y ** 2, 1.0)))
        return popt, pcov, True, chi2
    except Exception:
        return None, None, False, float("inf")


def asymgauss(x, N, mu, sigma_L, sigma_R):
    """Asymmetric (split) Gaussian: σ_L for x<µ, σ_R for x≥µ. C¹-continuous
    at x=µ (value and gradient match; second derivative is discontinuous).
    Area-normalised to N: ∫ asymgauss dx = N."""
    sigma_L = max(abs(sigma_L), 1e-300)
    sigma_R = max(abs(sigma_R), 1e-300)
    sigma   = np.where(x < mu, sigma_L, sigma_R)
    z       = (x - mu) / sigma
    norm    = math.sqrt(2.0 / math.pi) / (sigma_L + sigma_R)
    return N * norm * np.exp(-0.5 * z * z)


def fit_asymgauss(centers, counts, mu0, sig0):
    """Asymmetric-Gaussian fit. Returns (popt=(N,mu,sigma_L,sigma_R), pcov,
    fit_ok, chi2). Multiple starts to avoid σ_L≈σ_R degeneracies."""
    N0 = float(counts.sum()) * float(centers[1] - centers[0]) if len(centers) > 1 else float(counts.sum())
    lo = [0.0,    -np.inf,   1e-9, 1e-9]
    hi = [np.inf,  np.inf,   np.inf, np.inf]
    starts = [
        [N0, mu0, sig0,        sig0],            # symmetric (same as a single Gauss)
        [N0, mu0, sig0 * 0.5,  sig0 * 1.5],      # narrow-left, broad-right
        [N0, mu0, sig0 * 1.5,  sig0 * 0.5],      # broad-left, narrow-right
        [N0, mu0, sig0 * 0.3,  sig0 * 2.0],      # very asymmetric
        [N0, mu0, sig0 * 2.0,  sig0 * 0.3],
    ]
    return _best_fit(asymgauss, centers, counts, starts, lo, hi)


def asymgauss2g(x, N, mu, sigma_L, sigma_R, f_wide, mu_w, sigma_w):
    """Asymmetric Gaussian core + wide Gaussian. Both individually normalised
    so the convex sum (1−f_wide)·core + f_wide·wide is also normalised.
    Smooth everywhere (C¹ at x=µ for the core; C∞ for the wide), no piecewise
    handoff cliffs."""
    sigma_L = max(abs(sigma_L), 1e-300)
    sigma_R = max(abs(sigma_R), 1e-300)
    sigma_w = max(abs(sigma_w), 1e-300)
    sigma_x = np.where(x < mu, sigma_L, sigma_R)
    z_core  = (x - mu) / sigma_x
    norm_core = math.sqrt(2.0 / math.pi) / (sigma_L + sigma_R)
    core    = norm_core * np.exp(-0.5 * z_core * z_core)
    z_wide  = (x - mu_w) / sigma_w
    norm_wide = 1.0 / (sigma_w * math.sqrt(2.0 * math.pi))
    wide    = norm_wide * np.exp(-0.5 * z_wide * z_wide)
    return N * ((1.0 - f_wide) * core + f_wide * wide)


def fit_asymgauss2g(centers, counts, mu0, sig0, f_wide_max=0.5):
    """Asymmetric-Gaussian + wide-Gauss fit. Returns popt=(N,mu,σ_L,σ_R,
    f_wide,µ_w,σ_w)."""
    N0 = float(counts.sum()) * float(centers[1] - centers[0]) if len(centers) > 1 else float(counts.sum())
    # params: N, mu, sigma_L, sigma_R, f_wide, mu_w, sigma_w
    lo = [0.0,    -np.inf, 1e-9, 1e-9, 0.001, -np.inf, 1e-6]
    hi = [np.inf,  np.inf, np.inf, np.inf, float(f_wide_max), np.inf, np.inf]
    starts = [
        [N0, mu0, sig0,        sig0,        0.05, mu0,             sig0 * 5],
        [N0, mu0, sig0 * 0.5,  sig0,        0.10, mu0,             sig0 * 5],
        [N0, mu0, sig0,        sig0 * 0.5,  0.10, mu0,             sig0 * 5],
        [N0, mu0, sig0 * 0.3,  sig0 * 1.5,  0.15, mu0,             sig0 * 8],
        [N0, mu0, sig0 * 1.5,  sig0 * 0.3,  0.15, mu0,             sig0 * 8],
        [N0, mu0, sig0 * 0.5,  sig0 * 0.5,  0.20, mu0 - 2.0*sig0,  sig0 * 10],
        [N0, mu0, sig0 * 0.5,  sig0 * 0.5,  0.20, mu0 + 2.0*sig0,  sig0 * 10],
    ]
    return _best_fit(asymgauss2g, centers, counts, starts, lo, hi)


def asymgauss3g(x, N, mu, sigma_L, sigma_R,
                f_s, mu_s, sigma_s, f_o, mu_o, sigma_o):
    """Asymgauss core + 'shoulder' Gaussian + 'outlier' (very wide) Gaussian.
    Convex sum: (1−f_s−f_o)·core + f_s·shoulder + f_o·outlier. Each component
    individually normalised. Smooth everywhere."""
    sigma_L = max(abs(sigma_L), 1e-300)
    sigma_R = max(abs(sigma_R), 1e-300)
    sigma_s = max(abs(sigma_s), 1e-300)
    sigma_o = max(abs(sigma_o), 1e-300)
    sigma_x = np.where(x < mu, sigma_L, sigma_R)
    z_core  = (x - mu) / sigma_x
    norm_core = math.sqrt(2.0 / math.pi) / (sigma_L + sigma_R)
    core    = norm_core * np.exp(-0.5 * z_core * z_core)
    z_s     = (x - mu_s) / sigma_s
    n_s     = 1.0 / (sigma_s * math.sqrt(2.0 * math.pi))
    shoulder = n_s * np.exp(-0.5 * z_s * z_s)
    z_o     = (x - mu_o) / sigma_o
    n_o     = 1.0 / (sigma_o * math.sqrt(2.0 * math.pi))
    outlier = n_o * np.exp(-0.5 * z_o * z_o)
    f_core  = max(0.0, 1.0 - abs(f_s) - abs(f_o))
    return N * (f_core * core + abs(f_s) * shoulder + abs(f_o) * outlier)


def fit_asymgauss3g(centers, counts, mu0, sig0):
    """Asymgauss + 2-Gauss fit. Returns popt=(N,mu,σ_L,σ_R,f_s,µ_s,σ_s,
    f_o,µ_o,σ_o)."""
    N0 = float(counts.sum()) * float(centers[1] - centers[0]) if len(centers) > 1 else float(counts.sum())
    # params: N, mu, sigma_L, sigma_R, f_s, mu_s, sigma_s, f_o, mu_o, sigma_o
    lo = [0.0,    -np.inf, 1e-9, 1e-9, 0.001, -np.inf, 1e-6, 0.001, -np.inf, 1e-6]
    hi = [np.inf,  np.inf, np.inf, np.inf, 0.45, np.inf, np.inf, 0.30, np.inf, np.inf]
    starts = [
        # symmetric core, modest shoulder, small far-tail outlier
        [N0, mu0, sig0,        sig0,        0.20, mu0, sig0 * 4,  0.02, mu0, sig0 * 20],
        [N0, mu0, sig0 * 0.7,  sig0,        0.25, mu0, sig0 * 5,  0.03, mu0, sig0 * 25],
        [N0, mu0, sig0,        sig0 * 0.7,  0.25, mu0, sig0 * 5,  0.03, mu0, sig0 * 25],
        [N0, mu0, sig0 * 0.5,  sig0 * 1.2,  0.30, mu0, sig0 * 6,  0.05, mu0, sig0 * 30],
        [N0, mu0, sig0 * 1.2,  sig0 * 0.5,  0.30, mu0, sig0 * 6,  0.05, mu0, sig0 * 30],
        [N0, mu0, sig0 * 0.5,  sig0 * 0.5,  0.35, mu0 - sig0, sig0 * 8,  0.05, mu0, sig0 * 40],
        [N0, mu0, sig0 * 0.5,  sig0 * 0.5,  0.35, mu0 + sig0, sig0 * 8,  0.05, mu0, sig0 * 40],
    ]
    return _best_fit(asymgauss3g, centers, counts, starts, lo, hi)


def dcb_gaussbox(x, N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box):
    """Narrow DCB core + Gaussian-smeared box wide component.
    Wide: Box(−p_max, p_max) convolved with Gaussian(sigma_box) — flat plateau, fast erf edges.
    Params: N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box  (10 params, same as dcb_gauss).
    """
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    p_max = abs(p_max)
    sb = max(abs(sigma_box), 1e-10)
    sq2 = _SQRT2 * sb
    wide = 0.5 * (_sp_erf((x + p_max) / sq2) - _sp_erf((x - p_max) / sq2))
    wide_peak = max(float(_sp_erf(p_max / sq2)), 1e-10)
    return N * ((1.0 - f_wide) * core + f_wide * wide / wide_peak)


# ── Generic multi-start fitter ───────────────────────────────────────────────

def fit_dcb2g_iminuit(centers, counts, mu0, sig0, f_wide_max=0.95, mu_fix=None):
    """Poisson binned NLL with iminuit for distributions where chi² gets trapped.
    `f_wide_max`: cap on the wide-Gauss fraction (see fit_dcb2g).
    `mu_fix`: if set, freezes μ_c (use for physics-symmetric distributions)."""
    from iminuit import Minuit

    N0   = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)

    # Param order: N, mu_c, sc, aL, nL, aR, nR, fw, muw, sw
    def nll(N, mu_c, sc, aL, nL, aR, nR, fw, muw, sw):
        if sc <= 0 or sw <= 0:
            return 1e15
        pred = dcb_gauss(centers, abs(N), mu_c, abs(sc), abs(aL), abs(nL),
                         abs(aR), abs(nR), abs(fw), muw, abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    # Starting conditions tuned for narrow-core, left-bounded distributions:
    # small sigma_c, small aL (early left power-law), moderate aR for right tail
    starts = [
        [N0, mu0, sig0 * 0.5,  0.5,  3., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.4,  0.4,  2., 0.7, 3., 0.20, mu0 + 4*sig0, 10*sig0],
        [N0, mu0, sig0 * 0.5,  0.5,  4., 0.5, 2., 0.25, mu0,           8*sig0],
        [N0, mu0, sig0,        1.2,  5., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
        # broad LEFT component (for left-heavy distributions like fromele responses)
        [N0, mu0, sig0 * 0.5,  0.5,  3., 1.5, 3., 0.30, mu0 - 3*sig0,  6*sig0],
        [N0, mu0, sig0 * 0.4,  0.4,  2., 1.2, 3., 0.40, mu0 - 4*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.5,  0.5,  5., 1.5, 3., 0.50, mu0 - 6*sig0, 12*sig0],
        # very narrow spike + broad symmetric wings (ISR total-momentum distributions)
        [N0, mu0, sig0 * 0.04, 1.5,  5., 1.5,  5., 0.60, mu0, sig0 * 1.0],
        [N0, mu0, sig0 * 0.03, 1.0,  3., 1.0,  3., 0.70, mu0, sig0 * 0.8],
        [N0, mu0, sig0 * 0.05, 1.0,  5., 1.0,  5., 0.80, mu0, sig0 * 1.2],
        # ultra-narrow spike: wings extend further than MAD-based sig0 suggests
        [N0, mu0, sig0 * 0.02, 2.0, 10., 2.0, 10., 0.80, mu0, sig0 * 2.0],
        [N0, mu0, sig0 * 0.01, 1.5,  8., 1.5,  8., 0.88, mu0, sig0 * 1.5],
        [N0, mu0, sig0 * 0.03, 1.5,  7., 1.5,  7., 0.75, mu0, sig0 * 2.5],
    ]
    limits = [(1e-3, None), (None, None), (1e-6, None), (0.3, 8.),
              (1.01, 200.), (0.3, 8.), (1.01, 200.), (0.01, float(f_wide_max)),
              (None, None), (1e-4, None)]
    names  = ['N','mu_c','sc','aL','nL','aR','nR','fw','muw','sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        if mu_fix is not None:
            p0 = list(p0); p0[1] = float(mu_fix)
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            if mu_fix is not None:
                m.fixed["mu_c"] = True
            m.migrad()
            if not m.valid:
                m.migrad()   # second pass
            if m.valid:
                popt = list(m.values)
                pred = dcb_gauss(centers, *[abs(v) if j not in (1, 8) else v
                                            for j, v in enumerate(popt)])
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    # pcov not available from iminuit without HESSE — pass None; ok=True if converged
    return best_popt, None, True, best_chi2


def fit_dcb_gaussbox_iminuit(centers, counts, mu0, sig0):
    """Poisson NLL with iminuit for DCB + Gaussian-smeared box.
    Params: N, mu_c, sc, aL, nL, aR, nR, fw, p_max, sigma_box
    """
    from iminuit import Minuit
    N0 = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)
    x_span = 0.5 * (centers[-1] - centers[0])

    def nll(N, mu_c, sc, aL, nL, aR, nR, fw, p_max, sw):
        if sc <= 0 or sw <= 0 or p_max <= 0:
            return 1e15
        pred = dcb_gaussbox(centers, abs(N), mu_c, abs(sc), abs(aL), abs(nL),
                             abs(aR), abs(nR), abs(fw), abs(p_max), abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    starts = [
        [N0, mu0, sig0*0.005, 1.0, 50., 1.0, 50., 0.20, x_span*0.85, sig0*0.15],
        [N0, mu0, sig0*0.005, 1.5, 30., 1.5, 30., 0.18, x_span*0.80, sig0*0.15],
        [N0, mu0, sig0*0.010, 1.0,100., 1.0,100., 0.22, x_span*0.80, sig0*0.12],
        [N0, mu0, sig0*0.003, 1.2, 80., 1.2, 80., 0.16, x_span*0.75, sig0*0.10],
        [N0, mu0, sig0*0.007, 2.0, 20., 2.0, 20., 0.20, x_span*0.90, sig0*0.20],
        [N0, mu0, sig0*0.010, 1.5, 50., 1.5, 50., 0.25, x_span*0.85, sig0*0.18],
        [N0, mu0, sig0*0.003, 1.0,200., 1.0,200., 0.15, x_span*0.90, sig0*0.08],
        [N0, mu0, sig0*0.007, 1.0, 30., 1.0, 30., 0.30, x_span*0.80, sig0*0.20],
        [N0, mu0, sig0*0.015, 2.0, 15., 2.0, 15., 0.35, x_span*0.85, sig0*0.25],
        [N0, mu0, sig0*0.020, 1.5, 10., 1.5, 10., 0.40, x_span*0.80, sig0*0.30],
    ]
    limits = [(1e-3,None),(None,None),(1e-6,None),(0.3,8.),(1.01,200.),
              (0.3,8.),(1.01,200.),(0.01,0.40),(1.,None),(1e-4,None)]
    names = ['N','mu_c','sc','aL','nL','aR','nR','fw','p_max','sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            m.migrad()
            if not m.valid:
                m.migrad()
            if m.valid:
                popt = list(m.values)
                pred = dcb_gaussbox(centers, *[abs(v) if j != 1 else v
                                               for j, v in enumerate(popt)])
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    return best_popt, None, True, best_chi2


def fit_dcb_expleft2g_iminuit(centers, counts, mu0, sig0, constrain_mu0=False):
    """Poisson binned NLL with iminuit for the expleft2g model.
    Params: N, mu_c, sc, aL, kL, aR, nR, fw, muw, sw
    constrain_mu0: restrict mu_c near 0 and aR small to force hard-right-cutoff solution.
    """
    from iminuit import Minuit

    N0   = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)

    def nll(N, mu_c, sc, aL, kL, aR, nR, fw, muw, sw):
        if sc <= 0 or sw <= 0:
            return 1e15
        pred = dcb_expleft_gauss(centers, abs(N), mu_c, abs(sc), abs(aL), abs(kL),
                                 abs(aR), abs(nR), abs(fw), muw, abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    starts = [
        [N0, mu0, sig0 * 0.5,  1.0,  8., 0.8,   3., 0.15, mu0 + 3*sig0,  8*sig0],
        [N0, mu0, sig0 * 0.5,  0.8, 10., 0.7,   3., 0.20, mu0 + 4*sig0, 10*sig0],
        # left-biased wide component
        [N0, mu0, sig0 * 0.05, 0.5, 10., 0.5,   3., 0.50, mu0 - 3*sig0,  2*sig0],
        [N0, mu0, sig0 * 0.03, 0.5, 15., 0.5,   3., 0.60, mu0 - 5*sig0,  4*sig0],
        [N0, mu0, sig0 * 0.10, 1.0,  5., 1.0,   5., 0.35, mu0 - 3*sig0,  2*sig0],
        # hard right cutoff: small aR, large nR
        [N0, mu0, sig0 * 0.3,  0.5,  5., 0.1, 100., 0.40, mu0 - 3*sig0,  2*sig0],
        [N0, mu0, sig0 * 0.2,  0.5,  8., 0.1, 200., 0.50, mu0 - 4*sig0,  3*sig0],
        [N0, mu0, sig0 * 0.1,  0.3, 10., 0.1, 300., 0.60, mu0 - 5*sig0,  4*sig0],
        [N0, mu0, sig0 * 0.5,  1.0,  5., 0.1, 100., 0.30, mu0 - 2*sig0,  2*sig0],
    ]
    if constrain_mu0:
        # Discard unconstrained starts; use large-sigma_c starts matching physical ISR slope.
        starts = [
            [N0, 0.0,  2.0, 0.30, 0.20, 0.10, 200., 0.5, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  3.0, 0.30, 0.30, 0.10, 300., 0.4, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.20, 0.15, 0.05, 200., 0.6, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  4.0, 0.50, 0.40, 0.15, 200., 0.3, mu0 - 3*sig0, 5*sig0],
            [N0, -0.1, 2.5, 0.30, 0.25, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  2.0, 0.50, 0.20, 0.08, 300., 0.6, mu0 - 3*sig0, 5*sig0],
            [N0, 0.0,  1.0, 0.20, 0.10, 0.05, 400., 0.7, mu0 - 3*sig0, 4*sig0],
            [N0, -0.2, 3.0, 0.40, 0.30, 0.10, 200., 0.4, mu0 - 2*sig0, 4*sig0],
            [N0, 0.0,  5.0, 0.50, 0.50, 0.20, 300., 0.3, mu0 - 2*sig0, 5*sig0],
            [N0, 0.0,  2.0, 0.10, 0.20, 0.10, 200., 0.5, mu0 - 3*sig0, 3*sig0],
            [N0, -0.3, 2.0, 0.30, 0.20, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.30, 0.15, 0.05, 250., 0.55, mu0 - 2*sig0, 3*sig0],
        ]
        limits = [(1e-3, None), (-0.5, 0.1), (0.5, 10.), (0.05, 3.),
                  (0.01, 5.), (0.02, 0.4), (1.01, 500.), (0.01, 0.95),
                  (None, None), (1e-4, None)]
    else:
        limits = [(1e-3, None), (None, None), (1e-6, None), (0.1, 6.),
                  (0.1, 50.), (0.1, 8.), (1.01, 500.), (0.01, 0.95),
                  (None, None), (1e-4, None)]
    names = ['N', 'mu_c', 'sc', 'aL', 'kL', 'aR', 'nR', 'fw', 'muw', 'sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            m.migrad()
            if not m.valid:
                m.migrad()
            if m.valid:
                popt = list(m.values)
                pred = dcb_expleft_gauss(centers, *[abs(v) if j not in (1, 8) else v
                                                    for j, v in enumerate(popt)])
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    return best_popt, None, True, best_chi2


_GOOD_ENOUGH_CHI2_NDOF = 2.0   # stop trying more starts once we hit this

def _best_fit(fn, centers, counts, starts, lo, hi):
    errs = np.maximum(np.sqrt(counts), 1.0)
    nparams = len(starts[0])
    ndof = max(len(centers) - nparams, 1)
    best_popt, best_pcov, best_chi2 = None, None, np.inf
    for p0 in starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(fn, centers, counts, p0=p0,
                                       bounds=(lo, hi), sigma=errs,
                                       maxfev=8_000, ftol=1e-7, xtol=1e-7)
            chi2 = float(np.sum(((counts - fn(centers, *popt)) / errs) ** 2))
            if chi2 < best_chi2:
                best_popt, best_pcov, best_chi2 = popt, pcov, chi2
            if best_chi2 / ndof < _GOOD_ENOUGH_CHI2_NDOF:
                break   # good enough — skip remaining starts
        except Exception:
            pass
    if best_popt is None:
        return None, None, False, np.inf
    ok = np.all(np.isfinite(np.sqrt(np.diag(best_pcov))))
    return best_popt, best_pcov, ok, best_chi2


def fit_dcb(centers, counts, mu0, sig0):
    N0 = float(counts.max())
    lo = [0, -np.inf, 1e-6, 0.3, 1.01, 0.3, 1.01]
    hi = [np.inf, np.inf, np.inf, 8., 200., 8., 200.]
    starts = [
        [N0, mu0, sig0,        1.5,  5.,  1.5,  5.],
        [N0, mu0, sig0,        1.0,  3.,  1.0,  3.],
        [N0, mu0, sig0,        2.0, 10.,  2.0, 10.],
        # asymmetric: gradual left, steep right
        [N0, mu0, sig0,        0.5,  2.,  2.0,  8.],
        # asymmetric: steep left, gradual right
        [N0, mu0, sig0,        2.0,  8.,  0.5,  2.],
        # narrow core
        [N0, mu0, sig0 * 0.5,  0.5,  3.,  0.5,  3.],
    ]
    return _best_fit(dcb, centers, counts, starts, lo, hi)


def fit_dcb2g(centers, counts, mu0, sig0, f_wide_max=0.95, mu_fix=None):
    """`f_wide_max`: cap on wide-Gauss fraction. Set ≤ 0.5 for detector-resolution
    priors so the dcb-core stays the bulk; default 0.95 lets the spike priors
    have a narrow ~1% "core". `mu_fix`: clamp μ_c (physics-symmetric data)."""
    N0 = float(counts.max())
    # params: N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, mu_w, sigma_w
    if mu_fix is not None:
        mu_lo = float(mu_fix) - 1e-9
        mu_hi = float(mu_fix) + 1e-9
    else:
        mu_lo, mu_hi = -np.inf, np.inf
    lo = [0, mu_lo, 1e-6, 0.3, 1.01, 0.3, 1.01, 0.01, -np.inf, 1e-4]
    hi = [np.inf, mu_hi, np.inf, 8., 200., 8., 200., float(f_wide_max), np.inf, np.inf]
    starts = [
        [N0, mu0, sig0,        1.2,  5., 0.8, 3., 0.15, mu0,           5 * sig0],
        [N0, mu0, sig0,        1.5,  8., 0.5, 2., 0.25, mu0 + 2*sig0,  8 * sig0],
        [N0, mu0, sig0,        2.0, 10., 0.5, 2., 0.10, mu0 + 3*sig0,  6 * sig0],
        # sharp-core: early power-law transitions
        [N0, mu0, sig0 * 0.5,  0.5,  3., 0.5, 2., 0.15, mu0,           5 * sig0],
        [N0, mu0, sig0 * 0.5,  0.4,  2., 0.4, 2., 0.20, mu0 + 2*sig0,  8 * sig0],
        # broad LEFT component (for left-heavy distributions like fromele responses)
        [N0, mu0, sig0 * 0.5,  0.5,  3., 1.5, 3., 0.30, mu0 - 3*sig0,  6 * sig0],
        [N0, mu0, sig0 * 0.4,  0.4,  2., 1.2, 3., 0.40, mu0 - 4*sig0,  8 * sig0],
        [N0, mu0, sig0 * 0.5,  0.3,  2., 1.0, 2., 0.35, mu0 - 5*sig0, 10 * sig0],
        [N0, mu0, sig0 * 0.5,  0.5,  5., 1.5, 3., 0.50, mu0 - 6*sig0, 12 * sig0],
        # very narrow spike + broad symmetric wings (ISR total-momentum distributions)
        [N0, mu0, sig0 * 0.04, 1.5,  5., 1.5,  5., 0.60, mu0, sig0 * 1.0],
        [N0, mu0, sig0 * 0.03, 1.0,  3., 1.0,  3., 0.70, mu0, sig0 * 0.8],
        [N0, mu0, sig0 * 0.05, 1.0,  5., 1.0,  5., 0.80, mu0, sig0 * 1.2],
        [N0, mu0, sig0 * 0.04, 1.5,  5., 1.5,  5., 0.55, mu0, sig0 * 0.9],
        # ultra-narrow spike: wings extend further than MAD-based sig0 suggests
        [N0, mu0, sig0 * 0.02, 2.0, 10., 2.0, 10., 0.80, mu0, sig0 * 2.0],
        [N0, mu0, sig0 * 0.01, 1.5,  8., 1.5,  8., 0.88, mu0, sig0 * 1.5],
        [N0, mu0, sig0 * 0.03, 1.5,  7., 1.5,  7., 0.75, mu0, sig0 * 2.5],
    ]
    if mu_fix is not None:
        # Seeds must satisfy the tight (mu_fix±eps) bound or curve_fit rejects.
        for s in starts:
            s[1] = float(mu_fix)
    return _best_fit(dcb_gauss, centers, counts, starts, lo, hi)


def fit_dcb_gaussbox(centers, counts, mu0, sig0):
    """DCB narrow core + Gaussian-smeared box wide component.
    Params: N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box
    """
    N0 = float(counts.max())
    x_span = 0.5 * (centers[-1] - centers[0])
    # f_wide upper bound 0.40: plateau/peak ≈ 0.15–0.25 physically; high f_wide pulls N down
    # to match plateau, causing the optimizer to undershoot the spike peak.
    lo = [0, -np.inf, 1e-6, 0.3, 1.01, 0.3, 1.01, 0.01,  1., 1e-4]
    hi = [np.inf, np.inf, np.inf, 8., 200., 8., 200., 0.40, np.inf, np.inf]
    starts = [
        [N0, mu0, sig0*0.005, 1.0, 50., 1.0, 50., 0.20, x_span*0.85, sig0*0.15],
        [N0, mu0, sig0*0.005, 1.5, 30., 1.5, 30., 0.18, x_span*0.80, sig0*0.15],
        [N0, mu0, sig0*0.010, 1.0,100., 1.0,100., 0.22, x_span*0.80, sig0*0.12],
        [N0, mu0, sig0*0.003, 1.2, 80., 1.2, 80., 0.16, x_span*0.75, sig0*0.10],
        [N0, mu0, sig0*0.007, 2.0, 20., 2.0, 20., 0.20, x_span*0.90, sig0*0.20],
        [N0, mu0, sig0*0.010, 1.5, 50., 1.5, 50., 0.25, x_span*0.85, sig0*0.18],
        [N0, mu0, sig0*0.003, 1.0,200., 1.0,200., 0.15, x_span*0.90, sig0*0.08],
        [N0, mu0, sig0*0.007, 1.0, 30., 1.0, 30., 0.30, x_span*0.80, sig0*0.20],
        [N0, mu0, sig0*0.015, 2.0, 15., 2.0, 15., 0.35, x_span*0.85, sig0*0.25],
        [N0, mu0, sig0*0.020, 1.5, 10., 1.5, 10., 0.40, x_span*0.80, sig0*0.30],
    ]
    return _best_fit(dcb_gaussbox, centers, counts, starts, lo, hi)


def fit_dcb_expleft2g(centers, counts, mu0, sig0, constrain_mu0=False):
    """DCB with exponential left tail + Gaussian wide component.
    Params: N, mu_c, sigma_c, aL, kL, aR, nR, f_wide, mu_w, sigma_w
    kL > 0: exponential decay rate of left tail (kL > aL → faster than Gaussian).
    constrain_mu0: restrict mu_c near 0 and aR small for hard-right-cutoff distributions.
    """
    N0 = float(counts.max())
    if constrain_mu0:
        # Key insight: sigma_c must be LARGE (1-5 GeV) because the ISR exponential slope
        # in x-space is kL/sigma_c ≈ 0.10/GeV (1/e length ~10 GeV seen in data).
        # With sigma_c=2 GeV: kL ≈ 0.2. With sigma_c=3 GeV: kL ≈ 0.3.
        # mu_c near 0 (physical boundary) + hard right cutoff (small aR, large nR).
        lo = [0, -0.5, 0.5, 0.05, 0.01, 0.02, 1.01, 0.01, -np.inf, 1e-4]
        hi = [np.inf, 0.1, 10., 3., 5., 0.4, 500., 0.95, np.inf, np.inf]
        starts = [
            # sigma_c=1-5 GeV: kL=0.1-0.5 gives physical ISR slope ~0.10/GeV in x-space
            [N0, 0.0,  2.0, 0.30, 0.20, 0.10, 200., 0.5, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  3.0, 0.30, 0.30, 0.10, 300., 0.4, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.20, 0.15, 0.05, 200., 0.6, mu0 - 2*sig0, 3*sig0],
            [N0, 0.0,  4.0, 0.50, 0.40, 0.15, 200., 0.3, mu0 - 3*sig0, 5*sig0],
            [N0, -0.1, 2.5, 0.30, 0.25, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  2.0, 0.50, 0.20, 0.08, 300., 0.6, mu0 - 3*sig0, 5*sig0],
            [N0, 0.0,  1.0, 0.20, 0.10, 0.05, 400., 0.7, mu0 - 3*sig0, 4*sig0],
            [N0, -0.2, 3.0, 0.40, 0.30, 0.10, 200., 0.4, mu0 - 2*sig0, 4*sig0],
            [N0, 0.0,  5.0, 0.50, 0.50, 0.20, 300., 0.3, mu0 - 2*sig0, 5*sig0],
            [N0, 0.0,  2.0, 0.10, 0.20, 0.10, 200., 0.5, mu0 - 3*sig0, 3*sig0],
            [N0, -0.3, 2.0, 0.30, 0.20, 0.10, 300., 0.5, mu0 - 3*sig0, 4*sig0],
            [N0, 0.0,  1.5, 0.30, 0.15, 0.05, 250., 0.55, mu0 - 2*sig0, 3*sig0],
        ]
    else:
        # aR lower bound 0.1 (was 0.3): allows near-hard right cutoff when nR is large.
        # nR upper bound 500 (was 200): (nR/aR - aR + t)^{-nR} → 0 within one bin width.
        lo = [0, -np.inf, 1e-6, 0.1, 0.1,  0.1, 1.01, 0.01, -np.inf, 1e-4]
        hi = [np.inf, np.inf, np.inf, 6., 50., 8., 500., 0.95, np.inf, np.inf]
        starts = [
            # aL moderate, kL >> aL (sharp exponential cutoff), moderate right tail
            [N0, mu0, sig0 * 0.5,  1.0,  8., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
            [N0, mu0, sig0 * 0.5,  0.8, 10., 0.7, 3., 0.20, mu0 + 4*sig0, 10*sig0],
            [N0, mu0, sig0 * 0.5,  0.5, 15., 0.5, 2., 0.25, mu0,           8*sig0],
            [N0, mu0, sig0,        2.0,  5., 0.8, 3., 0.15, mu0 + 3*sig0,  8*sig0],
            [N0, mu0, sig0 * 0.4,  1.0, 20., 0.5, 3., 0.30, mu0 + 2*sig0,  6*sig0],
            # left-biased: wide Gaussian to the left of core.
            [N0, mu0, sig0 * 0.05, 0.5, 10., 0.5, 3., 0.50, mu0 - 3*sig0,  2*sig0],
            [N0, mu0, sig0 * 0.03, 0.5, 15., 0.5, 3., 0.60, mu0 - 5*sig0,  4*sig0],
            [N0, mu0, sig0 * 0.05, 0.5, 10., 0.5, 5., 0.40, mu0 - 4*sig0,  3*sig0],
            [N0, mu0, sig0 * 0.10, 1.0,  5., 1.0, 5., 0.35, mu0 - 3*sig0,  2*sig0],
            [N0, mu0, sig0 * 0.02, 0.3, 20., 0.3, 3., 0.70, mu0 - 6*sig0,  6*sig0],
            # hard right cutoff: small aR + large nR
            [N0, mu0, sig0 * 0.3,  0.5,  5., 0.1, 100., 0.40, mu0 - 3*sig0,  2*sig0],
            [N0, mu0, sig0 * 0.2,  0.5,  8., 0.1, 200., 0.50, mu0 - 4*sig0,  3*sig0],
            [N0, mu0, sig0 * 0.1,  0.3, 10., 0.1, 300., 0.60, mu0 - 5*sig0,  4*sig0],
            [N0, mu0, sig0 * 0.5,  1.0,  5., 0.1, 100., 0.30, mu0 - 2*sig0,  2*sig0],
        ]
    return _best_fit(dcb_expleft_gauss, centers, counts, starts, lo, hi)


def fit_dcb_expright2g(centers, counts, mu0, sig0):
    """Scipy fit: power-law left DCB + exponential right tail + Gaussian wide component."""
    N0 = float(counts.max())
    # Anchor sigma_c to the narrow right-side width (peak → x_max), not to MAD which is
    # inflated by the heavy ISR left tail.
    x_right_span = max(float(centers[-1]) - mu0, 0.5)
    s = max(x_right_span * 0.40, 0.15)

    lo = [0, -np.inf, 1e-6, 0.1, 1.01, 0.1, 0.05, 0.0, -np.inf, 1e-4]
    hi = [np.inf, np.inf, np.inf, 5., 50., 10., 100., 0.80, np.inf, np.inf]
    starts = [
        [N0, mu0, s,       0.5, 2.0, 1.5,  5.0, 0.05, mu0 - 4*s, 6*s],
        [N0, mu0, s,       0.3, 1.5, 1.0,  3.0, 0.10, mu0 - 5*s, 8*s],
        [N0, mu0, s*0.7,   0.5, 2.0, 2.0,  8.0, 0.05, mu0 - 5*s, 8*s],
        [N0, mu0, s*1.5,   0.4, 1.5, 1.5,  5.0, 0.05, mu0 - 4*s, 7*s],
        [N0, mu0, s,       0.7, 3.0, 2.0, 10.0, 0.05, mu0 - 5*s,10*s],
        [N0, mu0, s*0.5,   0.3, 1.2, 1.0,  5.0, 0.10, mu0 - 4*s, 6*s],
        [N0, mu0, s,       1.0, 2.0, 3.0, 15.0, 0.05, mu0 - 6*s,12*s],
        [N0, mu0, s*2.0,   0.5, 2.0, 2.0,  5.0, 0.10, mu0 - 4*s, 7*s],
        [N0, mu0, s*0.7,   0.4, 1.5, 1.5, 10.0, 0.20, mu0 - 6*s,12*s],
        [N0, mu0, s,       0.5, 1.5, 1.0,  3.0, 0.30, mu0 - 5*s,10*s],
    ]
    return _best_fit(dcb_expright_gauss, centers, counts, starts, lo, hi)


def fit_dcb_expright2g_iminuit(centers, counts, mu0, sig0):
    """Poisson NLL iminuit fit: power-law left DCB + exponential right + Gaussian wide."""
    from iminuit import Minuit
    N0   = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)
    x_right_span = max(float(centers[-1]) - mu0, 0.5)
    s = max(x_right_span * 0.40, 0.15)

    def nll(N, mu_c, sc, aL, nL, aR, kR, fw, mu_w, sw):
        if sc <= 0 or nL < 1.0 or aL <= 0 or aR <= 0 or kR <= 0 or sw <= 0:
            return 1e15
        pred = dcb_expright_gauss(centers, abs(N), mu_c, abs(sc),
                                   abs(aL), abs(nL), abs(aR), abs(kR),
                                   abs(fw), mu_w, abs(sw))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    starts = [
        [N0, mu0, s,     0.5, 2.0, 1.5,  5.0, 0.05, mu0 - 4*s,  6*s],
        [N0, mu0, s,     0.3, 1.5, 1.0,  3.0, 0.10, mu0 - 5*s,  8*s],
        [N0, mu0, s*0.7, 0.5, 2.0, 2.0,  8.0, 0.05, mu0 - 5*s,  8*s],
        [N0, mu0, s,     0.7, 3.0, 2.0, 10.0, 0.05, mu0 - 5*s, 10*s],
        [N0, mu0, s*0.5, 0.3, 1.2, 1.0,  5.0, 0.10, mu0 - 4*s,  6*s],
        [N0, mu0, s,     1.0, 2.0, 3.0, 15.0, 0.05, mu0 - 6*s, 12*s],
    ]
    limits = [(1e-3,None),(None,None),(1e-6,None),(0.1,5.),(1.01,50.),
              (0.1,10.),(0.05,100.),(0.,0.80),(None,None),(1e-4,None)]
    names  = ['N','mu_c','sc','aL','nL','aR','kR','fw','mu_w','sw']

    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            m.migrad()
            if not m.valid:
                m.migrad()
            if m.valid:
                popt = list(m.values)
                pred = dcb_expright_gauss(centers, abs(popt[0]), popt[1], abs(popt[2]),
                                          abs(popt[3]), abs(popt[4]), abs(popt[5]), abs(popt[6]),
                                          abs(popt[7]), popt[8], abs(popt[9]))
                chi2 = float(np.sum(((counts - pred) / errs) ** 2))
                if chi2 < best_chi2:
                    best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    return best_popt, None, True, best_chi2


def fit_dcb_expright3g_iminuit(centers, counts, mu0, sig0, nL_min=1.01, mu_fix=None):
    """Poisson NLL iminuit fit: dcber core + shoulder Gaussian + outlier Gaussian.
    `nL_min`: lower bound on the power-law tail index (raise to constrain the
    deep-tail mass).
    `mu_fix`: if set, freezes μ_c at this value (used when physics requires a
    symmetric or bounded distribution — see BRANCH_CONFIG)."""
    from iminuit import Minuit
    N0   = float(counts.max())
    errs = np.maximum(np.sqrt(counts), 1.0)
    x_lo = float(centers[0]); x_hi = float(centers[-1])
    L    = max(mu0 - x_lo, 1e-3)

    x_right_span = max(float(centers[-1]) - mu0, 0.5)
    s = max(x_right_span * 0.40, 0.15)

    sc_n   = max(sig0, 1e-4)
    sc_t   = max(sig0 * 0.1, 1e-4)
    ss_n   = max(min(sig0 * 5.0, L * 0.10), 0.02)
    so_n   = max(L * 0.25, 0.10)

    def nll(N, mu_c, sc, aL, nL, aR, kR, f_s, mu_s, sigma_s, f_o, mu_o, sigma_o):
        if (sc <= 0 or nL < 1.0 or aL <= 0 or aR <= 0 or kR <= 0
            or sigma_s <= 0 or sigma_o <= 0):
            return 1e15
        if abs(f_s) + abs(f_o) > 0.95:
            return 1e15
        pred = dcb_expright_3gauss(centers, abs(N), mu_c, abs(sc),
                                    abs(aL), abs(nL), abs(aR), abs(kR),
                                    abs(f_s), mu_s, abs(sigma_s),
                                    abs(f_o), mu_o, abs(sigma_o))
        pred = np.maximum(pred, 1e-300)
        return 2.0 * float(np.sum(pred - counts * np.log(pred)))

    starts = [
        # N, mu_c,  sc,    aL,  nL,  aR,  kR,    f_s, mu_s,         sigma_s, f_o,  mu_o,         sigma_o
        [N0, mu0, s,       0.5, 2.0, 1.5,  5.0, 0.10, mu0 - 0.05,   0.025,  0.03, mu0 - 0.30,   0.15],
        [N0, mu0, s*0.7,   0.4, 1.5, 1.5,  5.0, 0.05, mu0 - 0.05,   0.030,  0.05, mu0 - 0.25,   0.10],
        [N0, mu0, s,       0.6, 2.5, 2.0,  8.0, 0.15, mu0 - 0.06,   0.020,  0.03, mu0 - 0.30,   0.20],
        [N0, mu0, s*0.5,   0.3, 1.2, 1.0,  3.0, 0.12, mu0 - 0.04,   0.025,  0.05, mu0 - 0.25,   0.12],
        [N0, mu0, s,       0.5, 2.0, 2.5, 10.0, 0.08, mu0 - 0.05,   0.030,  0.04, mu0 - 0.30,   0.18],
        [N0, mu0, s*1.5,   0.5, 2.0, 1.5,  5.0, 0.20, mu0 - 0.05,   0.025,  0.02, mu0 - 0.20,   0.15],
        [N0, mu0, sc_n,    0.5, 2.0, 1.5,  5.0, 0.10, mu0 - L*0.05, ss_n,    0.05, mu0 - L*0.30, so_n],
        [N0, mu0, sc_t,    0.4, 2.0, 1.5,  5.0, 0.15, mu0 - L*0.05, ss_n,    0.05, mu0 - L*0.30, so_n*0.6],
        [N0, mu0, sc_t,    0.6, 2.5, 2.0,  8.0, 0.05, mu0 - L*0.03, ss_n*0.5,0.10, mu0 - L*0.40, so_n*1.2],
        [N0, mu0, sig0 * 3,  0.5, 2.0, 1.5,  5.0, 0.03, mu0 - L*0.10, ss_n*2, 0.05, mu0 - L*0.40, so_n],
        [N0, mu0, sig0 * 5,  0.6, 2.0, 2.0,  6.0, 0.03, mu0 - L*0.20, ss_n*3, 0.05, mu0 - L*0.50, so_n*1.5],
    ]
    nL_lo = float(nL_min)
    if mu_fix is not None:
        mu_lo = mu_hi = float(mu_fix)
        # Override seeds: mu_c = mu_fix, mu_s/mu_o anchored to the left
        # of mu_fix (the data is one-sided below mu_fix for m_loss).
        for s_ in starts:
            s_[1] = float(mu_fix)
            s_[8]  = float(mu_fix) - L * 0.05
            s_[11] = float(mu_fix) - L * 0.30
    else:
        mu_lo = mu0 - max(0.05, sig0 * 5)
        mu_hi = mu0 + max(0.02, sig0 * 2)
    mu_s_hi = float(mu_fix) if mu_fix is not None else mu0
    mu_o_hi = (float(mu_fix) - max(0.05, sig0 * 2)) if mu_fix is not None else (mu0 - max(0.05, sig0 * 2))
    limits = [(1e-3, None),
              (mu_lo, mu_hi),
              (1e-6, x_hi - x_lo),
              (0.05, 5.), (nL_lo, 50.), (0.1, 10.), (0.05, 100.),
              (0.0, 0.50),
              (x_lo, mu_s_hi),
              (1e-4, max(0.10, L * 0.30)),
              (0.0, 0.30),
              (x_lo, mu_o_hi),
              (1e-3, max(0.5, L))]
    for s in starts:
        if s[4] < nL_lo:
            s[4] = nL_lo
    names  = ['N','mu_c','sc','aL','nL','aR','kR',
              'f_s','mu_s','sigma_s','f_o','mu_o','sigma_o']

    # Heavy-tailed dists often have a near-flat direction (small Hessian
    # eigenvalue ⇒ m.valid=False) but a clean chi² — accept on finite chi².
    best_popt, best_chi2 = None, np.inf
    for p0 in starts:
        try:
            m = Minuit(nll, *p0, name=names)
            for i, (lo_i, hi_i) in enumerate(limits):
                m.limits[i] = (lo_i, hi_i)
            if mu_fix is not None:
                m.fixed["mu_c"] = True
            m.migrad()
            if not m.valid:
                m.migrad()
            if not np.isfinite(m.fval):
                continue
            popt = list(m.values)
            pred = dcb_expright_3gauss(
                centers, abs(popt[0]), popt[1], abs(popt[2]),
                abs(popt[3]), abs(popt[4]), abs(popt[5]), abs(popt[6]),
                abs(popt[7]), popt[8], abs(popt[9]),
                abs(popt[10]), popt[11], abs(popt[12]))
            chi2 = float(np.sum(((counts - pred) / errs) ** 2))
            if not np.isfinite(chi2):
                continue
            if chi2 < best_chi2:
                best_chi2, best_popt = chi2, popt
        except Exception:
            pass

    if best_popt is None:
        return None, None, False, np.inf
    return best_popt, None, True, best_chi2


def _flatten_raw(raw):
    arr = np.asarray(raw)
    if arr.dtype == object:
        return np.concatenate([np.asarray(x, dtype=float).ravel() for x in arr])
    return arr.astype(float).ravel()


def _norm_curve(fn, xfine, x_lo, x_hi):
    integ = _quad(fn, x_lo, x_hi, limit=300)[0]
    return fn(xfine) / max(integ, 1e-300)


def _comparison_plot(ecm, _fitted, pairs, title_suffix, out_dir, label_a, label_b,
                     combined_label=None):
    os.makedirs(out_dir, exist_ok=True)
    for tag, (ba, bb) in pairs.items():
        if ba not in _fitted or bb not in _fitted:
            continue
        fn_a, edges_a, _ = _fitted[ba]
        fn_b, edges_b, _ = _fitted[bb]
        x_lo = min(edges_a[0], edges_b[0])
        x_hi = max(edges_a[-1], edges_b[-1])
        xfine = np.linspace(x_lo, x_hi, 600)
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.plot(xfine, _norm_curve(fn_a, xfine, x_lo, x_hi),
                color="tab:blue",   lw=2, label=label_a.format(ba))
        ax.plot(xfine, _norm_curve(fn_b, xfine, x_lo, x_hi),
                color="tab:orange", lw=2, label=label_b.format(bb))
        if combined_label and tag in _fitted:
            fn_c, _, _ = _fitted[tag]
            ax.plot(xfine, _norm_curve(fn_c, xfine, x_lo, x_hi),
                    color="crimson", lw=1.5, ls=":",
                    label=combined_label.format(tag))
        ax.set_xlabel(tag, fontsize=11)
        ax.set_ylabel("Normalised PDF", fontsize=11)
        ax.set_title(f"{tag}  [ecm{ecm}]  — {title_suffix}", fontsize=11)
        ax.legend(fontsize=9, frameon=False)
        ax.set_ylim(bottom=0)
        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{tag}_comparison.{fmt}", dpi=150)
        plt.close(fig)


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.bool_):    return bool(o)
        return super().default(o)


def _fit_bin_task(args):
    """Module-level task wrapper for ProcessPoolExecutor (must be picklable)."""
    bname, ibin, vals_subset, ecm = args
    return bname, ibin, _fit_one(bname, vals_subset, ecm)


def _eval_normalised_pdf(p, x):
    """Return unit-area PDF (∫f dx = 1) at points `x` for the fit-result dict
    `p` produced by _fit_one. Dispatches by p['model']; multiplies the un-
    normalised shape by p['norm'] (= 1/integral of yfn/N_f, computed in
    _fit_one). Used by the binned-plot path to overlay the fitted PDF on the
    per-bin histogram."""
    model = p["model"]
    if model == "gauss":
        # Already area-normalised; norm should be ≈1 anyway.
        shape = gauss(x, 1.0, p["mu"], p["sigma"])
    elif model == "dcb":
        shape = dcb(x, 1.0, p["mu"], p["sigma"], p["aL"], p["nL"], p["aR"], p["nR"])
    elif model == "dcb2g":
        shape = dcb_gauss(x, 1.0, p["mu"], p["sigma"], p["aL"], p["nL"],
                          p["aR"], p["nR"], p["f_wide"], p["mu_wide"], p["sigma_wide"])
    elif model == "dcber2g":
        shape = dcb_expright_gauss(x, 1.0, p["mu"], p["sigma"], p["aL"], p["nL"],
                                    p["aR"], p["kR"], p["f_wide"], p["mu_wide"], p["sigma_wide"])
    elif model == "dcber3g":
        shape = dcb_expright_3gauss(x, 1.0, p["mu"], p["sigma"], p["aL"], p["nL"],
                                     p["aR"], p["kR"], p["f_s"], p["mu_s"], p["sigma_s"],
                                     p["f_o"], p["mu_o"], p["sigma_o"])
    elif model == "expleft2g":
        shape = dcb_expleft_gauss(x, 1.0, p["mu"], p["sigma"], p["aL"], p["kL"],
                                   p["aR"], p["nR"], p["f_wide"], p["mu_wide"], p["sigma_wide"])
    elif model == "dcbgb":
        shape = dcb_gaussbox(x, 1.0, p["mu"], p["sigma"], p["aL"], p["nL"],
                              p["aR"], p["nR"], p["f_wide"], p["p_max"], p["sigma_box"])
    elif model == "asymgauss":
        shape = asymgauss(x, 1.0, p["mu"], p["sigma_L"], p["sigma_R"])
    elif model == "asymgauss2g":
        shape = asymgauss2g(x, 1.0, p["mu"], p["sigma_L"], p["sigma_R"],
                             p["f_wide"], p["mu_wide"], p["sigma_wide"])
    elif model == "asymgauss3g":
        shape = asymgauss3g(x, 1.0, p["mu"], p["sigma_L"], p["sigma_R"],
                             p["f_s"], p["mu_s"], p["sigma_s"],
                             p["f_o"], p["mu_o"], p["sigma_o"])
    elif model == "spike_dcb2g":
        sc_loc  = p.get("spike_center", 0.0)
        sr      = abs(p["sig_res"])
        f_d     = p["f_delta"]
        delta_pdf = f_d * np.exp(-0.5 * ((x - sc_loc) / sr) ** 2) / (sr * math.sqrt(2.0 * math.pi))
        body = dcb_gauss(x, 1.0, p["mu"], p["sigma"], p["aL"], p["nL"],
                         p["aR"], p["nR"], p["f_wide"], p["mu_wide"], p["sigma_wide"])
        return delta_pdf + (1.0 - f_d) * body * p["norm"]
    else:
        raise ValueError(f"_eval_normalised_pdf: unknown model {model!r}")
    return shape * p["norm"]


def _plot_binned(ecm, plot_dir, bname, bins_list, edges_phys, bin_var_label,
                 vals_per_bin, cfg):
    """One figure per binned branch, with N_BINS_PRIOR sub-panels — each shows
    the per-bin histogram with the fitted PDF overlaid (top) and the pull
    (bottom). Panels are tiled in 2 rows × ⌈N/2⌉ cols; trailing cells are hidden
    if N is odd. The bin-edge legend in each title locates the bin in physical
    units of the binning variable."""
    nbins            = cfg.get("nbins", NBINS_DEF)
    clip_lo, clip_hi = cfg.get("clip", CLIP_DEF)

    n_rows_panels = 2
    n_cols_panels = (N_BINS_PRIOR + n_rows_panels - 1) // n_rows_panels
    # 2 GridSpec rows per panel row: hist+fit on top, pull below.
    fig, axes = plt.subplots(
        2 * n_rows_panels, n_cols_panels,
        figsize=(3.4 * n_cols_panels, 3.8 * n_rows_panels),
        gridspec_kw={"height_ratios": [3, 1] * n_rows_panels,
                     "hspace": 0.30, "wspace": 0.25},
        layout="constrained",
    )
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    for ibin in range(N_BINS_PRIOR):
        row_grp = ibin // n_cols_panels
        col     = ibin % n_cols_panels
        ax      = axes[2 * row_grp,     col]
        ax_res  = axes[2 * row_grp + 1, col]
        p      = bins_list[ibin]
        vals   = vals_per_bin[ibin] if vals_per_bin else None
        if p is None or vals is None or len(vals) < 100:
            ax.set_title(f"bin {ibin}: skip", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            ax_res.set_xticks([]); ax_res.set_yticks([])
            continue

        vals = vals[np.isfinite(vals)]
        lo_p, hi_p = np.percentile(vals, [clip_lo, clip_hi])
        vc = vals[(vals >= lo_p) & (vals <= hi_p)]
        counts, edges_h = np.histogram(vc, bins=nbins)
        centers = 0.5 * (edges_h[:-1] + edges_h[1:])
        bw = float(np.diff(edges_h)[0])
        N_total = float(counts.sum())

        xfine_uniform = np.linspace(edges_h[0], edges_h[-1], 600)
        _w = max(20.0 * abs(p.get("sigma", abs(p.get("sigma_L", 1.0)))),
                 5.0 * bw)
        xfine_peak = np.linspace(max(edges_h[0], p["mu"] - _w),
                                 min(edges_h[-1], p["mu"] + _w), 1500)
        xfine = np.unique(np.concatenate([xfine_uniform, xfine_peak]))
        pdf_norm = _eval_normalised_pdf(p, xfine)
        yfine = pdf_norm * N_total * bw

        ax.bar(centers, counts, width=bw, color="steelblue", alpha=0.55)
        ax.plot(xfine, yfine, color="crimson", lw=1.5)
        ax.set_title(
            f"bin {ibin}: {edges_phys[ibin]:.2f}-{edges_phys[ibin+1]:.2f}\n"
            f"N={int(N_total)}  χ²/ndf={p.get('chi2_ndof', 0):.2f}",
            fontsize=8,
        )
        ax.set_ylim(bottom=0)
        if col == 0:
            ax.set_ylabel("Entries", fontsize=9)

        pred = _eval_normalised_pdf(p, centers) * N_total * bw
        with np.errstate(invalid="ignore"):
            pull = np.where(counts > 0,
                            (counts - pred) / np.sqrt(np.maximum(counts, 1)),
                            0)
        ax_res.bar(centers, pull, width=bw, color="steelblue", alpha=0.6)
        ax_res.axhline(0, color="crimson", lw=0.5)
        ax_res.set_ylim(-5, 5)
        ax_res.set_xlabel(bname, fontsize=8)
        if col == 0:
            ax_res.set_ylabel("Pull", fontsize=9)

    for empty in range(N_BINS_PRIOR, n_rows_panels * n_cols_panels):
        row_grp = empty // n_cols_panels
        col     = empty % n_cols_panels
        axes[2 * row_grp,     col].axis("off")
        axes[2 * row_grp + 1, col].axis("off")

    fig.suptitle(f"{bname}  [ecm{ecm}]   binned by {bin_var_label}", fontsize=10)
    for fmt in ("png", "pdf"):
        fig.savefig(f"{plot_dir}/{bname}_binned.{fmt}", dpi=150)
    plt.close(fig)


def _fit_one(bname, vals_in, ecm):
    """Fit a single histogram of `vals_in` for branch `bname` and return a
    params dict with model, parameters, chi2_ndof, fit_ok, norm. No plotting.
    Returns None if the slice is too sparse to fit. Mirrors the inline fit
    dispatch in process_ecm — used for per-bin (binned-prior) fits.
    """
    cfg              = BRANCH_CONFIG.get(bname, {})
    model            = cfg.get("model", "dcb")
    nbins            = cfg.get("nbins", NBINS_DEF)
    clip_lo, clip_hi = cfg.get("clip", CLIP_DEF)

    vals = np.asarray(vals_in)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 100:
        return None

    lo_p, hi_p = np.percentile(vals, [clip_lo, clip_hi])
    vals_c = vals[(vals >= lo_p) & (vals <= hi_p)]

    counts, edges = np.histogram(vals_c, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask    = counts > 0

    mu0  = float(centers[np.argmax(counts)])
    sig0 = float(median_abs_deviation(vals_c, scale="normal"))

    if model == "dcber2g":
        popt, pcov, fit_ok, chi2 = fit_dcb_expright2g(centers[mask], counts[mask], mu0, sig0)
        _ndof_est = max(int(mask.sum()) - 10, 1)
        if popt is None or chi2 / _ndof_est > 5.0:
            popt2, _, fit_ok2, chi2_2 = fit_dcb_expright2g_iminuit(centers[mask], counts[mask], mu0, sig0)
            if popt2 is not None and (popt is None or chi2_2 < chi2):
                popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
        nparams = 10
    elif model == "dcber3g":
        popt, pcov, fit_ok, chi2 = fit_dcb_expright3g_iminuit(
            centers[mask], counts[mask], mu0, sig0,
            nL_min=float(cfg.get("nL_min", 1.01)),
            mu_fix=cfg.get("mu_fix", None))
        nparams = 13
    elif model == "dcbgb":
        popt, pcov, fit_ok, chi2 = fit_dcb_gaussbox_iminuit(centers[mask], counts[mask], mu0, sig0)
        _ndof_est = max(int(mask.sum()) - 10, 1)
        if popt is None or chi2 / _ndof_est > 3.0:
            popt2, pcov2, fit_ok2, chi2_2 = fit_dcb_gaussbox(centers[mask], counts[mask], mu0, sig0)
            if popt2 is not None and (popt is None or chi2_2 < chi2):
                popt, pcov, fit_ok, chi2 = popt2, pcov2, fit_ok2, chi2_2
        nparams = 10
    elif model == "dcb2g":
        fwmax = float(cfg.get("f_wide_max", 0.95))
        mu_fix = cfg.get("mu_fix", None)
        popt, pcov, fit_ok, chi2 = fit_dcb2g(centers[mask], counts[mask], mu0, sig0,
                                             f_wide_max=fwmax, mu_fix=mu_fix)
        _ndof_est = max(int(mask.sum()) - 10, 1)
        # Lower threshold (2.5 vs 5.0) to give iminuit a shot at narrow-core
        # distributions where curve_fit settles in a shallow secondary minimum.
        if popt is None or chi2 / _ndof_est > 2.5:
            popt2, _, fit_ok2, chi2_2 = fit_dcb2g_iminuit(centers[mask], counts[mask], mu0, sig0,
                                                          f_wide_max=fwmax, mu_fix=mu_fix)
            if popt2 is not None and chi2_2 < chi2:
                popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
        nparams = 10
    elif model == "expleft2g":
        constrain_mu0 = cfg.get("fix_mu0", False)
        popt, pcov, fit_ok, chi2 = fit_dcb_expleft2g(centers[mask], counts[mask], mu0, sig0, constrain_mu0=constrain_mu0)
        _ndof_est = max(int(mask.sum()) - 10, 1)
        if popt is None or chi2 / _ndof_est > 3.0:
            popt2, _, fit_ok2, chi2_2 = fit_dcb_expleft2g_iminuit(centers[mask], counts[mask], mu0, sig0, constrain_mu0=constrain_mu0)
            if popt2 is not None and (popt is None or chi2_2 < chi2):
                popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
        nparams = 10
    elif model == "asymgauss":
        popt, pcov, fit_ok, chi2 = fit_asymgauss(centers[mask], counts[mask], mu0, sig0)
        nparams = 4
    elif model == "asymgauss2g":
        fwmax = float(cfg.get("f_wide_max", 0.5))
        popt, pcov, fit_ok, chi2 = fit_asymgauss2g(centers[mask], counts[mask], mu0, sig0,
                                                   f_wide_max=fwmax)
        nparams = 7
    elif model == "asymgauss3g":
        popt, pcov, fit_ok, chi2 = fit_asymgauss3g(centers[mask], counts[mask], mu0, sig0)
        nparams = 10
    else:
        popt, pcov, fit_ok, chi2 = fit_dcb(centers[mask], counts[mask], mu0, sig0)
        nparams = 7

    ndof      = max(int(mask.sum()) - nparams, 1)
    chi2_ndof = chi2 / ndof

    if popt is None:
        # Fallback (rarely hit for a per-bin slice; mirrors the integrated path).
        popt = [float(counts.max()), mu0, sig0, 5., 100., 5., 100.]
        if model == "dcber2g":
            _s = max((float(centers[-1]) - mu0) * 0.4, 0.15)
            popt = [float(counts.max()), mu0, _s, 0.5, 2.0, 1.5, 5.0, 0.05, mu0 - 4*_s, 6*_s]
        elif model == "dcber3g":
            _s = max((float(centers[-1]) - mu0) * 0.4, 0.15)
            popt = [float(counts.max()), mu0, _s, 0.5, 2.0, 1.5, 5.0,
                    0.10, mu0 - 0.05, 0.025, 0.03, mu0 - 0.30, 0.15]
        elif model == "dcbgb":
            x_span = 0.5 * (centers[-1] - centers[0])
            popt += [0.01, x_span * 0.8, sig0 * 0.2]
        elif model in ("dcb2g", "expleft2g"):
            popt += [0.01, mu0, sig0 * 5]
        fit_ok    = False
        chi2_ndof = float("inf")

    if model == "dcber2g":
        N_f, mu_c, sc, aL, nL, aR, kR, fw, muw, sw = popt
        N_f, sc, aL, nL, aR, kR, fw, sw = abs(N_f), abs(sc), abs(aL), abs(nL), abs(aR), abs(kR), abs(fw), abs(sw)
        res = dict(model="dcber2g", mu=float(mu_c), sigma=float(sc),
                   aL=float(aL), nL=float(nL), aR=float(aR), kR=float(kR),
                   f_wide=float(fw), mu_wide=float(muw), sigma_wide=float(sw))
        yfn = lambda x, _N=N_f, _mc=float(mu_c), _sc=float(sc), _aL=float(aL), _nL=float(nL), _aR=float(aR), _kR=float(kR), _fw=float(fw), _mw=float(muw), _sw=float(sw): \
            dcb_expright_gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _kR, _fw, _mw, _sw)
    elif model == "dcber3g":
        N_f, mu_c, sc, aL, nL, aR, kR, fs, mus, ss, fo, muo, so = popt
        N_f, sc, aL, nL, aR, kR = abs(N_f), abs(sc), abs(aL), abs(nL), abs(aR), abs(kR)
        fs, ss, fo, so = abs(fs), abs(ss), abs(fo), abs(so)
        res = dict(model="dcber3g", mu=float(mu_c), sigma=float(sc),
                   aL=float(aL), nL=float(nL), aR=float(aR), kR=float(kR),
                   f_s=float(fs), mu_s=float(mus), sigma_s=float(ss),
                   f_o=float(fo), mu_o=float(muo), sigma_o=float(so))
        yfn = lambda x, _N=N_f, _mc=float(mu_c), _sc=float(sc), _aL=float(aL), _nL=float(nL), _aR=float(aR), _kR=float(kR), _fs=float(fs), _ms=float(mus), _ss=float(ss), _fo=float(fo), _mo=float(muo), _so=float(so): \
            dcb_expright_3gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _kR, _fs, _ms, _ss, _fo, _mo, _so)
    elif model == "dcbgb":
        N_f, mu_c, sc, aL, nL, aR, nR, fw, p_max, sb = popt
        N_f, sc, aL, nL, aR, nR, fw, p_max, sb = abs(N_f), abs(sc), abs(aL), abs(nL), abs(aR), abs(nR), abs(fw), abs(p_max), abs(sb)
        res = dict(model="dcbgb", mu=float(mu_c), sigma=float(sc),
                   aL=float(aL), nL=float(nL), aR=float(aR), nR=float(nR),
                   f_wide=float(fw), p_max=float(p_max), sigma_box=float(sb))
        yfn = lambda x, _N=N_f, _mc=float(mu_c), _sc=float(sc), _aL=float(aL), _nL=float(nL), _aR=float(aR), _nR=float(nR), _fw=float(fw), _pm=float(p_max), _sb=float(sb): \
            dcb_gaussbox(x, _N, _mc, _sc, _aL, _nL, _aR, _nR, _fw, _pm, _sb)
    elif model == "dcb2g":
        N_f, mu_c, sc, aL, nL, aR, nR, fw, muw, sw = popt
        N_f, sc, aL, nL, aR, nR, fw, sw = abs(N_f), abs(sc), abs(aL), abs(nL), abs(aR), abs(nR), abs(fw), abs(sw)
        res = dict(model="dcb2g", mu=float(mu_c), sigma=float(sc),
                   aL=float(aL), nL=float(nL), aR=float(aR), nR=float(nR),
                   f_wide=float(fw), mu_wide=float(muw), sigma_wide=float(sw))
        yfn = lambda x, _N=N_f, _mc=float(mu_c), _sc=float(sc), _aL=float(aL), _nL=float(nL), _aR=float(aR), _nR=float(nR), _fw=float(fw), _mw=float(muw), _sw=float(sw): \
            dcb_gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _nR, _fw, _mw, _sw)
    elif model == "expleft2g":
        N_f, mu_c, sc, aL, kL, aR, nR, fw, muw, sw = popt
        N_f, sc, aL, kL, aR, nR, fw, sw = abs(N_f), abs(sc), abs(aL), abs(kL), abs(aR), abs(nR), abs(fw), abs(sw)
        res = dict(model="expleft2g", mu=float(mu_c), sigma=float(sc),
                   aL=float(aL), kL=float(kL), aR=float(aR), nR=float(nR),
                   f_wide=float(fw), mu_wide=float(muw), sigma_wide=float(sw))
        yfn = lambda x, _N=N_f, _mc=float(mu_c), _sc=float(sc), _aL=float(aL), _kL=float(kL), _aR=float(aR), _nR=float(nR), _fw=float(fw), _mw=float(muw), _sw=float(sw): \
            dcb_expleft_gauss(x, _N, _mc, _sc, _aL, _kL, _aR, _nR, _fw, _mw, _sw)
    elif model == "asymgauss":
        N_f, mu_f, sL, sR = popt
        N_f, sL, sR = abs(N_f), abs(sL), abs(sR)
        # `sigma` (avg of L,R) is what the kinfit's y-rescaling reads via _y2x;
        # the asymgauss evaluator itself uses sigma_L and sigma_R per side.
        res = dict(model="asymgauss", mu=float(mu_f),
                   sigma=float(0.5 * (sL + sR)),
                   sigma_L=float(sL), sigma_R=float(sR))
        yfn = lambda x, _N=N_f, _m=float(mu_f), _sL=float(sL), _sR=float(sR): \
            asymgauss(x, _N, _m, _sL, _sR)
    elif model == "asymgauss2g":
        N_f, mu_f, sL, sR, fw, muw, sw = popt
        N_f, sL, sR, fw, sw = abs(N_f), abs(sL), abs(sR), abs(fw), abs(sw)
        res = dict(model="asymgauss2g", mu=float(mu_f),
                   sigma=float(0.5 * (sL + sR)),
                   sigma_L=float(sL), sigma_R=float(sR),
                   f_wide=float(fw), mu_wide=float(muw), sigma_wide=float(sw))
        yfn = lambda x, _N=N_f, _m=float(mu_f), _sL=float(sL), _sR=float(sR), _fw=float(fw), _mw=float(muw), _sw=float(sw): \
            asymgauss2g(x, _N, _m, _sL, _sR, _fw, _mw, _sw)
    elif model == "asymgauss3g":
        N_f, mu_f, sL, sR, fs, mus, ss, fo, muo, so = popt
        N_f, sL, sR = abs(N_f), abs(sL), abs(sR)
        fs, ss, fo, so = abs(fs), abs(ss), abs(fo), abs(so)
        res = dict(model="asymgauss3g", mu=float(mu_f),
                   sigma=float(0.5 * (sL + sR)),
                   sigma_L=float(sL), sigma_R=float(sR),
                   f_s=float(fs), mu_s=float(mus), sigma_s=float(ss),
                   f_o=float(fo), mu_o=float(muo), sigma_o=float(so))
        yfn = lambda x, _N=N_f, _m=float(mu_f), _sL=float(sL), _sR=float(sR), _fs=float(fs), _ms=float(mus), _ss=float(ss), _fo=float(fo), _mo=float(muo), _so=float(so): \
            asymgauss3g(x, _N, _m, _sL, _sR, _fs, _ms, _ss, _fo, _mo, _so)
    else:
        N_f, mu_f, sf, aL, nL, aR, nR = popt
        N_f, sf, aL, nL, aR, nR = abs(N_f), abs(sf), abs(aL), abs(nL), abs(aR), abs(nR)
        res = dict(model="dcb", mu=float(mu_f), sigma=float(sf),
                   aL=float(aL), nL=float(nL), aR=float(aR), nR=float(nR))
        yfn = lambda x, _N=N_f, _m=float(mu_f), _s=float(sf), _aL=float(aL), _nL=float(nL), _aR=float(aR), _nR=float(nR): \
            dcb(x, _N, _m, _s, _aL, _nL, _aR, _nR)

    res["chi2_ndof"] = round(float(chi2_ndof), 3)
    res["fit_ok"]    = bool(fit_ok)

    # Norm: integrate the unnormalised yfn (which carries N_f) over ±10·max(αL,αR)·σ
    # plus any wide-Gaussian span, then divide by N_f to get the unit-area shape.
    _sg = res.get("sigma", 1.0)
    _half = 10.0 * max(abs(res.get("aL", 2.0)), abs(res.get("aR", 2.0)), 1.0) * _sg
    _mu_ref = res["mu"]
    if "mu_wide" in res and "sigma_wide" in res:
        _half = max(_half, abs(res["mu_wide"] - _mu_ref) + 10.0 * abs(res["sigma_wide"]))
    _integ, _ = _quad(lambda x: yfn(x) / float(abs(N_f)),
                      _mu_ref - _half, _mu_ref + _half,
                      limit=500, epsrel=1e-6)
    res["norm"] = float(1.0 / max(_integ, 1e-300))
    return res


def process_ecm(ecm):
    INFILE   = INFILE_TMPL.format(ecm=ecm)
    plot_dir = f"{PLOTS_DIR}/ecm{ecm}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\n{'='*60}\nECM {ecm} GeV  —  {INFILE}\n{'='*60}")

    # ── Read tree (kinfit branches + virtual jet1+jet2 combined) ─────────────
    with uproot.open(INFILE) as f:
        tree = f["events"]
        available = set(tree.keys())

        branches = [b for b in KINFIT_BRANCHES if b in available]
        missing  = [b for b in KINFIT_BRANCHES if b not in available]
        if missing:
            print(f"WARNING: {len(missing)} kinfit branch(es) not in tree: {missing}")
        fit_only = os.environ.get("WW_FIT_ONLY", "").strip()
        if fit_only:
            requested = [b.strip() for b in fit_only.split(",") if b.strip()]
            unknown   = [b for b in requested if b not in branches]
            if unknown:
                print(f"WARNING: WW_FIT_ONLY names not in available branches: {unknown}")
            branches  = [b for b in branches if b in requested]
            print(f"  [{ecm}]  WW_FIT_ONLY active: {branches}")
        # Read kinfit branches + the dR branches needed for the matching cut +
        # the reco kinematics branches used to drive per-bin priors.
        read_branches = list(branches)
        for b in DR_BRANCHES:
            if b in available and b not in read_branches:
                read_branches.append(b)
        for b in BIN_VAR_BRANCHES:
            if b in available and b not in read_branches:
                read_branches.append(b)
        data_all = tree.arrays(read_branches, library="np")

        # Apply jet/quark matching cut: drop events where either jet's matched
        # quark is farther than DR_MAX. The cut lives here (rather than in
        # step1) so the saved tree retains the full dR distribution for
        # diagnostics.
        # Require ALL matched jets within DR_MAX (2 jets for ℓνqq, 4 for 4q).
        if all(b in data_all for b in DR_BRANCHES):
            dr_arrs = [_flatten_raw(data_all[b]) for b in DR_BRANCHES]
            mask = np.ones(dr_arrs[0].size, dtype=bool)
            for d in dr_arrs:
                mask &= (d < DR_MAX)
            n_in, n_out = mask.size, int(mask.sum())
            print(f"  dR cut (<{DR_MAX}) on {len(DR_BRANCHES)} jets: "
                  f"{n_out}/{n_in} = {100.*n_out/n_in:.2f}%")
            for b in list(branches) + [v for v in BIN_VAR_BRANCHES if v in data_all]:
                data_all[b] = _flatten_raw(data_all[b])[mask]
        else:
            print(f"  WARNING: dR branches missing, skipping cut")

        # Build virtual pooled jet branches (concat over all source jets — 2 for
        # ℓνqq, 4 for 4q).
        for cname, srcs in COMBINED_BRANCHES.items():
            if all(s in data_all for s in srcs):
                data_all[cname] = np.concatenate(
                    [_flatten_raw(data_all[s]) for s in srcs])
                if cname not in branches:
                    branches.append(cname)

    print(f"Fitting {len(branches)} branches from {INFILE}\n")
    results = {}
    _fitted = {}   # bname → (yfn, edges, norm) for comparison plots

    for bname in branches:
        cfg              = BRANCH_CONFIG.get(bname, {})
        model            = cfg.get("model", "dcb")
        nbins            = cfg.get("nbins", NBINS_DEF)
        clip_lo, clip_hi = cfg.get("clip", CLIP_DEF)

        vals = _flatten_raw(data_all[bname])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 100:
            print(f"  [{ecm}]  SKIP {bname}: {len(vals)} entries"); continue

        if cfg.get("symmetrize", False):
            vals = np.concatenate([vals, -vals])

        # spike_dcb2g splits the spike (|x|<delta_threshold) from the body and
        # fits each separately; the kinfit composes them.
        f_delta      = None
        sig_res      = None
        spike_center = None
        if model == "spike_dcb2g":
            delta_threshold = cfg.get("delta_threshold", 0.001)
            sig_res         = cfg.get("sig_res", 0.001)
            spike_center    = float(cfg.get("spike_center", 0.0))
            f_delta = float((np.abs(vals - spike_center) < delta_threshold).mean())
            vals = vals[np.abs(vals - spike_center) >= delta_threshold]
            if len(vals) < 100:
                print(f"  [{ecm}]  SKIP {bname}: {len(vals)} non-delta entries"); continue

        lo_p, hi_p = np.percentile(vals, [clip_lo, clip_hi])
        vals_c = vals[(vals >= lo_p) & (vals <= hi_p)]

        counts, edges = np.histogram(vals_c, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask    = counts > 0

        mu0  = float(centers[np.argmax(counts)])
        sig0 = float(median_abs_deviation(vals_c, scale="normal"))

        if model == "dcber2g":
            popt, pcov, fit_ok, chi2 = fit_dcb_expright2g(centers[mask], counts[mask], mu0, sig0)
            _ndof_est = max(int(mask.sum()) - 10, 1)
            if popt is None or chi2 / _ndof_est > 5.0:
                popt2, _, fit_ok2, chi2_2 = fit_dcb_expright2g_iminuit(
                    centers[mask], counts[mask], mu0, sig0)
                if popt2 is not None and (popt is None or chi2_2 < chi2):
                    popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
            nparams = 10
        elif model == "dcber3g":
            popt, pcov, fit_ok, chi2 = fit_dcb_expright3g_iminuit(
                centers[mask], counts[mask], mu0, sig0,
                nL_min=float(cfg.get("nL_min", 1.01)),
                mu_fix=cfg.get("mu_fix", None))
            nparams = 13
        elif model == "dcbgb":
            # Poisson NLL primary: correctly weights spike peak vs flat plateau.
            # chi² over-weights plateau bins (small sigma), forcing N down and undershooting peak.
            popt, pcov, fit_ok, chi2 = fit_dcb_gaussbox_iminuit(
                centers[mask], counts[mask], mu0, sig0)
            _ndof_est = max(int(mask.sum()) - 10, 1)
            if popt is None or chi2 / _ndof_est > 3.0:
                popt2, pcov2, fit_ok2, chi2_2 = fit_dcb_gaussbox(
                    centers[mask], counts[mask], mu0, sig0)
                if popt2 is not None and (popt is None or chi2_2 < chi2):
                    popt, pcov, fit_ok, chi2 = popt2, pcov2, fit_ok2, chi2_2
            nparams = 10
        elif model in ("dcb2g", "spike_dcb2g"):
            fwmax = float(cfg.get("f_wide_max", 0.95))
            mu_fix = cfg.get("mu_fix", None)
            popt, pcov, fit_ok, chi2 = fit_dcb2g(centers[mask], counts[mask], mu0, sig0,
                                                 f_wide_max=fwmax, mu_fix=mu_fix)
            # Run iminuit fallback once χ²/ndf > 2.5 — narrow-core distributions
            # (lep angular resolutions in extreme p bins) often need it.
            _ndof_est = max(int(mask.sum()) - 10, 1)
            if popt is None or chi2 / _ndof_est > 2.5:
                popt2, _, fit_ok2, chi2_2 = fit_dcb2g_iminuit(
                    centers[mask], counts[mask], mu0, sig0, f_wide_max=fwmax, mu_fix=mu_fix)
                if popt2 is not None and chi2_2 < chi2:
                    popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
            nparams = 10
        elif model == "expleft2g":
            constrain_mu0 = cfg.get("fix_mu0", False)
            popt, pcov, fit_ok, chi2 = fit_dcb_expleft2g(
                centers[mask], counts[mask], mu0, sig0, constrain_mu0=constrain_mu0)
            _ndof_est = max(int(mask.sum()) - 10, 1)
            if popt is None or chi2 / _ndof_est > 3.0:
                popt2, _, fit_ok2, chi2_2 = fit_dcb_expleft2g_iminuit(
                    centers[mask], counts[mask], mu0, sig0, constrain_mu0=constrain_mu0)
                if popt2 is not None and (popt is None or chi2_2 < chi2):
                    popt, pcov, fit_ok, chi2 = popt2, None, fit_ok2, chi2_2
            nparams = 10
        elif model == "gauss":
            popt, pcov, fit_ok, chi2 = fit_gauss(centers[mask], counts[mask], mu0, sig0)
            nparams = 3
        elif model == "asymgauss":
            popt, pcov, fit_ok, chi2 = fit_asymgauss(centers[mask], counts[mask], mu0, sig0)
            nparams = 4
        elif model == "asymgauss2g":
            fwmax = float(cfg.get("f_wide_max", 0.5))
            popt, pcov, fit_ok, chi2 = fit_asymgauss2g(centers[mask], counts[mask], mu0, sig0,
                                                       f_wide_max=fwmax)
            nparams = 7
        elif model == "asymgauss3g":
            popt, pcov, fit_ok, chi2 = fit_asymgauss3g(centers[mask], counts[mask], mu0, sig0)
            nparams = 10
        else:
            popt, pcov, fit_ok, chi2 = fit_dcb(centers[mask], counts[mask], mu0, sig0)
            nparams = 7

        ndof      = max(int(mask.sum()) - nparams, 1)
        chi2_ndof = chi2 / ndof

        if popt is None:
            print(f"  [{ecm}]  FAIL {bname}: all starts failed, using Gaussian-like fallback")
            if model == "gauss":
                popt = [float(counts.sum()) * float(centers[1] - centers[0]), mu0, sig0]
            elif model == "asymgauss":
                popt = [float(counts.sum()) * float(centers[1] - centers[0]), mu0, sig0, sig0]
            elif model == "asymgauss2g":
                popt = [float(counts.sum()) * float(centers[1] - centers[0]),
                        mu0, sig0, sig0, 0.05, mu0, sig0 * 5]
            elif model == "asymgauss3g":
                popt = [float(counts.sum()) * float(centers[1] - centers[0]),
                        mu0, sig0, sig0,
                        0.20, mu0, sig0 * 5,
                        0.03, mu0, sig0 * 25]
            else:
                popt = [float(counts.max()), mu0, sig0, 5., 100., 5., 100.]
                if model == "dcber2g":
                    _s = max((float(centers[-1]) - mu0) * 0.4, 0.15)
                    popt = [float(counts.max()), mu0, _s, 0.5, 2.0, 1.5, 5.0, 0.05,
                            mu0 - 4*_s, 6*_s]
                elif model == "dcber3g":
                    _s = max((float(centers[-1]) - mu0) * 0.4, 0.15)
                    popt = [float(counts.max()), mu0, _s, 0.5, 2.0, 1.5, 5.0,
                            0.10, mu0 - 0.05, 0.025,
                            0.03, mu0 - 0.30, 0.15]
                elif model == "dcbgb":
                    x_span = 0.5 * (centers[-1] - centers[0])
                    popt += [0.01, x_span * 0.8, sig0 * 0.2]
                elif model in ("dcb2g", "expleft2g"):
                    popt += [0.01, mu0, sig0 * 5]
            fit_ok    = False
            chi2_ndof = np.inf

        # ── unpack and sanitise ──────────────────────────────────────────────
        if model == "dcber2g":
            N_f, mu_c, sc, aL, nL, aR, kR, fw, muw, sw = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); kR = abs(float(kR))
            fw   = abs(float(fw)); muw = float(muw); sw = abs(float(sw))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcber2g",
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, kR=kR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _kR=kR,
                    _fw=fw, _mw=muw, _sw=sw):
                return dcb_expright_gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _kR, _fw, _mw, _sw)
            _xscan = np.linspace(centers[0], centers[-1], 2000)
            _x_peak = float(_xscan[np.argmax(yfn(_xscan))])
            lbl = (rf"DCBExpRight+G: $x_{{peak}}$={_x_peak:+.3g}, $\mu_c$={mu_c:+.3g}"
                   "\n"
                   rf"$\sigma_c$={sc:.3g}, $\alpha_L$={aL:.2f}, $n_L$={nL:.2f}, $k_R$={kR:.3g}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "dcber3g":
            N_f, mu_c, sc, aL, nL, aR, kR, fs, mus, ss, fo, muo, so = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); kR = abs(float(kR))
            fs   = abs(float(fs)); mus = float(mus); ss = abs(float(ss))
            fo   = abs(float(fo)); muo = float(muo); so = abs(float(so))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcber3g",
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, kR=kR,
                f_s=fs, mu_s=mus, sigma_s=ss,
                f_o=fo, mu_o=muo, sigma_o=so,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _kR=kR,
                    _fs=fs, _ms=mus, _ss=ss, _fo=fo, _mo=muo, _so=so):
                return dcb_expright_3gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _kR,
                                            _fs, _ms, _ss, _fo, _mo, _so)
            lbl = (rf"DCBExpRight+G+G: $\mu_c$={mu_c:+.3g}, $\sigma_c$={sc:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.2f}, $k_R$={kR:.3g}"
                   "\n"
                   rf"$f_s$={fs:.3f}, $\mu_s$={mus:.3g}, $\sigma_s$={ss:.3g}"
                   "\n"
                   rf"$f_o$={fo:.3f}, $\mu_o$={muo:.3g}, $\sigma_o$={so:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "dcbgb":
            N_f, mu_c, sc, aL, nL, aR, nR, fw, p_max, sb = popt
            mu_c = float(mu_c);  sc   = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            fw   = abs(float(fw)); p_max = abs(float(p_max)); sb = abs(float(sb))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcbgb",
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, nR=nR,
                f_wide=fw, p_max=p_max, sigma_box=sb,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _nR=nR,
                    _fw=fw, _pm=p_max, _sb=sb):
                return dcb_gaussbox(x, _N, _mc, _sc, _aL, _nL, _aR, _nR, _fw, _pm, _sb)
            lbl = (rf"DCB+Box: $\mu_c$={mu_c:+.3g}, $\sigma_c$={sc:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.1f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $p_{{max}}$={p_max:.2g}, $\sigma_{{box}}$={sb:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "dcb2g":
            N_f, mu_c, sc, aL, nL, aR, nR, fw, muw, sw = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            fw   = abs(float(fw)); muw = float(muw); sw = abs(float(sw))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcb2g",
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, nR=nR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _nR=nR,
                    _fw=fw, _mw=muw, _sw=sw):
                return dcb_gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _nR, _fw, _mw, _sw)
            lbl = (rf"DCB+G: $\mu_c$={mu_c:+.3g}, $\sigma_c$={sc:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.1f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "expleft2g":
            N_f, mu_c, sc, aL, kL, aR, nR, fw, muw, sw = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); kL = abs(float(kL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            fw   = abs(float(fw)); muw = float(muw); sw = abs(float(sw))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="expleft2g",
                mu=mu_c, sigma=sc, aL=aL, kL=kL, aR=aR, nR=nR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _kL=kL, _aR=aR, _nR=nR,
                    _fw=fw, _mw=muw, _sw=sw):
                return dcb_expleft_gauss(x, _N, _mc, _sc, _aL, _kL, _aR, _nR, _fw, _mw, _sw)
            # Report actual model maximum (physical peak), not the internal mu_c parameter.
            _xscan = np.linspace(centers[0], centers[-1], 2000)
            _x_peak = float(_xscan[np.argmax(yfn(_xscan))])
            lbl = (rf"ExpLeft+G: $x_{{peak}}$={_x_peak:+.3g}, $\mu_c$={mu_c:+.3g}"
                   "\n"
                   rf"$\sigma_c$={sc:.3g}, $k_L$={kL:.3g}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "spike_dcb2g":
            N_f, mu_c, sc, aL, nL, aR, nR, fw, muw, sw = popt
            mu_c = float(mu_c);  sc  = abs(float(sc))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            fw   = abs(float(fw)); muw = float(muw); sw = abs(float(sw))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="spike_dcb2g",
                f_delta=float(f_delta), sig_res=float(sig_res),
                mu=mu_c, sigma=sc, aL=aL, nL=nL, aR=aR, nR=nR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                # Symmetrized priors carry the kinfit barrier σ_b = 0.1·σ_res
                # for the plotter PDF; must match KF_M_LOSS_BARRIER_SIGMA_FRAC
                # in WWKinReco.h.
                barrier_sigma=(float(sig_res) * 0.1
                               if cfg.get("symmetrize", False) else None),
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _mc=mu_c, _sc=sc, _aL=aL, _nL=nL, _aR=aR, _nR=nR,
                    _fw=fw, _mw=muw, _sw=sw):
                return dcb_gauss(x, _N, _mc, _sc, _aL, _nL, _aR, _nR, _fw, _mw, _sw)
            lbl = (rf"$f_\Delta$={f_delta*100:.1f}% (no-ISR removed)+ DCB+G:"
                   "\n"
                   rf"$\mu_c$={mu_c:+.3g}, $\sigma_c$={sc:.3g}, "
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.1f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "gauss":
            N_f, mu_g, sg = popt
            N_f = abs(float(N_f)); mu_g = float(mu_g); sg = abs(float(sg))
            # mu_c / sc are referenced by the post-fit plotting code (xfine peak grid).
            mu_c, sc = mu_g, sg
            results[bname] = dict(
                model="gauss",
                mu=mu_g, sigma=sg,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _m=mu_g, _s=sg):
                return gauss(x, _N, _m, _s)
            lbl = (rf"Gauss: $\mu$={mu_g*1000:+.2f} MeV, $\sigma$={sg*1000:.2f} MeV"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "asymgauss":
            N_f, mu_g, sL, sR = popt
            N_f  = abs(float(N_f)); mu_g = float(mu_g)
            sL   = abs(float(sL));  sR   = abs(float(sR))
            sc   = 0.5 * (sL + sR)  # avg σ for kinfit y-rescaling via _y2x
            mu_c = mu_g
            results[bname] = dict(
                model="asymgauss",
                mu=mu_g, sigma=sc, sigma_L=sL, sigma_R=sR,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _m=mu_g, _sL=sL, _sR=sR):
                return asymgauss(x, _N, _m, _sL, _sR)
            lbl = (rf"AsymGauss: $\mu$={mu_g:+.3g}, "
                   rf"$\sigma_L$={sL:.3g}, $\sigma_R$={sR:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "asymgauss2g":
            N_f, mu_g, sL, sR, fw, muw, sw = popt
            N_f  = abs(float(N_f)); mu_g = float(mu_g)
            sL   = abs(float(sL));  sR   = abs(float(sR))
            fw   = abs(float(fw));  muw  = float(muw); sw = abs(float(sw))
            sc   = 0.5 * (sL + sR)
            mu_c = mu_g
            results[bname] = dict(
                model="asymgauss2g",
                mu=mu_g, sigma=sc, sigma_L=sL, sigma_R=sR,
                f_wide=fw, mu_wide=muw, sigma_wide=sw,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _m=mu_g, _sL=sL, _sR=sR, _fw=fw, _mw=muw, _sw=sw):
                return asymgauss2g(x, _N, _m, _sL, _sR, _fw, _mw, _sw)
            lbl = (rf"AsymGauss+G: $\mu$={mu_g:+.3g}, "
                   rf"$\sigma_L$={sL:.3g}, $\sigma_R$={sR:.3g}"
                   "\n"
                   rf"$f_w$={fw:.3f}, $\mu_w$={muw:.3g}, $\sigma_w$={sw:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        elif model == "asymgauss3g":
            N_f, mu_g, sL, sR, fs, mus, ss, fo, muo, so = popt
            N_f = abs(float(N_f)); mu_g = float(mu_g)
            sL  = abs(float(sL));  sR   = abs(float(sR))
            fs  = abs(float(fs));  ss   = abs(float(ss))
            fo  = abs(float(fo));  so   = abs(float(so))
            mus = float(mus);      muo  = float(muo)
            sc  = 0.5 * (sL + sR)
            mu_c = mu_g
            results[bname] = dict(
                model="asymgauss3g",
                mu=mu_g, sigma=sc, sigma_L=sL, sigma_R=sR,
                f_s=fs, mu_s=mus, sigma_s=ss,
                f_o=fo, mu_o=muo, sigma_o=so,
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _m=mu_g, _sL=sL, _sR=sR, _fs=fs, _ms=mus, _ss=ss, _fo=fo, _mo=muo, _so=so):
                return asymgauss3g(x, _N, _m, _sL, _sR, _fs, _ms, _ss, _fo, _mo, _so)
            lbl = (rf"AsymGauss+2G: $\mu$={mu_g:+.3g}, "
                   rf"$\sigma_L$={sL:.3g}, $\sigma_R$={sR:.3g}"
                   "\n"
                   rf"$f_s$={fs:.3f}, $\sigma_s$={ss:.3g}, "
                   rf"$f_o$={fo:.3f}, $\sigma_o$={so:.3g}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")
        else:
            N_f, mu_f, sf, aL, nL, aR, nR = popt
            mu_f = float(mu_f); sf  = abs(float(sf))
            aL   = abs(float(aL)); nL = abs(float(nL))
            aR   = abs(float(aR)); nR = abs(float(nR))
            N_f  = abs(float(N_f))
            results[bname] = dict(
                model="dcb",
                mu=float(mu_f), sigma=float(sf),
                aL=float(aL), nL=float(nL), aR=float(aR), nR=float(nR),
                chi2_ndof=round(float(chi2_ndof), 3), fit_ok=bool(fit_ok),
            )
            def yfn(x, _N=N_f, _m=mu_f, _s=sf, _aL=aL, _nL=nL, _aR=aR, _nR=nR):
                return dcb(x, _N, _m, _s, _aL, _nL, _aR, _nR)
            lbl = (rf"DCB: $\mu$={mu_f:+.3g}, $\sigma$={sf:.3g}"
                   "\n"
                   rf"$\alpha_L$={aL:.2f}, $n_L$={nL:.1f}, $\alpha_R$={aR:.2f}, $n_R$={nR:.1f}"
                   rf"   $\chi^2$/ndf={chi2_ndof:.2f}")

        # Normalise shape to unit integral; yfn includes N_f so divide it out.
        # Build integration range from fitted PDF shape (±10σ of core and wide
        # component) rather than histogram span, to capture broad wide-Gaussian tails.
        _p = results[bname]
        _sg = _p.get("sigma", 1.0)
        _half_core = 10.0 * max(abs(_p.get("aL", 2.0)), abs(_p.get("aR", 2.0)), 1.0) * _sg
        if "mu" in _p:
            _mu_ref = _p["mu"]
            _half = _half_core
            if "mu_wide" in _p and "sigma_wide" in _p:
                _half = max(_half, abs(_p["mu_wide"] - _mu_ref) + 10.0 * abs(_p["sigma_wide"]))
            _integ_lo, _integ_hi = _mu_ref - _half, _mu_ref + _half
        elif "x_cut" in _p:
            _xc = _p["x_cut"]; _sw = _p.get("sigma_wide", _sg)
            _integ_lo = _xc - 10.0 * max(_sw, _sg); _integ_hi = _xc + 0.5
        else:
            _integ_lo, _integ_hi = -_half_core, _half_core
        _integ, _ = _quad(lambda x: yfn(x) / N_f,
                          _integ_lo, _integ_hi,
                          limit=500, epsrel=1e-6)
        results[bname]["norm"] = float(1.0 / max(_integ, 1e-300))
        _fitted[bname] = (yfn, edges, results[bname]["norm"])

        print(f"  [{ecm}]  {bname:40s}  χ²/ndf={chi2_ndof:.2f}  "
              f"{'OK' if fit_ok else 'WARN'}  [{model}]")

        # ── plot ─────────────────────────────────────────────────────────────
        # Build xfine with a uniform global grid plus a dense sub-grid around
        # mu_c so very narrow cores (e.g. σ_c sub-mGeV for px/gen_WW_py) are
        # actually resolved by the displayed curve, not skipped between samples.
        xfine_uniform = np.linspace(edges[0], edges[-1], 600)
        _w = max(20.0 * sc, 5.0 * (edges[1] - edges[0]))
        xfine_peak = np.linspace(max(edges[0], mu_c - _w),
                                 min(edges[-1], mu_c + _w), 1500)
        xfine = np.unique(np.concatenate([xfine_uniform, xfine_peak]))
        yfine = yfn(xfine)
        bw    = np.diff(edges)[0]

        fig, (ax, ax_res) = plt.subplots(
            2, 1, figsize=(7, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
            layout="constrained",
        )
        if cfg.get("symmetrize", False):
            is_orig = centers <= 0
            ax.bar(centers[is_orig],  counts[is_orig],  width=bw, color="steelblue",
                   alpha=0.55, label="data (m_loss ≤ 0, physical)")
            ax.bar(centers[~is_orig], counts[~is_orig], width=bw, color="tab:orange",
                   alpha=0.55, label="mirror (−m_loss; symmetrized for fit)")
        else:
            ax.bar(centers, counts, width=bw, color="steelblue", alpha=0.55, label="data")
        ax.plot(xfine, yfine, color="crimson", lw=2, label=lbl)

        if model == "dcber2g":
            y_core = dcb_expright_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, kR, 0., muw, sw)
            y_wide = dcb_expright_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, kR, 1., muw, sw)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="DCBExpRight core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="Gaussian component")
        elif model == "dcbgb":
            y_core = dcb_gaussbox(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 0., p_max, sb)
            y_wide = dcb_gaussbox(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 1., p_max, sb)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="DCB core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="GaussBox component")
        elif model in ("dcb2g", "spike_dcb2g"):
            y_core = dcb_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 0., muw, sw)
            y_wide = dcb_gauss(xfine, N_f, mu_c, sc, aL, nL, aR, nR, 1., muw, sw)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="DCB core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="Gaussian component")
        elif model == "expleft2g":
            y_core = dcb_expleft_gauss(xfine, N_f, mu_c, sc, aL, kL, aR, nR, 0., muw, sw)
            y_wide = dcb_expleft_gauss(xfine, N_f, mu_c, sc, aL, kL, aR, nR, 1., muw, sw)
            ax.plot(xfine, y_core * (1 - fw), color="tab:orange", lw=1.2, ls="--",
                    label="ExpLeft core")
            ax.plot(xfine, y_wide * fw,       color="tab:green",  lw=1.2, ls=":",
                    label="Gaussian component")

        ax.set_ylabel("Entries", fontsize=11)
        ax.set_title(f"{bname}  [ecm{ecm}]", fontsize=11)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
        if cfg.get("log_y", False):
            in_clip = counts > 0
            if in_clip.any():
                ax.set_yscale("log")
                ax.set_ylim(max(0.5, float(counts[in_clip].min()) * 0.5),
                            float(counts.max()) * 1.5)
        else:
            ax.set_ylim(bottom=0)

        pull = np.where(counts > 0, (counts - yfn(centers)) / np.sqrt(np.maximum(counts, 1)), 0)
        ax_res.bar(centers, pull, width=bw, color="steelblue", alpha=0.6)
        ax_res.axhline(0, color="crimson", lw=1)
        ax_res.set_ylabel("Pull", fontsize=10)
        ax_res.set_xlabel(bname, fontsize=11)
        ax_res.set_ylim(-5, 5)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{plot_dir}/{bname}.{fmt}", dpi=150)

        plt.close(fig)

        # Zoom view: re-binned in the zoom window so narrow peaks are resolved.
        zoom_xlim = cfg.get("zoom_xlim")
        if zoom_xlim is not None:
            zlo, zhi = zoom_xlim
            z_vals = vals_c[(vals_c >= zlo) & (vals_c <= zhi)]
            if len(z_vals) > 50:
                z_counts, z_edges = np.histogram(z_vals, bins=120)
                z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
                z_bw      = float(np.diff(z_edges)[0])
                # The fitted PDF was scaled to the inclusive bin width `bw`;
                # rescale to the zoom bin width so curve and histogram match.
                scale = z_bw / bw
                z_xfine = np.linspace(zlo, zhi, 1500)
                z_yfine = yfn(z_xfine) * scale

                fig_z, (axz, axz_res) = plt.subplots(
                    2, 1, figsize=(7, 6), sharex=True,
                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
                    layout="constrained",
                )
                if cfg.get("symmetrize", False):
                    is_orig_z = z_centers <= 0
                    axz.bar(z_centers[is_orig_z],  z_counts[is_orig_z],  width=z_bw,
                            color="steelblue", alpha=0.55, label="data (m_loss ≤ 0)")
                    axz.bar(z_centers[~is_orig_z], z_counts[~is_orig_z], width=z_bw,
                            color="tab:orange", alpha=0.55, label="mirror (−m_loss)")
                else:
                    axz.bar(z_centers, z_counts, width=z_bw,
                            color="steelblue", alpha=0.55, label="data")
                axz.plot(z_xfine, z_yfine, color="crimson", lw=2, label=lbl)
                axz.set_ylabel("Entries", fontsize=11)
                axz.set_title(f"{bname}  [ecm{ecm}]  (zoom)", fontsize=11)
                axz.legend(fontsize=7.5, frameon=False, loc="upper left")
                axz.set_xlim(zlo, zhi)
                if cfg.get("log_y", False):
                    z_pos = z_counts > 0
                    if z_pos.any():
                        axz.set_yscale("log")
                        axz.set_ylim(max(0.5, float(z_counts[z_pos].min()) * 0.5),
                                     float(z_counts.max()) * 1.5)
                else:
                    axz.set_ylim(bottom=0)

                z_pred = yfn(z_centers) * scale
                z_pull = np.where(z_counts > 0,
                                  (z_counts - z_pred) / np.sqrt(np.maximum(z_counts, 1)),
                                  0)
                axz_res.bar(z_centers, z_pull, width=z_bw, color="steelblue", alpha=0.6)
                axz_res.axhline(0, color="crimson", lw=1)
                axz_res.set_ylabel("Pull", fontsize=10)
                axz_res.set_xlabel(bname, fontsize=11)
                axz_res.set_ylim(-5, 5)
                axz_res.set_xlim(zlo, zhi)
                for fmt in ("png", "pdf"):
                    fig_z.savefig(f"{plot_dir}/{bname}_zoom.{fmt}", dpi=150)
                plt.close(fig_z)

    # ── Jet1 vs jet2 comparison plots ────────────────────────────────────────
    # ℓνqq only: the comparison overlays the two per-jet fits vs the pooled fit
    # and assumes 2-source COMBINED_BRANCHES. 4q pools 4 jets → skip.
    if not IS_4Q:
        comp_dir = f"{plot_dir}/jet_comparisons"
        _comparison_plot(
            ecm, _fitted, COMBINED_BRANCHES, "jet1 vs jet2", comp_dir,
            label_a="{0}", label_b="{0}",
            combined_label="{0} (combined fit)",
        )
        print(f"  Jet comparison plots → {comp_dir}/")

    # ── Binned-prior pass: fit DCB per equal-occupancy bin of the binning var ─
    # Equal-occupancy edges per binning variable, computed once on the dR-cut
    # sample. Quantile arrays are built per binning-var key (for pooled jet
    # branches the binning var is the concat'd jet1+jet2 reco kinematic). Bin
    # fits are dispatched across an inner ProcessPool — they're independent
    # and dwarf the per-ECM serial cost.
    _skip_binned = os.environ.get("WW_SKIP_BINNED", "").strip() in ("1", "true", "yes")
    if _skip_binned:
        print(f"\n  [{ecm}]  WW_SKIP_BINNED=1 — binned-prior pass skipped")
    else:
        print(f"\n  [{ecm}]  Binned-prior pass: {N_BINS_PRIOR} bins per branch")
    if not _skip_binned:
        binvar_cache = {}
        def _bin_var_array(bcfg):
            key = bcfg
            if key in binvar_cache:
                return binvar_cache[key]
            arrs = []
            for v in bcfg:
                if v not in data_all:
                    binvar_cache[key] = None
                    return None
                a = _flatten_raw(data_all[v])
                if "costheta" in v:
                    a = np.abs(a)
                arrs.append(a)
            out = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
            binvar_cache[key] = out
            return out

        bin_specs = {}    # bname -> (list(bcfg), edges_list)
        bin_tasks = []    # list of (bname, ibin, vals_subset, ecm)
        bin_n     = {}    # (bname, ibin) -> N events
        bin_vals  = {}    # bname -> list of N_BINS_PRIOR np.ndarrays (kept for binned-plot pass)
        for bname, bcfg in BIN_CONFIG.items():
            if bname not in branches:
                continue
            v = _bin_var_array(bcfg)
            if v is None:
                print(f"  [{ecm}]  SKIP {bname}: binning var(s) {bcfg} not in tree")
                continue
            yvals = _flatten_raw(data_all[bname])
            if len(v) != len(yvals):
                print(f"  [{ecm}]  SKIP {bname}: binning-var length {len(v)} != {len(yvals)}")
                continue
            edges_q = np.quantile(v, np.linspace(0, 1, N_BINS_PRIOR + 1))
            edges_q[0]  -= 1e-9
            edges_q[-1] += 1e-9
            bin_specs[bname] = (list(bcfg), edges_q.tolist())
            bin_vals[bname]  = [None] * N_BINS_PRIOR

            # Optional per-bin jet-pool exclusion (4q). The pooled arrays are the
            # concat of the 4 jets in order, equal segments, so element i belongs
            # to jet (i // nseg)+1. Restrict the per-bin training subset only;
            # edges above are unchanged (computed on the full pooled distribution).
            excl  = POOL_BIN_EXCLUDE_4Q.get(bname)
            jetid = None
            if excl is not None:
                nseg = len(v) // 4
                if nseg * 4 == len(v):
                    jetid = np.repeat([1, 2, 3, 4], nseg)
                else:
                    print(f"  [{ecm}]  WARN {bname}: pooled length {len(v)} not 4×N; "
                          f"jet-pool exclusion disabled")

            for ibin in range(N_BINS_PRIOR):
                m = (v >= edges_q[ibin]) & (v < edges_q[ibin+1])
                if jetid is not None:
                    for jet, bins_excl in excl.items():
                        if ibin in bins_excl:
                            m = m & (jetid != jet)
                sub = yvals[m]
                bin_tasks.append((bname, ibin, sub, ecm))
                bin_n[(bname, ibin)] = int(m.sum())
                bin_vals[bname][ibin] = sub

        # Inner pool: 8 workers per ECM × 3 ECMs ≈ 24 cores busy on ironic (64 avail).
        # Tasks are independent; worker count is bounded so we don't oversubscribe.
        n_inner = min(8, max(1, len(bin_tasks)))
        print(f"  [{ecm}]  dispatching {len(bin_tasks)} binned fits across {n_inner} workers")
        bin_outputs = []
        if bin_tasks:
            with ProcessPoolExecutor(max_workers=n_inner) as inner_pool:
                for out in inner_pool.map(_fit_bin_task, bin_tasks):
                    bin_outputs.append(out)

        # Aggregate by branch (preserve bin order via index).
        agg = {b: [None]*N_BINS_PRIOR for b in bin_specs}
        for bname, ibin, p in bin_outputs:
            agg[bname][ibin] = p
            chi2 = p["chi2_ndof"] if p else float("nan")
            ok   = "OK" if (p and p.get("fit_ok")) else "WARN"
            print(f"    [{ecm}]  {bname:30s} bin {ibin}/{N_BINS_PRIOR}  "
                  f"N={bin_n.get((bname, ibin), -1)}  χ²/ndf={chi2:.2f}  {ok}")

        for bname, bins_list in agg.items():
            bin_var, edges = bin_specs[bname]
            results.setdefault(bname, {})
            results[bname]["binned"] = {
                "bin_var": bin_var,
                "edges":   edges,
                "bins":    bins_list,
            }
            # Per-branch binned plot: 5 sub-panels, one per quantile bin, with the
            # data histogram, fitted PDF, and pull. Mirrors the inclusive plot's
            # data+fit+pull layout for visual smoothness checks across the binning.
            cfg = BRANCH_CONFIG.get(bname, {})
            _plot_binned(ecm, plot_dir, bname, bins_list, edges,
                         "+".join(bin_var), bin_vals.get(bname), cfg)

    # ── BES correlation diagnostic: ρ(m_ee−ECM, pz_ee) ───────────────────────
    if "gen_ee_m_minus_ecm" in data_all and "gen_ee_pz" in data_all:
        a = _flatten_raw(data_all["gen_ee_m_minus_ecm"])
        b = _flatten_raw(data_all["gen_ee_pz"])
        if len(a) == len(b) and len(a) > 1:
            rho = float(np.corrcoef(a, b)[0, 1])
            print(f"  [{ecm}]  BES correlation  ρ(m_ee−ECM, pz_ee) = {rho:+.5f}")
            results["_bes_correlations"] = {
                "m_ee_pz_rho": rho,
                "n_events":    int(len(a)),
            }

    # ── JSON ──────────────────────────────────────────────────────────────────
    os.makedirs(FUNC_DIR, exist_ok=True)
    json_path = f"{FUNC_DIR}/dcb_results_ecm{ecm}.json"
    with open(json_path, "w") as fj:
        json.dump(results, fj, indent=2, cls=_NpEncoder)

    print(f"\nDone ecm{ecm}. Plots → {plot_dir}/  |  JSON → {json_path}")
    return ecm, results


def _cpp_struct_for_model(model):
    """Map fit-model name → (C++ struct typename, name prefix)."""
    if model == "dcb2g":     return ("DcbGaussParams",         "DCBG")
    if model == "expleft2g": return ("DcbExpLeftGaussParams",  "DCBELG")
    if model == "dcber2g":   return ("DcbExpRightGaussParams", "DCBERG")
    if model == "dcber3g":   return ("DcbExpRight3GaussParams","DCBER3G")
    if model == "gauss":     return ("GaussParams",            "GAUSS")
    if model == "asymgauss": return ("AsymGaussParams",        "AG")
    if model == "asymgauss2g": return ("AsymGauss2GParams",    "AG2G")
    if model == "asymgauss3g": return ("AsymGauss3GParams",    "AG3G")
    if model == "spike_dcb2g": return ("SpikeDcbGaussParams",  "SDCBG")
    return ("DcbParams", "DCB")


def _cpp_struct_initializer(p):
    """Inline brace-init body (params only, no `Type NAME =` prefix). Caller adds
    the trailing `, norm }` and the type/declaration. chi2/ok comment is
    appended by callers as needed. Returns the comma-separated parameter list
    matching the C++ struct field order in dcb_params.h."""
    if p["model"] == "dcb2g":
        return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
                f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
                f"{p['norm']:.10e}")
    if p["model"] == "expleft2g":
        return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['kL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
                f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
                f"{p['norm']:.10e}")
    if p["model"] == "dcber2g":
        return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['kR']:.6f}, "
                f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
                f"{p['norm']:.10e}")
    if p["model"] == "dcber3g":
        return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['kR']:.6f}, "
                f"{p['f_s']:.6f}, {p['mu_s']:+.6f}, {p['sigma_s']:.6f}, "
                f"{p['f_o']:.6f}, {p['mu_o']:+.6f}, {p['sigma_o']:.6f}, "
                f"{p['norm']:.10e}")
    if p["model"] == "gauss":
        return f"{p['mu']:+.6f}, {p['sigma']:.6f}"
    if p["model"] == "asymgauss":
        return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['sigma_L']:.6f}, {p['sigma_R']:.6f}")
    if p["model"] == "asymgauss2g":
        return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['sigma_L']:.6f}, {p['sigma_R']:.6f}, "
                f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}")
    if p["model"] == "asymgauss3g":
        return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['sigma_L']:.6f}, {p['sigma_R']:.6f}, "
                f"{p['f_s']:.6f}, {p['mu_s']:+.6f}, {p['sigma_s']:.6f}, "
                f"{p['f_o']:.6f}, {p['mu_o']:+.6f}, {p['sigma_o']:.6f}")
    if p["model"] == "spike_dcb2g":
        return (f"{p['f_delta']:.6f}, {p['sig_res']:.6f}, "
                f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
                f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
                f"{p['f_wide']:.6f}, {p['mu_wide']:+.6f}, {p['sigma_wide']:.6f}, "
                f"{p['norm']:.10e}")
    return (f"{p['mu']:+.6f}, {p['sigma']:.6f}, "
            f"{p['aL']:.6f}, {p['nL']:.6f}, {p['aR']:.6f}, {p['nR']:.6f}, "
            f"{p['norm']:.10e}")


def _cpp_binned_lines(bname, binned, ecm):
    """Emit per-bin constexpr structs + per-ECM std::array bundles + edges.
    Returns list of lines."""
    bins = binned.get("bins") or []
    if not bins or any(b is None for b in bins):
        return [f"// (binned fit incomplete for {bname} ecm{ecm} — skipped)"]
    model = bins[0]["model"]
    typename, prefix = _cpp_struct_for_model(model)
    tag   = bname.upper()
    out   = []
    for i, p in enumerate(bins):
        note = f"// bin{i}: chi2/ndf={p['chi2_ndof']:.2f}  {'OK' if p['fit_ok'] else 'WARN'}"
        out.append(
            f"constexpr {typename} {prefix}_{tag}_BIN{i}_{ecm} = "
            f"{{ {_cpp_struct_initializer(p)} }};  {note}"
        )
    bin_list = ", ".join(f"{prefix}_{tag}_BIN{i}_{ecm}" for i in range(len(bins)))
    out.append(
        f"constexpr std::array<{typename}, {len(bins)}> "
        f"{prefix}_{tag}_BINS_{ecm} = {{ {bin_list} }};"
    )
    edges = binned.get("edges") or []
    edge_str = ", ".join(f"{float(e):.6f}" for e in edges)
    out.append(
        f"constexpr std::array<double, {len(edges)}> "
        f"{prefix}_{tag}_EDGES_{ecm} = {{ {edge_str} }};"
    )
    return out


def _cpp_param_line(bname, p, ecm):
    typename, prefix = _cpp_struct_for_model(p["model"])
    note = f"// chi2/ndf={p['chi2_ndof']:.2f}  {'OK' if p['fit_ok'] else 'WARN'}"
    return (f"constexpr {typename} {prefix}_{bname.upper()}_{ecm} = "
            f"{{ {_cpp_struct_initializer(p)} }};  {note}")


def write_combined_header(all_results, params_only=False, out_name="dcb_params.h"):
    """
    Generate a kinfit-inputs C++ header from the fitted parameters.

    params_only=False (default): a single self-contained header with PDF struct
    definitions, normalised evaluators, fitted parameters, and the pick_bin
    helper (kinfit_inputs/dcb_params.h, ℓνqq).

    params_only=True (4q): emit ONLY the fitted-parameter macros wrapped in the
    WWFunctions namespace. Struct defs / evaluators / pick_bin are reused from
    dcb_params.h (included via WWKinReco4q.h → WWKinReco.h), so re-emitting them
    here would be a redefinition. Written to {FUNC_DIR}/{out_name}.

    all_results: dict  ecm -> {bname: params_dict}
    """
    if params_only:
        # Distinct namespace WWFunctions4q so the 4q param macros never collide
        # with the ℓνqq ones in dcb_params.h (same names, different sample/values).
        # `using namespace ::WWFunctions` makes the struct types (DcbGaussParams,
        # GaussParams, …) from dcb_params.h visible to the constexpr definitions.
        lines = [
            "#pragma once",
            "// Auto-generated by fit_resolutions.py (WW_CHANNEL=4q) — do not edit by hand.",
            "// PARAMS-ONLY: fitted-parameter macros for the 4q pooled-jet kinfit, in",
            "// namespace WWFunctions4q. Struct defs, evaluators, and pick_bin are reused",
            "// from dcb_params.h (included via WWKinReco4q.h → WWKinReco.h).",
            "",
            "namespace WWFunctions4q {",
            "using namespace ::WWFunctions;",
            "",
            "// ── Fitted parameters ────────────────────────────────────────────────",
        ]
        for ecm in sorted(all_results):
            results = all_results[ecm]
            lines.append("")
            lines.append(f"// ECM {ecm} GeV")
            for bname, p in sorted(results.items()):
                if "model" in p:
                    lines.append(_cpp_param_line(bname, p, ecm))
                if "binned" in p:
                    lines.extend(_cpp_binned_lines(bname, p["binned"], ecm))
        lines += ["", "} // namespace WWFunctions4q", ""]
        out_path = f"{FUNC_DIR}/{out_name}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines))
        print(f"Written {out_path} (params-only)")
        return

    lines = [
        "#pragma once",
        "// Auto-generated by fit_resolutions.py — do not edit by hand.",
        "// Contains PDF struct definitions, normalised evaluators, and fitted",
        "// parameters for all centre-of-mass energies.",
        "// Include this file instead of the individual per-ECM headers.",
        "",
        "#include <cmath>",
        "#include <algorithm>",
        "#include <array>",
        "",
        "namespace WWFunctions {",
        "",
        "// ── Struct definitions ───────────────────────────────────────────────",
        "",
        "struct DcbParams {",
        "    double mu, sigma, aL, nL, aR, nR;",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct DcbGaussParams {",
        "    double mu, sigma, aL, nL, aR, nR;     // narrow DCB core",
        "    double f_wide, mu_wide, sigma_wide;    // broad Gaussian component",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct DcbExpLeftGaussParams {",
        "    double mu, sigma, aL, kL, aR, nR;     // exp-left DCB core (kL: exp decay rate)",
        "    double f_wide, mu_wide, sigma_wide;    // broad Gaussian component",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct DcbExpRightGaussParams {",
        "    double mu, sigma, aL, nL;              // power-law left tail (ISR heavy tail)",
        "    double aR, kR;                         // exponential right tail: exp(-kR*(t-aR))",
        "    double f_wide, mu_wide, sigma_wide;    // broad Gaussian component",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct DcbExpRight3GaussParams {",
        "    double mu, sigma, aL, nL;              // dcber core (tracker + soft FSR)",
        "    double aR, kR;                         // exponential right cutoff",
        "    double f_s, mu_s, sigma_s;             // shoulder Gaussian (intermediate FSR)",
        "    double f_o, mu_o, sigma_o;             // outlier Gaussian (deep radiative tail)",
        "    double norm;                           // 1/integral, shape integrates to 1",
        "};",
        "",
        "struct GaussParams {",
        "    double mu, sigma;                      // single Gaussian (analytically normalised)",
        "};",
        "",
        "struct AsymGaussParams {",
        "    double mu, sigma;                      // sigma = (sigma_L+sigma_R)/2, used by _y2x for y-rescaling",
        "    double sigma_L, sigma_R;               // separate widths each side of mu (C¹ continuous at mu)",
        "};",
        "",
        "struct AsymGauss2GParams {",
        "    double mu, sigma;                      // sigma = (sigma_L+sigma_R)/2, used by _y2x",
        "    double sigma_L, sigma_R;               // asymmetric core widths each side of mu",
        "    double f_wide, mu_wide, sigma_wide;    // wide Gaussian component (heavy tails)",
        "};",
        "",
        "struct AsymGauss3GParams {",
        "    double mu, sigma;                      // sigma = (sigma_L+sigma_R)/2, used by _y2x",
        "    double sigma_L, sigma_R;               // asymmetric core widths each side of mu",
        "    double f_s, mu_s, sigma_s;             // shoulder Gaussian (intermediate-width)",
        "    double f_o, mu_o, sigma_o;             // outlier Gaussian (very-wide tail)",
        "};",
        "",
        "struct SpikeDcbGaussParams {",
        "    double f_delta, sigma_res;             // delta-spike (smoothed Gaussian at 0)",
        "    double mu, sigma, aL, nL, aR, nR;     // smooth narrow DCB core",
        "    double f_wide, mu_wide, sigma_wide;    // smooth broad Gaussian",
        "    double norm;                           // 1/integral of (smooth) shape",
        "};",
        "",
        "// ── Unnormalised shape helpers ───────────────────────────────────────",
        "",
        "namespace detail {",
        "inline double dcb_unnorm(double t, double aL, double nL,",
        "                          double aR, double nR) {",
        "    if (t < -aL) {",
        "        double AL = std::pow(nL/aL, nL) * std::exp(-0.5*aL*aL);",
        "        double BL = nL/aL - aL;",
        "        return AL * std::pow(std::max(BL - t, 1e-10), -nL);",
        "    }",
        "    if (t > aR) {",
        "        double AR = std::pow(nR/aR, nR) * std::exp(-0.5*aR*aR);",
        "        double BR = nR/aR - aR;",
        "        return AR * std::pow(std::max(BR + t, 1e-10), -nR);",
        "    }",
        "    return std::exp(-0.5*t*t);",
        "}",
        "inline double dcb_expleft_unnorm(double t, double aL, double kL,",
        "                                  double aR, double nR) {",
        "    if (t < -aL) return std::exp(-0.5*aL*aL) * std::exp(kL*(aL + t));",
        "    if (t > aR) {",
        "        double AR = std::pow(nR/aR, nR) * std::exp(-0.5*aR*aR);",
        "        double BR = nR/aR - aR;",
        "        return AR * std::pow(std::max(BR + t, 1e-10), -nR);",
        "    }",
        "    return std::exp(-0.5*t*t);",
        "}",
        "} // namespace detail",
        "",
        "// ── Evaluators: return -2*log(P(x)), P(x) normalised to unit integral ─",
        "",
        "inline double dcb_neg2logpdf(double x, const DcbParams& p) {",
        "    double f = detail::dcb_unnorm((x - p.mu)/p.sigma, p.aL, p.nL, p.aR, p.nR);",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "inline double dcb_gauss_neg2logpdf(double x, const DcbGaussParams& p) {",
        "    double core = detail::dcb_unnorm((x - p.mu)/p.sigma, p.aL, p.nL, p.aR, p.nR);",
        "    double wide = std::exp(-0.5 * std::pow((x - p.mu_wide)/p.sigma_wide, 2));",
        "    double f    = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "inline double dcb_expleft_gauss_neg2logpdf(double x, const DcbExpLeftGaussParams& p) {",
        "    double core = detail::dcb_expleft_unnorm((x - p.mu)/p.sigma, p.aL, p.kL, p.aR, p.nR);",
        "    double wide = std::exp(-0.5 * std::pow((x - p.mu_wide)/p.sigma_wide, 2));",
        "    double f    = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "inline double gauss_neg2logpdf(double x, const GaussParams& p) {",
        "    double t = (x - p.mu) / p.sigma;",
        "    return t*t + 2.0 * std::log(p.sigma) + std::log(2.0 * M_PI);",
        "}",
        "",
        "inline double asymgauss_neg2logpdf(double x, const AsymGaussParams& p) {",
        "    // Asymmetric (split) Gaussian: σ_L for x<µ, σ_R for x>=µ. C¹-continuous at µ.",
        "    // Normalised PDF: f(x) = sqrt(2/π) / (σ_L + σ_R) · exp(-z²/2)",
        "    //   z = (x - µ) / σ(x),  σ(x) = σ_L if x<µ else σ_R",
        "    double dx = x - p.mu;",
        "    double s  = (dx < 0.0) ? p.sigma_L : p.sigma_R;",
        "    double z  = dx / s;",
        "    return z*z + 2.0 * std::log(p.sigma_L + p.sigma_R) + std::log(M_PI / 2.0);",
        "}",
        "",
        "inline double asymgauss2g_neg2logpdf(double x, const AsymGauss2GParams& p) {",
        "    // Asymmetric Gaussian core + wide Gaussian. Both individually normalised,",
        "    // so the convex sum (1−f_wide)·core + f_wide·wide is also normalised.",
        "    double dx = x - p.mu;",
        "    double s_core  = (dx < 0.0) ? p.sigma_L : p.sigma_R;",
        "    double z_core  = dx / s_core;",
        "    double n_core  = std::sqrt(2.0 / M_PI) / (p.sigma_L + p.sigma_R);",
        "    double core    = n_core * std::exp(-0.5 * z_core * z_core);",
        "    double z_wide  = (x - p.mu_wide) / p.sigma_wide;",
        "    double n_wide  = 1.0 / (p.sigma_wide * std::sqrt(2.0 * M_PI));",
        "    double wide    = n_wide * std::exp(-0.5 * z_wide * z_wide);",
        "    double f       = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    return -2.0 * std::log(std::max(f, 1e-300));",
        "}",
        "",
        "inline double asymgauss3g_neg2logpdf(double x, const AsymGauss3GParams& p) {",
        "    // Asymmetric core + shoulder Gauss + outlier (very wide) Gauss. All",
        "    // individually normalised → convex sum is normalised.",
        "    double dx = x - p.mu;",
        "    double s_core  = (dx < 0.0) ? p.sigma_L : p.sigma_R;",
        "    double z_core  = dx / s_core;",
        "    double n_core  = std::sqrt(2.0 / M_PI) / (p.sigma_L + p.sigma_R);",
        "    double core    = n_core * std::exp(-0.5 * z_core * z_core);",
        "    double z_s     = (x - p.mu_s) / p.sigma_s;",
        "    double n_s     = 1.0 / (p.sigma_s * std::sqrt(2.0 * M_PI));",
        "    double shldr   = n_s * std::exp(-0.5 * z_s * z_s);",
        "    double z_o     = (x - p.mu_o) / p.sigma_o;",
        "    double n_o     = 1.0 / (p.sigma_o * std::sqrt(2.0 * M_PI));",
        "    double outl    = n_o * std::exp(-0.5 * z_o * z_o);",
        "    double f_core  = std::max(0.0, 1.0 - p.f_s - p.f_o);",
        "    double f       = f_core * core + p.f_s * shldr + p.f_o * outl;",
        "    return -2.0 * std::log(std::max(f, 1e-300));",
        "}",
        "",
        "inline double spike_dcb_gauss_neg2logpdf(double x, const SpikeDcbGaussParams& p) {",
        "    // Composite PDF: f_delta * G(0, sigma_res) + (1 - f_delta) * dcb_gauss(x).",
        "    // Used for distributions with a true delta-at-0 spike (e.g. ISR pz with",
        "    // ~40% of events at exactly 0); sigma_res smooths the delta for kinfit",
        "    // numerical stability.",
        "    double inv_sr   = 1.0 / p.sigma_res;",
        "    double delta_pdf = p.f_delta * inv_sr * std::exp(-0.5 * x * x * inv_sr * inv_sr)",
        "                       / std::sqrt(2.0 * M_PI);",
        "    double core = detail::dcb_unnorm((x - p.mu)/p.sigma, p.aL, p.nL, p.aR, p.nR);",
        "    double wide = std::exp(-0.5 * std::pow((x - p.mu_wide)/p.sigma_wide, 2));",
        "    double shape = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    double smooth_pdf = (1.0 - p.f_delta) * shape * p.norm;",
        "    return -2.0 * std::log(std::max(delta_pdf + smooth_pdf, 1e-300));",
        "}",
        "",
        "inline double dcb_expright_gauss_neg2logpdf(double x, const DcbExpRightGaussParams& p) {",
        "    double t    = (x - p.mu) / p.sigma;",
        "    double core;",
        "    if (t < -p.aL) {",
        "        double AL = std::pow(p.nL/p.aL, p.nL) * std::exp(-0.5*p.aL*p.aL);",
        "        double BL = p.nL/p.aL - p.aL;",
        "        core = AL * std::pow(std::max(BL - t, 1e-10), -p.nL);",
        "    } else if (t > p.aR) {",
        "        core = std::exp(-0.5*p.aR*p.aR - p.kR*(t - p.aR));",
        "    } else {",
        "        core = std::exp(-0.5*t*t);",
        "    }",
        "    double wide = std::exp(-0.5 * std::pow((x - p.mu_wide)/p.sigma_wide, 2));",
        "    double f    = (1.0 - p.f_wide) * core + p.f_wide * wide;",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "inline double dcb_expright_3gauss_neg2logpdf(double x, const DcbExpRight3GaussParams& p) {",
        "    double t    = (x - p.mu) / p.sigma;",
        "    double core;",
        "    if (t < -p.aL) {",
        "        double AL = std::pow(p.nL/p.aL, p.nL) * std::exp(-0.5*p.aL*p.aL);",
        "        double BL = p.nL/p.aL - p.aL;",
        "        core = AL * std::pow(std::max(BL - t, 1e-10), -p.nL);",
        "    } else if (t > p.aR) {",
        "        core = std::exp(-0.5*p.aR*p.aR - p.kR*(t - p.aR));",
        "    } else {",
        "        core = std::exp(-0.5*t*t);",
        "    }",
        "    double shoulder = std::exp(-0.5 * std::pow((x - p.mu_s)/p.sigma_s, 2));",
        "    double outlier  = std::exp(-0.5 * std::pow((x - p.mu_o)/p.sigma_o, 2));",
        "    double f_core   = std::max(0.0, 1.0 - p.f_s - p.f_o);",
        "    double f        = f_core * core + p.f_s * shoulder + p.f_o * outlier;",
        "    return -2.0 * (std::log(std::max(f, 1e-300)) + std::log(p.norm));",
        "}",
        "",
        "// ── Fitted parameters ────────────────────────────────────────────────",
    ]

    for ecm in sorted(all_results):
        results = all_results[ecm]
        lines.append(f"")
        lines.append(f"// ECM {ecm} GeV")
        for bname, p in sorted(results.items()):
            # Emit the integrated (kinematics-averaged) prior; legacy callers
            # still rely on the unbinned constants.
            if "model" in p:
                lines.append(_cpp_param_line(bname, p, ecm))
            # Per-bin priors + bin-edge array (only for branches in BIN_CONFIG).
            if "binned" in p:
                lines.extend(_cpp_binned_lines(bname, p["binned"], ecm))

    # Generic helper: pick the right per-bin DCB prior given the binning value.
    # Templated so the same code works for DcbGaussParams, DcbExpLeftGaussParams,
    # DcbExpRightGaussParams and DcbParams.
    lines += [
        "",
        "// ── Per-bin lookup helper ────────────────────────────────────────────",
        "// Pick the prior whose bin the binning value v falls in. Edges are the",
        "// same equal-occupancy quantile edges used in fit_resolutions.py; values",
        "// outside the [first, last] range are clamped to the edge bin.",
        "template <typename T, std::size_t Nb>",
        "inline const T& pick_bin(const std::array<T, Nb>& bins,",
        "                         const std::array<double, Nb + 1>& edges,",
        "                         double v) {",
        "    if (v <= edges[0])      return bins[0];",
        "    if (v >= edges[Nb])     return bins[Nb - 1];",
        "    // Linear scan; Nb is small (5).",
        "    for (std::size_t i = 0; i < Nb; ++i) {",
        "        if (v < edges[i + 1]) return bins[i];",
        "    }",
        "    return bins[Nb - 1];",
        "}",
        "",
        "} // namespace WWFunctions",
        "",
    ]

    out_path = f"{FUNC_DIR}/dcb_params.h"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Written {out_path}")


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=len(ECM_LIST)) as pool:
        all_results = dict(pool.map(process_ecm, ECM_LIST))
    if IS_4Q:
        write_combined_header(all_results, params_only=True,
                              out_name="dcb_params_4q.h")
    else:
        write_combined_header(all_results)
    publish(PLOTS_DIR, os.environ.get("FIT_PUBSUB",
                                      "resolutions_4q" if IS_4Q else "resolutions"))
