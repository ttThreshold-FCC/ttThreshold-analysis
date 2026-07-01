#!/usr/bin/env python3
"""
Plot kinfit results from step2 treemaker output.

Three stages, run in order:
  1. mW overlay       — per-ECM and ECM-comparison plots of W-mass branches
                        (reco / kinfit pre/post / kinfit combined)
                        → outputs/plots/lnuqq/allbranches/  (publish: "mW_overlay")
  2. kinfit variables — per-ECM and ECM-comparison plots for every kinfit branch,
                        with input-PDF overlays from fit_resolutions.py JSON,
                        reco/gen comparisons, and equal-N NLL slices.
                        → outputs/plots/kinfit_vars/         (publish: "kinfit_vars")
  3. correlations     — distribution of post-fit ρ for every parameter pair
                        + 2D summary heatmap (mean / median / mode).
                        → outputs/plots/kinfit_correlations/ (publish: "kinfit_correlations")
"""

import os, json, math
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from eos_publish import publish

# ── Shared constants ─────────────────────────────────────────────────────────

ECM_LIST    = [157, 160, 163]
INDIR       = os.environ.get("STEP2_INDIR", "outputs/treemaker/lnuqq/step2/semihad")
INFILE_TMPL = INDIR + "/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
JSON_TMPL   = "kinfit_inputs/dcb_results_ecm{ecm}.json"
TREE_NAME   = "events"
ECM_COLORS  = {157: "tab:purple", 160: "tab:orange", 163: "tab:cyan"}


# =============================================================================
# Stage 1 — mW overlay
# =============================================================================

MW_OUTDIR = "outputs/plots/lnuqq/allbranches"

# Last column is_kinfit: branches that respect the kinfit_valid mask in the
# `_valid` variant. Pre-fit reco branches are unfiltered in both variants.
MW_HISTS_CFG = [
    ("reco_Wlep_m",   "Wlep reco (pre-fit)", "tab:blue",  ":",  False),
    ("reco_Whad_m",   "Whad reco (pre-fit)", "tab:green", ":",  False),
    ("kinfit_Wlep_m", "Wlep (kinfit)",       "tab:blue",  "--", True),
    ("kinfit_Whad_m", "Whad (kinfit)",       "tab:green", "--", True),
    ("kinfit_mW",     "W (kinfit combined)", "tab:red",   "-",  True),
]

MW_REF   = 80.419
MW_XLIM  = (50, 100)
MW_NBINS = 100


def _mw_load(t, branch, valid_only, is_kinfit):
    """Load mW histogram. valid_only masks kinfit branches by kinfit_valid;
    pre-fit reco branches are never masked (no validity concept)."""
    if branch not in t.keys():
        return None, None, 0
    if valid_only and is_kinfit:
        arrs = t.arrays([branch, "kinfit_valid"], library="np")
        vals = arrs[branch][arrs["kinfit_valid"].astype(bool)]
    else:
        vals = t[branch].array(library="np")
    counts, edges = np.histogram(vals, bins=MW_NBINS, range=MW_XLIM)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = int(counts.sum())  # N entries in histogram range
    c = counts.astype(float)
    if norm > 0:
        c = c / norm
    return centers, c, norm


def plot_mW_overlay_per_ecm(ecm, t, variant):
    """One plot showing all mW histogram variants for a single ECM.
    variant is 'valid' or 'all' — controls kinfit_valid filtering."""
    valid_only = (variant == "valid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for branch, label, color, ls, is_kinfit in MW_HISTS_CFG:
        x, c, n = _mw_load(t, branch, valid_only, is_kinfit)
        if x is None:
            print(f"  WARNING [{ecm}]: {branch} not found")
            continue
        ax.step(x, c, where="mid", color=color, linestyle=ls, linewidth=2,
                label=f"{label}  (N={n})")

    ax.axvline(MW_REF, color="grey", linestyle="-", linewidth=1.5,
               label=f"$m_W$ = {MW_REF:.3f} GeV")
    ax.set_title(rf"$\sqrt{{s}}$ = {ecm} GeV  —  kinfit {variant}", fontsize=13)
    ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
    ax.set_ylabel("A.U.", fontsize=13)
    ax.set_xlim(MW_XLIM)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(f"{MW_OUTDIR}/mW_overlay_ecm{ecm}_{variant}.{fmt}", dpi=150)
    plt.close(fig)


def plot_mW_ecm_comparison(trees, variant):
    """One plot per histogram showing all ECMs overlaid.
    variant is 'valid' or 'all'."""
    valid_only = (variant == "valid")
    for branch, label, _, _, is_kinfit in MW_HISTS_CFG:
        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = False
        for ecm, t in trees.items():
            if t is None:
                continue
            x, c, n = _mw_load(t, branch, valid_only, is_kinfit)
            if x is None:
                print(f"  WARNING [{ecm}]: {branch} not found")
                continue
            ax.step(x, c, where="mid", color=ECM_COLORS[ecm], linewidth=2,
                    label=rf"$\sqrt{{s}}$ = {ecm} GeV  (N={n})")
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.axvline(MW_REF, color="grey", linestyle="-", linewidth=1.5,
                   label=f"$m_W$ = {MW_REF:.3f} GeV")
        ax.set_title(f"{label}  —  kinfit {variant}", fontsize=13)
        ax.set_xlabel(r"$m_W$ [GeV]", fontsize=13)
        ax.set_ylabel("A.U.", fontsize=13)
        ax.set_xlim(MW_XLIM)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=11, loc="upper left")
        fig.tight_layout()
        for fmt in ("png", "pdf"):
            fig.savefig(f"{MW_OUTDIR}/ecm_comparison_{branch}_{variant}.{fmt}", dpi=150)
        plt.close(fig)


def run_mW_overlay():
    os.makedirs(MW_OUTDIR, exist_ok=True)
    trees = {}
    for ecm in ECM_LIST:
        path = INFILE_TMPL.format(ecm=ecm)
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping ecm{ecm}")
            trees[ecm] = None
            continue
        if os.path.getsize(path) < 1_000_000:
            print(f"WARNING: {path} only {os.path.getsize(path)} bytes (placeholder?) — skipping ecm{ecm}")
            trees[ecm] = None
            continue
        try:
            trees[ecm] = uproot.open(path)[TREE_NAME]
        except Exception as e:
            print(f"WARNING: {path} present but events tree not yet finalized ({e}) — skipping ecm{ecm}")
            trees[ecm] = None

    for variant in ("valid", "all"):
        for ecm, t in trees.items():
            if t is not None:
                plot_mW_overlay_per_ecm(ecm, t, variant)
                print(f"Saved mW_overlay_ecm{ecm}_{variant}.[png|pdf]")
        plot_mW_ecm_comparison(trees, variant)
        print(f"Saved ecm_comparison_*_{variant}.[png|pdf]  →  {MW_OUTDIR}/")

    publish(MW_OUTDIR, os.environ.get("MW_PUBSUB", "mW_overlay"))


# =============================================================================
# Stage 2 — kinfit-variable plots with PDF overlays
# =============================================================================

KINFIT_OUTDIR = "outputs/plots/kinfit_vars"

_SQRT2   = math.sqrt(2.0)
_LOG_MAX = math.log(np.finfo(np.float64).max)
_LOG_MIN = -_LOG_MAX

# Mapping from kinfit output branch → fitted resolution branch in JSON
KINFIT_TO_RESOL = {
    "kinfit_s1":  "jet1_p_resp",
    "kinfit_s2":  "jet2_p_resp",
    "kinfit_sl":  "lep_p_resp",
    "kinfit_sn":  "met_p_resp",
    "kinfit_t1":  "jet1_theta_resol",
    "kinfit_t2":  "jet2_theta_resol",
    "kinfit_tl":  "lep_theta_resol",
    "kinfit_tn":  "met_theta_resol",
    "kinfit_p1":  "jet1_phi_resol",
    "kinfit_p2":  "jet2_phi_resol",
    "kinfit_pl":  "lep_phi_resol",
    "kinfit_pn":  "met_phi_resol",
    # BES nuisances (Gaussian priors)
    "kinfit_bes_m_minus_ecm": "gen_ee_m_minus_ecm",
    "kinfit_bes_pz":          "gen_ee_pz",
    # Derived pulls (computed at plot time from saved kinfit_WW_* + bes_*).
    # Priors applied in WWKinReco.h::kinFit chi² but not stored as own branches.
    "kinfit_m_loss":  "gen_WW_m_minus_m_ee",
    "kinfit_isr_px":  "gen_isr_px",
    "kinfit_isr_py":  "gen_isr_py",
    "kinfit_isr_pz":  "gen_isr_pz",
}

# Mapping: kinfit post-fit branch → reco / gen counterpart, derived from the
# convention <level>_<object>_<quantity>. Objects with all three levels:
#   constituents (jet1, jet2, lep, nu): p, pt, theta, phi
#   Ws (Wlep, Whad):                    m, p, pt, px, py, pz
#   WW system:                          m, m_minus_ecm, px, py, pz, p_imbalance_tot
def _build_kinfit_maps():
    reco, gen = {}, {}
    # reco-level neutrino is the detector MET (no real nu reco); gen-level uses
    # "quark" naming for the matched parton (jets at gen level would be misleading).
    _reco_obj = {"jet1": "jet1", "jet2": "jet2", "lep": "lep", "nu": "met"}
    _gen_obj  = {"jet1": "quark1", "jet2": "quark2", "lep": "lep", "nu": "nu"}
    for obj in ("jet1", "jet2", "lep", "nu"):
        for q in ("p", "pt", "theta", "phi"):
            kf = f"kinfit_{obj}_{q}"
            reco[kf] = f"reco_{_reco_obj[obj]}_{q}"
            gen[kf]  = f"gen_{_gen_obj[obj]}_{q}"
    for obj in ("Wlep", "Whad"):
        for q in ("m", "p", "pt", "px", "py", "pz"):
            kf = f"kinfit_{obj}_{q}"
            reco[kf] = f"reco_{obj}_{q}"
            gen[kf]  = f"gen_{obj}_{q}"
    for q in ("m", "m_minus_ecm", "px", "py", "pz", "p_imbalance_tot"):
        kf = f"kinfit_WW_{q}"
        reco[kf] = f"reco_WW_{q}"
        gen[kf]  = f"gen_WW_{q}"
    return reco, gen

KINFIT_TO_RECO, KINFIT_TO_GEN = _build_kinfit_maps()

# Resolution / response gen-truth branches + reco kinematics needed by
# plot_pulls (per-bin σ_prior lookup). Defined here as constants since
# PULL_TRUTH itself lives further down.
_PULL_GEN_BRANCHES  = (
    "jet1_p_resp", "jet2_p_resp", "lep_p_resp", "met_p_resp",
    "jet1_theta_resol", "jet2_theta_resol", "lep_theta_resol", "met_theta_resol",
    "jet1_phi_resol", "jet2_phi_resol", "lep_phi_resol", "met_phi_resol",
    "gen_ee_m_minus_ecm", "gen_ee_pz",
    "gen_WW_m_minus_m_ee",
    "gen_isr_px", "gen_isr_py", "gen_isr_pz",
)
_PULL_BIN_BRANCHES  = ("reco_jet1_p", "reco_jet2_p", "reco_lep_p", "reco_met_p")

EXTRA_BRANCHES = sorted(
    set(KINFIT_TO_RECO.values())
    | set(KINFIT_TO_GEN.values())
    | set(_PULL_GEN_BRANCHES)
    | set(_PULL_BIN_BRANCHES)
)

# Subdirectory assignment for per-ECM plots
_PULL_BRANCHES = {
    "kinfit_s1","kinfit_s2","kinfit_sl","kinfit_sn",
    "kinfit_t1","kinfit_t2","kinfit_tn","kinfit_tl",
    "kinfit_p1","kinfit_p2","kinfit_pn","kinfit_pl",
    "kinfit_bes_m_minus_ecm","kinfit_bes_pz",
    "kinfit_m_loss","kinfit_isr_px","kinfit_isr_py","kinfit_isr_pz",
}

# Derived pull branches computed at plot time from existing kinfit branches.
# Each entry: callable(branches_data, ecm) → np.ndarray of derived values.
_DERIVED_PULLS = {
    "kinfit_m_loss":  lambda d, ecm: d["kinfit_WW_m"]  - (ecm + d["kinfit_bes_m_minus_ecm"]),
    "kinfit_isr_px":  lambda d, ecm: -d["kinfit_WW_px"],
    "kinfit_isr_py":  lambda d, ecm: -d["kinfit_WW_py"],
    "kinfit_isr_pz":  lambda d, ecm: d["kinfit_bes_pz"] - d["kinfit_WW_pz"],
}

def _subdir(bname):
    # `_PULL_BRANCHES` (legacy name) actually contains *posterior* branches
    # whose plot compares the histogram of MAP estimates to the prior PDF in
    # physical units. The "pulls/" subdirectory is reserved for the normalized
    # residual plot generated by plot_pulls() — (s_fit − s_gen) / σ_prior.
    if bname in _PULL_BRANCHES:
        return "posteriors"
    if bname in KINFIT_TO_RECO and bname in KINFIT_TO_GEN:
        return "fit_vs_reco_gen"
    if bname in KINFIT_TO_GEN:
        return "fit_vs_gen"
    return "misc"

# All kinfit branches from treemaker_lnuqq_step2.py
KINFIT_BRANCHES = [
    "kinfit_mW", "kinfit_gW",
    "kinfit_s1", "kinfit_s2", "kinfit_sl", "kinfit_sn",
    "kinfit_t1", "kinfit_t2", "kinfit_tn", "kinfit_tl",
    "kinfit_p1", "kinfit_p2", "kinfit_pn", "kinfit_pl",
    "kinfit_bes_m_minus_ecm", "kinfit_bes_pz",
    # Derived pulls — computed at plot time, not stored as TTree branches.
    "kinfit_m_loss", "kinfit_isr_px", "kinfit_isr_py", "kinfit_isr_pz",
    "kinfit_chi2", "kinfit_chi2_ndof", "kinfit_valid", "kinfit_status",
    # constituent kinematics
    "kinfit_jet1_p",  "kinfit_jet2_p",  "kinfit_lep_p",  "kinfit_nu_p",
    "kinfit_jet1_pt", "kinfit_jet2_pt", "kinfit_lep_pt", "kinfit_nu_pt",
    "kinfit_jet1_theta", "kinfit_jet2_theta", "kinfit_lep_theta", "kinfit_nu_theta",
    "kinfit_jet1_phi",   "kinfit_jet2_phi",   "kinfit_lep_phi",   "kinfit_nu_phi",
    # W bosons
    "kinfit_Wlep_m", "kinfit_Wlep_p", "kinfit_Wlep_pt",
    "kinfit_Wlep_px", "kinfit_Wlep_py", "kinfit_Wlep_pz",
    "kinfit_Whad_m", "kinfit_Whad_p", "kinfit_Whad_pt",
    "kinfit_Whad_px", "kinfit_Whad_py", "kinfit_Whad_pz",
    # WW system (post-fit derived; no direct prior — overlaid via ISR balance)
    "kinfit_WW_px", "kinfit_WW_py", "kinfit_WW_pz",
    "kinfit_WW_m", "kinfit_WW_p_imbalance_tot",
]


# ── Model functions (copied from fit_resolutions.py — pure math) ─────────────

def _dcb_core(t, aL, nL, aR, nR):
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
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)

def _dcb_expleft_core(t, aL, kL, aR, nR):
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
    core = _dcb_expleft_core((x - mu_c) / sigma_c, aL, kL, aR, nR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)

def _dcb_expright_core(t, aL, nL, aR, kR):
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
    core = _dcb_expright_core((x - mu_c) / sigma_c, aL, nL, aR, kR)
    wide = np.exp(-0.5 * ((x - mu_w) / sigma_w) ** 2)
    return N * ((1.0 - f_wide) * core + f_wide * wide)

def dcb_expright_3gauss(x, N, mu_c, sigma_c, aL, nL, aR, kR,
                        f_s, mu_s, sigma_s, f_o, mu_o, sigma_o):
    core      = _dcb_expright_core((x - mu_c) / sigma_c, aL, nL, aR, kR)
    shoulder  = np.exp(-0.5 * ((x - mu_s) / sigma_s) ** 2)
    outlier   = np.exp(-0.5 * ((x - mu_o) / sigma_o) ** 2)
    f_core    = max(0.0, 1.0 - abs(f_s) - abs(f_o))
    return N * (f_core * core + abs(f_s) * shoulder + abs(f_o) * outlier)

def dcb_gaussbox(x, N, mu_c, sigma_c, aL, nL, aR, nR, f_wide, p_max, sigma_box):
    from scipy.special import erf as _sp_erf
    core = _dcb_core((x - mu_c) / sigma_c, aL, nL, aR, nR)
    p_max = abs(p_max)
    sb = max(abs(sigma_box), 1e-10)
    sq2 = _SQRT2 * sb
    wide = 0.5 * (_sp_erf((x + p_max) / sq2) - _sp_erf((x - p_max) / sq2))
    wide_peak = max(float(_sp_erf(p_max / sq2)), 1e-10)
    return N * ((1.0 - f_wide) * core + f_wide * wide / wide_peak)


def _make_pdf(p):
    """Return a callable f(x) → normalized PDF value, from JSON param dict."""
    model = p["model"]

    if model == "gauss":
        mu, sg = p["mu"], p["sigma"]
        inv_sg = 1.0 / abs(sg)
        norm_g = inv_sg / math.sqrt(2.0 * math.pi)
        def fn(x): return norm_g * np.exp(-0.5 * ((x - mu) * inv_sg) ** 2)
        return fn

    if model == "asymgauss":
        mu  = float(p["mu"])
        sL  = abs(float(p["sigma_L"]))
        sR  = abs(float(p["sigma_R"]))
        # Normalised PDF: f(x) = sqrt(2/π) / (σ_L+σ_R) · exp(-z²/2)
        norm = math.sqrt(2.0 / math.pi) / (sL + sR)
        def fn(x):
            xa = np.asarray(x, dtype=float)
            sigma = np.where(xa < mu, sL, sR)
            z = (xa - mu) / sigma
            return norm * np.exp(-0.5 * z * z)
        return fn

    if model == "asymgauss2g":
        mu   = float(p["mu"])
        sL   = abs(float(p["sigma_L"]))
        sR   = abs(float(p["sigma_R"]))
        fw   = float(p["f_wide"])
        muw  = float(p["mu_wide"])
        sw   = abs(float(p["sigma_wide"]))
        n_core = math.sqrt(2.0 / math.pi) / (sL + sR)
        n_wide = 1.0 / (sw * math.sqrt(2.0 * math.pi))
        def fn(x):
            xa = np.asarray(x, dtype=float)
            sigma = np.where(xa < mu, sL, sR)
            z_core = (xa - mu) / sigma
            core = n_core * np.exp(-0.5 * z_core * z_core)
            z_wide = (xa - muw) / sw
            wide = n_wide * np.exp(-0.5 * z_wide * z_wide)
            return (1.0 - fw) * core + fw * wide
        return fn

    if model == "asymgauss3g":
        mu   = float(p["mu"])
        sL   = abs(float(p["sigma_L"]))
        sR   = abs(float(p["sigma_R"]))
        fs   = float(p["f_s"]); mus = float(p["mu_s"]); ss = abs(float(p["sigma_s"]))
        fo   = float(p["f_o"]); muo = float(p["mu_o"]); so = abs(float(p["sigma_o"]))
        n_core = math.sqrt(2.0 / math.pi) / (sL + sR)
        n_s    = 1.0 / (ss * math.sqrt(2.0 * math.pi))
        n_o    = 1.0 / (so * math.sqrt(2.0 * math.pi))
        f_core = max(0.0, 1.0 - fs - fo)
        def fn(x):
            xa = np.asarray(x, dtype=float)
            sigma = np.where(xa < mu, sL, sR)
            z_core = (xa - mu) / sigma
            core = n_core * np.exp(-0.5 * z_core * z_core)
            z_s = (xa - mus) / ss; shldr = n_s * np.exp(-0.5 * z_s * z_s)
            z_o = (xa - muo) / so; outl  = n_o * np.exp(-0.5 * z_o * z_o)
            return f_core * core + fs * shldr + fo * outl
        return fn

    norm = p["norm"]
    if model == "dcb":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, nR = p["aL"], p["nL"], p["aR"], p["nR"]
        def fn(x): return norm * dcb(x, 1.0, mu, sg, aL, nL, aR, nR)

    elif model == "dcb2g":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, nR = p["aL"], p["nL"], p["aR"], p["nR"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * dcb_gauss(x, 1.0, mu, sg, aL, nL, aR, nR, fw, mw, sw)

    elif model == "expleft2g":
        mu, sg = p["mu"], p["sigma"]
        aL, kL, aR, nR = p["aL"], p["kL"], p["aR"], p["nR"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * dcb_expleft_gauss(x, 1.0, mu, sg, aL, kL, aR, nR, fw, mw, sw)

    elif model == "dcber2g":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, kR = p["aL"], p["nL"], p["aR"], p["kR"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        def fn(x): return norm * dcb_expright_gauss(x, 1.0, mu, sg, aL, nL, aR, kR, fw, mw, sw)

    elif model == "dcber3g":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, kR = p["aL"], p["nL"], p["aR"], p["kR"]
        fs, mus, ss = p["f_s"], p["mu_s"], p["sigma_s"]
        fo, muo, so = p["f_o"], p["mu_o"], p["sigma_o"]
        def fn(x): return norm * dcb_expright_3gauss(x, 1.0, mu, sg, aL, nL, aR, kR,
                                                       fs, mus, ss, fo, muo, so)

    elif model == "dcbgb":
        mu, sg = p["mu"], p["sigma"]
        aL, nL, aR, nR = p["aL"], p["nL"], p["aR"], p["nR"]
        fw, pm, sb = p["f_wide"], p["p_max"], p["sigma_box"]
        def fn(x): return norm * dcb_gaussbox(x, 1.0, mu, sg, aL, nL, aR, nR, fw, pm, sb)

    elif model == "spike_dcb2g":
        # f_delta · N(0, sig_res²) + (1−f_delta) · dcb2g(...). barrier_sigma
        # (when set, e.g. for symmetrized m_loss) mirrors the kinfit's quadratic
        # barrier on x>0 by suppressing the PDF there: exp(-(x/σ_b)²/2). Must
        # stay in sync with KF_M_LOSS_BARRIER_SIGMA_FRAC in WWKinReco.h.
        f_d, sr   = p["f_delta"], abs(p["sig_res"])
        mu, sg    = p["mu"], p["sigma"]
        aL, nL, aR, nR = p["aL"], p["nL"], p["aR"], p["nR"]
        fw, mw, sw = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        inv_sr = 1.0 / sr
        norm_g = inv_sr / math.sqrt(2.0 * math.pi)
        _bs = p.get("barrier_sigma")
        sig_b = abs(_bs) if _bs else None
        def fn(x):
            xa    = np.asarray(x, dtype=float)
            spike = f_d * norm_g * np.exp(-0.5 * (xa * inv_sr) ** 2)
            cont  = (1.0 - f_d) * dcb_gauss(xa, 1.0, mu, sg, aL, nL, aR, nR, fw, mw, sw)
            shape = norm * (spike + cont)
            if sig_b is not None:
                shape = np.where(xa > 0,
                                  shape * np.exp(-0.5 * (xa / sig_b) ** 2),
                                  shape)
            return shape

    elif model == "spike_dcber2g":
        # f_delta · N(spike_center, sig_res²) + (1−f_delta) · dcber2g(...)
        f_d, sr, sc_loc = p["f_delta"], abs(p["sig_res"]), p["spike_center"]
        mu, sg          = p["mu"], p["sigma"]
        aL, nL, aR, kR  = p["aL"], p["nL"], p["aR"], p["kR"]
        fw, mw, sw      = p["f_wide"], p["mu_wide"], p["sigma_wide"]
        inv_sr = 1.0 / sr
        norm_g = inv_sr / math.sqrt(2.0 * math.pi)
        def fn(x):
            spike = f_d * norm_g * np.exp(-0.5 * ((x - sc_loc) * inv_sr) ** 2)
            cont  = (1.0 - f_d) * dcb_expright_gauss(x, 1.0, mu, sg, aL, nL, aR, kR, fw, mw, sw)
            return norm * (spike + cont)

    else:
        return None

    return fn


# ── Axis range heuristics ─────────────────────────────────────────────────────

# Branches with a known fixed range; everything else is data-driven.
# Ranges chosen to span the post-fit (green) distribution; the prior PDF
# (red) often has heavier tails which would leave the fitted peak crammed.
_FIXED_RANGE = {
    "kinfit_mW":    (100, 50,   100),
    "kinfit_Wlep_m": (100, 50,   100),
    "kinfit_Whad_m": (100, 50,   100),
    "kinfit_gW":    (100,  1.9,  2.2),
    "kinfit_valid": (  3, -0.5,  2.5),
    "kinfit_chi2_ndof": (100, 0, 50),
    "kinfit_jet1_p":  (100,  0,   80),
    "kinfit_jet2_p":  (100,  0,   80),
    "kinfit_lep_p": (100,  0,   80),
    "kinfit_nu_p":  (100,  0,   80),
    "kinfit_Wlep_px": (100, -50, 50),
    "kinfit_Wlep_py": (100, -50, 50),
    "kinfit_Wlep_pz": (100, -50, 50),
    "kinfit_Whad_px": (100, -50, 50),
    "kinfit_Whad_py": (100, -50, 50),
    "kinfit_Whad_pz": (100, -50, 50),
    # Total WW system momentum: constrained near 0 by px/py/gen_WW_pz PDF
    # (gen has a sub-bin spike at 0 from collinear ISR; fit follows it tightly)
    "kinfit_WW_px": (100, -0.05, 0.05),
    "kinfit_WW_py": (100, -0.05, 0.05),
    "kinfit_WW_pz": (100, -0.3, 0.3),
    # BES nuisances: Gaussian priors with σ ≈ 119 MeV → ±400 MeV is ~3.5σ.
    "kinfit_bes_m_minus_ecm": (100, -0.4, 0.4),
    "kinfit_bes_pz":          (100, -0.4, 0.4),
    # Derived pulls: m_loss = WW_m − m_ee_fit (peaks below 0; ISR mass loss).
    # ISR 3-momentum: collinear-spike-dominated, σ ~few hundred MeV transverse,
    # ~few GeV longitudinal — see gen_isr distributions in dcb_results JSON.
    "kinfit_m_loss":  (100, -3.0, 0.5),
    "kinfit_isr_px":  (100, -2.0, 2.0),
    "kinfit_isr_py":  (100, -2.0, 2.0),
    "kinfit_isr_pz":  (100, -8.0, 8.0),
    "kinfit_jet1_theta":  (100, 0,   3.2),
    "kinfit_jet2_theta":  (100, 0,   3.2),
    "kinfit_nu_theta":  (100, 0,   3.2),
    "kinfit_jet1_phi":    (100, -3.2, 3.2),
    "kinfit_jet2_phi":    (100, -3.2, 3.2),
    "kinfit_nu_phi":    (100, -3.2, 3.2),
    # WW system post-fit: mass near ECM, mass-minus-ECM near 0 (slightly below, ISR)
    "kinfit_WW_m":           (100, 150, 170),
    # Pull/scale parameters: span post-fit data, not the wider prior tails.
    "kinfit_s1":  (100, 0.85, 1.15),
    "kinfit_s2":  (100, 0.85, 1.15),
    "kinfit_sl":  (100, 0.85, 1.15),
    "kinfit_sn":  (100, 0.92, 1.05),
    "kinfit_t1":  (100, -0.15, 0.15),
    "kinfit_t2":  (100, -0.15, 0.15),
    "kinfit_tn":  (100, -0.03, 0.03),
    "kinfit_tl":  (100, -5e-5, 5e-5),
    "kinfit_p1":  (100, -0.15, 0.15),
    "kinfit_p2":  (100, -0.15, 0.15),
    "kinfit_pn":  (100, -0.005, 0.005),
    "kinfit_pl":  (100, -3e-4, 3e-4),
    "kinfit_WW_p_imbalance_tot": (100, 0.0, 0.1),
}

def _auto_range(vals):
    lo, hi = np.percentile(vals[np.isfinite(vals)], [0.5, 99.5])
    margin = max(abs(hi - lo) * 0.1, abs(lo) * 1e-3, 1e-12)
    return lo - margin, hi + margin

def _pdf_natural_range(p, nsigma=5):
    sg = p.get("sigma", 1.0)
    aL = abs(p.get("aL", 2.0));  aR = abs(p.get("aR", 2.0))
    half = nsigma * max(aL, aR, 1.0) * sg
    if "mu" in p:
        mu = p["mu"]
        if "mu_wide" in p and "sigma_wide" in p:
            half = max(half, abs(p["mu_wide"] - mu) + nsigma * abs(p["sigma_wide"]))
        return mu - half, mu + half
    if "x_cut" in p:
        xc = p["x_cut"];  sw = p.get("sigma_wide", sg)
        return xc - nsigma * max(sw, sg), xc + 0.5
    return -half, half

def _binning(var, vals=None, pdf_params=None):
    if var in _FIXED_RANGE:
        return _FIXED_RANGE[var]

    data_lo = data_hi = None
    if vals is not None and len(vals) > 0:
        finite = vals[np.isfinite(vals)]
        if len(finite) > 0:
            data_lo, data_hi = _auto_range(finite)

    if pdf_params is not None:
        pdf_lo, pdf_hi = _pdf_natural_range(pdf_params)
        if data_lo is None:
            return (100, pdf_lo, pdf_hi)
        data_w = data_hi - data_lo
        pdf_w  = pdf_hi  - pdf_lo
        if data_w < 0.2 * pdf_w:
            return (100, pdf_lo, pdf_hi)
        return (100, data_lo, data_hi)

    if data_lo is not None:
        return (100, data_lo, data_hi)
    return (100, -5, 5)


# ── Per-ECM single-branch plot ────────────────────────────────────────────────

def _plot_branch(ax, bname, vals, ecm, pdf_fn=None, pdf_params=None,
                 reco_vals=None, gen_vals=None, color="steelblue", xlim=None):
    all_combined = np.concatenate([v for v in [vals, reco_vals, gen_vals]
                                   if v is not None and len(v)])
    nbins, xlo, xhi = _binning(bname, all_combined, pdf_params=pdf_params)
    if xlim is not None:
        xlo, xhi = xlim

    densities = []  # collect for y-limit computation under log scale

    def _draw_hist(v, label, clr, as_bar=False):
        v_c = v[(v >= xlo) & (v <= xhi)]
        counts, edges = np.histogram(v_c, bins=nbins, range=(xlo, xhi))
        bw = np.diff(edges)[0]
        density = counts / max(counts.sum() * bw, 1e-300)
        centers = 0.5 * (edges[:-1] + edges[1:])
        densities.append(density)
        if as_bar:
            ax.bar(centers, density, width=bw, color=clr, alpha=0.55, label=label)
        else:
            ax.step(centers, density, where="mid", color=clr, lw=2, label=label)

    if reco_vals is not None and len(reco_vals):
        _draw_hist(reco_vals, "reco", "lightskyblue", as_bar=True)
    if gen_vals is not None and len(gen_vals):
        _draw_hist(gen_vals, "gen", "tab:orange")
    _draw_hist(vals, "fitted", "tab:green")

    if pdf_fn is not None and pdf_params is not None:
        xfine = np.linspace(xlo, xhi, 600)
        yfine = pdf_fn(xfine)
        integ_plot = float(np.trapz(yfine, xfine))
        yfine = yfine / max(integ_plot, 1e-300)
        ax.plot(xfine, yfine, color="crimson", lw=2, label="input PDF")

    ax.set_xlabel(bname, fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_xlim(xlo, xhi)
    if bname == "kinfit_m_loss":
        ax.set_yscale("log")
        nonzero = [d[d > 0].min() for d in densities if (d > 0).any()]
        if nonzero:
            ax.set_ylim(bottom=0.5 * min(nonzero))
    else:
        ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, frameon=False)


# ── Normalized pull plots ─────────────────────────────────────────────────────
# pull(X) = (kinfit_X − gen_X) / σ_prior_per_bin
# For binned priors (jet/lep/MET resolutions and responses) σ comes from the
# per-pT-bin DCB2G/DCBER3G core. For un-binned priors (BES, ISR, m_loss) σ is
# the top-level prior σ. A pull histogram with σ ≈ 1 means the fit MAP is no
# better than guessing the prior mean; σ < 1 means the constraints add info on
# top of the prior; σ > 1 means the fit is worse than prior alone (bug).

# kinfit_branch → (gen_truth_branch, reco_binning_var_or_None)
# Truth conventions:
#   kinfit_s* : kinfit returns the multiplicative response s = p_reco / p_gen
#               → truth is the gen-level response branch (jet1_p_resp, etc.)
#   kinfit_t* : kinfit returns θ_shift = θ_reco − θ_gen
#               → truth is the gen-level θ residual (jet1_theta_resol, etc.)
#   kinfit_p* : same for φ
#   kinfit_bes_m_minus_ecm  : truth = gen_ee_m_minus_ecm (post-BES e+e- mass − ECM)
#   kinfit_bes_pz           : truth = gen_ee_pz
#   kinfit_m_loss (derived) : truth = gen_WW_m_minus_m_ee
#   kinfit_isr_{px,py,pz}   : truth = gen_isr_{px,py,pz}
PULL_TRUTH = {
    "kinfit_s1": ("jet1_p_resp",      "reco_jet1_p"),
    "kinfit_s2": ("jet2_p_resp",      "reco_jet2_p"),
    "kinfit_sl": ("lep_p_resp",       "reco_lep_p"),
    "kinfit_sn": ("met_p_resp",       "reco_met_p"),
    "kinfit_t1": ("jet1_theta_resol", "reco_jet1_p"),
    "kinfit_t2": ("jet2_theta_resol", "reco_jet2_p"),
    "kinfit_tl": ("lep_theta_resol",  "reco_lep_p"),
    "kinfit_tn": ("met_theta_resol",  "reco_met_p"),
    "kinfit_p1": ("jet1_phi_resol",   "reco_jet1_p"),
    "kinfit_p2": ("jet2_phi_resol",   "reco_jet2_p"),
    "kinfit_pl": ("lep_phi_resol",    "reco_lep_p"),
    "kinfit_pn": ("met_phi_resol",    "reco_met_p"),
    "kinfit_bes_m_minus_ecm": ("gen_ee_m_minus_ecm",  None),
    "kinfit_bes_pz":          ("gen_ee_pz",           None),
    "kinfit_m_loss":          ("gen_WW_m_minus_m_ee", None),
    "kinfit_isr_px":          ("gen_isr_px",          None),
    "kinfit_isr_py":          ("gen_isr_py",          None),
    "kinfit_isr_pz":          ("gen_isr_pz",          None),
}


def _per_event_prior_sigma(prior_params, bin_var_values=None):
    """Return per-event σ_prior from a JSON prior block.

    If `prior_params` has a `binned` dict and `bin_var_values` is supplied, look
    up σ per event by binning. Otherwise fall back to the top-level σ.
    Returns σ as a numpy array of the same shape as `bin_var_values` (or a
    scalar if no binning).
    """
    if "binned" in prior_params and bin_var_values is not None:
        edges_full = np.asarray(prior_params["binned"]["edges"], dtype=float)
        sigmas = np.array([b["sigma"] for b in prior_params["binned"]["bins"]], dtype=float)
        # searchsorted on the *interior* edges so bin index ∈ [0, len(bins)-1].
        idx = np.searchsorted(edges_full[1:-1], np.asarray(bin_var_values, dtype=float))
        idx = np.clip(idx, 0, len(sigmas) - 1)
        return sigmas[idx]
    return float(prior_params["sigma"])


def _prior_z_pdf(prior_params, bin_var_values, z_grid):
    """Prior PDF on z = (x − µ)/σ, evaluated on z_grid.

    Same change of variables the kinfit applies to its parameters in y-space.
    For per-bin priors, average over bins weighted by the per-event bin
    population so the curve matches what a "prior alone" pull histogram would
    look like for the actual event sample. Returns a normalized array of the
    same shape as z_grid (∫f dz = 1 over the displayed range), or None if the
    prior model isn't supported by `_make_pdf`.
    """
    z_grid = np.asarray(z_grid, dtype=float)

    def _bin_pdf(bp):
        mu, sg = float(bp["mu"]), abs(float(bp["sigma"]))
        pdf_fn = _make_pdf(bp)
        if pdf_fn is None:
            return None
        # Pull is defined as z = (fit − gen)/σ. In the prior-dominated limit
        # fit ≈ µ, so z = −(gen − µ)/σ — the prior in z-space mirrored about 0.
        # Sample at (µ − σ·z) so the overlay aligns with the pull shape directly.
        y = pdf_fn(mu - sg * z_grid) * sg
        integ = float(np.trapz(y, z_grid))
        if integ <= 0:
            return None
        return y / integ

    if "binned" in prior_params and bin_var_values is not None:
        edges_full = np.asarray(prior_params["binned"]["edges"], dtype=float)
        bins = prior_params["binned"]["bins"]
        idx = np.searchsorted(edges_full[1:-1], np.asarray(bin_var_values, dtype=float))
        idx = np.clip(idx, 0, len(bins) - 1)
        weights = np.bincount(idx, minlength=len(bins)).astype(float)
        if weights.sum() <= 0:
            return None
        weights /= weights.sum()
        result = np.zeros_like(z_grid)
        any_ok = False
        for w, bp in zip(weights, bins):
            if w == 0.0:
                continue
            yz = _bin_pdf(bp)
            if yz is None:
                continue
            result += w * yz
            any_ok = True
        return result if any_ok else None

    return _bin_pdf(prior_params)


def plot_pulls(ecm, raw_arrays, json_results):
    """Normalized pulls per nuisance: (kinfit_X − gen_X) / σ_prior.

    Bin σ_prior per event using the same `reco_*_p` binning used to fit the
    prior (per-pT-bin DCB2G). For un-binned priors (BES, ISR, m_loss) use the
    top-level σ. Overlay the prior PDF in z = (x − µ)/σ space (mirrored about
    0 so it aligns with the pull definition), the "prior-alone" baseline:
    pull narrower than overlay ⇒ constraint adds info; matches overlay ⇒
    prior dominates per event; wider ⇒ fit broken.
    """
    out_dir = f"{KINFIT_OUTDIR}/ecm{ecm}/pulls"
    os.makedirs(out_dir, exist_ok=True)

    NB = 100
    # Per-pull range is data-driven: bracket [q0.005, q0.995] with 5% padding
    # so 99% of events are visible plus a margin into the tails.
    Q_LO, Q_HI = 0.005, 0.995
    PAD_FRAC   = 0.05
    # Sanity cap on absolute extent so a single decade-out outlier doesn't
    # blow the plot up. ±20 σ_prior is well past anything physically meaningful
    # for a converged fit.
    CAP        = 20.0

    for bname, (gen_bname, bin_var) in PULL_TRUTH.items():
        if bname not in raw_arrays or gen_bname not in raw_arrays:
            continue
        prior_name = KINFIT_TO_RESOL.get(bname)
        if prior_name is None or prior_name not in json_results:
            continue
        prior_params = json_results[prior_name]

        fit_arr = np.asarray(raw_arrays[bname],     dtype=float).ravel()
        gen_arr = np.asarray(raw_arrays[gen_bname], dtype=float).ravel()
        if bin_var and bin_var in raw_arrays:
            bin_arr = np.asarray(raw_arrays[bin_var], dtype=float).ravel()
            sigma   = _per_event_prior_sigma(prior_params, bin_arr)
        else:
            sigma   = _per_event_prior_sigma(prior_params, None)

        residual = fit_arr - gen_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            pull = residual / sigma
        finite = np.isfinite(pull)
        pull   = pull[finite]
        if pull.size == 0:
            continue

        # Data-driven range: q0.005..q0.995 with 5% padding → ~99% events visible.
        q_lo_v, q_hi_v = float(np.quantile(pull, Q_LO)), float(np.quantile(pull, Q_HI))
        if q_hi_v <= q_lo_v:
            q_lo_v, q_hi_v = -1.0, 1.0
        span = q_hi_v - q_lo_v
        XLO  = max(q_lo_v - PAD_FRAC * span, -CAP)
        XHI  = min(q_hi_v + PAD_FRAC * span,  CAP)
        edges   = np.linspace(XLO, XHI, NB + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        bw      = edges[1] - edges[0]

        # In-range stats for the diagnostic numbers.
        in_range = pull[(pull >= XLO) & (pull <= XHI)]
        mu_pull  = float(np.mean(in_range)) if in_range.size else float("nan")
        sd_pull  = float(np.std(in_range))  if in_range.size else float("nan")
        n_total  = int(pull.size)
        n_in     = int(in_range.size)

        counts, _ = np.histogram(pull, bins=edges)
        density   = counts / max(counts.sum() * bw, 1.0)

        # Prior PDF in z = (x − µ)/σ space — same change of variables the
        # kinfit applies internally to its fit parameters. For binned priors,
        # averaged over bins by the per-event population.
        bin_arr_for_overlay = (
            np.asarray(raw_arrays[bin_var], dtype=float).ravel()
            if (bin_var and bin_var in raw_arrays) else None
        )
        ref_pdf = _prior_z_pdf(prior_params, bin_arr_for_overlay, centres)

        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.step(edges[:-1], density, where="post", color="tab:green", lw=1.5,
                label=f"pull (N={n_total}, in-range={n_in})")
        if ref_pdf is not None:
            label = f"prior in z-space ({prior_params['model']})"
            if bin_var:
                label += "  [bin-avg]"
            ax.plot(centres, ref_pdf, color="crimson", lw=1.8, label=label)
        ax.axvline(0.0, color="0.5", lw=0.8, ls="--")
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"({bname} − {gen_bname}) / σ_prior", fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        title = (f"{bname} pull  [ecm{ecm}]"
                 f"\nσ_prior from {prior_name}"
                 f"{'  [per-bin]' if bin_var else '  [pooled]'}"
                 f"   ⟨pull⟩={mu_pull:+.3f}, σ_pull={sd_pull:.3f}")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  pull plots → {out_dir}/")


def plot_per_ecm(ecm, branches_data, json_results):
    base = f"{KINFIT_OUTDIR}/ecm{ecm}"

    for bname in [b for b in KINFIT_BRANCHES if b in branches_data]:
        vals = branches_data[bname]
        resol_name = KINFIT_TO_RESOL.get(bname)
        pdf_fn = None
        pdf_params = None
        if resol_name and resol_name in json_results:
            pdf_params = json_results[resol_name]
            pdf_fn = _make_pdf(pdf_params)

        reco_bname = KINFIT_TO_RECO.get(bname)
        gen_bname  = KINFIT_TO_GEN.get(bname)
        reco_vals  = branches_data.get(reco_bname) if reco_bname else None
        gen_vals   = branches_data.get(gen_bname)  if gen_bname  else None

        out_dir = f"{base}/{_subdir(bname)}"
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        _plot_branch(ax, bname, vals, ecm, pdf_fn=pdf_fn, pdf_params=pdf_params,
                     reco_vals=reco_vals, gen_vals=gen_vals)
        title = f"{bname}  [ecm{ecm}]"
        if resol_name:
            title += f"\ninput PDF: {resol_name}"
            if resol_name in json_results:
                title += f"  ({json_results[resol_name]['model']}, χ²/ndf={json_results[resol_name]['chi2_ndof']:.2f})"
        ax.set_title(title, fontsize=10)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  per-ECM plots → {base}/")


# ── Chi2-slice comparison plots ───────────────────────────────────────────────

def plot_chi2_slices(ecm, raw_arrays, json_results):
    """Per fit_vs_reco_gen branch: 3 panels (reco | gen | fit), each showing
    3 overlaid equal-N NLL slices (NLL = chi2/2)."""
    chi2  = raw_arrays.get("kinfit_chi2")
    if chi2 is None:
        return

    out_dir = f"{KINFIT_OUTDIR}/ecm{ecm}/nll_slices"
    os.makedirs(out_dir, exist_ok=True)

    SLICE_COLORS = ["tab:blue", "tab:orange", "tab:red"]

    for bname in KINFIT_BRANCHES:
        reco_bname = KINFIT_TO_RECO.get(bname)
        gen_bname  = KINFIT_TO_GEN.get(bname)
        if not (reco_bname and gen_bname):
            continue
        if bname not in raw_arrays or reco_bname not in raw_arrays or gen_bname not in raw_arrays:
            continue

        fit_arr  = np.asarray(raw_arrays[bname],      dtype=float)
        reco_arr = np.asarray(raw_arrays[reco_bname], dtype=float)
        gen_arr  = np.asarray(raw_arrays[gen_bname],  dtype=float)
        nll_arr  = np.asarray(chi2,                   dtype=float)

        mask = (np.isfinite(fit_arr) & np.isfinite(reco_arr) &
                np.isfinite(gen_arr) & np.isfinite(nll_arr))

        fit_arr  = fit_arr[mask]
        reco_arr = reco_arr[mask]
        gen_arr  = gen_arr[mask]
        nll_arr  = nll_arr[mask]

        if len(nll_arr) < 30:
            continue

        q33, q67 = np.percentile(nll_arr, [100.0 / 3, 200.0 / 3])
        slice_masks = [
            nll_arr < q33,
            (nll_arr >= q33) & (nll_arr < q67),
            nll_arr >= q67,
        ]
        slice_labels = [
            f"NLL < {q33:.1f}",
            f"{q33:.1f} ≤ NLL < {q67:.1f}",
            f"NLL ≥ {q67:.1f}",
        ]

        nbins, xlo, xhi = _binning(bname, np.concatenate([fit_arr, reco_arr, gen_arr]))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), layout="constrained")
        fig.suptitle(f"{bname}  [ecm{ecm}]  — equal-N NLL slices", fontsize=11)

        for ax, (arr, panel_label) in zip(axes, [
            (reco_arr, "reco"),
            (gen_arr,  "gen"),
            (fit_arr,  "fitted"),
        ]):
            for sl_mask, sl_label, clr in zip(slice_masks, slice_labels, SLICE_COLORS):
                v_c = arr[sl_mask]
                v_c = v_c[(v_c >= xlo) & (v_c <= xhi)]
                counts, edges = np.histogram(v_c, bins=nbins, range=(xlo, xhi))
                bw = np.diff(edges)[0]
                density = counts / max(counts.sum() * bw, 1e-300)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centers, density, where="mid", color=clr, lw=2,
                        label=f"{sl_label}  (N={int(sl_mask.sum())})")
            ax.set_title(panel_label, fontsize=11)
            ax.set_xlabel(bname, fontsize=10)
            ax.set_ylabel("Probability density", fontsize=10)
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, frameon=False)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  NLL slice plots → {out_dir}/")


# ── Per-status comparison plots ──────────────────────────────────────────────
# Migrad status codes (see WWKinReco.h: KinFitResult::status):
#   0 = OK, 1 = PD-forced cov, 2 = Hesse failed, 3 = EDM>tol,
#   4 = max calls, 5 = other, −1 = early-return (invalid input).
# Two buckets shown: converged (0|1) vs EDM>tol (3). Codes 4/5 sit below 0.05 %
# at all ECMs and are dropped. Pull branches and other branches go to separate
# subdirectories (by_status/pulls/ and by_status/other/).

_STATUS_BUCKETS = [
    (lambda s: (s == 0) | (s == 1), "status=0|1 (converged)", "tab:blue"),
    (lambda s: s == 3,              "status=3 (EDM>tol)",     "tab:red"),
]


def plot_by_status(ecm, raw_arrays, json_results):
    """Per-branch histograms split by Migrad status, three subdirs:
    - by_status/posteriors/ : MAP histograms in physical units, prior PDF
                              overlay (only for nuisances with priors).
    - by_status/pulls/      : (kinfit_X − gen_X)/σ_prior split by status,
                              with prior PDF in z-space mirrored about 0.
    - by_status/other/      : non-prior branches (kinematics, chi², …).
    """
    status = raw_arrays.get("kinfit_status")
    if status is None:
        print(f"  [ecm{ecm}]  kinfit_status missing — skipping by-status plots")
        return

    base = f"{KINFIT_OUTDIR}/ecm{ecm}/by_status"
    os.makedirs(f"{base}/posteriors", exist_ok=True)
    os.makedirs(f"{base}/pulls",      exist_ok=True)
    os.makedirs(f"{base}/other",      exist_ok=True)

    status = np.asarray(status, dtype=int)

    # ── Posteriors and "other" branches: physical units, prior PDF overlay ──
    for bname in KINFIT_BRANCHES:
        if bname in ("kinfit_status", "kinfit_valid"):
            continue
        if bname not in raw_arrays:
            continue
        vals = np.asarray(raw_arrays[bname], dtype=float)
        finite = np.isfinite(vals)
        # Only count status≥0 events (status=−1 are early-returns with placeholder values).
        active = finite & (status >= 0)
        if active.sum() < 30:
            continue

        is_posterior = bname in _PULL_BRANCHES
        resol_name   = KINFIT_TO_RESOL.get(bname) if is_posterior else None
        pdf_params   = json_results.get(resol_name) if resol_name else None
        pdf_fn       = _make_pdf(pdf_params) if pdf_params else None

        nbins, xlo, xhi = _binning(bname, vals[active], pdf_params=pdf_params)

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for pred, lbl, clr in _STATUS_BUCKETS:
            m = pred(status) & active
            n = int(m.sum())
            if n < 1:
                continue
            v = vals[m]
            v = v[(v >= xlo) & (v <= xhi)]
            counts, edges = np.histogram(v, bins=nbins, range=(xlo, xhi))
            bw = np.diff(edges)[0]
            density = counts / max(counts.sum() * bw, 1e-300)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.step(centers, density, where="mid", color=clr, lw=2,
                    label=f"{lbl}  (N={n})")

        if pdf_fn is not None:
            xfine = np.linspace(xlo, xhi, 600)
            yfine = pdf_fn(xfine)
            integ = float(np.trapz(yfine, xfine))
            yfine = yfine / max(integ, 1e-300)
            ax.plot(xfine, yfine, color="crimson", lw=1.8, label="prior PDF")

        ax.set_xlabel(bname, fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        ax.set_title(f"{bname}  [ecm{ecm}]  — split by Migrad status", fontsize=10)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, frameon=False)

        sub = "posteriors" if is_posterior else "other"
        for fmt in ("png", "pdf"):
            fig.savefig(f"{base}/{sub}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    # ── Normalised pulls split by status, prior overlay in z-space ──
    NB = 100
    Q_LO, Q_HI = 0.005, 0.995
    PAD_FRAC   = 0.05
    CAP        = 20.0

    for bname, (gen_bname, bin_var) in PULL_TRUTH.items():
        if bname not in raw_arrays or gen_bname not in raw_arrays:
            continue
        prior_name = KINFIT_TO_RESOL.get(bname)
        if prior_name is None or prior_name not in json_results:
            continue
        prior_params = json_results[prior_name]

        fit_arr = np.asarray(raw_arrays[bname],     dtype=float).ravel()
        gen_arr = np.asarray(raw_arrays[gen_bname], dtype=float).ravel()
        if bin_var and bin_var in raw_arrays:
            bin_arr = np.asarray(raw_arrays[bin_var], dtype=float).ravel()
            sigma   = _per_event_prior_sigma(prior_params, bin_arr)
        else:
            bin_arr = None
            sigma   = _per_event_prior_sigma(prior_params, None)

        with np.errstate(divide="ignore", invalid="ignore"):
            pull = (fit_arr - gen_arr) / sigma
        valid = np.isfinite(pull) & (status >= 0)
        if valid.sum() < 30:
            continue

        # Range from the full active sample so both status buckets share the axis.
        pull_active = pull[valid]
        q_lo_v, q_hi_v = float(np.quantile(pull_active, Q_LO)), float(np.quantile(pull_active, Q_HI))
        if q_hi_v <= q_lo_v:
            q_lo_v, q_hi_v = -1.0, 1.0
        span = q_hi_v - q_lo_v
        XLO  = max(q_lo_v - PAD_FRAC * span, -CAP)
        XHI  = min(q_hi_v + PAD_FRAC * span,  CAP)
        edges   = np.linspace(XLO, XHI, NB + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        bw      = edges[1] - edges[0]

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for pred, lbl, clr in _STATUS_BUCKETS:
            m = pred(status) & valid
            n = int(m.sum())
            if n < 1:
                continue
            p_in = pull[m]
            p_in = p_in[(p_in >= XLO) & (p_in <= XHI)]
            sd_p = float(np.std(p_in)) if p_in.size else float("nan")
            counts, _ = np.histogram(pull[m], bins=edges)
            density   = counts / max(counts.sum() * bw, 1.0)
            ax.step(edges[:-1], density, where="post", color=clr, lw=1.5,
                    label=f"{lbl}  (N={n}, σ_pull={sd_p:.2f})")

        ref_pdf = _prior_z_pdf(prior_params, bin_arr[valid] if bin_arr is not None else None, centres)
        if ref_pdf is not None:
            label = f"prior in z-space ({prior_params['model']})"
            if bin_var:
                label += "  [bin-avg]"
            ax.plot(centres, ref_pdf, color="crimson", lw=1.8, label=label)

        ax.axvline(0.0, color="0.5", lw=0.8, ls="--")
        ax.set_xlim(XLO, XHI)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(f"({bname} − {gen_bname}) / σ_prior", fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        ax.set_title(f"{bname} pull  [ecm{ecm}]  — split by Migrad status", fontsize=10)
        ax.legend(fontsize=9, frameon=False, loc="upper right")
        for fmt in ("png", "pdf"):
            fig.savefig(f"{base}/pulls/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  [ecm{ecm}]  by-status plots → {base}/{{posteriors,pulls,other}}/")


# ── ECM comparison plots ──────────────────────────────────────────────────────

def plot_kinfit_ecm_comparison(all_data, all_json):
    branch_names = [b for b in KINFIT_BRANCHES
                    if any(b in all_data[e] for e in ECM_LIST if e in all_data)]

    for bname in branch_names:
        out_dir = f"{KINFIT_OUTDIR}/ecm_comparison/{_subdir(bname)}"
        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

        resol_name = KINFIT_TO_RESOL.get(bname)

        pdf_params_ref = None
        if resol_name:
            for _e in ECM_LIST:
                if _e in all_json and resol_name in all_json[_e]:
                    pdf_params_ref = all_json[_e][resol_name]
                    break

        all_vals_combined = np.concatenate([
            all_data[ecm][bname] for ecm in ECM_LIST
            if ecm in all_data and bname in all_data[ecm]
        ]) if any(ecm in all_data and bname in all_data[ecm] for ecm in ECM_LIST) else np.array([])
        nbins, xlo, xhi = _binning(bname,
                                    all_vals_combined if len(all_vals_combined) else None,
                                    pdf_params=pdf_params_ref)

        for ecm in ECM_LIST:
            if ecm not in all_data or bname not in all_data[ecm]:
                continue
            vals = all_data[ecm][bname]
            vals_c = vals[(vals >= xlo) & (vals <= xhi)]
            counts, edges = np.histogram(vals_c, bins=nbins, range=(xlo, xhi))
            centers = 0.5 * (edges[:-1] + edges[1:])
            bw = np.diff(edges)[0]
            density = counts / max(counts.sum() * bw, 1e-300)
            ax.step(centers, density, where="mid",
                    color=ECM_COLORS[ecm], lw=2, label=rf"$\sqrt{{s}}$ = {ecm} GeV")

            if resol_name and ecm in all_json and resol_name in all_json[ecm]:
                p = all_json[ecm][resol_name]
                pdf_fn = _make_pdf(p)
                if pdf_fn is not None:
                    xfine = np.linspace(xlo, xhi, 600)
                    yfine = pdf_fn(xfine)
                    integ_plot = float(np.trapz(yfine, xfine))
                    yfine = yfine / max(integ_plot, 1e-300)
                    ax.plot(xfine, yfine,
                            color=ECM_COLORS[ecm], lw=1.5, ls="--",
                            label=f"PDF ecm{ecm}")

        ax.set_xlabel(bname, fontsize=11)
        ax.set_ylabel("Probability density", fontsize=11)
        title = bname
        if resol_name:
            title += f"  (input PDF: {resol_name})"
        ax.set_title(title, fontsize=11)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, frameon=False)

        for fmt in ("png", "pdf"):
            fig.savefig(f"{out_dir}/{bname}.{fmt}", dpi=150)
        plt.close(fig)

    print(f"  ECM comparison plots → {KINFIT_OUTDIR}/ecm_comparison/")


def _load_ecm(ecm):
    """Load one ECM's TTree + JSON priors. Returns (ecm, branches_data,
    raw_data, json_dict, log) with branches_data filtered to finite entries
    and raw_data preserving the full (NaN-aware) arrays needed by plot_pulls /
    plot_chi2_slices / plot_by_status. Returns (..., None, None, None, log) on
    failure so the dispatcher can skip this ECM.
    """
    log = []
    infile = INFILE_TMPL.format(ecm=ecm)
    if not os.path.exists(infile):
        log.append(f"WARNING: {infile} not found — skipping ecm{ecm}")
        return ecm, None, None, None, log
    if os.path.getsize(infile) < 1_000_000:
        log.append(f"WARNING: {infile} only {os.path.getsize(infile)} bytes (placeholder?) — skipping ecm{ecm}")
        return ecm, None, None, None, log

    log.append(f"\n{'='*55}\nECM {ecm} GeV  —  {infile}\n{'='*55}")

    try:
        f = uproot.open(infile)
        tree = f[TREE_NAME]
    except Exception as e:
        log.append(f"WARNING: {infile} present but events tree not yet finalized ({e}) — skipping ecm{ecm}")
        return ecm, None, None, None, log
    with f:
        available = set(tree.keys())
        to_load    = [b for b in KINFIT_BRANCHES if b in available]
        missing    = [b for b in KINFIT_BRANCHES if b not in available and b not in _DERIVED_PULLS]
        extra_load = [b for b in EXTRA_BRANCHES  if b in available and b not in set(to_load)]
        extra_miss = [b for b in EXTRA_BRANCHES  if b not in available]
        if missing:
            log.append(f"  WARNING: kinfit branches not in tree: {missing}")
        if extra_miss:
            log.append(f"  WARNING: comparison branches not in tree (re-run treemaker?): {extra_miss}")
        raw = tree.arrays(to_load + extra_load, library="np")

    raw_data = {}
    branches_data = {}
    for bname in to_load + extra_load:
        arr = np.asarray(raw[bname], dtype=float).ravel()
        raw_data[bname] = arr
        branches_data[bname] = arr[np.isfinite(arr)]

    for dname, dfn in _DERIVED_PULLS.items():
        try:
            arr = np.asarray(dfn(raw_data, ecm), dtype=float).ravel()
        except KeyError as e:
            log.append(f"  WARNING: derived pull {dname} skipped — missing input {e}")
            continue
        raw_data[dname] = arr
        branches_data[dname] = arr[np.isfinite(arr)]

    json_path = JSON_TMPL.format(ecm=ecm)
    if os.path.exists(json_path):
        with open(json_path) as fj:
            json_dict = json.load(fj)
        log.append(f"  Loaded JSON: {json_path}")
    else:
        log.append(f"  WARNING: JSON not found ({json_path}) — no PDF overlay for ecm{ecm}")
        json_dict = {}

    return ecm, branches_data, raw_data, json_dict, log


# 4 independent plot phases per ECM. With 3 ECMs × 4 phases = 12 tasks,
# distributed across a 12-worker pool, we saturate `--ncores 12` cleanly.
_PLOT_PHASES = ("per_ecm", "pulls", "chi2_slices", "by_status")


def _plot_phase(args):
    """Worker: dispatch one (ecm, phase) tuple. Args bundle the loaded data
    so each task is self-contained — pickling cost is one ECM's arrays per
    task (~40 MB), amortized against ~30 s of plotting per phase."""
    ecm, phase, branches_data, raw_data, json_dict = args
    if phase == "per_ecm":
        plot_per_ecm(ecm, branches_data, json_dict)
    elif phase == "pulls":
        plot_pulls(ecm, raw_data, json_dict)
    elif phase == "chi2_slices":
        plot_chi2_slices(ecm, raw_data, json_dict)
    elif phase == "by_status":
        plot_by_status(ecm, raw_data, json_dict)
    return ecm, phase


def run_kinfit_vars():
    os.makedirs(KINFIT_OUTDIR, exist_ok=True)

    # Phase 1: load all ECMs in parallel (one worker per ECM — IO/CPU bound
    # roughly equally, no benefit from over-subscribing here).
    all_data    = {}
    all_json    = {}
    all_raw     = {}
    with ProcessPoolExecutor(max_workers=len(ECM_LIST)) as load_pool:
        for ecm, branches_data, raw_data, json_dict, log in load_pool.map(_load_ecm, ECM_LIST):
            for line in log:
                print(line)
            if branches_data is None:
                continue
            all_data[ecm] = branches_data
            all_raw[ecm]  = raw_data
            all_json[ecm] = json_dict

    # Phase 2: distribute (ecm, plot_phase) tasks across up to 12 workers.
    tasks = [
        (ecm, phase, all_data[ecm], all_raw[ecm], all_json[ecm])
        for ecm in all_data
        for phase in _PLOT_PHASES
    ]
    n_workers = min(12, max(len(tasks), 1))
    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers) as plot_pool:
            for ecm, phase in plot_pool.map(_plot_phase, tasks):
                print(f"  [ecm{ecm}]  {phase} done")

    plot_kinfit_ecm_comparison(all_data, all_json)
    print(f"\nDone. All plots in {KINFIT_OUTDIR}/")
    publish(KINFIT_OUTDIR, os.environ.get("KINFIT_PUBSUB", "kinfit_vars"))


# =============================================================================
# Stage 3 — kinfit correlation matrix
# =============================================================================

CORR_OUTDIR = "outputs/plots/kinfit_correlations"

# Index → name in the row-major 16x16 stored as `kinfit_corr`. Mirrors
# FCCAnalyses::WWFunctions::KF_PARAM_NAMES (WWKinReco.h) — must stay in sync.
CORR_PARAM_NAMES = [
    "mW", "gW", "s1", "s2", "sl", "sn",
    "t1", "t2", "tn", "p1", "p2", "pn",
    "tl", "pl", "bes_m", "bes_pz",
]
N_CORR_PAR = len(CORR_PARAM_NAMES)
CORR_SUMMARY_KINDS = ("mean", "median", "mode")
_MODE_EDGES   = np.linspace(-1.0, 1.0, 101)
_MODE_CENTRES = 0.5 * (_MODE_EDGES[:-1] + _MODE_EDGES[1:])


def _load_corr_matrices(ecm):
    """Return (strict, loose_only) (N, 16, 16) post-fit correlation arrays for
    one ECM. strict = kinfit_valid==1; loose_only = kinfit_valid_loose==1 AND
    kinfit_valid==0. Either can be None (missing file/branch, or empty mask)."""
    infile = INFILE_TMPL.format(ecm=ecm)
    if not os.path.exists(infile) or os.path.getsize(infile) < 1_000_000:
        return None, None

    try:
        with uproot.open(infile) as f:
            tree = f[TREE_NAME]
            if "kinfit_corr" not in tree.keys():
                print(f"  WARNING [ecm{ecm}]: kinfit_corr branch missing")
                return None, None
            arrs = tree.arrays(
                ["kinfit_corr", "kinfit_valid", "kinfit_valid_loose"],
                library="np")
    except Exception as e:
        print(f"  WARNING [ecm{ecm}]: cannot read kinfit_corr ({e})")
        return None, None

    flat         = arrs["kinfit_corr"]
    valid_strict = arrs["kinfit_valid"].astype(bool)
    loose_only   = arrs["kinfit_valid_loose"].astype(bool) & ~valid_strict

    def _to_mat(mask):
        rows = flat[mask]
        if len(rows) == 0:
            return None
        return (np.asarray(rows.tolist(), dtype=np.float32)
                  .reshape(-1, N_CORR_PAR, N_CORR_PAR))
    return _to_mat(valid_strict), _to_mat(loose_only)


def _summary_matrix(mats, kind):
    """Per-pair scalar summary across events. NaNs (fixed-gW row/col, invalid
    cov slots) are ignored per cell."""
    if kind == "mean":
        return np.nanmean(mats, axis=0)
    if kind == "median":
        return np.nanmedian(mats, axis=0)
    if kind == "mode":
        # Per-cell histogram peak on [-1, 1] in 100 bins. Constant cells (e.g.
        # diagonal=1) bypass the histogram so the result is exact.
        out = np.full((N_CORR_PAR, N_CORR_PAR), np.nan, dtype=float)
        for i in range(N_CORR_PAR):
            for j in range(N_CORR_PAR):
                v = mats[:, i, j]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                if v.std() == 0:
                    out[i, j] = float(v[0])
                    continue
                counts, _ = np.histogram(v, bins=_MODE_EDGES)
                out[i, j] = _MODE_CENTRES[int(np.argmax(counts))]
        return out
    raise ValueError(f"unknown summary kind: {kind}")


def _plot_corr_heatmap(M, fname, title, cbar_label, outdir, vmax=1.0):
    """Render a labelled 16×16 heatmap of M. vmax sets the symmetric colour
    range; per-cell text colour flips to white above |M|/vmax > 0.5."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    ax.set_xticks(np.arange(N_CORR_PAR))
    ax.set_yticks(np.arange(N_CORR_PAR))
    ax.set_xticklabels(CORR_PARAM_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CORR_PARAM_NAMES, fontsize=9)
    text_thresh = 0.5 * vmax
    for i in range(N_CORR_PAR):
        for j in range(N_CORR_PAR):
            v = M[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(v) > text_thresh else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(f"{outdir}/{fname}.{fmt}", dpi=150)
    plt.close(fig)


def _plot_corr_distributions(strict, loose_only, ecm, outdir):
    """16x16 grid: strict (filled blue) vs loose-only (orange step) overlaid
    per cell. Lower triangle hidden (matrix is symmetric)."""
    fig, axes = plt.subplots(N_CORR_PAR, N_CORR_PAR,
                              figsize=(2 * N_CORR_PAR, 2 * N_CORR_PAR),
                              sharex=True, sharey=False)
    for i in range(N_CORR_PAR):
        for j in range(N_CORR_PAR):
            ax = axes[i, j]
            ax.set_xlim(-1.0, 1.0)
            if j < i:
                ax.set_visible(False)
                continue
            if i == 0:
                ax.set_title(CORR_PARAM_NAMES[j], fontsize=8)
            if j == N_CORR_PAR - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(CORR_PARAM_NAMES[i], fontsize=8, rotation=270, va="bottom")
            ax.tick_params(labelsize=6)
            if i == j:
                ax.text(0.5, 0.5, "1", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="grey")
                ax.set_yticks([])
                continue
            for arr, label, color, fill in (
                (strict,     "strict",     "tab:blue",   True),
                (loose_only, "loose-only", "tab:orange", False),
            ):
                if arr is None:
                    continue
                v = arr[:, i, j]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                ax.hist(v, bins=50, range=(-1.0, 1.0), density=True,
                         histtype="stepfilled" if fill else "step",
                         color=color, alpha=0.45 if fill else 1.0,
                         edgecolor=color, lw=1.0, label=label)
            ax.text(0.97, 0.97,
                    rf"$\rho$({CORR_PARAM_NAMES[i]}, {CORR_PARAM_NAMES[j]})",
                    transform=ax.transAxes, fontsize=6.5,
                    ha="right", va="top")
            if i == 0 and j == 1:
                ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(rf"Per-pair ρ — strict vs loose-only — $\sqrt{{s}}$ = {ecm} GeV",
                  fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    for fmt in ("png", "pdf"):
        fig.savefig(f"{outdir}/distributions_ecm{ecm}.{fmt}", dpi=120)
    plt.close(fig)


def run_kinfit_correlations():
    os.makedirs(CORR_OUTDIR, exist_ok=True)
    cmp_outdir = os.path.join(CORR_OUTDIR, "strict_vs_loose")
    os.makedirs(cmp_outdir, exist_ok=True)

    n_done = 0
    for ecm in ECM_LIST:
        strict, loose_only = _load_corr_matrices(ecm)
        if strict is None or len(strict) == 0:
            print(f"  [ecm{ecm}] no strict-valid kinfit_corr data — skipping")
            continue
        n_loose = len(loose_only) if loose_only is not None else 0
        print(f"  [ecm{ecm}] strict={len(strict)}  loose-only={n_loose}")

        ecm_label = rf"$\sqrt{{s}}$ = {ecm} GeV"
        _plot_corr_distributions(strict, loose_only, ecm, CORR_OUTDIR)
        for kind in CORR_SUMMARY_KINDS:
            M_s = _summary_matrix(strict, kind)
            _plot_corr_heatmap(
                M_s, f"heatmap_{kind}_ecm{ecm}",
                f"Post-fit correlations ({kind}) — {ecm_label}",
                rf"$\langle\rho\rangle_{{\rm {kind}}}$",
                CORR_OUTDIR)
            if loose_only is None:
                continue
            M_l = _summary_matrix(loose_only, kind)
            _plot_corr_heatmap(
                M_l, f"heatmap_{kind}_loose_ecm{ecm}",
                f"Post-fit correlations ({kind}, loose-only) — {ecm_label}",
                rf"$\langle\rho\rangle_{{\rm {kind}}}$",
                cmp_outdir)
            D = M_l - M_s
            finite = D[np.isfinite(D)]
            vmax_d = max(0.05, float(np.nanmax(np.abs(finite))) if len(finite) else 0.05)
            _plot_corr_heatmap(
                D, f"heatmap_diff_{kind}_ecm{ecm}",
                f"Δ post-fit correlations ({kind}, loose − strict) — {ecm_label}",
                rf"$\Delta\langle\rho\rangle_{{\rm {kind}}}$ (loose − strict)",
                cmp_outdir, vmax=vmax_d)
        print(f"  [ecm{ecm}] correlation plots done")
        # Free per-ECM matrices so peak memory is one-ECM, not all-ECMs.
        del strict, loose_only
        n_done += 1

    if n_done == 0:
        print("No correlation data found — skipping stage 3")
        return

    print(f"Done. Correlation plots in {CORR_OUTDIR}/")
    print(f"      strict-vs-loose 2D heatmaps in {cmp_outdir}/")
    publish(CORR_OUTDIR, os.environ.get("CORR_PUBSUB", "kinfit_correlations"))


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print(" Stage 1 / 3 — mW overlay")
    print("=" * 60)
    run_mW_overlay()

    print()
    print("=" * 60)
    print(" Stage 2 / 3 — kinfit variables")
    print("=" * 60)
    run_kinfit_vars()

    print()
    print("=" * 60)
    print(" Stage 3 / 3 — kinfit correlations")
    print("=" * 60)
    run_kinfit_correlations()


if __name__ == "__main__":
    main()
