# Step 2 of 2 — WW → 4q (fully hadronic) kinematic fit.
# Re-clusters p8_ee_WW_ecm160 into 4 jets and runs the best-pairing kinematic
# fit (WWFunctions/WWKinReco4q.h): all 3 jet→W partitions, keep the lowest χ².
# Requires kinfit_inputs_4q/dcb_params_4q.h (fit_resolutions.py WW_CHANNEL=4q)
# and kinfit_inputs/logz_table.bin to exist before compiling.
import os, ROOT
import treemaker_common as tc

# Sample override. Default = the WW signal. Set WW_SAMPLE=p8_ee_ZZ_ecm160 (and
# WW_GENTRUTH=0) to apply the WW-hypothesis fit to a background sample as a χ²
# discriminant — ZZ→4q has no W gen-truth, so the gen chain is skipped.
WW_SAMPLE   = os.environ.get("WW_SAMPLE", "p8_ee_WW_ecm160")
WW_GENTRUTH = os.environ.get("WW_GENTRUTH", "1").strip().lower() in ("1", "true", "yes")

# Stage 1 subsample (override via WW_FRACTION). The step2 fit is heavier than
# step1, so default to a smaller slice; bump for higher-statistics validation.
_frac = float(os.environ.get("WW_FRACTION", "0.002"))
processList = {
    WW_SAMPLE: {"fraction": _frac, "crossSection": 1},
}

# gW handling for the 4q fit (mW is always Gaussian-constrained at the SM value).
#   "fixed" → pin gW;  "constrained" → Gaussian prior (default);  "free" → no prior.
KIN_FIT_GW_MODE = os.environ.get("KF_GW_MODE", "constrained")
# Binned (per-jet-p) pooled jet priors via pick_bin (default), or inclusive scalars.
KIN_FIT_USE_BINNED = os.environ.get("KF_USE_BINNED", "true").lower() in ("1", "true", "yes")
# Closure mode: "mloss" (default, invariant-mass m_loss + barrier) or "kfit"
# (symmetric Gaussian energy-balance closure). σ_R tunable via env KF4Q_SIGMA_R.
KIN_FIT_ISR_MODE = os.environ.get("KF4Q_ISR_MODE", "mloss")
# DIAGNOSTIC: full fit on all 3 pairings + per-term χ² breakdown (KF4Q_DIAG=1).
KIN_FIT_DIAG = os.environ.get("KF4Q_DIAG", "0").strip().lower() in ("1", "true", "yes")
# Reco-level genuine-4-jet selection: require sqrt(d_45) < this [GeV] (0 disables).
# Rejects hard-5th-jet/radiative events for a well-defined 4-jet system.
WW_SQRTD45_MAX = float(os.environ.get("WW_SQRTD45_MAX", "7.0"))

channel = "had"

prodTag      = "FCCee/winter2023/IDEA/"
_tag = os.environ.get("WW_TAG", "").strip()
_default = "outputs/treemaker/4q/step2/{}{}".format(channel, "_" + _tag if _tag else "")
outputDir    = os.environ.get("STEP2_OUTDIR", _default)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h",
                "WWFunctions/WWKinReco4q.h"]

# Always-present reco + kinfit branches.
all_branches = [
    # ── reco jet kinematics (4 jets) ───────────────────────────────────
    "reco_jet1_p", "reco_jet1_pt", "reco_jet1_theta", "reco_jet1_phi", "reco_jet1_costheta", "reco_jet1_mass",
    "reco_jet2_p", "reco_jet2_pt", "reco_jet2_theta", "reco_jet2_phi", "reco_jet2_costheta", "reco_jet2_mass",
    "reco_jet3_p", "reco_jet3_pt", "reco_jet3_theta", "reco_jet3_phi", "reco_jet3_costheta", "reco_jet3_mass",
    "reco_jet4_p", "reco_jet4_pt", "reco_jet4_theta", "reco_jet4_phi", "reco_jet4_costheta", "reco_jet4_mass",
    # ── Durham jet-resolution scales ───────────────────────────────────
    "d_23", "d_34", "d_45",
    # ── kinfit4q: pairing + χ² discriminant ────────────────────────────
    "kinfit4q_pairing",
    "kinfit4q_chi2_p0", "kinfit4q_chi2_p1", "kinfit4q_chi2_p2",
    "kinfit4q_dchi2", "kinfit4q_n_pairings_valid",
    "kinfit4q_chi2", "kinfit4q_chi2_ndof",
    "kinfit4q_valid", "kinfit4q_valid_loose", "kinfit4q_status", "kinfit4q_edm",
    "kinfit4q_winner_pass", "kinfit4q_mW", "kinfit4q_gW",
    # ── kinfit4q post-fit W / WW kinematics ────────────────────────────
    "kinfit4q_Wa_m", "kinfit4q_Wb_m", "kinfit4q_Wa_p", "kinfit4q_Wb_p",
    "kinfit4q_WW_m", "kinfit4q_WW_pt", "kinfit4q_WW_pz",
]

# Gen-truth branches (signal only; skipped for background/data-like samples).
if WW_GENTRUTH:
    all_branches += [
        "gen_quark1_p", "gen_quark2_p", "gen_quark3_p", "gen_quark4_p",
        "jet1_matched_q_dR", "jet2_matched_q_dR", "jet3_matched_q_dR", "jet4_matched_q_dR",
        "gen_W1_m", "gen_W2_m", "gen_WW_m", "gen_WW_m_minus_ecm",
        "gen_WW_m_minus_m_ee",
        # ISR cross-check: electron-based (genstat==21) vs WW-system proxy.
        "gen_isr_px", "gen_isr_py", "gen_isr_pz",
        "gen_isr_WW_px", "gen_isr_WW_py", "gen_isr_WW_pz",
        "jet1_p_resp", "jet2_p_resp", "jet3_p_resp", "jet4_p_resp",
        "gen_pairing_true", "kinfit4q_pairing_correct",
    ]

jetClusteringHelper = None

_dataset_iter = iter(processList.keys())

class RDFanalysis:

    def analysers(df):
        global jetClusteringHelper

        _dataset = next(_dataset_iter)
        _ecm = tc.parse_ecm(_dataset)
        print(f"[treemaker 4q step2] dataset={_dataset}  ecm={_ecm}")
        if str(_ecm) not in tc.AVAILABLE_ECM:
            raise ValueError(f"ecm={_ecm} parsed from '{_dataset}' not in AVAILABLE_ECM={tc.AVAILABLE_ECM}")
        ROOT.gInterpreter.ProcessLine(
            f'FCCAnalyses::WWFunctions::setKinFitParams4q({_ecm}, '
            f'{"true" if KIN_FIT_USE_BINNED else "false"}, '
            f'"{KIN_FIT_ISR_MODE}");')
        ROOT.gInterpreter.ProcessLine(
            'std::cout << "[DEBUG 4q step2] ECM from WWFunctions = " << FCCAnalyses::WWFunctions::ECM << std::endl;')

        df = tc.select_isoleps(df)
        df = tc.apply_channel_filter(df, channel)
        df, jetClusteringHelper = tc.cluster_jets_4q(df)
        df = tc.define_reco_jets_kinematics_4q(df)
        df = tc.filter_genuine_4jet(df, WW_SQRTD45_MAX)   # genuine 4-jet selection

        # Gen-truth chain (signal only). For background/data-like samples
        # (WW_GENTRUTH=0) we apply the WW-hypothesis fit directly to the 4 reco
        # jets with no gen requirement.
        if WW_GENTRUTH:
            df = tc.select_gen_fromW(df)
            df = tc.define_gen_kinematics_4q(df)
            df = tc.define_beam_kinematics(df, post_isr_mode="p8", gen_ww_p4="WW_4q_gen")
            df = tc.match_jets_to_quarks_4q(df)
            df = tc.define_resolutions_4q(df)

        df = tc.run_kinfit_4q(df, gw_mode=KIN_FIT_GW_MODE, with_truth=WW_GENTRUTH)

        if KIN_FIT_DIAG:
            df, _diag_branches = tc.run_kinfit_4q_diag(df, gw_mode=KIN_FIT_GW_MODE)
            all_branches.extend(_diag_branches)

        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print()
        print()

        return df

    def output():
        return all_branches
