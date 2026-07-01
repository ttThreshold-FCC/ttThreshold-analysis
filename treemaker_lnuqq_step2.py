# Step 2 of 2 — runs kinematic fit using DCB params from fit_dcb_resolutions.py.
# Requires kinfit_inputs/dcb_params.h + kinfit_inputs/logz_table.bin to exist
# before compiling/running (produced by fit_resolutions.py and
# tools/build_logz_table respectively).
import os, urllib, ROOT
import treemaker_common as tc

processList = {
    "wzp6_ee_munumuqq_noCut_ecm160": {"fraction": 1, "crossSection": 1},
    "wzp6_ee_munumuqq_noCut_ecm157": {"fraction": 1, "crossSection": 1},
    "wzp6_ee_munumuqq_noCut_ecm163": {"fraction": 1, "crossSection": 1},
}

# Run a single ECM in parallel mode: WW_ECM={157,160,163} restricts processList
# to that one dataset so 3 fccanalysis processes can be launched concurrently.
_ecm_env = os.environ.get("WW_ECM", "").strip()
if _ecm_env:
    _key = f"wzp6_ee_munumuqq_noCut_ecm{_ecm_env}"
    if _key not in processList:
        raise ValueError(f"WW_ECM={_ecm_env} not in processList ({list(processList)})")
    processList = {_key: processList[_key]}

# ── kinematic fit configuration ────────────────────────────────────────────
# Optimizer is ROOT Minuit2 (robust, ~200-500 chi² evaluations per event).
# An older custom BFGS variant was archived to WWFunctions/old/WWKinFitBFGS.h
# on 2026-05-04 — see the file header for the test outcome and removal rationale.
# gW handling:
#   "fixed"       → pin gW = KF_GW_FIXED via Minuit FixVariable.
#   "constrained" → fit gW with Gaussian prior σ = KF_GW_PRIOR_SIGMA_REL × KF_GW_FIXED.
#   "free"        → fit gW with no prior (data-only).
KIN_FIT_GW_MODE = os.environ.get("KF_GW_MODE", "constrained")
# Jet-prior convention. "pool": jet1 and jet2 share the pooled prior — pass-1 covers
# both pT-orderings, swap fallback off (~10-25% faster). "swap": separate per-jet
# priors with jet1↔jet2 swap fallback on non-converged events.
KIN_FIT_JET_PRIOR_MODE = os.environ.get("KF_JET_PRIOR_MODE", "swap")
# True → use binned priors (pick_bin per event from reco kinematics).
# False → use the inclusive (kinematics-averaged) scalar priors.
KIN_FIT_USE_BINNED = os.environ.get("KF_USE_BINNED", "true").lower() in ("1", "true", "yes")
# ISR / neutrino treatment: "mloss" (default; MET-measured ν + |m_loss|) or
# "kfit" (explicit ISR photon + ν derived from 4-mom conservation + Gaussian
# energy closure). For kfit, pair with KF_JET_PRIOR_MODE=pool.
KIN_FIT_ISR_MODE = os.environ.get("KF_ISR_MODE", "mloss")

# Run ONNX flavour tagging? Currently outputs are not consumed by any branch
# in FULL_BRANCHES, but the helper is wired up here for future use.
RUN_FLAVOUR_TAGGING = False

channel = "CHANNELNAMEHERE"
if channel not in ["lep", "semihad", "had"]:
    channel = "semihad"
print(channel)

prodTag      = "FCCee/winter2023/IDEA/"
# Output dir resolution:
#   1. STEP2_OUTDIR (full path override) — legacy
#   2. WW_TAG (suffix only) — preferred, dedicated-dir convention
#   3. default = outputs/treemaker/lnuqq/step2/<channel>/
_tag = os.environ.get("WW_TAG", "").strip()
_default = "outputs/treemaker/lnuqq/step2/{}{}".format(channel, "_" + _tag if _tag else "")
outputDir    = os.environ.get("STEP2_OUTDIR", _default)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h", "WWFunctions/WWKinReco.h"]

all_branches = [
    # ── constituent kinematics: jet1 / jet2 / lep / nu ─────────────────
    # reco
    "reco_jet1_p", "reco_jet1_pt", "reco_jet1_theta", "reco_jet1_phi", "reco_jet1_eta", "reco_jet1_costheta", "reco_jet1_mass",
    "reco_jet2_p", "reco_jet2_pt", "reco_jet2_theta", "reco_jet2_phi", "reco_jet2_eta", "reco_jet2_costheta", "reco_jet2_mass",
    "reco_lep_p",  "reco_lep_pt",  "reco_lep_theta",  "reco_lep_phi",  "reco_lep_eta",  "reco_lep_costheta",
    "reco_met_p",  "reco_met_pt",  "reco_met_theta",  "reco_met_phi",  "reco_met_eta",  "reco_met_costheta",
    # gen
    "gen_quark1_p", "gen_quark1_pt", "gen_quark1_theta", "gen_quark1_phi", "gen_quark1_eta", "gen_quark1_costheta",
    "gen_quark2_p", "gen_quark2_pt", "gen_quark2_theta", "gen_quark2_phi", "gen_quark2_eta", "gen_quark2_costheta",
    "gen_lep_p",    "gen_lep_pt",    "gen_lep_theta",    "gen_lep_phi",    "gen_lep_eta",    "gen_lep_costheta",
    "gen_nu_p",     "gen_nu_pt",     "gen_nu_theta",     "gen_nu_phi",     "gen_nu_eta",     "gen_nu_costheta",
    # kinfit (no eta / costheta / mass)
    "kinfit_jet1_p", "kinfit_jet1_pt", "kinfit_jet1_theta", "kinfit_jet1_phi",
    "kinfit_jet2_p", "kinfit_jet2_pt", "kinfit_jet2_theta", "kinfit_jet2_phi",
    "kinfit_lep_p",  "kinfit_lep_pt",  "kinfit_lep_theta",  "kinfit_lep_phi",
    "kinfit_nu_p",   "kinfit_nu_pt",   "kinfit_nu_theta",   "kinfit_nu_phi",

    # ── W kinematics: Wlep / Whad ──────────────────────────────────────
    "reco_Wlep_m",   "reco_Wlep_p",   "reco_Wlep_pt",  "reco_Wlep_px",  "reco_Wlep_py",  "reco_Wlep_pz",  "reco_Wlep_costheta",  "reco_Wlep_phi",
    "gen_Wlep_m",    "gen_Wlep_p",    "gen_Wlep_pt",   "gen_Wlep_px",   "gen_Wlep_py",   "gen_Wlep_pz",   "gen_Wlep_costheta",   "gen_Wlep_phi",
    "kinfit_Wlep_m", "kinfit_Wlep_p", "kinfit_Wlep_pt","kinfit_Wlep_px","kinfit_Wlep_py","kinfit_Wlep_pz",
    "reco_Whad_m",   "reco_Whad_p",   "reco_Whad_pt",  "reco_Whad_px",  "reco_Whad_py",  "reco_Whad_pz",  "reco_Whad_costheta",  "reco_Whad_phi",
    "gen_Whad_m",    "gen_Whad_p",    "gen_Whad_pt",   "gen_Whad_px",   "gen_Whad_py",   "gen_Whad_pz",   "gen_Whad_costheta",   "gen_Whad_phi",
    "kinfit_Whad_m", "kinfit_Whad_p", "kinfit_Whad_pt","kinfit_Whad_px","kinfit_Whad_py","kinfit_Whad_pz",

    # ── WW system ──────────────────────────────────────────────────────
    "reco_WW_m",   "reco_WW_m_minus_ecm",   "reco_WW_px",   "reco_WW_py",   "reco_WW_pz",   "reco_WW_p_imbalance_tot",
    "gen_WW_m",    "gen_WW_m_minus_ecm",    "gen_WW_px",    "gen_WW_py",    "gen_WW_pz",    "gen_WW_p_imbalance_tot",
    "kinfit_WW_m", "kinfit_WW_m_minus_ecm", "kinfit_WW_px", "kinfit_WW_py", "kinfit_WW_pz", "kinfit_WW_p_imbalance_tot",

    # ── beam / ISR gen-level (for pull-plot truth) ─────────────────────
    "gen_ee_m_minus_ecm", "gen_ee_pz",
    "gen_WW_m_minus_m_ee",
    "gen_isr_px", "gen_isr_py", "gen_isr_pz",

    # ── resolutions / responses (cross-level by construction) ──────────
    "jet1_p_resp", "jet1_theta_resol", "jet1_phi_resol", "jet1_eta_resol", "jet1_costheta_resol",
    "jet2_p_resp", "jet2_theta_resol", "jet2_phi_resol", "jet2_eta_resol", "jet2_costheta_resol",
    "lep_p_resp",  "lep_theta_resol",  "lep_phi_resol",  "lep_eta_resol",  "lep_costheta_resol",
    "met_p_resp",  "met_theta_resol",  "met_phi_resol",  "met_eta_resol",  "met_costheta_resol",
    "Wlep_m_resol", "Wlep_p_resol",
    "Whad_m_resol", "Whad_p_resol",
    "WW_m_resol", "WW_px_resol", "WW_py_resol", "WW_pz_resol",

    # ── kinfit-internal: parameters and fit quality ────────────────────
    "kinfit_mW", "kinfit_gW",
    "kinfit_s1", "kinfit_s2", "kinfit_sl", "kinfit_sn",
    "kinfit_t1", "kinfit_t2", "kinfit_tn", "kinfit_tl",
    "kinfit_p1", "kinfit_p2", "kinfit_pn", "kinfit_pl",
    "kinfit_bes_m_minus_ecm", "kinfit_bes_pz",
    "kinfit_chi2", "kinfit_chi2_ndof", "kinfit_valid", "kinfit_valid_loose", "kinfit_status", "kinfit_edm",
    "kinfit_winner_pass", "kinfit_n_passes_run", "kinfit_priors_swapped",
    # Post-fit correlation matrix (16x16, row-major flat; see KF_PARAM_NAMES in WWKinReco.h).
    "kinfit_corr",

    # ── misc derived ───────────────────────────────────────────────────
    "n_lep_reco", "n_reco_jets", "deltaM",
    "d_12", "d_32",
    "jet1_matched_q_dR", "jet2_matched_q_dR",
]

# ── ONNX flavour-tagging model (loaded only if RUN_FLAVOUR_TAGGING) ─────────
model_name = "fccee_flavtagging_edm4hep_wc"
model_dir  = "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
local_preproc = f"{model_dir}/{model_name}.json"
local_model   = f"{model_dir}/{model_name}.onnx"
url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc   = f"{url_model_dir}/{model_name}.json"
url_model     = f"{url_model_dir}/{model_name}.onnx"

def _get_file_path(url, filename):
    if os.path.exists(filename):
        return os.path.abspath(filename)
    urllib.request.urlretrieve(url, os.path.basename(url))
    return os.path.basename(url)

if RUN_FLAVOUR_TAGGING:
    from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
    weaver_preproc = _get_file_path(url_preproc, local_preproc)
    weaver_model   = _get_file_path(url_model,   local_model)

# Module-level helpers, set inside analysers(); fccanalysis reads them from
# here to register jet (and flavour-tagging) collections.
jetClusteringHelper = None
jetFlavourHelper    = None

_dataset_iter = iter(processList.keys())

class RDFanalysis:

    def analysers(df):
        global jetClusteringHelper, jetFlavourHelper

        _dataset = next(_dataset_iter)
        _ecm = tc.parse_ecm(_dataset)
        print(f"[treemaker step2] dataset={_dataset}  ecm={_ecm}")
        if str(_ecm) not in tc.AVAILABLE_ECM:
            raise ValueError(f"ecm={_ecm} parsed from '{_dataset}' not in AVAILABLE_ECM={tc.AVAILABLE_ECM}")
        ROOT.gInterpreter.ProcessLine(
            f'FCCAnalyses::WWFunctions::setKinFitParams({_ecm}, "{KIN_FIT_JET_PRIOR_MODE}", '
            f'{"true" if KIN_FIT_USE_BINNED else "false"}, "{KIN_FIT_ISR_MODE}");')
        ROOT.gInterpreter.ProcessLine(
            'std::cout << "[DEBUG step2] ECM from WWFunctions = " << FCCAnalyses::WWFunctions::ECM << std::endl;')

        df = tc.select_isoleps(df)
        df = tc.apply_channel_filter(df, channel)
        df, jetClusteringHelper = tc.cluster_jets(df, channel)

        if RUN_FLAVOUR_TAGGING:
            collections_noleps = {
                "GenParticles": "Particle",
                "PFParticles": "ReconstructedParticlesNoMuNoEl",
                "PFTracks": "EFlowTrack",
                "PFPhotons": "EFlowPhoton",
                "PFNeutralHadrons": "EFlowNeutralHadron",
                "TrackState": "EFlowTrack_1",
                "TrackerHits": "TrackerHits",
                "CalorimeterHits": "CalorimeterHits",
                "dNdx": "EFlowTrack_2",
                "PathLength": "EFlowTrack_L",
                "Bz": "magFieldBz",
            }
            jetFlavourHelper = JetFlavourHelper(
                collections_noleps,
                jetClusteringHelper.jets,
                jetClusteringHelper.constituents,
            )
            df = jetFlavourHelper.define(df)
            df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        df = tc.define_reco_lep_met(df)
        df = tc.define_reco_jets_kinematics(df)

        df = tc.select_gen_fromele(df)
        df = tc.define_gen_kinematics(df)
        df = tc.define_beam_kinematics(df)

        df = tc.define_reco_W_WW(df)
        df = tc.match_jets_to_quarks(df)
        df = tc.define_resolutions(df)

        df = tc.run_kinfit(df, gw_mode=KIN_FIT_GW_MODE)

        # Diagnostic (not enforced): count events that would pass a dR<0.1
        # jet/quark matching cut. Side count, evaluated in the same event loop
        # as Report() below.
        n_total_lazy = df.Count()
        n_match_lazy = df.Filter(
            "jet1_matched_q_dR < 0.1 && jet2_matched_q_dR < 0.1").Count()

        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print()
        n_total = n_total_lazy.GetValue()
        n_match = n_match_lazy.GetValue()
        frac = (n_match / n_total) if n_total else 0.0
        print(f"  [diag] jet/quark dR<0.1 (not enforced): "
              f"{n_match}/{n_total} ({100*frac:.2f}%)")
        print()

        return df

    def output():
        return all_branches
