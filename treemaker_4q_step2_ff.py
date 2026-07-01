# Forward-fold inputs for WW→4q on the INCLUSIVE p8_ee_WW sample, NO kinematic fit.
# Produces gen + reco W masses (the conv_mw_4q forward-fold inputs) so the estimator can run at ecm240/365
# where the per-ECM kinfit DCB priors do not exist. = treemaker_4q_step2.py minus run_kinfit_4q.
import os, ROOT
import treemaker_common as tc

WW_SAMPLE = os.environ.get("WW_SAMPLE", "p8_ee_WW_ecm240")
_frac = float(os.environ.get("WW_FRACTION", "0.01"))
processList = {WW_SAMPLE: {"fraction": _frac, "crossSection": 1}}
WW_SQRTD45_MAX = float(os.environ.get("WW_SQRTD45_MAX", "7.0"))

channel = "had"
prodTag = "FCCee/winter2023/IDEA/"
_tag = os.environ.get("WW_TAG", "").strip()
_default = "outputs/treemaker/4q/step2_ff/{}{}".format(channel, "_" + _tag if _tag else "")
outputDir = os.environ.get("STEP2_OUTDIR", _default)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h"]   # NO WWKinReco4q.h ⇒ no per-ECM kinfit dep

all_branches = [
    "reco_jet1_p", "reco_jet1_pt", "reco_jet1_theta", "reco_jet1_phi", "reco_jet1_costheta", "reco_jet1_mass",
    "reco_jet2_p", "reco_jet2_pt", "reco_jet2_theta", "reco_jet2_phi", "reco_jet2_costheta", "reco_jet2_mass",
    "reco_jet3_p", "reco_jet3_pt", "reco_jet3_theta", "reco_jet3_phi", "reco_jet3_costheta", "reco_jet3_mass",
    "reco_jet4_p", "reco_jet4_pt", "reco_jet4_theta", "reco_jet4_phi", "reco_jet4_costheta", "reco_jet4_mass",
    "d_23", "d_34", "d_45",
    "gen_quark1_p", "gen_quark2_p", "gen_quark3_p", "gen_quark4_p",
    "jet1_matched_q_dR", "jet2_matched_q_dR", "jet3_matched_q_dR", "jet4_matched_q_dR",
    "gen_W1_m", "gen_W2_m", "gen_WW_m", "gen_WW_m_minus_ecm", "gen_WW_m_minus_m_ee",
    "gen_isr_px", "gen_isr_py", "gen_isr_pz",
    "gen_isr_WW_px", "gen_isr_WW_py", "gen_isr_WW_pz",
    # W-grouped gen quark 4-vectors ([0,1]=W1, [2,3]=W2): true + wrong pairings vs ISR
    "gen_qW0_px", "gen_qW0_py", "gen_qW0_pz", "gen_qW0_e",
    "gen_qW1_px", "gen_qW1_py", "gen_qW1_pz", "gen_qW1_e",
    "gen_qW2_px", "gen_qW2_py", "gen_qW2_pz", "gen_qW2_e",
    "gen_qW3_px", "gen_qW3_py", "gen_qW3_pz", "gen_qW3_e",
    "jet1_p_resp", "jet2_p_resp", "jet3_p_resp", "jet4_p_resp",
    "gen_pairing_true",
]

jetClusteringHelper = None
_dataset_iter = iter(processList.keys())

class RDFanalysis:
    def analysers(df):
        global jetClusteringHelper
        _dataset = next(_dataset_iter)
        _ecm = tc.parse_ecm(_dataset)
        print(f"[treemaker 4q FF step2] dataset={_dataset}  ecm={_ecm}  (NO kinfit)")
        if str(_ecm) not in tc.AVAILABLE_ECM:
            raise ValueError(f"ecm={_ecm} not in AVAILABLE_ECM={tc.AVAILABLE_ECM}")
        ROOT.gInterpreter.ProcessLine(f"FCCAnalyses::WWFunctions::ECM = {_ecm};")  # set ECM directly (no kinfit setup)

        df = tc.select_isoleps(df)
        df = tc.apply_channel_filter(df, channel)
        df, jetClusteringHelper = tc.cluster_jets_4q(df)
        df = tc.define_reco_jets_kinematics_4q(df)
        df = tc.filter_genuine_4jet(df, WW_SQRTD45_MAX)
        df = tc.select_gen_fromW(df)
        df = tc.define_gen_kinematics_4q(df)
        df = tc.define_beam_kinematics(df, post_isr_mode="p8", gen_ww_p4="WW_4q_gen")
        df = tc.match_jets_to_quarks_4q(df)
        df = tc.define_resolutions_4q(df)
        # NO tc.run_kinfit_4q() — forward-fold needs only gen+reco masses.
        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print(); print()
        return df

    def output():
        return all_branches
