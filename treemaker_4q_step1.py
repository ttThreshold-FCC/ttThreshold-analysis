# Step 1 of 2 — WW → 4q (fully hadronic) resolution-input producer.
# Mirrors treemaker_lnuqq_step1.py but for the all-hadronic channel: clusters 4
# jets, matches them to the 4 gen quarks (W-grouped via WWFunctions::
# sel_quarks_fromW), and writes the per-jet response/resolution branches consumed
# by fit_resolutions.py (4q mode). Does NOT include WWKinReco*.h, so it builds
# without the DCB params header.
import os
import ROOT
import treemaker_common as tc

# Stage 1 subsample: p8_ee_WW_ecm160 is large (~hundreds of M events). A small
# fraction yields ample clean 4-jet events for the pooled resolution fit and
# pairing study. Override via WW_FRACTION for larger/smaller runs.
_frac = float(os.environ.get("WW_FRACTION", "0.005"))
processList = {
    "p8_ee_WW_ecm160": {"fraction": _frac, "crossSection": 1},
}

# Fully hadronic channel (0 isolated leptons). The gen-level all-hadronic
# requirement (4 quarks from 2 W's) is enforced in select_gen_fromW.
channel = "had"

prodTag      = "FCCee/winter2023/IDEA/"
# Optional WW_TAG env var → suffixes the output directory so parallel tests
# don't collide and side-by-side comparison stays trivial.
_tag = os.environ.get("WW_TAG", "").strip()
_suffix = "_{}".format(_tag) if _tag else ""
outputDir    = "outputs/treemaker/4q/step1/{}{}".format(channel, _suffix)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h"]

# Branches consumed by fit_resolutions.py (4q mode) + matching/diagnostic
# branches. dR cuts are applied inside fit_resolutions on the loaded arrays, so
# step1 keeps every event.
all_branches = [
    # per-jet response / angular resolutions (4 jets, pooled downstream)
    "jet1_p_resp", "jet2_p_resp", "jet3_p_resp", "jet4_p_resp",
    "jet1_theta_resol", "jet2_theta_resol", "jet3_theta_resol", "jet4_theta_resol",
    "jet1_phi_resol",   "jet2_phi_resol",   "jet3_phi_resol",   "jet4_phi_resol",
    # jet/quark matching quality (per jet)
    "jet1_matched_q_dR", "jet2_matched_q_dR", "jet3_matched_q_dR", "jet4_matched_q_dR",
    # WW-system gen kinematics (BES / ISR / m_loss priors are channel-independent)
    "gen_WW_px", "gen_WW_py", "gen_WW_pz",
    "gen_WW_m", "gen_WW_m_minus_ecm",
    "gen_W1_m", "gen_W2_m",
    "gen_ee_m_minus_ecm", "gen_ee_pz",
    "gen_WW_m_minus_m_ee",
    "gen_isr_px", "gen_isr_py", "gen_isr_pz",
    # reco kinematics for equal-occupancy binning variable lookups
    "reco_jet1_p", "reco_jet2_p", "reco_jet3_p", "reco_jet4_p",
    "reco_jet1_costheta", "reco_jet2_costheta", "reco_jet3_costheta", "reco_jet4_costheta",
    # diagnostics
    "gen_pairing_true",
    "d_23", "d_34", "d_45",
]

# Module-level helper, set inside analysers(); fccanalysis reads it from here
# to register the jet collections.
jetClusteringHelper = None

_dataset_iter = iter(processList.keys())

class RDFanalysis:

    def analysers(df):
        global jetClusteringHelper

        _dataset = next(_dataset_iter)
        _ecm = tc.parse_ecm(_dataset)
        print(f"[treemaker 4q step1] dataset={_dataset}  ecm={_ecm}")
        if str(_ecm) not in tc.AVAILABLE_ECM:
            raise ValueError(f"ecm={_ecm} parsed from '{_dataset}' not in AVAILABLE_ECM={tc.AVAILABLE_ECM}")
        ROOT.gInterpreter.ProcessLine(f"FCCAnalyses::WWFunctions::ECM = {_ecm};")
        ROOT.gInterpreter.ProcessLine(
            'std::cout << "[DEBUG 4q step1] ECM from WWFunctions = " << FCCAnalyses::WWFunctions::ECM << std::endl;')

        df = tc.select_isoleps(df)
        df = tc.apply_channel_filter(df, channel)
        df, jetClusteringHelper = tc.cluster_jets_4q(df)
        df = tc.define_reco_jets_kinematics_4q(df)

        df = tc.select_gen_fromW(df)
        df = tc.define_gen_kinematics_4q(df)
        df = tc.define_beam_kinematics(df, post_isr_mode="p8", gen_ww_p4="WW_4q_gen")

        df = tc.match_jets_to_quarks_4q(df)
        df = tc.define_resolutions_4q(df)

        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print()
        print()

        return df

    def output():
        return all_branches
