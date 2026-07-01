# Forward-fold inputs for WW→ℓνqq (semileptonic) on the INCLUSIVE p8_ee_WW sample, NO kinematic fit.
# Produces gen + reco Wlep/Whad masses (the conv_mw forward-fold inputs) so the μνqq/eνqq estimator can run
# at any ECM (160/240/365) where the per-ECM kinfit DCB priors do not exist. = treemaker_lnuqq_step2.py minus run_kinfit.
#
# WHY (HANDOFF_DAY11 NEXT-1 + STEP-0 C5 fix): the dedicated wzp6_ee_munumuqq sample is MUON-ONLY, so the channel
# comparison wrongly used a μ+e combined BR on muon-only resolution. The inclusive p8 sample gives BOTH e and μ
# directly from the W, so we select them via the W parent (p8 keeps the W in history) and produce a properly
# μ+e-combined ℓνqq tree (WW_LEPTON_PDG controls the flavour subset).
#
# source setup.sh   (sources ../FCCAnalyses/setup.sh — provides `fccanalysis`)
# WW_SAMPLE=p8_ee_WW_ecm240 WW_TAG=ff_lnuqq240 fccanalysis run treemaker_lnuqq_step2_ff.py
import os, ROOT
import treemaker_common as tc

WW_SAMPLE = os.environ.get("WW_SAMPLE", "p8_ee_WW_ecm240")
_frac = float(os.environ.get("WW_FRACTION", "1.0"))
processList = {WW_SAMPLE: {"fraction": _frac, "crossSection": 1}}

# Flavour subset of the leptonic W: "both"/""→ e+μ combined (default, BR≈0.291),
# "13"→ μνqq only (BR≈0.146), "11"→ eνqq only.
_lep = os.environ.get("WW_LEPTON_PDG", "both").strip().lower()
if _lep in ("", "both", "emu", "0"):
    LEPTON_PDGS = (11, 13)
elif _lep in ("13", "mu", "muon"):
    LEPTON_PDGS = (13,)
elif _lep in ("11", "e", "el", "electron"):
    LEPTON_PDGS = (11,)
else:
    raise ValueError(f"WW_LEPTON_PDG={_lep!r} must be both|13|11")

channel = "semihad"
prodTag = "FCCee/winter2023/IDEA/"
_tag = os.environ.get("WW_TAG", "").strip()
_default = "outputs/treemaker/lnuqq/step2_ff/{}{}".format(channel, "_" + _tag if _tag else "")
outputDir = os.environ.get("STEP2_OUTDIR", _default)
includePaths = ["examples/functions.h", "WWFunctions/WWFunctions.h"]   # NO WWKinReco.h ⇒ no per-ECM kinfit dep

all_branches = [
    # constituent reco kinematics
    "reco_jet1_p", "reco_jet1_pt", "reco_jet1_theta", "reco_jet1_phi", "reco_jet1_costheta", "reco_jet1_mass",
    "reco_jet2_p", "reco_jet2_pt", "reco_jet2_theta", "reco_jet2_phi", "reco_jet2_costheta", "reco_jet2_mass",
    "reco_lep_p",  "reco_lep_pt",  "reco_lep_theta",  "reco_lep_phi",  "reco_lep_costheta",
    "reco_met_p",  "reco_met_pt",  "reco_met_theta",  "reco_met_phi",
    # constituent gen kinematics
    "gen_lep_p", "gen_lep_theta", "gen_lep_phi",
    "gen_nu_p",  "gen_nu_theta",  "gen_nu_phi",
    # W kinematics (the forward-fold observables)
    "reco_Wlep_m", "reco_Wlep_p", "reco_Wlep_costheta", "reco_Wlep_phi",
    "gen_Wlep_m",  "gen_Wlep_p",  "gen_Wlep_costheta",  "gen_Wlep_phi",
    "reco_Whad_m", "reco_Whad_p", "reco_Whad_costheta", "reco_Whad_phi",
    "gen_Whad_m",  "gen_Whad_p",  "gen_Whad_costheta",  "gen_Whad_phi",
    # WW system
    "reco_WW_m", "reco_WW_m_minus_ecm",
    "gen_WW_m",  "gen_WW_m_minus_ecm",
    # misc
    "n_lep_reco", "n_reco_jets", "deltaM", "d_12", "d_32",
]

jetClusteringHelper = None
_dataset_iter = iter(processList.keys())

class RDFanalysis:
    def analysers(df):
        global jetClusteringHelper
        _dataset = next(_dataset_iter)
        _ecm = tc.parse_ecm(_dataset)
        print(f"[treemaker lnuqq FF step2] dataset={_dataset}  ecm={_ecm}  leptons={LEPTON_PDGS}  (NO kinfit)")
        if str(_ecm) not in tc.AVAILABLE_ECM:
            raise ValueError(f"ecm={_ecm} not in AVAILABLE_ECM={tc.AVAILABLE_ECM}")
        ROOT.gInterpreter.ProcessLine(f"FCCAnalyses::WWFunctions::ECM = {_ecm};")  # set ECM directly (no kinfit setup)

        df = tc.select_isoleps(df)
        df = tc.apply_channel_filter(df, channel)            # exactly 1 isolated lepton (e or μ)
        df, jetClusteringHelper = tc.cluster_jets(df, channel)  # exactly 2 jets
        df = tc.define_reco_lep_met(df)
        df = tc.define_reco_jets_kinematics(df)

        df = tc.select_gen_fromW_semilep(df, lepton_pdgs=LEPTON_PDGS)
        df = tc.define_gen_kinematics(df, "gen_leps_fromW", "gen_neutrinos_fromW", "gen_lightquarks_fromW")
        df = tc.define_reco_W_WW(df)
        # NO beam/ISR, NO jet-quark match, NO resolutions, NO kinfit — the fold needs only gen+reco W masses.

        print(f"\n[cutflow] dataset={_dataset}")
        df.Report().Print(); print()
        return df

    def output():
        return all_branches
