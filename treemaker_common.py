# Shared analyser building blocks for treemaker_lnuqq_step{1,2}.py.
# Each helper takes an RDataFrame and returns the extended frame, except
# `cluster_jets` which also returns the ExclusiveJetClusteringHelper instance
# (the entry script must keep it as a module-level global so the framework
# can register the jet collections).
import re
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

AVAILABLE_ECM = ['157', '160', '163', '240', '365']

# FSR dressing parameters (see project_lep_p_resp_fsr_dressing.md).
# Tuned 2026-05-05 from photon-investigation analysis: dR<0.1 around the iso
# lepton, E_γ>0.5 GeV. Recovers ~30% of deep-tail (lep_p_resp<0.88) events
# with negligible core impact and 0.1-0.2% overshoot rate.
FSR_DRESS_DR_MAX = 0.1
FSR_DRESS_E_MIN  = 0.5

def parse_ecm(name):
    m = re.search(r'_ecm(\d+)', name)
    if not m:
        raise ValueError(f"Cannot parse ecm from sample name: {name}")
    return int(m.group(1))


# ── selection ────────────────────────────────────────────────────────────────
def select_isoleps(df):
    df = df.Alias("Muon0",     "Muon#0.index")
    df = df.Alias("Electron0", "Electron#0.index")

    df = df.Define("muons_all",
        "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
    df = df.Define("electrons_all",
        "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")

    df = df.Define("muons_sel",
        "FCCAnalyses::ReconstructedParticle::sel_p(12)(muons_all)")
    df = df.Define("electrons_sel",
        "FCCAnalyses::ReconstructedParticle::sel_p(12)(electrons_all)")

    df = df.Define("muons_iso",
        "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(muons_sel, ReconstructedParticles)")
    df = df.Define("electrons_iso",
        "FCCAnalyses::ZHfunctions::coneIsolation(0.01, 0.5)(electrons_sel, ReconstructedParticles)")

    df = df.Define("muons_sel_iso",
        "FCCAnalyses::ZHfunctions::sel_iso(0.7)(muons_sel, muons_iso)")
    df = df.Define("electrons_sel_iso",
        "FCCAnalyses::ZHfunctions::sel_iso(0.7)(electrons_sel, electrons_iso)")
    return df


def apply_channel_filter(df, channel, veto=True):
    # veto=False skips the isolated-lepton COUNT cut but still builds the
    # lepton/FSR-removed collection the jet clustering runs on. Used by studies
    # that gen-select the final state (e.g. the 4q BW-pairing study) and don't
    # want the reco lepton veto removing good gen-4q events.
    if veto:
        if channel == "had":
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 0",
                           "channel: 0 isolated leptons (had)")
        elif channel == "semihad":
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 1",
                           "channel: 1 isolated lepton (semihad)")
        else:
            df = df.Filter("muons_sel_iso.size() + electrons_sel_iso.size() == 2",
                           "channel: 2 isolated leptons (lep)")

    df = df.Define("Isoleps_bare", "ROOT::VecOps::Concatenate(muons_sel_iso, electrons_sel_iso)")

    # FSR dressing: absorb nearby photons (RPs with type==22) into each iso
    # lepton; the same photons must be removed from the jet input.
    df = df.Define("Isoleps",
        f"FCCAnalyses::WWFunctions::dress_isoleps(Isoleps_bare, ReconstructedParticles, "
        f"{FSR_DRESS_DR_MAX}, {FSR_DRESS_E_MIN})")
    df = df.Define("FSR_photons",
        f"FCCAnalyses::WWFunctions::dressed_photons(Isoleps_bare, ReconstructedParticles, "
        f"{FSR_DRESS_DR_MAX}, {FSR_DRESS_E_MIN})")

    df = df.Define("ReconstructedParticlesNoMuons",
        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, muons_sel_iso)")
    df = df.Define("ReconstructedParticlesNoMuNoEl",
        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuons, electrons_sel_iso)")
    df = df.Define("ReconstructedParticlesNoMuNoElNoFSR",
        "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticlesNoMuNoEl, FSR_photons)")
    return df


def cluster_jets(df, channel):
    nJets = 2 if channel == "semihad" else 4
    # Cluster on the FSR-dressed reduced collection so absorbed photons don't
    # double-count on the hadronic side.
    helper = ExclusiveJetClusteringHelper("ReconstructedParticlesNoMuNoElNoFSR", nJets)
    df = helper.define(df)
    df = df.Define("jets_p4",
        f"JetConstituentsUtils::compute_tlv_jets({helper.jets})")
    # ExclusiveJetClusteringHelper passes arg_sorted=1 to clustering_ee_kt → jets
    # are returned E-sorted (descending), not pT-sorted. jet1 = highest-E, jet2 =
    # second-highest-E. For WW at √s ≈ 2·mW, leading-E ≈ leading-pT in most
    # events, but they can disagree for forward/asymmetric configurations.
    df = df.Define("jet1", "jets_p4[0]")
    df = df.Define("jet2", "jets_p4[1]")
    df = df.Define("n_reco_jets", "(int)jets_p4.size()")
    df = df.Filter("n_reco_jets == 2", "exactly 2 reco jets")
    df = df.Define("d_12", "JetClusteringUtils::get_exclusive_dmerge(_jet, 1)")
    df = df.Define("d_32", "JetClusteringUtils::get_exclusive_dmerge(_jet, 2)")
    return df, helper


# ── reco kinematics ──────────────────────────────────────────────────────────
def define_reco_lep_met(df):
    df = df.Define("Isoleps_p4_reco", "FCCAnalyses::ReconstructedParticle::get_tlv(Isoleps, 0)")
    df = df.Define("missing_p_p4",    "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET, 0)")
    df = df.Define("n_lep_reco",      "(int)Isoleps.size()")

    df = df.Define("reco_lep_p",        "Isoleps_p4_reco.P()")
    df = df.Define("reco_lep_pt",       "Isoleps_p4_reco.Pt()")
    df = df.Define("reco_lep_eta",      "Isoleps_p4_reco.Eta()")
    df = df.Define("reco_lep_theta",    "Isoleps_p4_reco.Theta()")
    df = df.Define("reco_lep_phi",      "Isoleps_p4_reco.Phi()")
    df = df.Define("reco_lep_costheta", "Isoleps_p4_reco.CosTheta()")

    df = df.Define("reco_met_p",        "missing_p_p4.P()")
    df = df.Define("reco_met_pt",       "missing_p_p4.Pt()")
    df = df.Define("reco_met_theta",    "missing_p_p4.Theta()")
    df = df.Define("reco_met_phi",      "missing_p_p4.Phi()")
    df = df.Define("reco_met_eta",      "missing_p_p4.Eta()")
    df = df.Define("reco_met_costheta", "missing_p_p4.CosTheta()")

    df = df.Define("Wlep_reco",
        "FCCAnalyses::WWFunctions::sum_p4({Isoleps_p4_reco, missing_p_p4})")
    return df


def define_reco_jets_kinematics(df):
    df = df.Define("reco_jet1_p",        "jet1.P()")
    df = df.Define("reco_jet1_pt",       "jet1.Pt()")
    df = df.Define("reco_jet1_theta",    "jet1.Theta()")
    df = df.Define("reco_jet1_phi",      "jet1.Phi()")
    df = df.Define("reco_jet1_eta",      "jet1.Eta()")
    df = df.Define("reco_jet1_costheta", "jet1.CosTheta()")
    df = df.Define("reco_jet1_mass",     "jet1.M()")
    df = df.Define("reco_jet2_p",        "jet2.P()")
    df = df.Define("reco_jet2_pt",       "jet2.Pt()")
    df = df.Define("reco_jet2_theta",    "jet2.Theta()")
    df = df.Define("reco_jet2_phi",      "jet2.Phi()")
    df = df.Define("reco_jet2_eta",      "jet2.Eta()")
    df = df.Define("reco_jet2_costheta", "jet2.CosTheta()")
    df = df.Define("reco_jet2_mass",     "jet2.M()")

    # Reco jets treated as massless: jet mass is dominated by clustering /
    # detector, not the parton mass; matches kinfit's (p,theta,phi) jet param.
    df = df.Define("jet1_massless", "FCCAnalyses::WWFunctions::tlv_setmass(jet1, 0.)")
    df = df.Define("jet2_massless", "FCCAnalyses::WWFunctions::tlv_setmass(jet2, 0.)")
    df = df.Define("Whad_reco",
        "FCCAnalyses::WWFunctions::sum_p4({jet1_massless, jet2_massless})")
    return df


# ── gen-level (W decay products via parent==e±, since W is not in MC history) ─
def select_gen_fromele(df):
    df = df.Alias("Particle0", "Particle#0.index")
    df = df.Define("gen_leps_fromele",
        "FCCAnalyses::WWFunctions::sel_genleps_fromele(13)(Particle, Particle0)")
    df = df.Define("gen_neutrinos_fromele",
        "FCCAnalyses::WWFunctions::sel_genleps_fromele(14)(Particle, Particle0)")
    df = df.Define("gen_lightquarks_fromele",
        "FCCAnalyses::WWFunctions::sel_lightQuarks_fromele()(Particle, Particle0)")

    df = df.Filter("gen_leps_fromele.size() == 1",
                   "gen: exactly 1 muon fromele (W daughter)")
    df = df.Filter("gen_neutrinos_fromele.size() == 1",
                   "gen: exactly 1 nu_mu fromele (W daughter)")
    df = df.Filter("gen_lightquarks_fromele.size() == 2",
                   "gen: exactly 2 light quarks fromele (W daughters)")
    return df


# ── gen-level ℓνqq for the inclusive Pythia8 p8_ee_WW sample (W KEPT in history) ─
def select_gen_fromW_semilep(df, lepton_pdgs=(11, 13)):
    """Gen ℓνqq selection on the inclusive p8_ee_WW sample: pick the charged lepton,
    neutrino and 2 light quarks by their immediate W parent (|PDG|==24) — the p8 analog
    of select_gen_fromele (which uses the e±-parent proxy for the W-less Whizard samples).
    lepton_pdgs selects the flavour(s): (11,13)=e+μ combined [default], (13,)=μ-only,
    (11,)=e-only. τ→ℓ events drop out automatically (the e/μ has a τ parent, not W)."""
    df = df.Alias("Particle0", "Particle#0.index")
    lep_cols, nu_cols = [], []
    for lpdg in lepton_pdgs:
        npdg = lpdg + 1   # 11→12 (νe), 13→14 (νμ)
        df = df.Define(f"gen_lep{lpdg}_fromW",
            f"FCCAnalyses::WWFunctions::sel_genleps_fromW({lpdg})(Particle, Particle0)")
        df = df.Define(f"gen_nu{npdg}_fromW",
            f"FCCAnalyses::WWFunctions::sel_genleps_fromW({npdg})(Particle, Particle0)")
        lep_cols.append(f"gen_lep{lpdg}_fromW")
        nu_cols.append(f"gen_nu{npdg}_fromW")

    def _concat(cols):
        expr = cols[0]
        for c in cols[1:]:
            expr = f"ROOT::VecOps::Concatenate({expr}, {c})"
        return expr

    df = df.Define("gen_leps_fromW", _concat(lep_cols))
    df = df.Define("gen_neutrinos_fromW", _concat(nu_cols))
    df = df.Define("gen_lightquarks_fromW",
        "FCCAnalyses::WWFunctions::sel_lightQuarks_fromW()(Particle, Particle0)")

    df = df.Filter("gen_leps_fromW.size() == 1",
                   "gen: exactly 1 charged lepton fromW (leptonic-W daughter)")
    df = df.Filter("gen_neutrinos_fromW.size() == 1",
                   "gen: exactly 1 neutrino fromW (leptonic-W daughter)")
    df = df.Filter("gen_lightquarks_fromW.size() == 2",
                   "gen: exactly 2 light quarks fromW (hadronic-W daughters)")
    return df


def define_beam_kinematics(df, post_isr_mode="whizard", gen_ww_p4=None):
    """Beam e± at the two relevant chain depths:
      depth=1 (post-BES, pre-ISR) → m(ee)−ECM gives BES;
      post-ISR e± (into hard process) → (depth1 − post_isr) gives the total ISR
      4-momentum.

    post_isr_mode selects how the post-ISR (hard-process-incoming) e± is found:
      "whizard" (default) → depth-2 chain walk (sel_post_isr_electrons). Correct
                 for the Whizard wzp6_ee_munumuqq ℓνqq samples.
      "p8"      → generatorStatus==21 (sel_post_isr_electrons_status). Robust to
                 Pythia8's variable-length ISR e-chain; the depth-2 walk drops
                 ~40% of p8 events (the hard-process e sits at depth 1/2/3).

    gen_ww_p4 (a TLorentzVector column name, e.g. "WW_4q_gen") enables a SECOND,
    generator-agnostic ISR estimate from the WW system: for a fully-reconstructed
    final state the post-ISR e+e- 4-momentum equals the WW 4-momentum (momentum
    conservation), so gen_isr_WW = gen_ee(depth1) − gen_WW. Emitted alongside the
    electron-based gen_isr for cross-check; needs no e-history walk → no event loss."""
    df = df.Define("gen_beams",
        "FCCAnalyses::WWFunctions::sel_beam_electrons()(Particle, Particle0)")
    df = df.Filter("gen_beams.size() == 2",
                   "gen: exactly 2 post-BES beam e± (depth=1)")
    df = df.Define("gen_beams_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_beams)")
    df = df.Define("gen_ee_p4",
        "FCCAnalyses::WWFunctions::sum_p4({gen_beams_tlv[0], gen_beams_tlv[1]})")
    df = df.Define("gen_ee_m_minus_ecm", "gen_ee_p4.M() - FCCAnalyses::WWFunctions::ECM")
    # Longitudinal momentum of the depth-1 e+e- system = E+ − E−, the BES
    # asymmetry between the two beams. Independent of m(ee)−ECM (sum vs diff of
    # independent Gaussians) but same width.
    df = df.Define("gen_ee_pz", "gen_ee_p4.Pz()")

    if post_isr_mode == "p8":
        df = df.Define("gen_post_isr_e",
            "FCCAnalyses::WWFunctions::sel_post_isr_electrons_status(21)(Particle)")
        df = df.Filter("gen_post_isr_e.size() == 2",
                       "gen: exactly 2 post-ISR e± (genstat==21)")
    elif post_isr_mode == "whizard":
        df = df.Define("gen_post_isr_e",
            "FCCAnalyses::WWFunctions::sel_post_isr_electrons()(Particle, Particle0)")
        df = df.Filter("gen_post_isr_e.size() == 2",
                       "gen: exactly 2 post-ISR e± (depth=2)")
    else:
        raise ValueError(f"post_isr_mode={post_isr_mode!r} must be 'whizard' or 'p8'")
    df = df.Define("gen_post_isr_e_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_post_isr_e)")
    df = df.Define("gen_ee_postisr_p4",
        "FCCAnalyses::WWFunctions::sum_p4({gen_post_isr_e_tlv[0], gen_post_isr_e_tlv[1]})")
    df = df.Define("gen_isr_p4", "gen_ee_p4 - gen_ee_postisr_p4")
    df = df.Define("gen_isr_px", "gen_isr_p4.Px()")
    df = df.Define("gen_isr_py", "gen_isr_p4.Py()")
    df = df.Define("gen_isr_pz", "gen_isr_p4.Pz()")

    # WW-system ISR proxy (generator-agnostic cross-check): gen_ee − gen_WW.
    if gen_ww_p4 is not None:
        df = df.Define("gen_isr_WW_p4", f"gen_ee_p4 - {gen_ww_p4}")
        df = df.Define("gen_isr_WW_px", "gen_isr_WW_p4.Px()")
        df = df.Define("gen_isr_WW_py", "gen_isr_WW_p4.Py()")
        df = df.Define("gen_isr_WW_pz", "gen_isr_WW_p4.Pz()")

    # m(WW) − m(ee) — pure ISR mass-loss, with BES variance subtracted off vs
    # the older m(WW) − ECM. In the no-ISR limit it is exactly 0; with ISR it
    # is < 0 (energy loss to the ISR photons).
    df = df.Define("gen_WW_m_minus_m_ee", "gen_WW_m - gen_ee_p4.M()")
    return df


def define_gen_kinematics(df, lep_col="gen_leps_fromele",
                          nu_col="gen_neutrinos_fromele",
                          quark_col="gen_lightquarks_fromele"):
    # lep_col/nu_col/quark_col let the p8 ℓνqq path (select_gen_fromW_semilep) reuse
    # this builder with the *_fromW collections; defaults keep the Whizard fromele path
    # bit-identical.
    df = df.Define("gen_leps_fromele_tlv",
        f"FCCAnalyses::MCParticle::get_tlv({lep_col})")
    df = df.Define("gen_neutrinos_fromele_tlv",
        f"FCCAnalyses::MCParticle::get_tlv({nu_col})")
    df = df.Define("gen_lightquarks_fromele_tlv",
        f"FCCAnalyses::MCParticle::get_tlv({quark_col})")

    df = df.Define("lep_p4_gen", "gen_leps_fromele_tlv[0]")
    df = df.Define("nu_p4_gen",  "gen_neutrinos_fromele_tlv[0]")
    df = df.Define("gen_q1_p4",  "gen_lightquarks_fromele_tlv[0]")
    df = df.Define("gen_q2_p4",  "gen_lightquarks_fromele_tlv[1]")

    df = df.Define("gen_lep_p",        "lep_p4_gen.P()")
    df = df.Define("gen_lep_pt",       "lep_p4_gen.Pt()")
    df = df.Define("gen_lep_theta",    "lep_p4_gen.Theta()")
    df = df.Define("gen_lep_phi",      "lep_p4_gen.Phi()")
    df = df.Define("gen_lep_eta",      "lep_p4_gen.Eta()")
    df = df.Define("gen_lep_costheta", "lep_p4_gen.CosTheta()")
    df = df.Define("gen_nu_p",         "nu_p4_gen.P()")
    df = df.Define("gen_nu_pt",        "nu_p4_gen.Pt()")
    df = df.Define("gen_nu_theta",     "nu_p4_gen.Theta()")
    df = df.Define("gen_nu_phi",       "nu_p4_gen.Phi()")
    df = df.Define("gen_nu_eta",       "nu_p4_gen.Eta()")
    df = df.Define("gen_nu_costheta",  "nu_p4_gen.CosTheta()")

    # Gen quarks keep their native (MC) masses; reco jets are massless.
    df = df.Define("Wlep_gen", "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen})")
    df = df.Define("Whad_gen", "FCCAnalyses::WWFunctions::sum_p4({gen_q1_p4, gen_q2_p4})")
    df = df.Define("Wlnuqq_gen",
        "FCCAnalyses::WWFunctions::sum_p4({lep_p4_gen, nu_p4_gen, gen_q1_p4, gen_q2_p4})")

    df = df.Define("gen_Wlep_m",        "Wlep_gen.M()")
    df = df.Define("gen_Wlep_p",        "Wlep_gen.P()")
    df = df.Define("gen_Wlep_pt",       "Wlep_gen.Pt()")
    df = df.Define("gen_Wlep_px",       "Wlep_gen.Px()")
    df = df.Define("gen_Wlep_py",       "Wlep_gen.Py()")
    df = df.Define("gen_Wlep_pz",       "Wlep_gen.Pz()")
    df = df.Define("gen_Wlep_costheta", "Wlep_gen.CosTheta()")
    df = df.Define("gen_Wlep_phi",      "Wlep_gen.Phi()")

    df = df.Define("gen_Whad_m",        "Whad_gen.M()")
    df = df.Define("gen_Whad_p",        "Whad_gen.P()")
    df = df.Define("gen_Whad_pt",       "Whad_gen.Pt()")
    df = df.Define("gen_Whad_px",       "Whad_gen.Px()")
    df = df.Define("gen_Whad_py",       "Whad_gen.Py()")
    df = df.Define("gen_Whad_pz",       "Whad_gen.Pz()")
    df = df.Define("gen_Whad_costheta", "Whad_gen.CosTheta()")
    df = df.Define("gen_Whad_phi",      "Whad_gen.Phi()")

    df = df.Define("gen_WW_m",               "Wlnuqq_gen.M()")
    df = df.Define("gen_WW_m_minus_ecm",     "gen_WW_m - FCCAnalyses::WWFunctions::ECM")
    df = df.Define("gen_WW_px",              "Wlnuqq_gen.Px()")
    df = df.Define("gen_WW_py",              "Wlnuqq_gen.Py()")
    df = df.Define("gen_WW_pz",              "Wlnuqq_gen.Pz()")
    df = df.Define("gen_WW_p_imbalance_tot", "Wlnuqq_gen.P()")
    return df


def match_jets_to_quarks(df):
    df = df.Define("matched_gen_quarks",
        "FCCAnalyses::WWFunctions::matchJets2(jet1, jet2, gen_q1_p4, gen_q2_p4)")
    df = df.Define("jet1_matched_q_p4", "matched_gen_quarks.first")
    df = df.Define("jet2_matched_q_p4", "matched_gen_quarks.second")

    df = df.Define("jet1_matched_q_dR", "(double)jet1.DeltaR(jet1_matched_q_p4)")
    df = df.Define("jet2_matched_q_dR", "(double)jet2.DeltaR(jet2_matched_q_p4)")

    df = df.Define("gen_quark1_p",        "jet1_matched_q_p4.P()")
    df = df.Define("gen_quark1_pt",       "jet1_matched_q_p4.Pt()")
    df = df.Define("gen_quark1_theta",    "jet1_matched_q_p4.Theta()")
    df = df.Define("gen_quark1_phi",      "jet1_matched_q_p4.Phi()")
    df = df.Define("gen_quark1_eta",      "jet1_matched_q_p4.Eta()")
    df = df.Define("gen_quark1_costheta", "jet1_matched_q_p4.CosTheta()")
    df = df.Define("gen_quark2_p",        "jet2_matched_q_p4.P()")
    df = df.Define("gen_quark2_pt",       "jet2_matched_q_p4.Pt()")
    df = df.Define("gen_quark2_theta",    "jet2_matched_q_p4.Theta()")
    df = df.Define("gen_quark2_phi",      "jet2_matched_q_p4.Phi()")
    df = df.Define("gen_quark2_eta",      "jet2_matched_q_p4.Eta()")
    df = df.Define("gen_quark2_costheta", "jet2_matched_q_p4.CosTheta()")
    return df


# ── reco W / WW system (needs Wlep_reco AND Whad_reco) ───────────────────────
def define_reco_W_WW(df):
    df = df.Define("reco_Wlep_m",        "Wlep_reco.M()")
    df = df.Define("reco_Wlep_p",        "Wlep_reco.P()")
    df = df.Define("reco_Wlep_pt",       "Wlep_reco.Pt()")
    df = df.Define("reco_Wlep_px",       "Wlep_reco.Px()")
    df = df.Define("reco_Wlep_py",       "Wlep_reco.Py()")
    df = df.Define("reco_Wlep_pz",       "Wlep_reco.Pz()")
    df = df.Define("reco_Wlep_costheta", "Wlep_reco.CosTheta()")
    df = df.Define("reco_Wlep_phi",      "Wlep_reco.Phi()")

    df = df.Define("reco_Whad_m",        "Whad_reco.M()")
    df = df.Define("reco_Whad_p",        "Whad_reco.P()")
    df = df.Define("reco_Whad_pt",       "Whad_reco.Pt()")
    df = df.Define("reco_Whad_px",       "Whad_reco.Px()")
    df = df.Define("reco_Whad_py",       "Whad_reco.Py()")
    df = df.Define("reco_Whad_pz",       "Whad_reco.Pz()")
    df = df.Define("reco_Whad_costheta", "Whad_reco.CosTheta()")
    df = df.Define("reco_Whad_phi",      "Whad_reco.Phi()")

    df = df.Define("WW_reco", "(Wlep_reco + Whad_reco)")
    df = df.Define("reco_WW_m",               "WW_reco.M()")
    df = df.Define("reco_WW_m_minus_ecm",     "reco_WW_m - FCCAnalyses::WWFunctions::ECM")
    df = df.Define("reco_WW_px",              "WW_reco.Px()")
    df = df.Define("reco_WW_py",              "WW_reco.Py()")
    df = df.Define("reco_WW_pz",              "WW_reco.Pz()")
    df = df.Define("reco_WW_p_imbalance_tot", "WW_reco.P()")

    df = df.Define("deltaM",
        "FCCAnalyses::WWFunctions::deltaM(n_lep_reco, n_reco_jets, Wlep_reco, Whad_reco)")
    return df


# ── resolutions / responses (cross-level: reco − gen, reco / gen) ───────────
def define_resolutions(df):
    df = df.Define("lep_p_resp",         "reco_lep_p / gen_lep_p")
    df = df.Define("lep_theta_resol",    "reco_lep_theta - gen_lep_theta")
    df = df.Define("lep_phi_resol",      "TVector2::Phi_mpi_pi(reco_lep_phi - gen_lep_phi)")
    df = df.Define("lep_eta_resol",      "reco_lep_eta - gen_lep_eta")
    df = df.Define("lep_costheta_resol", "reco_lep_costheta - gen_lep_costheta")

    df = df.Define("met_p_resp",         "reco_met_p / gen_nu_p")
    df = df.Define("met_theta_resol",    "reco_met_theta - gen_nu_theta")
    df = df.Define("met_phi_resol",      "TVector2::Phi_mpi_pi(reco_met_phi - gen_nu_phi)")
    df = df.Define("met_eta_resol",      "reco_met_eta - gen_nu_eta")
    df = df.Define("met_costheta_resol", "reco_met_costheta - gen_nu_costheta")

    df = df.Define("jet1_p_resp",         "reco_jet1_p / gen_quark1_p")
    df = df.Define("jet1_theta_resol",    "reco_jet1_theta - gen_quark1_theta")
    df = df.Define("jet1_phi_resol",      "TVector2::Phi_mpi_pi(reco_jet1_phi - gen_quark1_phi)")
    df = df.Define("jet1_eta_resol",      "reco_jet1_eta - gen_quark1_eta")
    df = df.Define("jet1_costheta_resol", "reco_jet1_costheta - gen_quark1_costheta")
    df = df.Define("jet2_p_resp",         "reco_jet2_p / gen_quark2_p")
    df = df.Define("jet2_theta_resol",    "reco_jet2_theta - gen_quark2_theta")
    df = df.Define("jet2_phi_resol",      "TVector2::Phi_mpi_pi(reco_jet2_phi - gen_quark2_phi)")
    df = df.Define("jet2_eta_resol",      "reco_jet2_eta - gen_quark2_eta")
    df = df.Define("jet2_costheta_resol", "reco_jet2_costheta - gen_quark2_costheta")

    df = df.Define("Wlep_m_resol", "reco_Wlep_m - gen_Wlep_m")
    df = df.Define("Wlep_p_resol", "reco_Wlep_p - gen_Wlep_p")
    df = df.Define("Whad_m_resol", "reco_Whad_m - gen_Whad_m")
    df = df.Define("Whad_p_resol", "reco_Whad_p - gen_Whad_p")
    df = df.Define("WW_m_resol",   "reco_WW_m - gen_WW_m")
    df = df.Define("WW_px_resol",  "reco_WW_px - gen_WW_px")
    df = df.Define("WW_py_resol",  "reco_WW_py - gen_WW_py")
    df = df.Define("WW_pz_resol",  "reco_WW_pz - gen_WW_pz")
    return df


# ════════════════════════════════════════════════════════════════════════════
#  WW → 4q (fully hadronic) building blocks
#  ───────────────────────────────────────────────────────────────────────────
#  Parallel to the ℓνqq helpers above but for the all-hadronic channel: 4 jets,
#  4 gen quarks grouped by parent-W (p8 keeps the W in the MC history — see
#  WWFunctions::sel_quarks_fromW). Reuses select_isoleps, apply_channel_filter
#  (channel="had"), and define_beam_kinematics unchanged. FSR dressing is a
#  no-op (0 isolated leptons) so we cluster on the same reduced collection.
# ════════════════════════════════════════════════════════════════════════════

def cluster_jets_4q(df):
    # Exclusive kt into exactly 4 jets on the (lepton/FSR-removed, here ≡ full)
    # reconstructed collection. Jets are E-sorted (descending) like ℓνqq.
    helper = ExclusiveJetClusteringHelper("ReconstructedParticlesNoMuNoElNoFSR", 4)
    df = helper.define(df)
    df = df.Define("jets_p4",
        f"JetConstituentsUtils::compute_tlv_jets({helper.jets})")
    df = df.Define("jet1", "jets_p4[0]")
    df = df.Define("jet2", "jets_p4[1]")
    df = df.Define("jet3", "jets_p4[2]")
    df = df.Define("jet4", "jets_p4[3]")
    df = df.Define("n_reco_jets", "(int)jets_p4.size()")
    df = df.Filter("n_reco_jets == 4", "exactly 4 reco jets")
    # Durham dmerge scales (y_{n,n+1}) — jet-resolution observables, kept as
    # diagnostics / future background discriminants.
    df = df.Define("d_23", "JetClusteringUtils::get_exclusive_dmerge(_jet, 2)")
    df = df.Define("d_34", "JetClusteringUtils::get_exclusive_dmerge(_jet, 3)")
    df = df.Define("d_45", "JetClusteringUtils::get_exclusive_dmerge(_jet, 4)")
    return df, helper


def filter_genuine_4jet(df, sqrtd45_max=7.0):
    """Reco-level genuine-4-jet selection (standard ee 4-jet cut): reject hard
    5th-jet / radiative events via the Durham 4→5 splitting scale, requiring
    √d_45 < sqrtd45_max [GeV] (d_45 is in GeV²). Data-applicable — no gen truth.
    Studied on WW→4q (sqrt-d binning): the clean (well-matched) population
    dominates below √d_45 ≈ 5–7 GeV; the cut roughly doubles the all-4 jet→quark
    matching efficiency and the surviving 4-jet system is much better defined.
    sqrtd45_max<=0 disables the cut."""
    if sqrtd45_max and sqrtd45_max > 0:
        df = df.Filter(f"d_45 < {sqrtd45_max * sqrtd45_max}",
                       f"reco genuine 4-jet: sqrt(d_45) < {sqrtd45_max:g} GeV")
    return df


def define_reco_jets_kinematics_4q(df):
    for i in (1, 2, 3, 4):
        df = df.Define(f"reco_jet{i}_p",        f"jet{i}.P()")
        df = df.Define(f"reco_jet{i}_pt",       f"jet{i}.Pt()")
        df = df.Define(f"reco_jet{i}_theta",    f"jet{i}.Theta()")
        df = df.Define(f"reco_jet{i}_phi",      f"jet{i}.Phi()")
        df = df.Define(f"reco_jet{i}_eta",      f"jet{i}.Eta()")
        df = df.Define(f"reco_jet{i}_costheta", f"jet{i}.CosTheta()")
        df = df.Define(f"reco_jet{i}_mass",     f"jet{i}.M()")
        # Reco jets treated as massless (matches the kinfit (p,theta,phi) param).
        df = df.Define(f"jet{i}_massless",
            f"FCCAnalyses::WWFunctions::tlv_setmass(jet{i}, 0.)")
    return df


# ── gen-level (4 light quarks from the two W's; W present in p8 history) ──────
def select_gen_fromW(df):
    df = df.Alias("Particle0", "Particle#0.index")
    # W-grouped quarks: size 4 → [Wa_q0, Wa_q1, Wb_q0, Wb_q1]; empty if the
    # event is not a clean 2×(W→qq) topology (e.g. one W decayed leptonically).
    df = df.Define("gen_quarks_W",
        "FCCAnalyses::WWFunctions::sel_quarks_fromW()(Particle, Particle0)")
    df = df.Filter("gen_quarks_W.size() == 4",
                   "gen: 4 light quarks from 2 hadronic W's")
    return df


def select_gen_fromZ(df):
    """Gen-level ZZ→4q filter: exactly 4 quarks from 2 Z's. Used to restrict the
    inclusive ZZ sample to the fully-hadronic final state (the WW-hypothesis BW
    pairing control). No pairing truth — the gof discriminant is generator-blind."""
    df = df.Alias("Particle0", "Particle#0.index")
    df = df.Define("gen_quarks_Z",
        "FCCAnalyses::WWFunctions::sel_quarks_fromBoson(23)(Particle, Particle0)")
    df = df.Filter("gen_quarks_Z.size() == 4",
                   "gen: 4 quarks from 2 hadronic Z's")
    return df


def define_gen_kinematics_4q(df):
    df = df.Define("gen_quarks_W_tlv",
        "FCCAnalyses::MCParticle::get_tlv(gen_quarks_W)")
    df = df.Define("gen_q0_p4", "gen_quarks_W_tlv[0]")
    df = df.Define("gen_q1_p4", "gen_quarks_W_tlv[1]")
    df = df.Define("gen_q2_p4", "gen_quarks_W_tlv[2]")
    df = df.Define("gen_q3_p4", "gen_quarks_W_tlv[3]")

    # The two true W's (gen grouping): W1 = q0+q1, W2 = q2+q3. Quark masses kept.
    df = df.Define("W1_gen", "FCCAnalyses::WWFunctions::sum_p4({gen_q0_p4, gen_q1_p4})")
    df = df.Define("W2_gen", "FCCAnalyses::WWFunctions::sum_p4({gen_q2_p4, gen_q3_p4})")
    df = df.Define("WW_4q_gen",
        "FCCAnalyses::WWFunctions::sum_p4({gen_q0_p4, gen_q1_p4, gen_q2_p4, gen_q3_p4})")

    # W-grouped gen quark 4-vectors ([0,1]=W1, [2,3]=W2) — lets a downstream study
    # form the TRUE and both WRONG pairings + any angular separation, vs ISR.
    for i, src in [(0, "gen_q0_p4"), (1, "gen_q1_p4"), (2, "gen_q2_p4"), (3, "gen_q3_p4")]:
        df = df.Define(f"gen_qW{i}_px", f"{src}.Px()")
        df = df.Define(f"gen_qW{i}_py", f"{src}.Py()")
        df = df.Define(f"gen_qW{i}_pz", f"{src}.Pz()")
        df = df.Define(f"gen_qW{i}_e",  f"{src}.E()")

    for W, src in [("W1", "W1_gen"), ("W2", "W2_gen")]:
        df = df.Define(f"gen_{W}_m",  f"{src}.M()")
        df = df.Define(f"gen_{W}_p",  f"{src}.P()")
        df = df.Define(f"gen_{W}_pt", f"{src}.Pt()")
        df = df.Define(f"gen_{W}_px", f"{src}.Px()")
        df = df.Define(f"gen_{W}_py", f"{src}.Py()")
        df = df.Define(f"gen_{W}_pz", f"{src}.Pz()")

    # gen_WW_* names match the ℓνqq convention so define_beam_kinematics
    # (gen_WW_m_minus_m_ee) and downstream resolution code reuse unchanged.
    df = df.Define("gen_WW_m",               "WW_4q_gen.M()")
    df = df.Define("gen_WW_m_minus_ecm",     "gen_WW_m - FCCAnalyses::WWFunctions::ECM")
    df = df.Define("gen_WW_px",              "WW_4q_gen.Px()")
    df = df.Define("gen_WW_py",              "WW_4q_gen.Py()")
    df = df.Define("gen_WW_pz",              "WW_4q_gen.Pz()")
    df = df.Define("gen_WW_p_imbalance_tot", "WW_4q_gen.P()")
    return df


def match_jets_to_quarks_4q(df):
    # Global min-ΣΔR assignment: perm[i] = gen-quark index (0..3) matched to jet i.
    df = df.Define("jet_match_perm",
        "FCCAnalyses::WWFunctions::matchJets4(jet1, jet2, jet3, jet4, "
        "gen_q0_p4, gen_q1_p4, gen_q2_p4, gen_q3_p4)")

    for i in (1, 2, 3, 4):
        k = i - 1
        df = df.Define(f"gen_quark{i}_p4", f"gen_quarks_W_tlv[jet_match_perm[{k}]]")
        df = df.Define(f"jet{i}_matched_q_dR",
            f"(double)jet{i}.DeltaR(gen_quark{i}_p4)")
        df = df.Define(f"gen_quark{i}_p",        f"gen_quark{i}_p4.P()")
        df = df.Define(f"gen_quark{i}_pt",       f"gen_quark{i}_p4.Pt()")
        df = df.Define(f"gen_quark{i}_theta",    f"gen_quark{i}_p4.Theta()")
        df = df.Define(f"gen_quark{i}_phi",      f"gen_quark{i}_p4.Phi()")
        df = df.Define(f"gen_quark{i}_eta",      f"gen_quark{i}_p4.Eta()")
        df = df.Define(f"gen_quark{i}_costheta", f"gen_quark{i}_p4.CosTheta()")
        # W-group label of jet i (quarks 0,1 → W1 = 0; quarks 2,3 → W2 = 1).
        df = df.Define(f"jet{i}_wlab", f"(int)(jet_match_perm[{k}] >= 2)")

    # True reco-jet pairing index in {0,1,2} (−1 if matching doesn't split 2-2).
    df = df.Define("gen_pairing_true",
        "FCCAnalyses::WWFunctions::pairing_index_from_groups("
        "jet1_wlab, jet2_wlab, jet3_wlab, jet4_wlab)")
    return df


def define_match_quality_4q(df):
    """Global jet→quark matching-quality variables (Δθ, Δφ over all 4 pairs of the
    globally-chosen assignment). gen_match_dist = Σ_i √(Δθ_i²+Δφ_i²) is a single
    event-level matching score to cut on, replacing the 4 per-jet dR<0.1 cuts.
    gen_match_dmax = max_i √(Δθ_i²+Δφ_i²) is the worst single jet (≈ the per-jet cut)."""
    for i in (1, 2, 3, 4):
        df = df.Define(f"jet{i}_dtheta", f"(double)(jet{i}.Theta() - gen_quark{i}_theta)")
        df = df.Define(f"jet{i}_dphi",
            f"(double)TVector2::Phi_mpi_pi(jet{i}.Phi() - gen_quark{i}_phi)")
        df = df.Define(f"jet{i}_dang",
            f"std::sqrt(jet{i}_dtheta*jet{i}_dtheta + jet{i}_dphi*jet{i}_dphi)")
    df = df.Define("gen_match_dist", "jet1_dang + jet2_dang + jet3_dang + jet4_dang")
    df = df.Define("gen_match_dmax",
        "std::max(std::max(jet1_dang, jet2_dang), std::max(jet3_dang, jet4_dang))")
    return df


def define_resolutions_4q(df):
    for i in (1, 2, 3, 4):
        df = df.Define(f"jet{i}_p_resp",      f"reco_jet{i}_p / gen_quark{i}_p")
        df = df.Define(f"jet{i}_theta_resol", f"reco_jet{i}_theta - gen_quark{i}_theta")
        df = df.Define(f"jet{i}_phi_resol",
            f"TVector2::Phi_mpi_pi(reco_jet{i}_phi - gen_quark{i}_phi)")
        df = df.Define(f"jet{i}_eta_resol",      f"reco_jet{i}_eta - gen_quark{i}_eta")
        df = df.Define(f"jet{i}_costheta_resol", f"reco_jet{i}_costheta - gen_quark{i}_costheta")
    return df


_GW_MODE_TO_INT = {"fixed": 0, "constrained": 1, "free": 2}

# ── kinematic fit (step2 only) ───────────────────────────────────────────────
def run_kinfit(df, gw_mode="fixed"):
    if gw_mode not in _GW_MODE_TO_INT:
        raise ValueError(
            f"gw_mode={gw_mode!r} must be one of {list(_GW_MODE_TO_INT)}"
        )
    gw_mode_int = _GW_MODE_TO_INT[gw_mode]
    df = df.Define("kinfit",
        "FCCAnalyses::WWFunctions::kinFit("
        "reco_jet1_p, reco_jet1_theta, reco_jet1_phi,"
        "reco_jet2_p, reco_jet2_theta, reco_jet2_phi,"
        "reco_lep_p,  reco_lep_theta,  reco_lep_phi,"
        f"reco_met_p, reco_met_theta, reco_met_phi, {gw_mode_int})")

    for tag in ["mW","gW",
                "s1","s2","sl","sn",
                "t1","t2","tn","tl",
                "p1","p2","pn","pl",
                "bes_m_minus_ecm","bes_pz",
                "chi2","chi2_ndof","valid","valid_loose","status","edm",
                "winner_pass","n_passes_run","priors_swapped"]:
        df = df.Define(f"kinfit_{tag}", f"kinfit.{tag}")

    # Post-fit correlation matrix (16x16, row-major flat). Index order is
    # FCCAnalyses::WWFunctions::KF_PARAM_NAMES; entry [i*N+j] is corr(i,j).
    # Stored as a single ROOT::RVecF column to keep the per-event payload as
    # one branch instead of 120 scalars.
    df = df.Define("kinfit_corr",
        "ROOT::RVecF _c(FCCAnalyses::WWFunctions::KF_NPAR_TOTAL "
        "* FCCAnalyses::WWFunctions::KF_NPAR_TOTAL);"
        " for (int _i = 0; _i < FCCAnalyses::WWFunctions::KF_NPAR_TOTAL; ++_i)"
        "  for (int _j = 0; _j < FCCAnalyses::WWFunctions::KF_NPAR_TOTAL; ++_j)"
        "   _c[_i * FCCAnalyses::WWFunctions::KF_NPAR_TOTAL + _j] = kinfit.corr[_i][_j];"
        " return _c;")

    # Postfit scalars projected from TLVs in KinFitResult.
    for obj, src in [("jet1","j1"), ("jet2","j2"), ("lep","lep"), ("nu","nu")]:
        df = df.Define(f"kinfit_{obj}_p",     f"kinfit.{src}.P()")
        df = df.Define(f"kinfit_{obj}_pt",    f"kinfit.{src}.Pt()")
        df = df.Define(f"kinfit_{obj}_theta", f"kinfit.{src}.Theta()")
        df = df.Define(f"kinfit_{obj}_phi",   f"kinfit.{src}.Phi()")

    df = df.Define("Wlep_kinfit", "kinfit.lep + kinfit.nu")
    df = df.Define("Whad_kinfit", "kinfit.j1  + kinfit.j2")
    df = df.Define("WW_kinfit",   "Wlep_kinfit + Whad_kinfit")

    for W, src in [("Wlep","Wlep_kinfit"), ("Whad","Whad_kinfit")]:
        df = df.Define(f"kinfit_{W}_m",  f"{src}.M()")
        df = df.Define(f"kinfit_{W}_p",  f"{src}.P()")
        df = df.Define(f"kinfit_{W}_pt", f"{src}.Pt()")
        df = df.Define(f"kinfit_{W}_px", f"{src}.Px()")
        df = df.Define(f"kinfit_{W}_py", f"{src}.Py()")
        df = df.Define(f"kinfit_{W}_pz", f"{src}.Pz()")

    df = df.Define("kinfit_WW_m",               "WW_kinfit.M()")
    df = df.Define("kinfit_WW_m_minus_ecm",     "kinfit_WW_m - FCCAnalyses::WWFunctions::ECM")
    df = df.Define("kinfit_WW_px",              "WW_kinfit.Px()")
    df = df.Define("kinfit_WW_py",              "WW_kinfit.Py()")
    df = df.Define("kinfit_WW_pz",              "WW_kinfit.Pz()")
    df = df.Define("kinfit_WW_p_imbalance_tot", "WW_kinfit.P()")
    return df


# ── WW → 4q kinematic fit (step2 only) ───────────────────────────────────────
def run_kinfit_4q(df, gw_mode="constrained", with_truth=True):
    """Best-pairing 4q kinematic fit (WWKinReco4q.h). Runs all 3 jet→W
    partitions, keeps the lowest-χ² one, and emits pairing / χ² / W kinematics
    branches. with_truth=True also emits the gen-truth pairing-correctness flag
    (needs gen_pairing_true); set False for background / data-like samples where
    no W gen-truth exists (e.g. ZZ→4q, applying the WW hypothesis as a χ²
    discriminant)."""
    if gw_mode not in _GW_MODE_TO_INT:
        raise ValueError(f"gw_mode={gw_mode!r} must be one of {list(_GW_MODE_TO_INT)}")
    gw_mode_int = _GW_MODE_TO_INT[gw_mode]
    df = df.Define("kinfit4q",
        "FCCAnalyses::WWFunctions::kinFit4q_bestpairing("
        "reco_jet1_p, reco_jet1_theta, reco_jet1_phi,"
        "reco_jet2_p, reco_jet2_theta, reco_jet2_phi,"
        "reco_jet3_p, reco_jet3_theta, reco_jet3_phi,"
        f"reco_jet4_p, reco_jet4_theta, reco_jet4_phi, {gw_mode_int})")

    # Pairing-level outputs (the headline: which partition wins + χ² separation).
    df = df.Define("kinfit4q_pairing",           "kinfit4q.pairing")
    df = df.Define("kinfit4q_chi2_p0",           "kinfit4q.chi2_p0")
    df = df.Define("kinfit4q_chi2_p1",           "kinfit4q.chi2_p1")
    df = df.Define("kinfit4q_chi2_p2",           "kinfit4q.chi2_p2")
    df = df.Define("kinfit4q_dchi2",             "kinfit4q.dchi2")
    df = df.Define("kinfit4q_n_pairings_valid",  "kinfit4q.n_pairings_valid")

    # Winning-fit quality + parameters.
    for tag, expr in [
        ("chi2",        "kinfit4q.fit.chi2"),
        ("chi2_ndof",   "kinfit4q.fit.chi2_ndof"),
        ("valid",       "kinfit4q.fit.valid"),
        ("valid_loose", "kinfit4q.fit.valid_loose"),
        ("status",      "kinfit4q.fit.status"),
        ("edm",         "kinfit4q.fit.edm"),
        ("winner_pass", "kinfit4q.fit.winner_pass"),
        ("mW",          "kinfit4q.fit.mW"),
        ("gW",          "kinfit4q.fit.gW"),
    ]:
        df = df.Define(f"kinfit4q_{tag}", expr)

    # Post-fit W / WW kinematics (W_a = first pair, W_b = second pair).
    for W, src in [("Wa", "kinfit4q.Wa"), ("Wb", "kinfit4q.Wb"), ("WW", "kinfit4q.WW")]:
        df = df.Define(f"kinfit4q_{W}_m",  f"{src}.M()")
        df = df.Define(f"kinfit4q_{W}_p",  f"{src}.P()")
        df = df.Define(f"kinfit4q_{W}_pt", f"{src}.Pt()")
        df = df.Define(f"kinfit4q_{W}_pz", f"{src}.Pz()")

    # Pairing correctness vs gen truth (gen_pairing_true uses the same index
    # convention as kinFit4q_bestpairing / pairing_index_from_groups).
    if with_truth:
        df = df.Define("kinfit4q_pairing_correct",
            "(int)(gen_pairing_true >= 0 && kinfit4q.pairing == gen_pairing_true)")
    return df


# ── Standalone BW jet→W pairing (no kinematic fit; WWFunctions/BWPairing.h) ───
def run_bw_pairing(df, with_truth=True):
    """Fast BW pairing discriminant on the 4 reco jets. Emits the most-probable
    partition, the per-partition gof / posterior probability / di-jet masses, and
    (signal only) the gen-truth correctness flag. No Minuit — pure arithmetic."""
    df = df.Define("bwpair",
        "FCCAnalyses::WWFunctions::bwPairing(jet1, jet2, jet3, jet4)")
    df = df.Define("bwpair_pairing",   "bwpair.pairing")
    df = df.Define("bwpair_gof_best",  "bwpair.gof_best")
    df = df.Define("bwpair_prob_best", "bwpair.prob_best")
    df = df.Define("bwpair_dgof",      "bwpair.dgof")
    for k in range(3):
        df = df.Define(f"bwpair_gof{k}",  f"bwpair.gof[{k}]")
        df = df.Define(f"bwpair_prob{k}", f"bwpair.prob[{k}]")
        df = df.Define(f"bwpair_ma{k}",   f"bwpair.m_a[{k}]")
        df = df.Define(f"bwpair_mb{k}",   f"bwpair.m_b[{k}]")
    if with_truth:
        df = df.Define("bwpair_correct",
            "(int)(gen_pairing_true >= 0 && bwpair.pairing == gen_pairing_true)")
    return df


# ── DIAGNOSTIC: full fit on all 3 pairings + per-term χ² breakdown ────────────
# Emits, per jet→W partition k∈{0,1,2}: full-fit chi2 / valid / status / ndof and
# the 8-way per-term decomposition (bw, bes, isr, m_loss, scale_pen, angular, mw,
# gw). Heavier than production (3 full fits/event); use on a diagnostic subsample
# to root-cause the discriminant + chi2-magnitude. Returns the extra branch names
# so the caller can append them to the output list.
def run_kinfit_4q_diag(df, gw_mode="constrained"):
    gw_mode_int = _GW_MODE_TO_INT[gw_mode]
    df = df.Define("kf4qdiag",
        "FCCAnalyses::WWFunctions::kinFit4q_diag("
        "reco_jet1_p, reco_jet1_theta, reco_jet1_phi,"
        "reco_jet2_p, reco_jet2_theta, reco_jet2_phi,"
        "reco_jet3_p, reco_jet3_theta, reco_jet3_phi,"
        f"reco_jet4_p, reco_jet4_theta, reco_jet4_phi, {gw_mode_int})")
    branches = ["kf4qdiag_argmin"]
    df = df.Define("kf4qdiag_argmin", "kf4qdiag.argmin")
    _terms = ["bw", "bes", "isr", "m_loss", "scale_pen", "angular", "mw", "gw", "total", "gof"]
    for k in range(3):
        for tag, expr in [
            (f"chi2_p{k}",        f"(float)kf4qdiag.chi2[{k}]"),
            (f"valid_p{k}",       f"(int)kf4qdiag.valid[{k}]"),
            (f"valid_loose_p{k}", f"(int)kf4qdiag.valid_loose[{k}]"),
            (f"status_p{k}",      f"(int)kf4qdiag.status[{k}]"),
            (f"ndof_p{k}",        f"(float)kf4qdiag.ndof[{k}]"),
            (f"edm_p{k}",         f"(float)kf4qdiag.edm[{k}]"),
            (f"fast_status_p{k}", f"(int)kf4qdiag.fast_status[{k}]"),
            (f"fast_valid_p{k}",  f"(int)kf4qdiag.fast_valid[{k}]"),
            (f"fast_edm_p{k}",    f"(float)kf4qdiag.fast_edm[{k}]"),
        ]:
            name = f"kf4qdiag_{tag}"
            df = df.Define(name, expr); branches.append(name)
        for tm in _terms:
            name = f"kf4qdiag_t_{tm}_p{k}"
            df = df.Define(name, f"(float)kf4qdiag.terms[{k}].{tm}"); branches.append(name)
    return df, branches


