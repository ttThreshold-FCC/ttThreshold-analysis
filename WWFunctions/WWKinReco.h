#ifndef WWKinReco_H
#define WWKinReco_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <functional>
#include <random>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "kinfit_inputs/dcb_params.h"
#include "WWFunctions/WWFunctions.h"

namespace FCCAnalyses { namespace WWFunctions {

// Pull in DCB evaluators and fitted params from the generated headers.
using namespace ::WWFunctions;

// ── Per-ECM parameter bundles ───────────────────────────────────────────────
//
// All resolutions (jet/lep p_resp, θ/φ resolutions) are now BINNED — each holds
// a 5-element std::array of DCB params plus the equal-occupancy quantile edges
// (6 doubles). The per-event prior is selected at kinFit() entry by feeding the
// reco kinematic into pick_bin(): jet*_p / lep_p drive p_resp + θ_resol; jet
// |cosθ| drives jet φ_resol; lep_p drives lep φ_resol. MET and gen-level
// constraints remain unbinned (no kinematics dependence in the resol study).
constexpr std::size_t KF_NBINS = 5;

struct KinFitParamSet {
    // Binned variants (per-bin DCB params + equal-occupancy quantile edges).
    std::array<DcbGaussParams,        KF_NBINS> jet1_p_resp_bins;
    std::array<double,                KF_NBINS + 1> jet1_p_resp_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet2_p_resp_bins;
    std::array<double,                KF_NBINS + 1> jet2_p_resp_edges;
    std::array<DcbExpRight3GaussParams, KF_NBINS> lep_p_resp_bins;      // dcber3g (FSR-dressed)
    std::array<double,                  KF_NBINS + 1> lep_p_resp_edges;
    AsymGauss3GParams                            met_p_resp;
    std::array<DcbGaussParams,        KF_NBINS> jet1_phi_resol_bins;    // dcb2g
    std::array<double,                KF_NBINS + 1> jet1_phi_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet1_theta_resol_bins;  // dcb2g
    std::array<double,                KF_NBINS + 1> jet1_theta_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet2_phi_resol_bins;    // dcb2g
    std::array<double,                KF_NBINS + 1> jet2_phi_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> jet2_theta_resol_bins;  // dcb2g
    std::array<double,                KF_NBINS + 1> jet2_theta_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> lep_phi_resol_bins;     // dcb2g
    std::array<double,                KF_NBINS + 1> lep_phi_resol_edges;
    std::array<DcbGaussParams,        KF_NBINS> lep_theta_resol_bins;   // dcb2g
    std::array<double,                KF_NBINS + 1> lep_theta_resol_edges;
    AsymGauss3GParams                            met_phi_resol;
    AsymGauss3GParams                            met_theta_resol;
    // BES nuisances (Gaussian) — sum and asymmetry of beam-energy fluctuations.
    GaussParams                                  ee_m_minus_ecm;
    GaussParams                                  ee_pz;
    // ISR 3-momentum (spike_dcb2g) — delta-at-0 for no-ISR events + dcb_gauss
    // tail for radiated events. Px/Py from transverse balance, Pz from
    // longitudinal balance with the BES pz nuisance.
    SpikeDcbGaussParams                          isr_px;
    SpikeDcbGaussParams                          isr_py;
    SpikeDcbGaussParams                          isr_pz;
    // m_loss prior: spike_dcb2g fit on |m_loss| (data symmetrized in
    // fit_resolutions.py); kinfit adds a barrier for m_loss>0.
    SpikeDcbGaussParams                          ww_m_minus_m_ee;
    // Inclusive (kinematics-averaged) variants — used when kf_use_binned_priors=false.
    DcbGaussParams                               jet1_p_resp_incl;
    DcbGaussParams                               jet2_p_resp_incl;
    DcbExpRight3GaussParams                      lep_p_resp_incl;
    DcbGaussParams                               jet1_phi_resol_incl;
    DcbGaussParams                               jet1_theta_resol_incl;
    DcbGaussParams                               jet2_phi_resol_incl;
    DcbGaussParams                               jet2_theta_resol_incl;
    DcbGaussParams                               lep_phi_resol_incl;
    DcbGaussParams                               lep_theta_resol_incl;
};

// Three jet-prior conventions, selected at setKinFitParams() time:
//  - POOL  : jet1 and jet2 share the pooled prior DCBG_JET_*_BINS_{ECM}. No
//            pT-ordering dependence; jet1↔jet2 swap fallback is a no-op so it's
//            disabled.
//  - SWAP  : per-jet priors DCBG_JET{1,2}_*_BINS_{ECM}; swap fallback enabled
//            so events whose pT ordering disagrees with the prior fit ordering
//            can still converge.
//  - FIXED : per-jet priors DCBG_JET{1,2}_*_BINS_{ECM}; swap fallback disabled
//            (sanity check: how often do we lose convergence without swap?).
#define KF_POOL_BUNDLE(E) \
    DCBG_JET_P_RESP_BINS_##E,        DCBG_JET_P_RESP_EDGES_##E, \
    DCBG_JET_P_RESP_BINS_##E,        DCBG_JET_P_RESP_EDGES_##E, \
    DCBER3G_LEP_P_RESP_BINS_##E,      DCBER3G_LEP_P_RESP_EDGES_##E, \
    AG3G_MET_P_RESP_##E, \
    DCBG_JET_PHI_RESOL_BINS_##E,     DCBG_JET_PHI_RESOL_EDGES_##E, \
    DCBG_JET_THETA_RESOL_BINS_##E,   DCBG_JET_THETA_RESOL_EDGES_##E, \
    DCBG_JET_PHI_RESOL_BINS_##E,     DCBG_JET_PHI_RESOL_EDGES_##E, \
    DCBG_JET_THETA_RESOL_BINS_##E,   DCBG_JET_THETA_RESOL_EDGES_##E, \
    DCBG_LEP_PHI_RESOL_BINS_##E,     DCBG_LEP_PHI_RESOL_EDGES_##E, \
    DCBG_LEP_THETA_RESOL_BINS_##E,   DCBG_LEP_THETA_RESOL_EDGES_##E, \
    AG3G_MET_PHI_RESOL_##E,          AG3G_MET_THETA_RESOL_##E, \
    GAUSS_GEN_EE_M_MINUS_ECM_##E, GAUSS_GEN_EE_PZ_##E, \
    SDCBG_GEN_ISR_PX_##E, SDCBG_GEN_ISR_PY_##E, SDCBG_GEN_ISR_PZ_##E, \
    SDCBG_GEN_WW_M_MINUS_M_EE_##E, \
    /* inclusive — POOL: jet1 and jet2 share the pooled scalar */ \
    DCBG_JET_P_RESP_##E, DCBG_JET_P_RESP_##E, DCBER3G_LEP_P_RESP_##E, \
    DCBG_JET_PHI_RESOL_##E, DCBG_JET_THETA_RESOL_##E, \
    DCBG_JET_PHI_RESOL_##E, DCBG_JET_THETA_RESOL_##E, \
    DCBG_LEP_PHI_RESOL_##E, DCBG_LEP_THETA_RESOL_##E

#define KF_SEP_BUNDLE(E) \
    DCBG_JET1_P_RESP_BINS_##E,       DCBG_JET1_P_RESP_EDGES_##E, \
    DCBG_JET2_P_RESP_BINS_##E,       DCBG_JET2_P_RESP_EDGES_##E, \
    DCBER3G_LEP_P_RESP_BINS_##E,      DCBER3G_LEP_P_RESP_EDGES_##E, \
    AG3G_MET_P_RESP_##E, \
    DCBG_JET1_PHI_RESOL_BINS_##E,    DCBG_JET1_PHI_RESOL_EDGES_##E, \
    DCBG_JET1_THETA_RESOL_BINS_##E,  DCBG_JET1_THETA_RESOL_EDGES_##E, \
    DCBG_JET2_PHI_RESOL_BINS_##E,    DCBG_JET2_PHI_RESOL_EDGES_##E, \
    DCBG_JET2_THETA_RESOL_BINS_##E,  DCBG_JET2_THETA_RESOL_EDGES_##E, \
    DCBG_LEP_PHI_RESOL_BINS_##E,     DCBG_LEP_PHI_RESOL_EDGES_##E, \
    DCBG_LEP_THETA_RESOL_BINS_##E,   DCBG_LEP_THETA_RESOL_EDGES_##E, \
    AG3G_MET_PHI_RESOL_##E,          AG3G_MET_THETA_RESOL_##E, \
    GAUSS_GEN_EE_M_MINUS_ECM_##E, GAUSS_GEN_EE_PZ_##E, \
    SDCBG_GEN_ISR_PX_##E, SDCBG_GEN_ISR_PY_##E, SDCBG_GEN_ISR_PZ_##E, \
    SDCBG_GEN_WW_M_MINUS_M_EE_##E, \
    /* inclusive — SEP: per-jet scalar */ \
    DCBG_JET1_P_RESP_##E, DCBG_JET2_P_RESP_##E, DCBER3G_LEP_P_RESP_##E, \
    DCBG_JET1_PHI_RESOL_##E, DCBG_JET1_THETA_RESOL_##E, \
    DCBG_JET2_PHI_RESOL_##E, DCBG_JET2_THETA_RESOL_##E, \
    DCBG_LEP_PHI_RESOL_##E, DCBG_LEP_THETA_RESOL_##E

static const KinFitParamSet KF_PARAMS_157_POOL = { KF_POOL_BUNDLE(157) };
static const KinFitParamSet KF_PARAMS_160_POOL = { KF_POOL_BUNDLE(160) };
static const KinFitParamSet KF_PARAMS_163_POOL = { KF_POOL_BUNDLE(163) };
static const KinFitParamSet KF_PARAMS_157_SEP  = { KF_SEP_BUNDLE(157) };
static const KinFitParamSet KF_PARAMS_160_SEP  = { KF_SEP_BUNDLE(160) };
static const KinFitParamSet KF_PARAMS_163_SEP  = { KF_SEP_BUNDLE(163) };

// ── Active kinfit parameters (set per-dataset via setKinFitParams) ─────────
// Binned priors live in std::array<...> bundles plus matching edge arrays.
// kinFit() picks the per-event prior via pick_bin() at the function entry.
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet1_p_resp_bins      = DCBG_JET1_P_RESP_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet1_p_resp_edges      = DCBG_JET1_P_RESP_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet2_p_resp_bins      = DCBG_JET2_P_RESP_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet2_p_resp_edges      = DCBG_JET2_P_RESP_EDGES_160;
inline std::array<DcbExpRight3GaussParams, KF_NBINS> kf_lep_p_resp_bins       = DCBER3G_LEP_P_RESP_BINS_160;
inline std::array<double,                  KF_NBINS + 1> kf_lep_p_resp_edges       = DCBER3G_LEP_P_RESP_EDGES_160;
inline AsymGauss3GParams                            kf_met_p_resp            = AG3G_MET_P_RESP_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet1_phi_resol_bins   = DCBG_JET1_PHI_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet1_phi_resol_edges   = DCBG_JET1_PHI_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet1_theta_resol_bins = DCBG_JET1_THETA_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet1_theta_resol_edges = DCBG_JET1_THETA_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet2_phi_resol_bins   = DCBG_JET2_PHI_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet2_phi_resol_edges   = DCBG_JET2_PHI_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_jet2_theta_resol_bins = DCBG_JET2_THETA_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_jet2_theta_resol_edges = DCBG_JET2_THETA_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_lep_phi_resol_bins    = DCBG_LEP_PHI_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_lep_phi_resol_edges    = DCBG_LEP_PHI_RESOL_EDGES_160;
inline std::array<DcbGaussParams,        KF_NBINS> kf_lep_theta_resol_bins  = DCBG_LEP_THETA_RESOL_BINS_160;
inline std::array<double,                KF_NBINS + 1> kf_lep_theta_resol_edges  = DCBG_LEP_THETA_RESOL_EDGES_160;
inline AsymGauss3GParams                            kf_met_phi_resol         = AG3G_MET_PHI_RESOL_160;
inline AsymGauss3GParams                            kf_met_theta_resol       = AG3G_MET_THETA_RESOL_160;
inline GaussParams                                  kf_ee_m_minus_ecm        = GAUSS_GEN_EE_M_MINUS_ECM_160;
inline GaussParams                                  kf_ee_pz                 = GAUSS_GEN_EE_PZ_160;
inline SpikeDcbGaussParams                          kf_isr_px                = SDCBG_GEN_ISR_PX_160;
inline SpikeDcbGaussParams                          kf_isr_py                = SDCBG_GEN_ISR_PY_160;
inline SpikeDcbGaussParams                          kf_isr_pz                = SDCBG_GEN_ISR_PZ_160;
inline SpikeDcbGaussParams                          kf_ww_m_minus_m_ee       = SDCBG_GEN_WW_M_MINUS_M_EE_160;

// Inclusive (kinematics-averaged) scalars — used when kf_use_binned_priors=false.
// Same data as the bin arrays would collapse to with one bin spanning all events.
inline DcbGaussParams         kf_jet1_p_resp_incl      = DCBG_JET1_P_RESP_160;
inline DcbGaussParams         kf_jet2_p_resp_incl      = DCBG_JET2_P_RESP_160;
inline DcbExpRight3GaussParams kf_lep_p_resp_incl      = DCBER3G_LEP_P_RESP_160;
inline DcbGaussParams         kf_jet1_phi_resol_incl   = DCBG_JET1_PHI_RESOL_160;
inline DcbGaussParams         kf_jet1_theta_resol_incl = DCBG_JET1_THETA_RESOL_160;
inline DcbGaussParams         kf_jet2_phi_resol_incl   = DCBG_JET2_PHI_RESOL_160;
inline DcbGaussParams         kf_jet2_theta_resol_incl = DCBG_JET2_THETA_RESOL_160;
inline DcbGaussParams         kf_lep_phi_resol_incl    = DCBG_LEP_PHI_RESOL_160;
inline DcbGaussParams         kf_lep_theta_resol_incl  = DCBG_LEP_THETA_RESOL_160;

// True when the jet1↔jet2 swap fallback should be tried on non-converged
// events. Enabled only in "swap" mode; "pool" priors are jet-symmetric so swap
// is a no-op, and "fixed" mode deliberately skips the swap to measure the
// natural-ordering convergence rate alone.
inline bool kf_jet_swap_enabled = false;
// True → kinFit picks per-event prior from the binned arrays via pick_bin().
// False → use the inclusive (kinematics-averaged) scalar priors.
inline bool kf_use_binned_priors = true;

// ISR / neutrino treatment, selected at setKinFitParams() time:
//  - "mloss" (default): neutrino built from the measured MET (x[5,8,11] = MET
//            p_resp/θ/φ); ISR via the WW recoil priors; energy via the
//            |m_loss| spike-DCB + barrier. The original lnuqq fit.
//  - "kfit" : explicit ISR photon (px_γ,py_γ,k) occupies x[5,8,11] with the
//            measured ISR-momentum priors; the neutrino is DERIVED from
//            4-momentum conservation (p_ν = beam − visible − ISR, E_ν=|p_ν|,
//            massless). The MET measurement is dropped (≡ −visible). Energy
//            conservation enters as a single Gaussian closure residual
//            r = (ECM+bes_m) − E_WW − E_γ at the RECO-level width
//            KF_ISR_SYS_SIGMA. See jax_prototype/kinfit_lnuqq_jax.py (kfit).
inline std::string kf_isr_mode = "mloss";
// y-rescale (preconditioning) for the explicit ISR params in "kfit" mode,
// centered at 0 (ISR spectrum peaks at 0): px_γ,py_γ ~0.4, k longitudinal ~few.
static constexpr double KF_ISR_SX = 0.4;
static constexpr double KF_ISR_SY = 0.4;
static constexpr double KF_ISR_SZ = 2.0;
// Energy-closure prior width [GeV] for "kfit": the RECO-level closure spread
// (jet-energy-residual dominated, ~1.5) — NOT the truth ISR-system mass (~0.42),
// which over-constrains (see project_lnuqq_binned_jets_and_gof_tail).
static constexpr double KF_ISR_SYS_SIGMA = 1.5;
inline const double KF_ISR_SYS_LOG_NORM =
        std::log(2.0 * M_PI * KF_ISR_SYS_SIGMA * KF_ISR_SYS_SIGMA);

// Runtime knobs read by setKinFitParams (declared here so they precede that
// function's use of them). Full rationale at their use-sites below.
//  KF_SIMPLEX_MAX_CALLS: Simplex-pre-pass-only function-call cap (env
//    KF_SIMPLEX_MAXCALLS). The shared KF_MAX_FUNCTION_CALLS=100000 lets the
//    gradient-free Simplex grind to the ceiling every call (the dominant
//    per-event cost); a low cap slashes it and leaves Migrad/Hesse untouched.
//  KF_MW_FIX_VALUE: constrained-scan hypothesis [GeV] (env KF_MW_FIX). >0 pins
//    mW via FixVariable(0) — removes the flat mW<->jet-scale direction so the
//    fit converges ~100%; per-event chi2 then IS the profile -2lnL(mW), summed
//    over events across a grid to give the ensemble L(mW). <=0 = float mW.
inline int    KF_SIMPLEX_MAX_CALLS = 100000;
inline double KF_MW_FIX_VALUE      = -1.0;

// Forward-decl: setKinFitParams calls kf_init_logz_table() (defined below
// alongside the LogZTable definition) to allocate the precomputed log-Z grid.
// The init runs on the main thread before any RDataFrame workers exist, so
// no race; subsequent calls are no-ops.
static void kf_init_logz_table();

inline void setKinFitParams(int ecm, const std::string& jet_mode = "swap",
                             bool use_binned = true,
                             const std::string& isr_mode = "mloss") {
    ECM = static_cast<float>(ecm);
    const bool use_pool  = (jet_mode == "pool");
    const bool use_fixed = (jet_mode == "fixed");
    // Swap fallback only in "swap" (default) mode.
    kf_jet_swap_enabled  = !use_pool && !use_fixed;
    kf_use_binned_priors = use_binned;
    kf_isr_mode          = isr_mode;
    const KinFitParamSet* p =
        ecm == 157 ? (use_pool ? &KF_PARAMS_157_POOL : &KF_PARAMS_157_SEP) :
        ecm == 160 ? (use_pool ? &KF_PARAMS_160_POOL : &KF_PARAMS_160_SEP) :
        ecm == 163 ? (use_pool ? &KF_PARAMS_163_POOL : &KF_PARAMS_163_SEP) : nullptr;
    if (!p) return;
    kf_jet1_p_resp_bins       = p->jet1_p_resp_bins;
    kf_jet1_p_resp_edges      = p->jet1_p_resp_edges;
    kf_jet2_p_resp_bins       = p->jet2_p_resp_bins;
    kf_jet2_p_resp_edges      = p->jet2_p_resp_edges;
    kf_lep_p_resp_bins        = p->lep_p_resp_bins;
    kf_lep_p_resp_edges       = p->lep_p_resp_edges;
    kf_met_p_resp             = p->met_p_resp;
    kf_jet1_phi_resol_bins    = p->jet1_phi_resol_bins;
    kf_jet1_phi_resol_edges   = p->jet1_phi_resol_edges;
    kf_jet1_theta_resol_bins  = p->jet1_theta_resol_bins;
    kf_jet1_theta_resol_edges = p->jet1_theta_resol_edges;
    kf_jet2_phi_resol_bins    = p->jet2_phi_resol_bins;
    kf_jet2_phi_resol_edges   = p->jet2_phi_resol_edges;
    kf_jet2_theta_resol_bins  = p->jet2_theta_resol_bins;
    kf_jet2_theta_resol_edges = p->jet2_theta_resol_edges;
    kf_lep_phi_resol_bins     = p->lep_phi_resol_bins;
    kf_lep_phi_resol_edges    = p->lep_phi_resol_edges;
    kf_lep_theta_resol_bins   = p->lep_theta_resol_bins;
    kf_lep_theta_resol_edges  = p->lep_theta_resol_edges;
    kf_met_phi_resol          = p->met_phi_resol;
    kf_met_theta_resol        = p->met_theta_resol;
    kf_ee_m_minus_ecm         = p->ee_m_minus_ecm;
    kf_ee_pz                  = p->ee_pz;
    kf_isr_px                 = p->isr_px;
    kf_isr_py                 = p->isr_py;
    kf_isr_pz                 = p->isr_pz;
    kf_ww_m_minus_m_ee        = p->ww_m_minus_m_ee;
    kf_jet1_p_resp_incl       = p->jet1_p_resp_incl;
    kf_jet2_p_resp_incl       = p->jet2_p_resp_incl;
    kf_lep_p_resp_incl        = p->lep_p_resp_incl;
    kf_jet1_phi_resol_incl    = p->jet1_phi_resol_incl;
    kf_jet1_theta_resol_incl  = p->jet1_theta_resol_incl;
    kf_jet2_phi_resol_incl    = p->jet2_phi_resol_incl;
    kf_jet2_theta_resol_incl  = p->jet2_theta_resol_incl;
    kf_lep_phi_resol_incl     = p->lep_phi_resol_incl;
    kf_lep_theta_resol_incl   = p->lep_theta_resol_incl;
    // Simplex pre-pass call cap (default 100000 = unchanged). Bounds only the
    // gradient-free pre-pass; Migrad/Hesse keep KF_MAX_FUNCTION_CALLS.
    if (const char* e = std::getenv("KF_SIMPLEX_MAXCALLS")) {
        const int v = std::atoi(e);
        if (v > 0) KF_SIMPLEX_MAX_CALLS = v;
    }
    // Fixed-mW scan hypothesis (GeV). >0 pins mW; <=0 / unset floats it.
    if (const char* e = std::getenv("KF_MW_FIX")) {
        const double v = std::atof(e);
        KF_MW_FIX_VALUE = (v > 0.0) ? v : -1.0;
        if (KF_MW_FIX_VALUE > 0.0)
            std::fprintf(stderr, "[setKinFitParams] mW FIXED at %.4f GeV (constrained-scan mode)\n",
                         KF_MW_FIX_VALUE);
    }
    // Allocate the 3D log Z table on the heap if not yet built. Definition is
    // below the LogZTable / kf_build_logz_table block so we go through this
    // forward-declared init function.
    kf_init_logz_table();
}

// ── kinematic fit ──────────────────────────────────────────────────────────

// Kinematic fit constants.
// Momentum scale params (s1,s2,sl,sn) are response = p_reco/p_gen; angular
// params (t1,t2,tn,p1,p2,pn,tl,pl) are absolute shifts in radians. All priors
// (DCB family, GaussParams, SpikeDcbGaussParams) live in dcb_params.h and are
// selected per-dataset by setKinFitParams().
static constexpr double KF_MW_INIT = 80.419;
static constexpr double KF_GW_FIXED = 2.049;
// Pre-conditioning σ for mW (no Gaussian prior — y-rescale around KF_MW_INIT
// using a typical per-event posterior scale). gW uses KF_GW_PRIOR_SIGMA below.
static constexpr double KF_MW_PHYS_SIGMA = 2.0;
// (KF_MW_FIX_VALUE / constrained-scan mode declared above setKinFitParams.)
static constexpr int    KF_NDIM    = 15;   // free parameters when gW is fixed
                                            // (12 detector nuisances + mW + 2 BES)
// Total slots in the parameter array x[] (includes gW even when fixed).
static constexpr int    KF_NPAR_TOTAL = KF_NDIM + 1;
// Number of constraint terms in chi2: 4 momentum-response + 8 angular-resolution
// + 2 BES (m, pz) + 3 ISR (px, py, pz via balance) + 1 m(WW)−m(ee) + 2 BW.
// When gw_mode==Constrained a Gaussian prior on gW adds +1 constraint, applied at
// chi2_ndof time.
static constexpr int    KF_N_CONSTR = 20;
// "kfit" mode drops the 3 MET prior terms (p_resp + θ + φ) — the neutrino is
// derived, not measured — and replaces |m_loss| with 1 energy-closure term:
// 3 jet/lep p_resp + 6 jet/lep angular + 2 BES + 3 ISR + 1 closure + 2 BW = 17.
static constexpr int    KF_N_CONSTR_KFIT = 17;

// gW handling: pinned to KF_GW_FIXED, fitted with a Gaussian prior (σ =
// KF_GW_PRIOR_SIGMA), or fitted with no prior (data-only).
enum KFGwMode { KF_GW_FIXED_MODE = 0, KF_GW_CONSTRAINED_MODE = 1, KF_GW_FREE_MODE = 2 };

// Migrad / Minuit2 tuning. These get tweaked together when convergence
// behaviour changes (cascade order, status-3 recovery, etc.).
static constexpr int    KF_MAX_FUNCTION_CALLS = 100000;
// (KF_SIMPLEX_MAX_CALLS — Simplex-pre-pass-only cap — declared above
// setKinFitParams; it bounds only the gradient-free pre-pass, the dominant
// per-event cost, and leaves Migrad/Hesse on the full KF_MAX_FUNCTION_CALLS.)
// Runtime-settable (env KF_MIGRAD_TOLERANCE / KF_MIGRAD_STRATEGY via the 4q
// setKinFitParams4q; defaults preserve the original constexpr values).
inline double KF_MIGRAD_TOLERANCE = 1e-3;
inline int    KF_MIGRAD_STRATEGY  = 2;
// Initial Migrad finite-difference step in y-space (every parameter is
// y-rescaled to unit-σ priors). 0.01 = 1 % of prior σ — small enough to
// resolve posteriors that are 5–10 % of prior σ (BES, MET angular, lep tl)
// without spending too many extra evaluations on parameters whose posterior
// matches the prior.
static constexpr double KF_INIT_STEP = 0.01;

// Loose-valid EDM cap. Migrad tolerance is 1e-3 (set in configure); any status=3
// event with finite EDM under 10× that lands a fit point essentially indistinguishable
// from a converged one (matched chi²/ndof distributions; verified 2026-05-04).
static constexpr double KF_LOOSE_EDM_MAX = 1e-2;

// Random-restart pass. After the earlier passes have failed, re-run
// Simplex+Migrad from up to N draws of x_default jittered by 0.5σ in y-space.
// Seeded from the event's reco kinematics → reproducible across runs.
static constexpr int    KF_RESTART_N      = 2;
static constexpr double KF_RESTART_SIGMA  = 0.5;

// Barrier σ for m_loss > 0 (in units of the m_loss prior's sigma_res). Must
// match the value baked into `barrier_sigma` written by fit_resolutions.py.
static constexpr double KF_M_LOSS_BARRIER_SIGMA_FRAC = 0.1;

// Gaussian prior on gW (only active when gw_mode==KF_GW_CONSTRAINED_MODE). gW
// is also y-rescaled using this σ, so the prior collapses to y_gW² + log_norm
// in the χ². When gw_mode==KF_GW_FREE_MODE the same y-rescaling is used for
// pre-conditioning; only the prior term is dropped from the χ².
static constexpr double KF_GW_PRIOR_SIGMA_REL = 0.01;
static constexpr double KF_GW_PRIOR_SIGMA     = KF_GW_PRIOR_SIGMA_REL * KF_GW_FIXED;
// std::log isn't constexpr until C++26, so this is a runtime const initialized once.
inline const double KF_GW_PRIOR_LOG_NORM =
        std::log(2.0 * M_PI * KF_GW_PRIOR_SIGMA * KF_GW_PRIOR_SIGMA);

struct KinFitResult {
    float mW, gW;
    float s1, s2, sl, sn;
    float t1, t2, tn, tl;   // theta shifts: jet1, jet2, MET, lepton
    float p1, p2, pn, pl;   // phi shifts:   jet1, jet2, MET, lepton
    float chi2;
    float chi2_ndof;        // chi2 / (KF_N_CONSTR - n_free_params)
    int   status;           // raw minimizer status code (Minuit2: 0=OK, 1=PD-forced cov,
                            //   2=Hesse failed, 3=EDM>tol, 4=max calls, 5=other;
                            //   BFGS: 0=converged, 1=max-iter/LS-fail).
                            //   −1 if early-returned without fitting (invalid input p).
    int   valid;            // strict: status ∈ {0,1} (Migrad+Hesse clean)
    int   valid_loose;      // loose: valid OR (status==3 AND finite EDM AND edm<KF_LOOSE_EDM_MAX)
                            //   — Migrad-near-converged events whose EDM is within one
                            //   decade of tolerance. Empirically clean (~57-61% of
                            //   status=3 events; no chi²/ndof tail vs strict valid).
    // Diagnostics: which pass won and how many actually ran.
    //   winner_pass: 1=Simplex+Migrad-natural, 2=Simplex+Migrad-swapped,
    //                3=Hesse-refresh + Simplex+Migrad,
    //                4=random-restart Simplex+Migrad.
    //   n_passes_run: total passes executed before stopping (1..3+restarts).
    //   priors_swapped: 1 if the winning pass used jet1↔jet2-swapped priors.
    int   winner_pass;
    int   n_passes_run;
    int   priors_swapped;
    float edm;              // estimated distance to minimum at convergence (Minuit2)
    // Post-fit BES nuisances (depth-1 e+e- system): m_ee_fit − ECM (sum BES
    // component) and pz_ee_fit (asymmetry). Diagnostic.
    float bes_m_minus_ecm;
    float bes_pz;
    // Post-fit 4-vectors. All scalar projections (P, Pt, M, Px, ...) and the
    // Wlep/Whad/WW sums are derived in the consumer.
    TLorentzVector j1, j2, lep, nu;
    // Post-fit correlation matrix between the KF_NPAR_TOTAL fit parameters,
    // in y-space (the rescaling is diagonal so y/x-space correlations match).
    // Index order matches x[]: 0=mW, 1=gW, 2..5=s{1,2,l,n}, 6..8=t{1,2,n},
    // 9..11=p{1,2,n}, 12=tl, 13=pl, 14=bes_m, 15=bes_pz. Filled with NaN
    // when the fit hasn't run, when gW is fixed (its row/col), or when the
    // covariance is non-positive.
    float corr[KF_NPAR_TOTAL][KF_NPAR_TOTAL];
};

// Index order in KinFitResult::corr (and in Minuit2's parameter vector).
static constexpr const char* KF_PARAM_NAMES[KF_NPAR_TOTAL] = {
    "mW", "gW", "s1", "s2", "sl", "sn",
    "t1", "t2", "tn", "p1", "p2", "pn",
    "tl", "pl", "bes_m", "bes_pz",
};

// Massless 4-vector from spherical coordinates.
static TLorentzVector _vec_spherical(double p, double theta, double phi) {
    double st = std::sin(theta), ct = std::cos(theta);
    TLorentzVector v;
    v.SetPxPyPzE(p * st * std::cos(phi), p * st * std::sin(phi), p * ct, p);
    return v;
}

// Decode standardized y-coord to physical value: x = μ_prior + σ_prior · y.
// All prior PDFs (DcbParams, DcbGaussParams, DcbExpLeftGaussParams,
// DcbExpRightGaussParams, GaussParams) expose .mu/.sigma as their first two
// fields, so this template works for every kf_* struct. Preconditions the
// Hessian: in y-space all 14 nuisance directions (12 detector + 2 BES) have
// unit RMS, so Migrad's EDM tolerance is uniform. mW (and gW when free) stay
// in physical units.
template<typename PdfT>
static inline double _y2x(double y, const PdfT& p) { return p.mu + p.sigma * y; }

// Gauss-Legendre nodes / weights on [-1,1] (machine-precision). N=24 chosen
// from the gen-level convergence study (2026-05-05): max quadrature error on
// the log-Z integral is ≲30 MeV at all 3 ECMs (ecm157 27 MeV vs asymptotic).
// At N=24 with sym+branchless inner loop the helper costs ~½ of the N=32
// version — the speed/accuracy sweet spot for the inclusive m_W fit. Bump
// KF_GL_N to 32 if the running-width / Coulomb additions push us into a
// regime where ~30 MeV quadrature error matters.
static constexpr int KF_GL_N = 24;
static constexpr std::array<double, KF_GL_N> KF_GL24_X = {{
    -9.9518721999702131e-01, -9.7472855597130947e-01, -9.3827455200273280e-01, -8.8641552700440096e-01,
    -8.2000198597390295e-01, -7.4012419157855436e-01, -6.4809365193697555e-01, -5.4542147138883956e-01,
    -4.3379350762604513e-01, -3.1504267969616340e-01, -1.9111886747361631e-01, -6.4056892862605630e-02,
    +6.4056892862605630e-02, +1.9111886747361631e-01, +3.1504267969616340e-01, +4.3379350762604513e-01,
    +5.4542147138883956e-01, +6.4809365193697555e-01, +7.4012419157855436e-01, +8.2000198597390295e-01,
    +8.8641552700440096e-01, +9.3827455200273280e-01, +9.7472855597130947e-01, +9.9518721999702131e-01,
}};
static constexpr std::array<double, KF_GL_N> KF_GL24_W = {{
    +1.2341229799987091e-02, +2.8531388628933743e-02, +4.4277438817419551e-02, +5.9298584915436742e-02,
    +7.3346481411080411e-02, +8.6190161531953288e-02, +9.7618652104114065e-02, +1.0744427011596561e-01,
    +1.1550566805372561e-01, +1.2167047292780342e-01, +1.2583745634682830e-01, +1.2793819534675221e-01,
    +1.2793819534675221e-01, +1.2583745634682830e-01, +1.2167047292780342e-01, +1.1550566805372561e-01,
    +1.0744427011596561e-01, +9.7618652104114065e-02, +8.6190161531953288e-02, +7.3346481411080411e-02,
    +5.9298584915436742e-02, +4.4277438817419551e-02, +2.8531388628933743e-02, +1.2341229799987091e-02,
}};

// log Z(mW, gW, m_WW) — normalization integral of the BW × BW × √λ/s_WW joint
// PDF on the kinematic triangle {m_h+m_l < m_WW, m_h,m_l > 0}, as a function of
// the floating mW (and floating gW when fit_gW=true).
//
// Substitution t_i = atan((m_i² − mW²) / (mW·gW)) makes BW(m_i²) dm_i² = dt_i,
// so the BW peaks become uniform in t. Per-W limits:
//   t_min = atan(−mW / gW)            (m_i = 0)
//   t_max = atan((m_WW² − mW²) / (mW·gW))   (m_i = m_WW)
// At each (t_h, t_l) recover m_h, m_l = √(mW² + mW·gW · tan(t_i)). The integrand
// reduces to √λ / (4·m_h·m_l·s_WW) with the kinematic mask m_h+m_l < m_WW.
//
// Used by the BW chi² block to ADD 2·log Z to the un-normalized
// −2·log[BW·BW·√λ/s_WW] term, removing the mW-bias measured at gen level
// (~−1 GeV at ecm160) — see project_kinfit_bw_normalization.md.
//
// _ontf: direct on-the-fly evaluator. Used to BUILD the 3D table below; the
// kinfit chi² goes through the public log_Z_bw_phasespace which interpolates
// the table (50-100× faster per call). Available directly for debugging /
// validation against the table.
static inline double log_Z_bw_phasespace_ontf(double m_WW, double mW, double gW) {
    const double mwgw  = mW * gW;
    const double mW2   = mW * mW;
    const double s_ww  = m_WW * m_WW;
    const double t_min = std::atan(-mW2 / mwgw);
    const double t_max = std::atan((s_ww - mW2) / mwgw);
    const double half_d = 0.5 * (t_max - t_min);
    const double half_s = 0.5 * (t_max + t_min);

    // Precompute m and 1/m at each t-node (1D, shared between t_h and t_l axes).
    // Floor m² at 1e-12 to keep 1/m finite — the integrand is integrable at the
    // lower endpoint but the summand isn't (GL nodes are interior so this never
    // triggers in practice; defensive only).
    std::array<double, KF_GL_N> m_node, inv_m_node;
    for (int i = 0; i < KF_GL_N; ++i) {
        const double t  = half_d * KF_GL24_X[i] + half_s;
        const double m2 = mW2 + mwgw * std::tan(t);
        m_node[i]     = std::sqrt(std::max(m2, 1e-12));
        inv_m_node[i] = 1.0 / m_node[i];
    }

    // Integrand is symmetric under m_h ↔ m_l, so only sum the upper triangle
    // (i ≤ j) and double-count the off-diagonal. The kinematic mask is
    // equivalent to λ ≥ 0, so √λ via std::max keeps the inner loop branchless
    // and auto-vectorisable.
    double Z = 0.0;
    for (int i = 0; i < KF_GL_N; ++i) {
        const double m_h  = m_node[i];
        const double w_h  = KF_GL24_W[i];
        const double inv_mh_4sww = inv_m_node[i] / (4.0 * s_ww);
        // Diagonal i==j (single weight): dif=0, so λ = (s_ww − 4·m_h²)·s_ww.
        {
            const double sum2 = 4.0 * m_h * m_h;
            const double lam  = (s_ww - sum2) * s_ww;
            Z += w_h * w_h * std::sqrt(std::max(lam, 0.0)) * inv_mh_4sww * inv_m_node[i];
        }
        // Off-diagonal j > i (counted twice by symmetry)
        for (int j = i + 1; j < KF_GL_N; ++j) {
            const double m_l = m_node[j];
            const double sum2 = (m_h + m_l) * (m_h + m_l);
            const double dif2 = (m_h - m_l) * (m_h - m_l);
            const double lam  = (s_ww - sum2) * (s_ww - dif2);
            Z += 2.0 * w_h * KF_GL24_W[j] * std::sqrt(std::max(lam, 0.0))
                 * inv_mh_4sww * inv_m_node[j];
        }
    }
    Z *= half_d * half_d;                            // [t_min,t_max]² Jacobian
    return std::log(Z > 0.0 ? Z : 1e-300);
}

// ── 3D precompute table of log Z over (mW, gW, m_WW) ──────────────────────
//
// At ~1.8 µs per on-the-fly call, log_Z dominates the kinfit chi² (the rest
// of the chi² body is ~26 ns). Trilinear interpolation on a precomputed grid
// drops the per-call cost to ~13 ns — restoring v8 wall time. Out-of-grid
// values clamp to the nearest edge (graceful but biased near the boundary;
// Migrad never goes far from the grid in normal operation, see ranges below).
//
// Grid (chosen 2026-05-05). Trilinear interp error sub-MeV vs on-the-fly at
// gen-level (well below the ~30 MeV quadrature noise floor of the underlying
// GL24 rule). Range chosen wide enough to cover any plausible Migrad excursion
// without out-of-grid clamping:
//   mW  ∈ [50, 100]  GeV, 501 nodes (0.10 GeV)  — covers any per-event mW peak
//   gW  ∈ [1.95, 2.15] GeV, 21 nodes (0.01 GeV)  — covers ±5σ_prior of gW
//   mWW ∈ [100, 170] GeV, 281 nodes (0.25 GeV) — spans any plausible step1 m_WW
// Footprint: 2.96M doubles ≈ 23.6 MB. Build ~5 s (one-time, magic static).
static constexpr double KF_LOGZ_GRID_MW_LO  = 50.0,  KF_LOGZ_GRID_MW_HI  = 100.0;
static constexpr int    KF_LOGZ_GRID_N_MW   = 501;
static constexpr double KF_LOGZ_GRID_GW_LO  = 1.95,  KF_LOGZ_GRID_GW_HI  = 2.15;
static constexpr int    KF_LOGZ_GRID_N_GW   = 21;
static constexpr double KF_LOGZ_GRID_MWW_LO = 100.0, KF_LOGZ_GRID_MWW_HI = 170.0;
static constexpr int    KF_LOGZ_GRID_N_MWW  = 281;

struct LogZTable {
    double dmW, dgW, dmWW;
    double inv_dmW, inv_dgW, inv_dmWW;
    const double* data;  // mmap'd, row-major (mW, gW, m_WW); ((i*n_gW)+k)*n_mWW+j
};

// Magic number for the binary table file (must match tools/build_logz_table.cxx).
static constexpr uint64_t KF_LOGZ_TABLE_MAGIC = 0x4C5A544256303031ULL; // "LZTBV001"
static constexpr const char* KF_LOGZ_TABLE_PATH =
    "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco/kinfit_inputs/logz_table.bin";

// Global pointer, zero-initialized. setKinFitParams() mmaps the precomputed
// table file on first call (main thread, before RDataFrame spawns workers).
// Worker threads later dereference via log_Z_bw_phasespace. Plain pointer +
// kernel-managed mmap region avoids cling-JIT trouble with static-init of
// large objects (which segfaulted both magic-static and inline-global
// approaches earlier).
inline LogZTable* kf_logz_table_ptr = nullptr;

static void kf_init_logz_table() {
    if (kf_logz_table_ptr) return;
    int fd = ::open(KF_LOGZ_TABLE_PATH, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "[kinfit] cannot open log Z table %s: %s\n",
                     KF_LOGZ_TABLE_PATH, std::strerror(errno));
        return;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) { ::close(fd); return; }
    // Reject a truncated/stale table: the header-only check below cannot catch a
    // short write, and mmap over a short file SIGBUSes on reads past EOF.
    const long kf_logz_expect = 72L +
        (long)KF_LOGZ_GRID_N_MW * KF_LOGZ_GRID_N_GW * KF_LOGZ_GRID_N_MWW * (long)sizeof(double);
    if ((long)st.st_size != kf_logz_expect) {
        std::fprintf(stderr, "[kinfit] log Z table %s size %ld != expected %ld (truncated/stale, "
                     "rebuild via tools/build_logz_table); falling back to on-the-fly\n",
                     KF_LOGZ_TABLE_PATH, (long)st.st_size, kf_logz_expect);
        ::close(fd); return;
    }
    void* mmap_base = ::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mmap_base == MAP_FAILED) {
        std::fprintf(stderr, "[kinfit] mmap of log Z table failed: %s\n", std::strerror(errno));
        return;
    }
    // Validate header: magic + dimensions match compile-time constants.
    const char*   base  = static_cast<const char*>(mmap_base);
    uint64_t      magic;          std::memcpy(&magic,  base + 0,  sizeof(magic));
    int32_t       hdr_n[4];       std::memcpy(hdr_n,   base + 8,  sizeof(hdr_n));
    if (magic != KF_LOGZ_TABLE_MAGIC ||
        hdr_n[0] != KF_LOGZ_GRID_N_MW || hdr_n[1] != KF_LOGZ_GRID_N_GW ||
        hdr_n[2] != KF_LOGZ_GRID_N_MWW) {
        std::fprintf(stderr, "[kinfit] log Z table header mismatch (rebuild via tools/build_logz_table)\n");
        ::munmap(mmap_base, st.st_size);
        return;
    }
    LogZTable* T = new LogZTable;
    T->dmW  = (KF_LOGZ_GRID_MW_HI  - KF_LOGZ_GRID_MW_LO ) / (KF_LOGZ_GRID_N_MW  - 1);
    T->dgW  = (KF_LOGZ_GRID_GW_HI  - KF_LOGZ_GRID_GW_LO ) / (KF_LOGZ_GRID_N_GW  - 1);
    T->dmWW = (KF_LOGZ_GRID_MWW_HI - KF_LOGZ_GRID_MWW_LO) / (KF_LOGZ_GRID_N_MWW - 1);
    T->inv_dmW  = 1.0 / T->dmW;
    T->inv_dgW  = 1.0 / T->dgW;
    T->inv_dmWW = 1.0 / T->dmWW;
    // Header layout: 8 (magic) + 16 (4 int32) + 48 (6 doubles) = 72 bytes.
    T->data = reinterpret_cast<const double*>(base + 72);
    kf_logz_table_ptr = T;
    std::printf("[kinfit] log Z table mmap'd from %s (%ld bytes)\n",
                KF_LOGZ_TABLE_PATH, (long)st.st_size);
}

// DIAGNOSTIC 2026-05-05: print kf_logz_table_ptr seen by each worker thread
// once and abort. Tells us whether the value is null (fallback should engage),
// the correct heap address (lookup itself is broken), or garbage (cross-TU
// sharing failed — explains why fallback doesn't engage).
static inline double log_Z_bw_phasespace(double m_WW, double mW, double gW) {
    if (!kf_logz_table_ptr) return log_Z_bw_phasespace_ontf(m_WW, mW, gW);
    // NaN guard: Migrad's gradient probing can produce m_WW = sqrt(M2 < 0) = NaN.
    // Without this guard, (int)NaN below is UB and yields INT_MIN, making the
    // pointer arithmetic spray off the end of the mmap'd region → segfault.
    // Fall back to the on-the-fly evaluator (which propagates NaN cleanly).
    if (!std::isfinite(m_WW) || !std::isfinite(mW) || !std::isfinite(gW))
        return log_Z_bw_phasespace_ontf(m_WW, mW, gW);
    const LogZTable& T = *kf_logz_table_ptr;
    double fmW  = (mW   - KF_LOGZ_GRID_MW_LO ) * T.inv_dmW;
    double fgW  = (gW   - KF_LOGZ_GRID_GW_LO ) * T.inv_dgW;
    double fmWW = (m_WW - KF_LOGZ_GRID_MWW_LO) * T.inv_dmWW;
    if (fmW  < 0) fmW  = 0; else if (fmW  > KF_LOGZ_GRID_N_MW  - 1) fmW  = KF_LOGZ_GRID_N_MW  - 1;
    if (fgW  < 0) fgW  = 0; else if (fgW  > KF_LOGZ_GRID_N_GW  - 1) fgW  = KF_LOGZ_GRID_N_GW  - 1;
    if (fmWW < 0) fmWW = 0; else if (fmWW > KF_LOGZ_GRID_N_MWW - 1) fmWW = KF_LOGZ_GRID_N_MWW - 1;
    int i = (int)fmW;   if (i >= KF_LOGZ_GRID_N_MW  - 1) i = KF_LOGZ_GRID_N_MW  - 2;
    int k = (int)fgW;   if (k >= KF_LOGZ_GRID_N_GW  - 1) k = KF_LOGZ_GRID_N_GW  - 2;
    int j = (int)fmWW;  if (j >= KF_LOGZ_GRID_N_MWW - 1) j = KF_LOGZ_GRID_N_MWW - 2;
    const double wx = fmW - i, wy = fgW - k, wz = fmWW - j;
    const size_t row   = KF_LOGZ_GRID_N_MWW;
    const size_t plane = (size_t)KF_LOGZ_GRID_N_GW * KF_LOGZ_GRID_N_MWW;
    const double* p = T.data + (size_t)i * plane + (size_t)k * row + j;
    const double c000 = p[0],            c001 = p[1];
    const double c010 = p[row],          c011 = p[row + 1];
    const double c100 = p[plane],        c101 = p[plane + 1];
    const double c110 = p[plane + row],  c111 = p[plane + row + 1];
    const double c00 = (1-wz)*c000 + wz*c001;
    const double c01 = (1-wz)*c010 + wz*c011;
    const double c10 = (1-wz)*c100 + wz*c101;
    const double c11 = (1-wz)*c110 + wz*c111;
    const double c0 = (1-wy)*c00 + wy*c01;
    const double c1 = (1-wy)*c10 + wy*c11;
    return (1-wx)*c0 + wx*c1;
}


// ── Shared W-pair Breit-Wigner × phase-space normalized −2 ln PDF ──────────────
// Single source for the term used verbatim by the lnuqq kinFit chi², the 4q
// kinFit4q chi², and the 4q per-term diagnostic — factored so the lineshape
// convention (fixed-width BW × √λ/s_WW, normalized by log_Z over the kinematic
// triangle {m_h+m_l < m_WW}) cannot silently drift between the three sites and
// bias mW differently across channels (see project_kinfit_bw_normalization).
// Args: the two W (di-jet) masses m_h,m_l and s_WW = (Wh+Wl).M2(). Returns
// −2 ln[ BW(m_h)·BW(m_l)·√λ/s_WW / Z ], BW = mΓ/(δ²+m²Γ²), λ smoothly floored.
static inline double bw_phasespace_neg2ll(double mh, double ml, double s_ww,
                                          double mW, double gW) {
    const double mwgw = mW * gW;
    const double dh = mh*mh - mW*mW, dl = ml*ml - mW*mW;
    const double bw_h = mwgw / (dh*dh + mwgw*mwgw);
    const double bw_l = mwgw / (dl*dl + mwgw*mwgw);
    double lam = (s_ww - (mh+ml)*(mh+ml)) * (s_ww - (mh-ml)*(mh-ml));
    lam = std::sqrt(lam*lam + 1e-24);  // smooth |λ| floor — derivative continuous through 0
    return -2.0 * (std::log(bw_h) + std::log(bw_l))
         + 4.0 * std::log(M_PI)
         - std::log(lam) + 2.0 * std::log(s_ww)
         + 2.0 * log_Z_bw_phasespace(std::sqrt(s_ww), mW, gW);
}


KinFitResult kinFit(float jet1_p,    float jet1_theta,    float jet1_phi,
                    float jet2_p,    float jet2_theta,    float jet2_phi,
                    float Isolep_p,  float Isolep_theta,  float Isolep_phi,
                    float missing_p, float missing_p_theta, float missing_p_phi,
                    int gw_mode = KF_GW_FIXED_MODE) {

    const bool gw_free        = (gw_mode != KF_GW_FIXED_MODE);
    const bool gw_constrained = (gw_mode == KF_GW_CONSTRAINED_MODE);
    const bool kfit           = (kf_isr_mode == "kfit");  // explicit-ISR + derived-ν
    // Constrained-scan mode: pin mW at the hypothesis (y-coord) instead of floating.
    const bool   mw_fixed  = (KF_MW_FIX_VALUE > 0.0);
    const double x0_mW_fix = mw_fixed ? (KF_MW_FIX_VALUE - KF_MW_INIT) / KF_MW_PHYS_SIGMA : 0.0;

    KinFitResult result{};
    result.gW    = KF_GW_FIXED;
    result.valid = 0;
    result.status = -1;
    result.chi2  = 999.0f;
    result.chi2_ndof = 999.0f;
    for (int i = 0; i < KF_NPAR_TOTAL; ++i)
        for (int j = 0; j < KF_NPAR_TOTAL; ++j)
            result.corr[i][j] = std::numeric_limits<float>::quiet_NaN();

    if (Isolep_p < 0 || jet1_p <= 0 || jet2_p <= 0 || missing_p <= 0)
        return result;

    // Pick per-event priors. Jet priors are taken by VALUE so chi2fn can
    // capture them by reference. Each jet ALSO has an "alt" pick — the other
    // jet's prior at the SAME jet's own kinematics (binned: other binset, own
    // bin index; inclusive: other jet's scalar). swap_jet_priors() toggles
    // active <-> alt via std::swap (involution).
    const double j1_acth = std::abs(std::cos(jet1_theta));
    const double j2_acth = std::abs(std::cos(jet2_theta));
    DcbGaussParams p_jet1_p_resp      = kf_use_binned_priors ? pick_bin(kf_jet1_p_resp_bins,      kf_jet1_p_resp_edges,      jet1_p)   : kf_jet1_p_resp_incl;
    DcbGaussParams p_jet2_p_resp      = kf_use_binned_priors ? pick_bin(kf_jet2_p_resp_bins,      kf_jet2_p_resp_edges,      jet2_p)   : kf_jet2_p_resp_incl;
    DcbGaussParams p_jet1_theta_resol = kf_use_binned_priors ? pick_bin(kf_jet1_theta_resol_bins, kf_jet1_theta_resol_edges, jet1_p)   : kf_jet1_theta_resol_incl;
    DcbGaussParams p_jet2_theta_resol = kf_use_binned_priors ? pick_bin(kf_jet2_theta_resol_bins, kf_jet2_theta_resol_edges, jet2_p)   : kf_jet2_theta_resol_incl;
    DcbGaussParams p_jet1_phi_resol   = kf_use_binned_priors ? pick_bin(kf_jet1_phi_resol_bins,   kf_jet1_phi_resol_edges,   j1_acth)  : kf_jet1_phi_resol_incl;
    DcbGaussParams p_jet2_phi_resol   = kf_use_binned_priors ? pick_bin(kf_jet2_phi_resol_bins,   kf_jet2_phi_resol_edges,   j2_acth)  : kf_jet2_phi_resol_incl;
    DcbGaussParams p_jet1_p_resp_alt{};
    DcbGaussParams p_jet2_p_resp_alt{};
    DcbGaussParams p_jet1_theta_resol_alt{};
    DcbGaussParams p_jet2_theta_resol_alt{};
    DcbGaussParams p_jet1_phi_resol_alt{};
    DcbGaussParams p_jet2_phi_resol_alt{};
    if (kf_jet_swap_enabled) {
        p_jet1_p_resp_alt      = kf_use_binned_priors ? pick_bin(kf_jet2_p_resp_bins,      kf_jet2_p_resp_edges,      jet1_p)   : kf_jet2_p_resp_incl;
        p_jet2_p_resp_alt      = kf_use_binned_priors ? pick_bin(kf_jet1_p_resp_bins,      kf_jet1_p_resp_edges,      jet2_p)   : kf_jet1_p_resp_incl;
        p_jet1_theta_resol_alt = kf_use_binned_priors ? pick_bin(kf_jet2_theta_resol_bins, kf_jet2_theta_resol_edges, jet1_p)   : kf_jet2_theta_resol_incl;
        p_jet2_theta_resol_alt = kf_use_binned_priors ? pick_bin(kf_jet1_theta_resol_bins, kf_jet1_theta_resol_edges, jet2_p)   : kf_jet1_theta_resol_incl;
        p_jet1_phi_resol_alt   = kf_use_binned_priors ? pick_bin(kf_jet2_phi_resol_bins,   kf_jet2_phi_resol_edges,   j1_acth)  : kf_jet2_phi_resol_incl;
        p_jet2_phi_resol_alt   = kf_use_binned_priors ? pick_bin(kf_jet1_phi_resol_bins,   kf_jet1_phi_resol_edges,   j2_acth)  : kf_jet1_phi_resol_incl;
    }
    const DcbExpRight3GaussParams kf_lep_p_resp   = kf_use_binned_priors ? pick_bin(kf_lep_p_resp_bins,      kf_lep_p_resp_edges,      Isolep_p) : kf_lep_p_resp_incl;
    const DcbGaussParams        kf_lep_theta_resol = kf_use_binned_priors ? pick_bin(kf_lep_theta_resol_bins, kf_lep_theta_resol_edges, Isolep_p) : kf_lep_theta_resol_incl;
    const DcbGaussParams        kf_lep_phi_resol   = kf_use_binned_priors ? pick_bin(kf_lep_phi_resol_bins,   kf_lep_phi_resol_edges,   Isolep_p) : kf_lep_phi_resol_incl;

    // KF_NPAR_TOTAL parameters: x[0]=mW, x[1]=gW, x[2..5]=scales,
    // x[6..8]=jet/MET theta, x[9..11]=jet/MET phi, x[12..13]=lep angles,
    // x[14]=BES m, x[15]=BES pz.
    // When fit_gW=false, gW is pinned to KF_GW_FIXED (y_gW=0) via FixVariable(1).
    auto chi2fn = [=, &p_jet1_p_resp, &p_jet2_p_resp,
                       &p_jet1_theta_resol, &p_jet2_theta_resol,
                       &p_jet1_phi_resol,   &p_jet2_phi_resol](const double* x) -> double {
        // ALL 16 entries are standardized y-coords (y = (x_phys − μ)/σ): mW
        // around KF_MW_INIT with σ = KF_MW_PHYS_SIGMA, gW around KF_GW_FIXED
        // with σ = KF_GW_PRIOR_SIGMA, and 14 nuisances around their prior μ/σ.
        // Pre-conditions the Hessian to unit RMS in every direction.
        // s_i guard handles transient negative regions during Migrad line search.
        const double mW = KF_MW_INIT  + KF_MW_PHYS_SIGMA  * x[0];
        const double gW = KF_GW_FIXED + KF_GW_PRIOR_SIGMA * x[1];
        const double s1 = _y2x(x[2],  p_jet1_p_resp);
        const double s2 = _y2x(x[3],  p_jet2_p_resp);
        const double sl = _y2x(x[4],  kf_lep_p_resp);
        const double t1 = _y2x(x[6],  p_jet1_theta_resol);
        const double t2 = _y2x(x[7],  p_jet2_theta_resol);
        const double p1 = _y2x(x[9],  p_jet1_phi_resol);
        const double p2 = _y2x(x[10], p_jet2_phi_resol);
        const double tl = _y2x(x[12], kf_lep_theta_resol);
        const double pl = _y2x(x[13], kf_lep_phi_resol);
        const double bes_m  = _y2x(x[14], kf_ee_m_minus_ecm);   // = m_ee_fit − ECM
        const double bes_pz = _y2x(x[15], kf_ee_pz);            // = pz_ee_fit
        if (s1 <= 0.0 || s2 <= 0.0 || sl <= 0.0) return 1e10;

        TLorentzVector j1f = _vec_spherical(jet1_p/s1,   jet1_theta   - t1, jet1_phi   - p1);
        TLorentzVector j2f = _vec_spherical(jet2_p/s2,   jet2_theta   - t2, jet2_phi   - p2);
        TLorentzVector lf  = _vec_spherical(Isolep_p/sl, Isolep_theta - tl, Isolep_phi - pl);

        // Neutrino: "kfit" derives it from 4-momentum conservation against an
        // explicit ISR photon (px_γ,py_γ,k in slots x[5,8,11]); "mloss" builds it
        // from the measured MET. ISR / closure / MET-prior terms branch below.
        TLorentzVector nf;
        double pgx = 0.0, pgy = 0.0, kz = 0.0, Eg = 0.0;
        double met_scale_pen = 0.0, met_angular = 0.0;
        if (kfit) {
            pgx = KF_ISR_SX * x[5];  pgy = KF_ISR_SY * x[8];  kz = KF_ISR_SZ * x[11];
            Eg  = std::sqrt(pgx*pgx + pgy*pgy + kz*kz);
            const TLorentzVector Vis = j1f + j2f + lf;
            const double nux = -Vis.Px() - pgx;
            const double nuy = -Vis.Py() - pgy;
            const double nuz = bes_pz - Vis.Pz() - kz;
            nf.SetPxPyPzE(nux, nuy, nuz, std::sqrt(nux*nux + nuy*nuy + nuz*nuz));
        } else {
            const double sn = _y2x(x[5],  kf_met_p_resp);
            const double tn = _y2x(x[8],  kf_met_theta_resol);
            const double pn = _y2x(x[11], kf_met_phi_resol);
            if (sn <= 0.0) return 1e10;
            nf = _vec_spherical(missing_p/sn, missing_p_theta - tn, missing_p_phi - pn);
            met_scale_pen = asymgauss3g_neg2logpdf(sn, kf_met_p_resp);
            met_angular   = asymgauss3g_neg2logpdf(tn, kf_met_theta_resol)
                          + asymgauss3g_neg2logpdf(pn, kf_met_phi_resol);
        }

        TLorentzVector Wh = j1f + j2f;
        TLorentzVector Wl = lf  + nf;
        TLorentzVector WW = Wh  + Wl;

        // Joint BW × phase-space PDF, normalized over the kinematic triangle
        // {m_h+m_l < m_WW} so the per-event mW is unbiased (the log Z term) — see
        // project_kinfit_bw_normalization. Shared kernel (identical in kinFit4q).
        double bw_term = bw_phasespace_neg2ll(Wh.M(), Wl.M(), WW.M2(), mW, gW);

        // BES nuisance priors (Gaussian).
        double bes_term = gauss_neg2logpdf(bes_m,  kf_ee_m_minus_ecm)
                        + gauss_neg2logpdf(bes_pz, kf_ee_pz);

        // ISR + energy closure.
        //  kfit : priors on the explicit ISR photon (px_γ,py_γ,k) + a Gaussian
        //         energy-closure residual r = (ECM+bes_m) − E_WW − E_γ at the
        //         reco-level width KF_ISR_SYS_SIGMA.
        //  mloss: priors on the WW recoil (ISR_p = depth1_p − WW_p, px=py=0,
        //         pz=bes_pz) + |m_loss| spike-DCB with an m_loss>0 barrier.
        double isr_term, eclos_term;
        if (kfit) {
            isr_term = spike_dcb_gauss_neg2logpdf(pgx, kf_isr_px)
                     + spike_dcb_gauss_neg2logpdf(pgy, kf_isr_py)
                     + spike_dcb_gauss_neg2logpdf(kz,  kf_isr_pz);
            const double r = (ECM + bes_m) - WW.E() - Eg;
            eclos_term = (r / KF_ISR_SYS_SIGMA) * (r / KF_ISR_SYS_SIGMA) + KF_ISR_SYS_LOG_NORM;
        } else {
            isr_term = spike_dcb_gauss_neg2logpdf(-WW.Px(),         kf_isr_px)
                     + spike_dcb_gauss_neg2logpdf(-WW.Py(),         kf_isr_py)
                     + spike_dcb_gauss_neg2logpdf(bes_pz - WW.Pz(), kf_isr_pz);
            const double m_loss = WW.M() - (ECM + bes_m);
            eclos_term = spike_dcb_gauss_neg2logpdf(std::fabs(m_loss), kf_ww_m_minus_m_ee);
            if (m_loss > 0.0) {
                const double sigma_barrier =
                    kf_ww_m_minus_m_ee.sigma_res * KF_M_LOSS_BARRIER_SIGMA_FRAC;
                const double rb = m_loss / sigma_barrier;
                eclos_term += rb * rb;
            }
        }

        double scale_pen = dcb_gauss_neg2logpdf(s1, p_jet1_p_resp)
                         + dcb_gauss_neg2logpdf(s2, p_jet2_p_resp)
                         + dcb_expright_3gauss_neg2logpdf(sl, kf_lep_p_resp)
                         + met_scale_pen;   // MET p_resp term only in "mloss"

        double angular = dcb_gauss_neg2logpdf(t1, p_jet1_theta_resol)
                       + dcb_gauss_neg2logpdf(t2, p_jet2_theta_resol)
                       + dcb_gauss_neg2logpdf(tl, kf_lep_theta_resol)
                       + dcb_gauss_neg2logpdf(p1, p_jet1_phi_resol)
                       + dcb_gauss_neg2logpdf(p2, p_jet2_phi_resol)
                       + dcb_gauss_neg2logpdf(pl, kf_lep_phi_resol)
                       + met_angular;       // MET θ/φ terms only in "mloss"

        // gW prior (only in Constrained mode) collapses to y_gW² + log_norm
        // under the y-rescaling. In Free mode the prior is dropped; in Fixed
        // mode gW is pinned via FixVariable so the term is identically zero.
        double gw_term = gw_constrained ? (x[1] * x[1] + KF_GW_PRIOR_LOG_NORM) : 0.0;

        return bw_term + bes_term + isr_term + eclos_term + scale_pen + angular + gw_term;
    };

    std::function<double(const double*)> fObj = chi2fn;
    ROOT::Math::Functor functor(fObj, KF_NPAR_TOTAL);

    // Configure a Minuit2 minimizer (Migrad or Simplex) with a KF_NPAR_TOTAL-D
    // starting point. ALL variables are y-coords with σ=1 by construction:
    // 0=y_mW, 1=y_gW, 2..13=detector y-coords, 14..15=BES y-coords. Step 0.1
    // is uniformly ~10% of the prior (or pre-conditioning) σ. No limits — the
    // priors keep the fit in physical regions.
    auto configure = [&](ROOT::Math::Minimizer* m, const double* x0, bool with_strategy) {
        m->SetFunction(functor);
        m->SetMaxFunctionCalls(KF_MAX_FUNCTION_CALLS);
        m->SetTolerance(KF_MIGRAD_TOLERANCE);
        if (with_strategy) m->SetStrategy(KF_MIGRAD_STRATEGY);
        m->SetPrintLevel(-1);
        m->SetVariable(0,  "y_mW",     x0[0],  KF_INIT_STEP);
        m->SetVariable(1,  "y_gW",     x0[1],  KF_INIT_STEP);
        m->SetVariable(2,  "y_s1",     x0[2],  KF_INIT_STEP);
        m->SetVariable(3,  "y_s2",     x0[3],  KF_INIT_STEP);
        m->SetVariable(4,  "y_sl",     x0[4],  KF_INIT_STEP);
        m->SetVariable(5,  "y_sn",     x0[5],  KF_INIT_STEP);
        m->SetVariable(6,  "y_t1",     x0[6],  KF_INIT_STEP);
        m->SetVariable(7,  "y_t2",     x0[7],  KF_INIT_STEP);
        m->SetVariable(8,  "y_tn",     x0[8],  KF_INIT_STEP);
        m->SetVariable(9,  "y_p1",     x0[9],  KF_INIT_STEP);
        m->SetVariable(10, "y_p2",     x0[10], KF_INIT_STEP);
        m->SetVariable(11, "y_pn",     x0[11], KF_INIT_STEP);
        m->SetVariable(12, "y_tl",     x0[12], KF_INIT_STEP);
        m->SetVariable(13, "y_pl",     x0[13], KF_INIT_STEP);
        m->SetVariable(14, "y_bes_m",  x0[14], KF_INIT_STEP);
        m->SetVariable(15, "y_bes_pz", x0[15], KF_INIT_STEP);
        if (!gw_free) m->FixVariable(1);
        if (mw_fixed) m->FixVariable(0);   // constrained-scan: pin mW at hypothesis
    };

    double x_default[KF_NPAR_TOTAL] = {0,0, 0,0,0,0, 0,0,0, 0,0,0, 0,0, 0,0};
    if (mw_fixed) x_default[0] = x0_mW_fix;   // start (and stay) at the pinned mW
    std::unique_ptr<ROOT::Math::Minimizer> minimizer(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad")
    );

    // Single minimizer "mode": Simplex pre-pass + Migrad. Status 0 = minimum
    // found; 1 = covariance forced positive-definite (still valid postfit);
    // 3 = EDM > tol (the dominant residual failure). Tolerance was loosened
    // from 1e-6 to 1e-3 (Migrad default). Simplex (gradient-free) descends
    // through non-quadratic regions and avoids the Migrad-alone false-minimum
    // pathology seen at ecm163 (median fitted mW=79.05 with Migrad-alone vs
    // 80.18 with Simplex+Migrad).
    auto simplex_then_migrad = [&](const double* x_init) {
        std::unique_ptr<ROOT::Math::Minimizer> simplex(
            ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex")
        );
        configure(simplex.get(), x_init, /*with_strategy=*/false);
        simplex->SetMaxFunctionCalls(KF_SIMPLEX_MAX_CALLS);  // Simplex-only cap (see decl)
        simplex->Minimize();
        configure(minimizer.get(), simplex->X(), /*with_strategy=*/true);
        minimizer->Minimize();
        minimizer->Minimize();
        return minimizer->Status();
    };

    // Toggle the jet1↔jet2 prior assignment (momentum + theta + phi). Tests
    // the alternative jet-to-prior pairing for events whose pT-ordering
    // disagrees with the prior-fit ordering. Each std::swap exchanges the
    // active prior with its precomputed alt (other-jet binset, own kinematics)
    // — so the bin index per jet stays fixed; only the binset name flips.
    auto swap_jet_priors = [&]() {
        std::swap(p_jet1_p_resp,      p_jet1_p_resp_alt);
        std::swap(p_jet2_p_resp,      p_jet2_p_resp_alt);
        std::swap(p_jet1_theta_resol, p_jet1_theta_resol_alt);
        std::swap(p_jet2_theta_resol, p_jet2_theta_resol_alt);
        std::swap(p_jet1_phi_resol,   p_jet1_phi_resol_alt);
        std::swap(p_jet2_phi_resol,   p_jet2_phi_resol_alt);
    };

    struct PassResult { int status; double chi2; double edm; double x[KF_NPAR_TOTAL]; bool swapped; int pass_id; };
    auto snapshot = [&](int s, bool swapped_now, int pass_id) {
        PassResult r{};
        r.status   = s;
        r.chi2     = minimizer->MinValue();
        r.edm      = minimizer->Edm();
        r.swapped  = swapped_now;
        r.pass_id  = pass_id;
        const double* xref = minimizer->X();
        for (int i = 0; i < KF_NPAR_TOTAL; ++i) r.x[i] = xref[i];
        return r;
    };
    auto converged = [](int s) { return s == 0 || s == 1; };
    // Converged beats non-converged; otherwise lower chi² wins.
    auto pick_better = [&](PassResult& best, const PassResult& cand) {
        bool ob = converged(best.status), oc = converged(cand.status);
        if (oc && !ob)             { best = cand; return; }
        if (!oc && ob)             return;
        if (cand.chi2 < best.chi2) best = cand;
    };

    // Pass order: Simplex+Migrad always; cheap-Migrad-first removed because
    // Migrad-alone reports false success at biased mW basins (median 79.05 GeV
    // at ecm163 vs 80.18 GeV with Simplex pre-pass). Combined Minuit2
    // ("Minimize") has the same blindspot — it only falls back to Simplex when
    // Migrad reports failure, not when it converges to the wrong basin.
    //   1. Simplex+Migrad, natural priors
    //   2. Simplex+Migrad, swapped priors (only if kf_jet_swap_enabled)
    //   3. Hesse-refresh + Simplex+Migrad from best.x
    //   4. Random-restart Simplex+Migrad with jittered init
    int n_passes_run = 1;
    bool priors_swapped = false;
    PassResult best = snapshot(simplex_then_migrad(x_default), priors_swapped, /*pass=*/1);

    if (!converged(best.status) && kf_jet_swap_enabled) {
        swap_jet_priors(); priors_swapped = true;
        ++n_passes_run;
        pick_better(best, snapshot(simplex_then_migrad(x_default), priors_swapped, /*pass=*/2));
    }

    // Pass 3: Hesse refresh from best.x → Simplex+Migrad. Hesse recomputes the
    // Hessian numerically (breaks stale-Davidon plateaus and recovers from
    // NaN/Inf EDM). Then Simplex+Migrad descends from that refreshed point.
    if (!converged(best.status) && (!std::isfinite(best.edm) || best.edm < 1.0)) {
        if (priors_swapped != best.swapped) {
            swap_jet_priors();
            priors_swapped = best.swapped;
        }
        configure(minimizer.get(), best.x, /*with_strategy=*/true);
        minimizer->Hesse();
        ++n_passes_run;
        pick_better(best, snapshot(simplex_then_migrad(best.x), priors_swapped, /*pass=*/3));
    }

    // Pass 4: deterministic random-restart Simplex+Migrad. Catches events the
    // earlier passes leave with large or non-finite EDM. Seed is hashed from
    // event kinematics so the same event gets the same jitter on re-runs.
    if (!converged(best.status)) {
        if (priors_swapped != best.swapped) {
            swap_jet_priors();
            priors_swapped = best.swapped;
        }
        auto bits_of = [](double d) -> uint64_t {
            uint64_t b; std::memcpy(&b, &d, sizeof(b)); return b;
        };
        uint64_t s = bits_of(jet1_p) ^ (bits_of(jet2_p) << 1) ^ (bits_of(Isolep_p) << 2);
        std::mt19937_64 rng(s);
        std::normal_distribution<double> jitter(0.0, KF_RESTART_SIGMA);
        for (int t = 0; t < KF_RESTART_N; ++t) {
            double x_jitter[KF_NPAR_TOTAL];
            for (int i = 0; i < KF_NPAR_TOTAL; ++i) x_jitter[i] = x_default[i];
            for (int i = 2; i < KF_NPAR_TOTAL; ++i) x_jitter[i] += jitter(rng);  // jitter y-coords only
            ++n_passes_run;
            pick_better(best, snapshot(simplex_then_migrad(x_jitter), priors_swapped, /*pass=*/4));
            if (converged(best.status)) break;
        }
    }

    // Sync in-scope priors to the winner's orientation — result extraction
    // below reads them by reference.
    if (priors_swapped != best.swapped) swap_jet_priors();

    // Re-seed `minimizer` at the winning point and recompute the Hessian so
    // CovMatrix() reflects the winning pass (the live minimizer state may
    // belong to a later, losing pass — passes 2/3/4 reuse the same object).
    // Hesse() is a numerical Hessian at the supplied point, no extra descent.
    configure(minimizer.get(), best.x, /*with_strategy=*/true);
    minimizer->Hesse();

    int    status   = best.status;
    double chi2     = best.chi2;
    const double* x_final = best.x;

    result.status         = status;
    result.valid          = (status == 0 || status == 1) ? 1 : 0;
    result.valid_loose    = (result.valid
                             || (status == 3 && std::isfinite(best.edm)
                                 && best.edm < KF_LOOSE_EDM_MAX)) ? 1 : 0;
    result.chi2           = chi2;
    result.winner_pass    = best.pass_id;
    result.n_passes_run   = n_passes_run;
    result.priors_swapped = best.swapped ? 1 : 0;
    result.edm            = static_cast<float>(best.edm);
    int n_par    = gw_free ? KF_NPAR_TOTAL : KF_NDIM;
    if (mw_fixed) n_par -= 1;   // mW pinned, one fewer free parameter
    // +1 constraint from the gW Gaussian prior in Constrained mode only.
    int n_constr = (kfit ? KF_N_CONSTR_KFIT : KF_N_CONSTR) + (gw_constrained ? 1 : 0);
    result.chi2_ndof = (n_constr > n_par) ? result.chi2 / float(n_constr - n_par) : -1.0f;
    result.mW = static_cast<float>(KF_MW_INIT  + KF_MW_PHYS_SIGMA  * x_final[0]);
    result.gW = static_cast<float>(KF_GW_FIXED + KF_GW_PRIOR_SIGMA * x_final[1]);
    result.s1 = _y2x(x_final[2],  p_jet1_p_resp);
    result.s2 = _y2x(x_final[3],  p_jet2_p_resp);
    result.sl = _y2x(x_final[4],  kf_lep_p_resp);
    result.t1 = _y2x(x_final[6],  p_jet1_theta_resol);
    result.t2 = _y2x(x_final[7],  p_jet2_theta_resol);
    result.p1 = _y2x(x_final[9],  p_jet1_phi_resol);
    result.p2 = _y2x(x_final[10], p_jet2_phi_resol);
    result.tl = _y2x(x_final[12], kf_lep_theta_resol);
    result.pl = _y2x(x_final[13], kf_lep_phi_resol);
    // sn/tn/pn slots: "mloss" = MET p_resp/θ/φ; "kfit" = explicit ISR px_γ/py_γ/k.
    if (kfit) {
        result.sn = static_cast<float>(KF_ISR_SX * x_final[5]);
        result.tn = static_cast<float>(KF_ISR_SY * x_final[8]);
        result.pn = static_cast<float>(KF_ISR_SZ * x_final[11]);
    } else {
        result.sn = _y2x(x_final[5],  kf_met_p_resp);
        result.tn = _y2x(x_final[8],  kf_met_theta_resol);
        result.pn = _y2x(x_final[11], kf_met_phi_resol);
    }
    result.bes_m_minus_ecm = static_cast<float>(_y2x(x_final[14], kf_ee_m_minus_ecm));
    result.bes_pz          = static_cast<float>(_y2x(x_final[15], kf_ee_pz));

    // Post-fit correlation matrix from Minuit2's covariance at the winning x.
    // y-rescaling is diagonal so y-space and x-space correlations agree.
    // Off-diagonal entries with non-positive variances (e.g. fixed gW row/col)
    // are NaN, set above; here we overwrite only the well-defined slots.
    // Filled for valid_loose so strict-valid vs loose-only comparisons are
    // possible downstream (loose events are status=3 near-converged points;
    // Hesse above provides a numerical Hessian regardless of convergence).
    if (result.valid_loose) {
        for (int i = 0; i < KF_NPAR_TOTAL; ++i) {
            const double vii = minimizer->CovMatrix(i, i);
            if (!(vii > 0.0)) continue;
            result.corr[i][i] = 1.0f;
            for (int j = i + 1; j < KF_NPAR_TOTAL; ++j) {
                const double vjj = minimizer->CovMatrix(j, j);
                if (!(vjj > 0.0)) continue;
                const double rho = minimizer->CovMatrix(i, j) / std::sqrt(vii * vjj);
                result.corr[i][j] = static_cast<float>(rho);
                result.corr[j][i] = static_cast<float>(rho);
            }
        }
    }

    // Post-fit kinematics (shared — uses result fields filled above)
    TLorentzVector j1f = _vec_spherical(jet1_p/result.s1,    jet1_theta    - result.t1, jet1_phi    - result.p1);
    TLorentzVector j2f = _vec_spherical(jet2_p/result.s2,    jet2_theta    - result.t2, jet2_phi    - result.p2);
    TLorentzVector lf  = _vec_spherical(Isolep_p/result.sl,  Isolep_theta - result.tl, Isolep_phi - result.pl);
    TLorentzVector nf;
    if (kfit) {   // derive the neutrino from 4-mom conservation (sn/tn/pn hold ISR px/py/k)
        const TLorentzVector Vis = j1f + j2f + lf;
        const double nux = -Vis.Px() - result.sn;
        const double nuy = -Vis.Py() - result.tn;
        const double nuz = result.bes_pz - Vis.Pz() - result.pn;
        nf.SetPxPyPzE(nux, nuy, nuz, std::sqrt(nux*nux + nuy*nuy + nuz*nuz));
    } else {
        nf = _vec_spherical(missing_p/result.sn, missing_p_theta - result.tn, missing_p_phi - result.pn);
    }

    result.j1  = j1f;
    result.j2  = j2f;
    result.lep = lf;
    result.nu  = nf;
    return result;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
