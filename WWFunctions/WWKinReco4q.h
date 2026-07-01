#ifndef WWKinReco4q_H
#define WWKinReco4q_H

// ── WW → 4q (fully hadronic) kinematic fit ──────────────────────────────────
//
// Companion to WWKinReco.h (ℓνqq). Reuses ALL of that header's machinery via
// the include below: PDF struct defs + evaluators (dcb_params.h), the log-Z BW
// normalization table, _y2x / _vec_spherical, and the Minuit2 tuning constants.
// Only the fit topology differs: 4 measured jets, no lepton/MET, two HADRONIC
// W's, and a jet→W PAIRING choice (3 partitions → pick the lowest χ²).
//
// Parameter layout is the ℓνqq 16-slot layout with the 6 lepton/MET params
// (sl, sn, tl, tn, pl, pn) replaced by the 6 extra-jet params (s3, s4, t3, t4,
// φ3, φ4):
//   0 mW, 1 gW, 2..5 s1..s4, 6..9 t1..t4, 10..13 φ1..φ4, 14 bes_m, 15 bes_pz.
//
// mW is FLOATED with a tight Gaussian prior at the SM value (KF4Q_MW_PRIOR_SIGMA,
// ~10 MeV) — exactly like gW's "constrained" mode. The 4q chi² is meant primarily
// as a (a) jet→W pairing discriminant and (b) background discriminant, not a
// precision mW estimator.
//
// Priors. The pooled single-jet response/resolutions (DCBG_JET_*_4Q) apply to all
// four jets — no per-jet labeling or swap fallback. The WW-system BES+ISR+m_loss
// ARCHITECTURE is identical to ℓνqq so a future Whizard 4q sample (real BES +
// transverse ISR) drops in without restructuring. For the current p8 sample the
// degenerate components (BES ≈ 0, transverse ISR ≈ 0 / collinear) use VERY NARROW
// fixed Gaussian priors (KF4Q_*_PRIOR below); the non-degenerate longitudinal ISR
// (gen_isr_pz) and mass loss (gen_WW_m_minus_m_ee) use the fitted p8 priors.

#include "WWFunctions/WWKinReco.h"
#include "kinfit_inputs_4q/dcb_params_4q.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>

namespace FCCAnalyses { namespace WWFunctions {

// ── TEMPORARY profiling (export KF4Q_TIMING=1): split the per-event 4q fit cost
//    into Simplex / Migrad / Hesse. Effectively zero overhead when the env var
//    is unset (one cached-bool branch per scoped block). Prints a summary to
//    stderr at program exit. Remove once the bottleneck split is confirmed. ──
namespace kf4q_prof {
    inline std::atomic<long long> ns_simplex{0}, ns_migrad{0}, ns_hesse{0}, ns_event{0};
    inline std::atomic<long long> n_event{0}, n_simplex{0}, n_migrad{0}, n_hesse{0};
    inline bool enabled() { static const bool on = (std::getenv("KF4Q_TIMING") != nullptr); return on; }
    inline long long now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    struct Scoped {
        std::atomic<long long>* acc; std::atomic<long long>* cnt; long long t0; bool on;
        Scoped(std::atomic<long long>* a, std::atomic<long long>* c)
            : acc(a), cnt(c), t0(0), on(enabled()) { if (on) t0 = now_ns(); }
        ~Scoped() {
            if (!on) return;
            acc->fetch_add(now_ns() - t0, std::memory_order_relaxed);
            if (cnt) cnt->fetch_add(1, std::memory_order_relaxed);
        }
    };
    struct Reporter { ~Reporter(); };
    inline Reporter _reporter;   // static-duration: destructor fires at exit
    inline Reporter::~Reporter() {
        if (!enabled() || n_event.load() == 0) return;
        const long long ne = n_event.load();
        auto us = [](long long ns) { return ns / 1000.0; };
        std::fprintf(stderr,
          "\n[KF4Q_TIMING] events=%lld   per-event total=%.1f us\n"
          "   simplex: %10.1f us total   %7.2f us/call   %8lld calls   %7.1f us/event\n"
          "   migrad : %10.1f us total   %7.2f us/call   %8lld calls   %7.1f us/event\n"
          "   hesse  : %10.1f us total   %7.2f us/call   %8lld calls   %7.1f us/event\n",
          ne, us(ns_event.load()) / ne,
          us(ns_simplex.load()), us(ns_simplex.load()) / std::max(1LL, n_simplex.load()), n_simplex.load(), us(ns_simplex.load()) / ne,
          us(ns_migrad.load()),  us(ns_migrad.load())  / std::max(1LL, n_migrad.load()),  n_migrad.load(),  us(ns_migrad.load())  / ne,
          us(ns_hesse.load()),   us(ns_hesse.load())   / std::max(1LL, n_hesse.load()),   n_hesse.load(),   us(ns_hesse.load())   / ne);
    }
}

// ── 4q fit constants ────────────────────────────────────────────────────────
static constexpr int    KF4Q_NPAR_TOTAL = 16;   // mW, gW, 4 s, 4 t, 4 φ, 2 BES
static constexpr int    KF4Q_NDIM       = 15;   // free params when gW fixed
// Constraint terms: 4 jet-p + 8 jet-angular + 2 BES + 3 ISR + 1 m_loss + 2 BW
// + 1 mW prior. A floated/constrained gW adds +1 (applied at chi2_ndof time).
static constexpr int    KF4Q_N_CONSTR   = 21;

// mW handling. Two modes (set per-run by setKinFitParams4q from the environment):
//   - PRIOR mode (default): tight Gaussian prior on mW (SM ± KF4Q_MW_PRIOR_SIGMA);
//     mW is y-rescaled with this σ so the prior collapses to y_mW² + log_norm
//     (mirrors the gW prior). σ env-tunable via KF4Q_MW_PRIOR_SIGMA for the scan.
//   - FREE mode (env KF4Q_MW_FREE=1): NO Gaussian prior — mW floats for the mW
//     MEASUREMENT, y-rescaled by KF_MW_PHYS_SIGMA (2 GeV) exactly like the ℓνqq fit.
//     This also removes the stiffest Hessian direction (the 10 MeV prior) → better
//     convergence: the free-mW measurement goal and convergence goal are aligned.
inline double KF4Q_MW_PRIOR_SIGMA   = 0.010;   // 10 MeV (env KF4Q_MW_PRIOR_SIGMA)
inline double KF4Q_MW_PRIOR_LOG_NORM =
        std::log(2.0 * M_PI * 0.010 * 0.010);
inline bool   kf4q_mw_free = false;            // env KF4Q_MW_FREE=1 → no mW prior
// Active mW y-rescale: KF_MW_PHYS_SIGMA in free mode, else the prior σ.
inline double kf4q_mw_scale() { return kf4q_mw_free ? KF_MW_PHYS_SIGMA : KF4Q_MW_PRIOR_SIGMA; }

// VERY NARROW fixed Gaussian priors for the p8-degenerate WW-system components.
// p8 has negligible BES (gen_ee_* ≈ 0) and purely collinear ISR (gen_isr_px/py ≈ 0
// to machine precision), so these pin the corresponding nuisances near 0:
//   - bes_m, bes_pz : no beam-energy spread in p8.
//   - ISR px, py    : transverse-momentum balance (WW pT ≈ 0) — a near-exact
//                     conservation constraint at gen level.
// TODO: replace with priors FITTED from a Whizard 4q sample (real BES + ISR pT).
// σ kept small (enforce the constraint) but finite (Migrad y-rescaling stability).
inline const GaussParams KF4Q_BES_M_PRIOR  = {0.0, 0.05};
inline const GaussParams KF4Q_BES_PZ_PRIOR = {0.0, 0.05};
inline const GaussParams KF4Q_ISR_PX_PRIOR = {0.0, 0.10};
inline const GaussParams KF4Q_ISR_PY_PRIOR = {0.0, 0.10};

// Active 4q priors, set per-ECM by setKinFitParams4q(). Only ecm160 (p8) exists.
// The pooled-jet response/resolutions are BINNED in the jet's own reco kinematics
// (KF_NBINS=5 equal-occupancy bins from WWKinReco.h): p_resp & θ bin on jet p, φ
// on |cosθ| — mirroring the ℓνqq muon channel. Each of the 4 jets picks its own
// bin via pick_bin(); the inclusive scalars remain as a fallback (kf4q_use_binned).
inline bool kf4q_use_binned = true;
inline std::array<DcbGaussParams, KF_NBINS>     kf4q_jet_p_resp_bins       = WWFunctions4q::DCBG_JET_P_RESP_4Q_BINS_160;
inline std::array<double,         KF_NBINS + 1> kf4q_jet_p_resp_edges      = WWFunctions4q::DCBG_JET_P_RESP_4Q_EDGES_160;
inline std::array<DcbGaussParams, KF_NBINS>     kf4q_jet_theta_resol_bins  = WWFunctions4q::DCBG_JET_THETA_RESOL_4Q_BINS_160;
inline std::array<double,         KF_NBINS + 1> kf4q_jet_theta_resol_edges = WWFunctions4q::DCBG_JET_THETA_RESOL_4Q_EDGES_160;
inline std::array<DcbGaussParams, KF_NBINS>     kf4q_jet_phi_resol_bins    = WWFunctions4q::DCBG_JET_PHI_RESOL_4Q_BINS_160;
inline std::array<double,         KF_NBINS + 1> kf4q_jet_phi_resol_edges   = WWFunctions4q::DCBG_JET_PHI_RESOL_4Q_EDGES_160;
inline DcbGaussParams       kf4q_jet_p_resp_incl      = WWFunctions4q::DCBG_JET_P_RESP_4Q_160;
inline DcbGaussParams       kf4q_jet_theta_resol_incl = WWFunctions4q::DCBG_JET_THETA_RESOL_4Q_160;
inline DcbGaussParams       kf4q_jet_phi_resol_incl   = WWFunctions4q::DCBG_JET_PHI_RESOL_4Q_160;
inline SpikeDcbGaussParams  kf4q_isr_pz           = WWFunctions4q::SDCBG_GEN_ISR_PZ_160;
inline SpikeDcbGaussParams  kf4q_ww_m_minus_m_ee  = WWFunctions4q::SDCBG_GEN_WW_M_MINUS_M_EE_160;

// Closure mode (env KF4Q_ISR_MODE, set per-run by setKinFitParams4q):
//   "mloss" (default, original) : invariant-mass m_WW−m_ee spike-DCB prior + an
//                                 asymmetric barrier on the unphysical m_WW>m_ee side.
//   "kfit"                      : symmetric Gaussian ENERGY-balance closure
//                                 r = (ECM+bes_m) − E_WW − E_γ, replacing the m_loss
//                                 term. ISR stays recoil-derived (4q has no neutrino,
//                                 so momentum conservation already fixes ISR exactly —
//                                 the only lnuqq-kfit ingredient that transfers is the
//                                 barrier→symmetric-closure swap, which in ℓνqq removed
//                                 the −2 GeV Whad bias from the one-sided m_loss penalty).
// σ_R = reco-level energy-balance residual width; 4 jets ⇒ wider than the ℓνqq 1.5.
// Env KF4Q_SIGMA_R overrides; default to be MEASURED on the 4q sample (placeholder 3.0).
inline std::string kf4q_isr_mode      = "mloss";
inline double      KF4Q_ISR_SYS_SIGMA = 3.0;
inline double      KF4Q_ISR_SYS_LOG_NORM =
        std::log(2.0 * M_PI * KF4Q_ISR_SYS_SIGMA * KF4Q_ISR_SYS_SIGMA);

// Simplex pre-pass function-call cap (env KF4Q_SIMPLEX_MAXCALLS). The shared
// KF_MAX_FUNCTION_CALLS=100000 is applied to BOTH Simplex and Migrad; profiling
// (KF4Q_TIMING) shows Simplex hits that ceiling on ~every call (~100 ms/call =
// 96% of per-event time) because Nelder-Mead never satisfies its tolerance in
// 16-D. Migrad converges in ~1000 evals, so a low Simplex-only cap slashes the
// dominant cost while leaving Migrad untouched. Default 100000 = unchanged.
inline int         KF4Q_SIMPLEX_MAX_CALLS = 100000;

inline void setKinFitParams4q(int ecm, bool use_binned = true,
                              const std::string& isr_mode = "mloss") {
    ECM = static_cast<float>(ecm);
    kf4q_use_binned = use_binned;
    // Only the p8_ee_WW_ecm160 priors exist for now; other ECMs fall back to 160.
    kf4q_jet_p_resp_bins       = WWFunctions4q::DCBG_JET_P_RESP_4Q_BINS_160;
    kf4q_jet_p_resp_edges      = WWFunctions4q::DCBG_JET_P_RESP_4Q_EDGES_160;
    kf4q_jet_theta_resol_bins  = WWFunctions4q::DCBG_JET_THETA_RESOL_4Q_BINS_160;
    kf4q_jet_theta_resol_edges = WWFunctions4q::DCBG_JET_THETA_RESOL_4Q_EDGES_160;
    kf4q_jet_phi_resol_bins    = WWFunctions4q::DCBG_JET_PHI_RESOL_4Q_BINS_160;
    kf4q_jet_phi_resol_edges   = WWFunctions4q::DCBG_JET_PHI_RESOL_4Q_EDGES_160;
    kf4q_jet_p_resp_incl       = WWFunctions4q::DCBG_JET_P_RESP_4Q_160;
    kf4q_jet_theta_resol_incl  = WWFunctions4q::DCBG_JET_THETA_RESOL_4Q_160;
    kf4q_jet_phi_resol_incl    = WWFunctions4q::DCBG_JET_PHI_RESOL_4Q_160;
    kf4q_isr_pz          = WWFunctions4q::SDCBG_GEN_ISR_PZ_160;
    kf4q_ww_m_minus_m_ee = WWFunctions4q::SDCBG_GEN_WW_M_MINUS_M_EE_160;

    // Closure mode + energy-balance σ_R (kfit mode). Arg wins; env is the override.
    kf4q_isr_mode = isr_mode;
    if (const char* e = std::getenv("KF4Q_ISR_MODE")) kf4q_isr_mode = e;
    if (const char* e = std::getenv("KF4Q_SIGMA_R")) {
        const double v = std::atof(e);
        if (v > 0.0) KF4Q_ISR_SYS_SIGMA = v;
    }
    KF4Q_ISR_SYS_LOG_NORM =
        std::log(2.0 * M_PI * KF4Q_ISR_SYS_SIGMA * KF4Q_ISR_SYS_SIGMA);

    // Per-run tuning from the environment (defaults preserve current behavior).
    // mW: free-floating (measurement) or tunable-σ prior (convergence scan).
    if (const char* e = std::getenv("KF4Q_MW_FREE"))
        kf4q_mw_free = (std::atoi(e) != 0);
    if (const char* e = std::getenv("KF4Q_MW_PRIOR_SIGMA")) {
        const double v = std::atof(e);
        if (v > 0.0) KF4Q_MW_PRIOR_SIGMA = v;
    }
    KF4Q_MW_PRIOR_LOG_NORM =
        std::log(2.0 * M_PI * KF4Q_MW_PRIOR_SIGMA * KF4Q_MW_PRIOR_SIGMA);
    // Minimizer numerics (shared KF_* knobs; only this 4q process is affected).
    if (const char* e = std::getenv("KF_MIGRAD_TOLERANCE")) {
        const double v = std::atof(e);
        if (v > 0.0) KF_MIGRAD_TOLERANCE = v;
    }
    if (const char* e = std::getenv("KF_MIGRAD_STRATEGY"))
        KF_MIGRAD_STRATEGY = std::atoi(e);
    if (const char* e = std::getenv("KF4Q_SIMPLEX_MAXCALLS")) {
        const int v = std::atoi(e);
        if (v > 0) KF4Q_SIMPLEX_MAX_CALLS = v;
    }
    std::fprintf(stderr,
        "[setKinFitParams4q] mW_free=%d  mW_prior_sigma=%.4g  migrad_tol=%.1e  strategy=%d  "
        "isr_mode=%s  sigma_R=%.3g\n",
        (int)kf4q_mw_free, KF4Q_MW_PRIOR_SIGMA, KF_MIGRAD_TOLERANCE, KF_MIGRAD_STRATEGY,
        kf4q_isr_mode.c_str(), KF4Q_ISR_SYS_SIGMA);

    kf_init_logz_table();   // shared log-Z table (range covers ecm160 m_WW)
}

// Index order in KinFit4qResult::corr (matches the Minuit2 parameter vector).
static constexpr const char* KF4Q_PARAM_NAMES[KF4Q_NPAR_TOTAL] = {
    "mW", "gW", "s1", "s2", "s3", "s4",
    "t1", "t2", "t3", "t4", "p1", "p2", "p3", "p4",
    "bes_m", "bes_pz",
};

struct KinFit4qResult {
    float mW, gW;
    float s1, s2, s3, s4;      // jet momentum responses (p_reco/p_fit)
    float t1, t2, t3, t4;      // jet theta shifts
    float p1, p2, p3, p4;      // jet phi shifts
    float bes_m_minus_ecm, bes_pz;
    float chi2;
    float chi2_ndof;
    int   status;
    int   valid;               // status ∈ {0,1}
    int   valid_loose;         // valid OR (status==3 AND finite EDM < KF_LOOSE_EDM_MAX)
    int   winner_pass;
    int   n_passes_run;
    float edm;
    // Post-fit jets in THIS fit's W_a=(j1,j2), W_b=(j3,j4) convention.
    TLorentzVector j1, j2, j3, j4;
    float corr[KF4Q_NPAR_TOTAL][KF4Q_NPAR_TOTAL];
};

// Single-pairing 4q fit: treats (jet1,jet2) → W_a and (jet3,jet4) → W_b.
inline KinFit4qResult kinFit4q(
        float jet1_p, float jet1_theta, float jet1_phi,
        float jet2_p, float jet2_theta, float jet2_phi,
        float jet3_p, float jet3_theta, float jet3_phi,
        float jet4_p, float jet4_theta, float jet4_phi,
        int gw_mode = KF_GW_CONSTRAINED_MODE,
        bool fast = false) {

    // fast=true: cheap pairing-discriminant fit — a single strategy-0 Migrad from
    // the default start, NO Simplex pre-pass / retry ladder / Hesse-covariance.
    // Only chi2 (+ a rough validity flag) are meaningful; used by the best-pairing
    // wrapper to rank the 3 jet→W partitions before the full fit on the winner.
    const bool gw_free        = (gw_mode != KF_GW_FIXED_MODE);
    const bool gw_constrained = (gw_mode == KF_GW_CONSTRAINED_MODE);

    KinFit4qResult result{};
    result.gW    = KF_GW_FIXED;
    result.valid = 0;
    result.status = -1;
    result.chi2  = 999.0f;
    result.chi2_ndof = 999.0f;
    for (int i = 0; i < KF4Q_NPAR_TOTAL; ++i)
        for (int j = 0; j < KF4Q_NPAR_TOTAL; ++j)
            result.corr[i][j] = std::numeric_limits<float>::quiet_NaN();

    if (jet1_p <= 0 || jet2_p <= 0 || jet3_p <= 0 || jet4_p <= 0)
        return result;

    // Per-jet priors: each jet selects its own bin from the pooled binned arrays
    // (p_resp & θ on the jet's p; φ on its |cosθ|). Picked once at entry, captured
    // by value in chi2fn and reused in the result extraction below.
    const double j1_acth = std::abs(std::cos(jet1_theta));
    const double j2_acth = std::abs(std::cos(jet2_theta));
    const double j3_acth = std::abs(std::cos(jet3_theta));
    const double j4_acth = std::abs(std::cos(jet4_theta));
    const DcbGaussParams pr1 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet1_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams pr2 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet2_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams pr3 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet3_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams pr4 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet4_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams th1 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet1_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams th2 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet2_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams th3 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet3_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams th4 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet4_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams ph1 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j1_acth) : kf4q_jet_phi_resol_incl;
    const DcbGaussParams ph2 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j2_acth) : kf4q_jet_phi_resol_incl;
    const DcbGaussParams ph3 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j3_acth) : kf4q_jet_phi_resol_incl;
    const DcbGaussParams ph4 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j4_acth) : kf4q_jet_phi_resol_incl;

    auto chi2fn = [=](const double* x) -> double {
        const double mW = KF_MW_INIT  + kf4q_mw_scale() * x[0];
        const double gW = KF_GW_FIXED + KF_GW_PRIOR_SIGMA   * x[1];
        const double s1 = _y2x(x[2],  pr1);
        const double s2 = _y2x(x[3],  pr2);
        const double s3 = _y2x(x[4],  pr3);
        const double s4 = _y2x(x[5],  pr4);
        const double t1 = _y2x(x[6],  th1);
        const double t2 = _y2x(x[7],  th2);
        const double t3 = _y2x(x[8],  th3);
        const double t4 = _y2x(x[9],  th4);
        const double q1 = _y2x(x[10], ph1);
        const double q2 = _y2x(x[11], ph2);
        const double q3 = _y2x(x[12], ph3);
        const double q4 = _y2x(x[13], ph4);
        const double bes_m  = _y2x(x[14], KF4Q_BES_M_PRIOR);
        const double bes_pz = _y2x(x[15], KF4Q_BES_PZ_PRIOR);
        if (s1 <= 0.0 || s2 <= 0.0 || s3 <= 0.0 || s4 <= 0.0) return 1e10;

        TLorentzVector j1f = _vec_spherical(jet1_p/s1, jet1_theta - t1, jet1_phi - q1);
        TLorentzVector j2f = _vec_spherical(jet2_p/s2, jet2_theta - t2, jet2_phi - q2);
        TLorentzVector j3f = _vec_spherical(jet3_p/s3, jet3_theta - t3, jet3_phi - q3);
        TLorentzVector j4f = _vec_spherical(jet4_p/s4, jet4_theta - t4, jet4_phi - q4);

        TLorentzVector Wa = j1f + j2f;
        TLorentzVector Wb = j3f + j4f;
        TLorentzVector WW = Wa + Wb;

        // BW × BW × phase-space, normalized via the shared log-Z table.
        // Shared kernel bw_phasespace_neg2ll (WWKinReco.h) — identical to lnuqq kinFit.
        double bw_term = bw_phasespace_neg2ll(Wa.M(), Wb.M(), WW.M2(), mW, gW);

        // BES priors (narrow Gaussian — p8 has ≈0 BES).
        double bes_term = gauss_neg2logpdf(bes_m,  KF4Q_BES_M_PRIOR)
                        + gauss_neg2logpdf(bes_pz, KF4Q_BES_PZ_PRIOR);

        // ISR via 4-momentum balance with the depth-1 e+e- (px=py=0, pz=bes_pz).
        // Transverse: narrow-Gaussian (pT conservation). Longitudinal: fitted p8 prior.
        const double isr_px_val = -WW.Px();
        const double isr_py_val = -WW.Py();
        const double isr_pz_val = bes_pz - WW.Pz();
        double isr_term = gauss_neg2logpdf(isr_px_val, KF4Q_ISR_PX_PRIOR)
                        + gauss_neg2logpdf(isr_py_val, KF4Q_ISR_PY_PRIOR)
                        + spike_dcb_gauss_neg2logpdf(isr_pz_val, kf4q_isr_pz);

        // Closure: "mloss" = invariant-mass m_WW−m_ee ≤ 0 (fitted prior + asymmetric
        // barrier on the unphysical side); "kfit" = symmetric Gaussian ENERGY balance
        // r = (ECM+bes_m) − E_WW − E_γ, with E_γ the massless energy of the recoil-
        // derived ISR (3-momentum isr_{px,py,pz}_val). Replaces the one-sided m_loss
        // penalty that biased the dijet masses low.
        const double m_ee_fit = ECM + bes_m;
        double m_loss_term;
        if (kf4q_isr_mode == "kfit") {
            const double Eg = std::sqrt(isr_px_val*isr_px_val
                                      + isr_py_val*isr_py_val
                                      + isr_pz_val*isr_pz_val);
            const double r  = (m_ee_fit - WW.E() - Eg) / KF4Q_ISR_SYS_SIGMA;
            m_loss_term = r * r + KF4Q_ISR_SYS_LOG_NORM;
        } else {
            const double m_loss = WW.M() - m_ee_fit;
            m_loss_term = spike_dcb_gauss_neg2logpdf(std::fabs(m_loss),
                                                     kf4q_ww_m_minus_m_ee);
            if (m_loss > 0.0) {
                const double sigma_barrier =
                    kf4q_ww_m_minus_m_ee.sigma_res * KF_M_LOSS_BARRIER_SIGMA_FRAC;
                const double r = m_loss / sigma_barrier;
                m_loss_term += r * r;
            }
        }

        // Pooled-BINNED jet detector priors (each jet's own bin, picked at entry).
        double scale_pen = dcb_gauss_neg2logpdf(s1, pr1)
                         + dcb_gauss_neg2logpdf(s2, pr2)
                         + dcb_gauss_neg2logpdf(s3, pr3)
                         + dcb_gauss_neg2logpdf(s4, pr4);

        double angular = dcb_gauss_neg2logpdf(t1, th1)
                       + dcb_gauss_neg2logpdf(t2, th2)
                       + dcb_gauss_neg2logpdf(t3, th3)
                       + dcb_gauss_neg2logpdf(t4, th4)
                       + dcb_gauss_neg2logpdf(q1, ph1)
                       + dcb_gauss_neg2logpdf(q2, ph2)
                       + dcb_gauss_neg2logpdf(q3, ph3)
                       + dcb_gauss_neg2logpdf(q4, ph4);

        // mW prior (PRIOR mode only — dropped in free-floating mode) + gW prior.
        double mw_term = kf4q_mw_free ? 0.0 : (x[0] * x[0] + KF4Q_MW_PRIOR_LOG_NORM);
        double gw_term = gw_constrained ? (x[1] * x[1] + KF_GW_PRIOR_LOG_NORM) : 0.0;

        return bw_term + bes_term + isr_term + m_loss_term
             + scale_pen + angular + mw_term + gw_term;
    };

    std::function<double(const double*)> fObj = chi2fn;
    ROOT::Math::Functor functor(fObj, KF4Q_NPAR_TOTAL);

    auto configure = [&](ROOT::Math::Minimizer* m, const double* x0, bool with_strategy,
                         int strategy = -1) {
        m->SetFunction(functor);
        m->SetMaxFunctionCalls(KF_MAX_FUNCTION_CALLS);
        m->SetTolerance(KF_MIGRAD_TOLERANCE);
        if (with_strategy) m->SetStrategy(strategy >= 0 ? strategy : KF_MIGRAD_STRATEGY);
        m->SetPrintLevel(-1);
        const char* names[KF4Q_NPAR_TOTAL] = {
            "y_mW","y_gW","y_s1","y_s2","y_s3","y_s4",
            "y_t1","y_t2","y_t3","y_t4","y_p1","y_p2","y_p3","y_p4",
            "y_bes_m","y_bes_pz"};
        for (int i = 0; i < KF4Q_NPAR_TOTAL; ++i)
            m->SetVariable(i, names[i], x0[i], KF_INIT_STEP);
        if (!gw_free) m->FixVariable(1);
    };

    double x_default[KF4Q_NPAR_TOTAL] = {0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0};
    std::unique_ptr<ROOT::Math::Minimizer> minimizer(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", "Migrad"));

    auto simplex_then_migrad = [&](const double* x_init) {
        std::unique_ptr<ROOT::Math::Minimizer> simplex(
            ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex"));
        configure(simplex.get(), x_init, /*with_strategy=*/false);
        simplex->SetMaxFunctionCalls(KF4Q_SIMPLEX_MAX_CALLS);  // Simplex-only cap (see decl)
        { kf4q_prof::Scoped _t(&kf4q_prof::ns_simplex, &kf4q_prof::n_simplex); simplex->Minimize(); }
        configure(minimizer.get(), simplex->X(), /*with_strategy=*/true);
        // Two Migrad passes: the second restarts from the converged point and lets
        // Minuit re-estimate EDM/covariance — empirically rescues ~2× more fits to
        // converged status (valid 22%→48% at ecm160) for negligible cost (Migrad is
        // ~3% of the per-event time; Simplex dominates).
        { kf4q_prof::Scoped _t(&kf4q_prof::ns_migrad, &kf4q_prof::n_migrad); minimizer->Minimize(); }
        { kf4q_prof::Scoped _t(&kf4q_prof::ns_migrad, &kf4q_prof::n_migrad); minimizer->Minimize(); }
        return minimizer->Status();
    };

    struct PassResult { int status; double chi2; double edm; double x[KF4Q_NPAR_TOTAL]; int pass_id; };
    auto snapshot = [&](int s, int pass_id) {
        PassResult r{};
        r.status = s; r.chi2 = minimizer->MinValue(); r.edm = minimizer->Edm(); r.pass_id = pass_id;
        const double* xref = minimizer->X();
        for (int i = 0; i < KF4Q_NPAR_TOTAL; ++i) r.x[i] = xref[i];
        return r;
    };
    auto converged = [](int s) { return s == 0 || s == 1; };
    auto pick_better = [&](PassResult& best, const PassResult& cand) {
        bool ob = converged(best.status), oc = converged(cand.status);
        if (oc && !ob)             { best = cand; return; }
        if (!oc && ob)             return;
        if (cand.chi2 < best.chi2) best = cand;
    };

    // fast: single strategy-0 Migrad, no Simplex / retries (pairing discriminant).
    // full: Pass 1 Simplex+Migrad; Pass 2 Hesse-refresh + Simplex+Migrad; Pass 3
    // deterministic random-restart. No jet-swap pass (priors are pooled /
    // jet-symmetric, so swapping is a no-op).
    int n_passes_run = 1;
    PassResult best;
    if (fast) {
        configure(minimizer.get(), x_default, /*with_strategy=*/true, /*strategy=*/0);
        { kf4q_prof::Scoped _t(&kf4q_prof::ns_migrad, &kf4q_prof::n_migrad); minimizer->Minimize(); }
        best = snapshot(minimizer->Status(), /*pass=*/0);
    } else {
        best = snapshot(simplex_then_migrad(x_default), /*pass=*/1);

        if (!converged(best.status) && (!std::isfinite(best.edm) || best.edm < 1.0)) {
            configure(minimizer.get(), best.x, /*with_strategy=*/true);
            { kf4q_prof::Scoped _t(&kf4q_prof::ns_hesse, &kf4q_prof::n_hesse); minimizer->Hesse(); }
            ++n_passes_run;
            pick_better(best, snapshot(simplex_then_migrad(best.x), /*pass=*/2));
        }

        if (!converged(best.status)) {
            auto bits_of = [](double d) -> uint64_t {
                uint64_t b; std::memcpy(&b, &d, sizeof(b)); return b; };
            uint64_t s = bits_of(jet1_p) ^ (bits_of(jet2_p) << 1)
                       ^ (bits_of(jet3_p) << 2) ^ (bits_of(jet4_p) << 3);
            std::mt19937_64 rng(s);
            std::normal_distribution<double> jitter(0.0, KF_RESTART_SIGMA);
            for (int t = 0; t < KF_RESTART_N; ++t) {
                double x_jitter[KF4Q_NPAR_TOTAL];
                for (int i = 0; i < KF4Q_NPAR_TOTAL; ++i) x_jitter[i] = x_default[i];
                for (int i = 2; i < KF4Q_NPAR_TOTAL; ++i) x_jitter[i] += jitter(rng);
                ++n_passes_run;
                pick_better(best, snapshot(simplex_then_migrad(x_jitter), /*pass=*/3));
                if (converged(best.status)) break;
            }
        }

        // Recompute the Hessian at the winning point for the covariance.
        configure(minimizer.get(), best.x, /*with_strategy=*/true);
        { kf4q_prof::Scoped _t(&kf4q_prof::ns_hesse, &kf4q_prof::n_hesse); minimizer->Hesse(); }
    }

    const double* xf = best.x;
    result.status        = best.status;
    result.valid         = converged(best.status) ? 1 : 0;
    result.valid_loose   = (result.valid
                            || (best.status == 3 && std::isfinite(best.edm)
                                && best.edm < KF_LOOSE_EDM_MAX)) ? 1 : 0;
    result.chi2          = static_cast<float>(best.chi2);
    result.winner_pass   = best.pass_id;
    result.n_passes_run  = n_passes_run;
    result.edm           = static_cast<float>(best.edm);
    int n_par    = gw_free ? KF4Q_NPAR_TOTAL : KF4Q_NDIM;
    int n_constr = KF4Q_N_CONSTR + (gw_constrained ? 1 : 0);
    result.chi2_ndof = (n_constr > n_par) ? result.chi2 / float(n_constr - n_par) : -1.0f;

    result.mW = static_cast<float>(KF_MW_INIT  + kf4q_mw_scale() * xf[0]);
    result.gW = static_cast<float>(KF_GW_FIXED + KF_GW_PRIOR_SIGMA   * xf[1]);
    result.s1 = _y2x(xf[2],  pr1);
    result.s2 = _y2x(xf[3],  pr2);
    result.s3 = _y2x(xf[4],  pr3);
    result.s4 = _y2x(xf[5],  pr4);
    result.t1 = _y2x(xf[6],  th1);
    result.t2 = _y2x(xf[7],  th2);
    result.t3 = _y2x(xf[8],  th3);
    result.t4 = _y2x(xf[9],  th4);
    result.p1 = _y2x(xf[10], ph1);
    result.p2 = _y2x(xf[11], ph2);
    result.p3 = _y2x(xf[12], ph3);
    result.p4 = _y2x(xf[13], ph4);
    result.bes_m_minus_ecm = static_cast<float>(_y2x(xf[14], KF4Q_BES_M_PRIOR));
    result.bes_pz          = static_cast<float>(_y2x(xf[15], KF4Q_BES_PZ_PRIOR));

    if (!fast && result.valid_loose) {
        for (int i = 0; i < KF4Q_NPAR_TOTAL; ++i) {
            const double vii = minimizer->CovMatrix(i, i);
            if (!(vii > 0.0)) continue;
            result.corr[i][i] = 1.0f;
            for (int j = i + 1; j < KF4Q_NPAR_TOTAL; ++j) {
                const double vjj = minimizer->CovMatrix(j, j);
                if (!(vjj > 0.0)) continue;
                const double rho = minimizer->CovMatrix(i, j) / std::sqrt(vii * vjj);
                result.corr[i][j] = static_cast<float>(rho);
                result.corr[j][i] = static_cast<float>(rho);
            }
        }
    }

    result.j1 = _vec_spherical(jet1_p/result.s1, jet1_theta - result.t1, jet1_phi - result.p1);
    result.j2 = _vec_spherical(jet2_p/result.s2, jet2_theta - result.t2, jet2_phi - result.p2);
    result.j3 = _vec_spherical(jet3_p/result.s3, jet3_theta - result.t3, jet3_phi - result.p3);
    result.j4 = _vec_spherical(jet4_p/result.s4, jet4_theta - result.t4, jet4_phi - result.p4);
    return result;
}

// ── DIAGNOSTIC: per-term χ² breakdown ───────────────────────────────────────
// Re-evaluates each additive χ² term from a fit's PHYSICAL parameters, mirroring
// chi2fn term-by-term. Single source of truth check: terms.total must equal the
// fit's reported chi2 (MinValue) to ~1e-4. Used only by kinFit4q_diag to expose
// which prior/constraint dominates and to test the pairing discriminant — NOT on
// the production path. (TEMPORARY, strip with the kf4q_prof profiling.)
struct KinFit4qChi2Terms {
    double bw, bes, isr, m_loss, scale_pen, angular, mw, gw, total;
    // Proper goodness-of-fit: each term referenced to its own mode (min value),
    // so gof = Σ (term − term_at_mode) ≥ 0 and is ≈ χ²(ndof)-distributed — unlike
    // `total`, which carries the large negative prior log-norm offsets. ndof =
    // KF4Q_N_CONSTR(+gw) − n_par. For the BW term the reference is both W's on the
    // pole (m_h=m_l=mW), isolating the dijet-mass-compatibility residual.
    double gof;
};

inline KinFit4qChi2Terms kf4q_chi2_terms(
        float jet1_p, float jet1_theta, float jet1_phi,
        float jet2_p, float jet2_theta, float jet2_phi,
        float jet3_p, float jet3_theta, float jet3_phi,
        float jet4_p, float jet4_theta, float jet4_phi,
        double mW, double gW,
        double s1, double s2, double s3, double s4,
        double t1, double t2, double t3, double t4,
        double q1, double q2, double q3, double q4,
        double bes_m, double bes_pz, int gw_mode) {
    KinFit4qChi2Terms T{};
    // Identical bin selection to kinFit4q (p_resp & θ on jet p, φ on |cosθ|).
    const double j1_acth = std::abs(std::cos(jet1_theta));
    const double j2_acth = std::abs(std::cos(jet2_theta));
    const double j3_acth = std::abs(std::cos(jet3_theta));
    const double j4_acth = std::abs(std::cos(jet4_theta));
    const DcbGaussParams pr1 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet1_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams pr2 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet2_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams pr3 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet3_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams pr4 = kf4q_use_binned ? pick_bin(kf4q_jet_p_resp_bins, kf4q_jet_p_resp_edges, (double)jet4_p) : kf4q_jet_p_resp_incl;
    const DcbGaussParams th1 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet1_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams th2 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet2_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams th3 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet3_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams th4 = kf4q_use_binned ? pick_bin(kf4q_jet_theta_resol_bins, kf4q_jet_theta_resol_edges, (double)jet4_p) : kf4q_jet_theta_resol_incl;
    const DcbGaussParams ph1 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j1_acth) : kf4q_jet_phi_resol_incl;
    const DcbGaussParams ph2 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j2_acth) : kf4q_jet_phi_resol_incl;
    const DcbGaussParams ph3 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j3_acth) : kf4q_jet_phi_resol_incl;
    const DcbGaussParams ph4 = kf4q_use_binned ? pick_bin(kf4q_jet_phi_resol_bins, kf4q_jet_phi_resol_edges, j4_acth) : kf4q_jet_phi_resol_incl;

    TLorentzVector j1f = _vec_spherical(jet1_p/s1, jet1_theta - t1, jet1_phi - q1);
    TLorentzVector j2f = _vec_spherical(jet2_p/s2, jet2_theta - t2, jet2_phi - q2);
    TLorentzVector j3f = _vec_spherical(jet3_p/s3, jet3_theta - t3, jet3_phi - q3);
    TLorentzVector j4f = _vec_spherical(jet4_p/s4, jet4_theta - t4, jet4_phi - q4);
    TLorentzVector Wa = j1f + j2f, Wb = j3f + j4f, WW = Wa + Wb;

    // Shared kernel for T.bw (identical to chi2fn); bw_h/bw_l/mwgw kept below for r_bw.
    double mh = Wa.M(), ml = Wb.M(), mwgw = mW * gW;
    double dh = mh*mh - mW*mW, dl = ml*ml - mW*mW;
    double bw_h = mwgw / (dh*dh + mwgw*mwgw);
    double bw_l = mwgw / (dl*dl + mwgw*mwgw);
    double s_ww = WW.M2();
    T.bw = bw_phasespace_neg2ll(mh, ml, s_ww, mW, gW);

    T.bes = gauss_neg2logpdf(bes_m, KF4Q_BES_M_PRIOR)
          + gauss_neg2logpdf(bes_pz, KF4Q_BES_PZ_PRIOR);

    const double isr_px_val = -WW.Px(), isr_py_val = -WW.Py(), isr_pz_val = bes_pz - WW.Pz();
    T.isr = gauss_neg2logpdf(isr_px_val, KF4Q_ISR_PX_PRIOR)
          + gauss_neg2logpdf(isr_py_val, KF4Q_ISR_PY_PRIOR)
          + spike_dcb_gauss_neg2logpdf(isr_pz_val, kf4q_isr_pz);

    // Closure term + its mode-referenced residual (r_mloss), mirroring chi2fn.
    const double m_ee_fit = ECM + bes_m;
    double r_mloss;
    if (kf4q_isr_mode == "kfit") {
        const double Eg = std::sqrt(isr_px_val*isr_px_val + isr_py_val*isr_py_val
                                  + isr_pz_val*isr_pz_val);
        const double rr = (m_ee_fit - WW.E() - Eg) / KF4Q_ISR_SYS_SIGMA;
        T.m_loss = rr * rr + KF4Q_ISR_SYS_LOG_NORM;
        r_mloss  = rr * rr;                 // mode at r=0
    } else {
        const double m_loss = WW.M() - m_ee_fit;
        T.m_loss = spike_dcb_gauss_neg2logpdf(std::fabs(m_loss), kf4q_ww_m_minus_m_ee);
        if (m_loss > 0.0) {
            const double sigma_barrier = kf4q_ww_m_minus_m_ee.sigma_res * KF_M_LOSS_BARRIER_SIGMA_FRAC;
            const double r = m_loss / sigma_barrier;
            T.m_loss += r * r;
        }
        r_mloss = T.m_loss - spike_dcb_gauss_neg2logpdf(0.0, kf4q_ww_m_minus_m_ee);
    }

    T.scale_pen = dcb_gauss_neg2logpdf(s1, pr1) + dcb_gauss_neg2logpdf(s2, pr2)
                + dcb_gauss_neg2logpdf(s3, pr3) + dcb_gauss_neg2logpdf(s4, pr4);
    T.angular = dcb_gauss_neg2logpdf(t1, th1) + dcb_gauss_neg2logpdf(t2, th2)
              + dcb_gauss_neg2logpdf(t3, th3) + dcb_gauss_neg2logpdf(t4, th4)
              + dcb_gauss_neg2logpdf(q1, ph1) + dcb_gauss_neg2logpdf(q2, ph2)
              + dcb_gauss_neg2logpdf(q3, ph3) + dcb_gauss_neg2logpdf(q4, ph4);

    double y_mW = (mW - KF_MW_INIT) / kf4q_mw_scale();
    T.mw = kf4q_mw_free ? 0.0 : (y_mW * y_mW + KF4Q_MW_PRIOR_LOG_NORM);
    double y_gW = (gW - KF_GW_FIXED) / KF_GW_PRIOR_SIGMA;
    T.gw = (gw_mode == KF_GW_CONSTRAINED_MODE) ? (y_gW * y_gW + KF_GW_PRIOR_LOG_NORM) : 0.0;

    T.total = T.bw + T.bes + T.isr + T.m_loss + T.scale_pen + T.angular + T.mw + T.gw;

    // ── Mode-referenced residuals → goodness-of-fit (≥0, ≈χ²) ──────────────
    // Each prior/constraint term minus its value at its own mode (min). DCB modes
    // ≈ p.mu; Gaussian/spike modes at 0 (their μ); BW reference at the W pole.
    double r_bw  = -2.0 * (std::log(bw_h) + std::log(bw_l)) - 4.0 * std::log(mwgw);
    double r_bes = (gauss_neg2logpdf(bes_m,  KF4Q_BES_M_PRIOR)  - gauss_neg2logpdf(KF4Q_BES_M_PRIOR.mu,  KF4Q_BES_M_PRIOR))
                 + (gauss_neg2logpdf(bes_pz, KF4Q_BES_PZ_PRIOR) - gauss_neg2logpdf(KF4Q_BES_PZ_PRIOR.mu, KF4Q_BES_PZ_PRIOR));
    double r_isr = (gauss_neg2logpdf(isr_px_val, KF4Q_ISR_PX_PRIOR) - gauss_neg2logpdf(KF4Q_ISR_PX_PRIOR.mu, KF4Q_ISR_PX_PRIOR))
                 + (gauss_neg2logpdf(isr_py_val, KF4Q_ISR_PY_PRIOR) - gauss_neg2logpdf(KF4Q_ISR_PY_PRIOR.mu, KF4Q_ISR_PY_PRIOR))
                 + (spike_dcb_gauss_neg2logpdf(isr_pz_val, kf4q_isr_pz) - spike_dcb_gauss_neg2logpdf(0.0, kf4q_isr_pz));
    double r_scale = (dcb_gauss_neg2logpdf(s1, pr1) - dcb_gauss_neg2logpdf(pr1.mu, pr1))
                   + (dcb_gauss_neg2logpdf(s2, pr2) - dcb_gauss_neg2logpdf(pr2.mu, pr2))
                   + (dcb_gauss_neg2logpdf(s3, pr3) - dcb_gauss_neg2logpdf(pr3.mu, pr3))
                   + (dcb_gauss_neg2logpdf(s4, pr4) - dcb_gauss_neg2logpdf(pr4.mu, pr4));
    double r_ang = 0.0;
    { const DcbGaussParams* TH[8] = {&th1,&th2,&th3,&th4,&ph1,&ph2,&ph3,&ph4};
      const double V[8] = {t1,t2,t3,t4,q1,q2,q3,q4};
      for (int i=0;i<8;++i) r_ang += dcb_gauss_neg2logpdf(V[i], *TH[i]) - dcb_gauss_neg2logpdf(TH[i]->mu, *TH[i]); }
    double r_mw = kf4q_mw_free ? 0.0 : y_mW * y_mW;
    double r_gw = (gw_mode == KF_GW_CONSTRAINED_MODE) ? (y_gW * y_gW) : 0.0;
    T.gof = r_bw + r_bes + r_isr + r_mloss + r_scale + r_ang + r_mw + r_gw;
    return T;
}

// ── DIAGNOSTIC: full-quality fit on ALL THREE pairings + per-term breakdown ──
// Decouples discriminant quality (does the TRUE pairing have lowest full-fit χ²?)
// from minimizer quality (does the fast tier-1 fit rank correctly?). Heavier than
// production (3 full fits/event) — gate to a diagnostic subsample. (TEMPORARY.)
struct KinFit4qDiag {
    float chi2[3];
    int   valid[3], status[3], valid_loose[3];
    float ndof[3], edm[3];
    // Plain Migrad-only (fast) fit per pairing — to test whether the Simplex
    // pre-pass + retry ladder actually improves convergence vs bare Migrad.
    int   fast_status[3], fast_valid[3];
    float fast_chi2[3], fast_edm[3];
    KinFit4qChi2Terms terms[3];
    int   argmin;            // pairing with lowest full-fit χ²
};

inline KinFit4qDiag kinFit4q_diag(
        float j1_p, float j1_theta, float j1_phi,
        float j2_p, float j2_theta, float j2_phi,
        float j3_p, float j3_theta, float j3_phi,
        float j4_p, float j4_theta, float j4_phi,
        int gw_mode = KF_GW_CONSTRAINED_MODE) {
    const double P[4]  = {j1_p, j2_p, j3_p, j4_p};
    const double TH[4] = {j1_theta, j2_theta, j3_theta, j4_theta};
    const double PH[4] = {j1_phi, j2_phi, j3_phi, j4_phi};
    static const int order[3][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2}};

    KinFit4qDiag D{};
    D.argmin = -1;
    for (int k = 0; k < 3; ++k) {
        const int a = order[k][0], b = order[k][1], c = order[k][2], d = order[k][3];
        KinFit4qResult r = kinFit4q(P[a], TH[a], PH[a], P[b], TH[b], PH[b],
                                    P[c], TH[c], PH[c], P[d], TH[d], PH[d],
                                    gw_mode, /*fast=*/false);
        D.chi2[k]        = r.chi2;
        D.valid[k]       = r.valid;
        D.valid_loose[k] = r.valid_loose;
        D.status[k]      = r.status;
        D.ndof[k]        = r.chi2_ndof;
        D.edm[k]         = r.edm;
        D.terms[k]  = kf4q_chi2_terms(
            P[a], TH[a], PH[a], P[b], TH[b], PH[b], P[c], TH[c], PH[c], P[d], TH[d], PH[d],
            r.mW, r.gW, r.s1, r.s2, r.s3, r.s4, r.t1, r.t2, r.t3, r.t4,
            r.p1, r.p2, r.p3, r.p4, r.bes_m_minus_ecm, r.bes_pz, gw_mode);
        KinFit4qResult rf = kinFit4q(P[a], TH[a], PH[a], P[b], TH[b], PH[b],
                                     P[c], TH[c], PH[c], P[d], TH[d], PH[d],
                                     gw_mode, /*fast=*/true);
        D.fast_status[k] = rf.status; D.fast_valid[k] = rf.valid;
        D.fast_chi2[k]   = rf.chi2;   D.fast_edm[k]   = rf.edm;
        if (D.argmin < 0 || D.chi2[k] < D.chi2[D.argmin]) D.argmin = k;
    }
    return D;
}

// Best-pairing 4q fit. The 4 jets admit 3 partitions into 2 W's:
//   pairing 0: (j1 j2)(j3 j4)   1: (j1 j3)(j2 j4)   2: (j1 j4)(j2 j3)
// Run kinFit4q for each, pick the lowest χ² (preferring valid fits). The pairing
// index follows WWFunctions::pairing_index_from_groups so it is directly
// comparable to gen_pairing_true.
struct KinFit4qBest {
    KinFit4qResult fit;          // winning single-pairing fit
    int   pairing;               // winner ∈ {0,1,2}
    float chi2_p0, chi2_p1, chi2_p2;
    int   valid_p0, valid_p1, valid_p2;
    float dchi2;                 // 2nd-best − best (pairing separation)
    int   n_pairings_valid;
    // Post-fit W's / WW in the winner's W_a=(first pair), W_b=(second pair).
    TLorentzVector Wa, Wb, WW;
};

inline KinFit4qBest kinFit4q_bestpairing(
        float j1_p, float j1_theta, float j1_phi,
        float j2_p, float j2_theta, float j2_phi,
        float j3_p, float j3_theta, float j3_phi,
        float j4_p, float j4_theta, float j4_phi,
        int gw_mode = KF_GW_CONSTRAINED_MODE) {

    kf4q_prof::Scoped _ev(&kf4q_prof::ns_event, &kf4q_prof::n_event);  // per-event total (TEMP profiling)

    const double P[4]  = {j1_p, j2_p, j3_p, j4_p};
    const double TH[4] = {j1_theta, j2_theta, j3_theta, j4_theta};
    const double PH[4] = {j1_phi, j2_phi, j3_phi, j4_phi};
    // jet indices feeding (W_a jet1, W_a jet2, W_b jet1, W_b jet2) per pairing.
    static const int order[3][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2}};

    KinFit4qBest out{};

    // ── Tier 1: cheap pairing discriminant. Run a fast (Migrad-only, no Simplex /
    // retries / Hesse) fit for each of the 3 partitions and rank them by χ². The
    // wrong pairings give badly-mismatched W masses → high χ², so even the rough
    // Migrad-alone χ² separates pairings; the precise fit is deferred to tier 2.
    float   chi2s[3];
    int     valids[3];
    for (int k = 0; k < 3; ++k) {
        const int a = order[k][0], b = order[k][1], c = order[k][2], d = order[k][3];
        KinFit4qResult cheap = kinFit4q(P[a], TH[a], PH[a], P[b], TH[b], PH[b],
                                        P[c], TH[c], PH[c], P[d], TH[d], PH[d],
                                        gw_mode, /*fast=*/true);
        chi2s[k]  = cheap.chi2;
        valids[k] = cheap.valid;
    }
    out.chi2_p0 = chi2s[0]; out.chi2_p1 = chi2s[1]; out.chi2_p2 = chi2s[2];
    out.valid_p0 = valids[0]; out.valid_p1 = valids[1]; out.valid_p2 = valids[2];
    out.n_pairings_valid = valids[0] + valids[1] + valids[2];

    // Winner: prefer valid pairings; among the eligible set, lowest χ². If none
    // valid, fall back to lowest χ² over all three.
    int best = -1;
    for (int k = 0; k < 3; ++k) {
        bool eligible = (out.n_pairings_valid > 0) ? (valids[k] == 1) : true;
        if (!eligible) continue;
        if (best < 0 || chi2s[k] < chi2s[best]) best = k;
    }
    // Second-best χ² among the same eligibility class, for the Δχ² separation.
    float second = std::numeric_limits<float>::infinity();
    for (int k = 0; k < 3; ++k) {
        if (k == best) continue;
        bool eligible = (out.n_pairings_valid > 0) ? (valids[k] == 1) : true;
        if (eligible && chi2s[k] < second) second = chi2s[k];
    }

    out.pairing = best;
    out.dchi2   = std::isfinite(second) ? (second - chi2s[best]) : -1.0f;

    // ── Tier 2: one full-quality fit (Simplex + Migrad + retry ladder + Hesse
    // covariance) on the winning pairing only. This is the result reported
    // downstream (kinfit4q_chi2 / valid / mW / covariance / post-fit W's).
    const int a = order[best][0], b = order[best][1], c = order[best][2], d = order[best][3];
    out.fit = kinFit4q(P[a], TH[a], PH[a], P[b], TH[b], PH[b],
                       P[c], TH[c], PH[c], P[d], TH[d], PH[d], gw_mode, /*fast=*/false);
    out.Wa = out.fit.j1 + out.fit.j2;
    out.Wb = out.fit.j3 + out.fit.j4;
    out.WW = out.Wa + out.Wb;
    return out;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
