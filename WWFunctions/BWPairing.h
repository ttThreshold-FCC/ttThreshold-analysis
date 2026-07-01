#ifndef BWPairing_H
#define BWPairing_H

// ── Standalone BW jet→W pairing discriminant ────────────────────────────────
//
// Given 4 jets, decide which of the 3 partitions into 2 di-jets is most likely
// to be the two W's — using ONLY the Breit-Wigner compatibility of the two
// di-jet masses with the W resonance. No kinematic fit, no Minuit: it is pure
// arithmetic on the raw jet 4-vectors and runs in ~microseconds.
//
// This is the discriminating core of the full WW→4q kinematic fit (where the
// per-term study showed the BW term is the ONLY part that separates pairings)
// distilled into a self-contained tool, usable as (a) a fast pairing chooser and
// (b) a WW-vs-background (e.g. ZZ→4q) χ²-like discriminant under the W hypothesis.
//
// Outputs per call:
//   pairing  — most probable partition ∈ {0,1,2}
//   gof[k]   — "goodness of fit" of partition k: −2·log[BW(m_a)·BW(m_b)] referenced
//              to the W pole (both di-jets exactly on mW), so gof ≥ 0 and gof = 0
//              means both masses sit on the pole. Lower = more W-like.
//   prob[k]  — posterior probability that partition k is the correct one, under a
//              flat prior: prob[k] = L_k / Σ_j L_j with L_k = BW(m_a)·BW(m_b).
//              The three probabilities sum to 1 by construction.
//   m_a[k], m_b[k] — the two di-jet masses of partition k (a = first pair).
//   dgof     — gof(2nd best) − gof(best): the pairing separation.
//
// Pairing index convention matches kinFit4q_bestpairing / pairing_index_from_groups:
//   0: (j1 j2)(j3 j4)   1: (j1 j3)(j2 j4)   2: (j1 j4)(j2 j3)

#include <TLorentzVector.h>
#include <cmath>
#include <limits>

namespace FCCAnalyses { namespace WWFunctions {

// PDG-ish defaults (kept independent of the Minuit-pulling WWKinReco.h so this
// header stays dependency-light). Override per call if desired.
static constexpr double BWPAIR_MW    = 80.385;
static constexpr double BWPAIR_GAMMA = 2.085;

struct BWPairingResult {
    int   pairing;        // most probable partition ∈ {0,1,2}
    float gof[3];         // pole-referenced −2 log[BW_a·BW_b] (≥ 0)
    float prob[3];        // posterior over partitions, Σ = 1
    float m_a[3], m_b[3]; // di-jet masses (a = first pair, b = second pair)
    float gof_best;       // gof[pairing]
    float prob_best;      // prob[pairing]
    float dgof;           // gof(2nd best) − gof(best)
};

// Relativistic Breit-Wigner shape value at mass m (peak = 1/(mW·Γ) at m = mW).
inline double _bwpair_val(double m, double mW, double Gamma) {
    const double mwg = mW * Gamma;
    const double d   = m * m - mW * mW;
    return mwg / (d * d + mwg * mwg);
}

inline BWPairingResult bwPairing(const TLorentzVector& j1, const TLorentzVector& j2,
                                 const TLorentzVector& j3, const TLorentzVector& j4,
                                 double mW = BWPAIR_MW, double Gamma = BWPAIR_GAMMA) {
    static const int order[3][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2}};
    const TLorentzVector* J[4] = {&j1, &j2, &j3, &j4};

    BWPairingResult R{};
    const double mwg       = mW * Gamma;
    const double pole_ref  = 4.0 * std::log(mwg);   // −2·2·log(1/mwg): both on pole
    double L[3], gof[3];
    for (int k = 0; k < 3; ++k) {
        const TLorentzVector Wa = *J[order[k][0]] + *J[order[k][1]];
        const TLorentzVector Wb = *J[order[k][2]] + *J[order[k][3]];
        const double ma = Wa.M(), mb = Wb.M();
        R.m_a[k] = static_cast<float>(ma);
        R.m_b[k] = static_cast<float>(mb);
        const double bwa = _bwpair_val(ma, mW, Gamma);
        const double bwb = _bwpair_val(mb, mW, Gamma);
        gof[k]   = -2.0 * (std::log(bwa) + std::log(bwb)) - pole_ref;
        L[k]     = bwa * bwb;
        R.gof[k] = static_cast<float>(gof[k]);
    }

    // Posterior over partitions: softmax(−gof/2) ≡ L_k / Σ L_j (the pole_ref is a
    // common offset and cancels). Computed in the stable softmax form.
    double gmin = gof[0];
    for (int k = 1; k < 3; ++k) gmin = std::min(gmin, gof[k]);
    double w[3], wsum = 0.0;
    for (int k = 0; k < 3; ++k) { w[k] = std::exp(-0.5 * (gof[k] - gmin)); wsum += w[k]; }
    for (int k = 0; k < 3; ++k) R.prob[k] = static_cast<float>(w[k] / wsum);

    int best = 0;
    for (int k = 1; k < 3; ++k) if (gof[k] < gof[best]) best = k;
    double second = std::numeric_limits<double>::infinity();
    for (int k = 0; k < 3; ++k) if (k != best) second = std::min(second, gof[k]);

    R.pairing   = best;
    R.gof_best  = static_cast<float>(gof[best]);
    R.prob_best = R.prob[best];
    R.dgof      = static_cast<float>(second - gof[best]);
    return R;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
