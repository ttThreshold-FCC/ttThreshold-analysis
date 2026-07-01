#ifndef WWFunctions_H
#define WWFunctions_H

#include <cmath>
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include "TMath.h"
#include "TVector2.h"
#include "TLorentzVector.h"
#include "ROOT/RVec.hxx"
#include "Math/Vector4D.h"
#include "edm4hep/MCParticleData.h"

namespace FCCAnalyses { namespace WWFunctions {

inline float ECM = 160.0f;

// ── selectors keeping particles whose mother is e± (proxy for "from W decay"
//    in samples where the W is not stored in the MC history). Walk the proper
//    parents_begin/parents_end relation range — robust against the flat
//    Particle0[i] indexing assumption used in FCCAnalyses::MCParticle::sel_*.

namespace _selectors_detail {
// True if particle p has at least one immediate parent with |PDG| == target_abs_pdg
// (walks the proper parents_begin/parents_end relation range).
inline bool _has_parent_pdg(const edm4hep::MCParticleData& p,
                            const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in,
                            const ROOT::VecOps::RVec<int>& parents_relation,
                            int target_abs_pdg) {
    for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
        if (j >= parents_relation.size()) break;
        int parent_idx = parents_relation[j];
        if (parent_idx < 0 || parent_idx >= (int)in.size()) continue;
        if (std::abs(in[parent_idx].PDG) == target_abs_pdg) return true;
    }
    return false;
}
inline bool _has_electron_parent(const edm4hep::MCParticleData& p,
                                  const ROOT::VecOps::RVec<edm4hep::MCParticleData>& in,
                                  const ROOT::VecOps::RVec<int>& parents_relation) {
    return _has_parent_pdg(p, in, parents_relation, 11);
}
}

struct sel_genleps_fromele {
    int m_pdg;
    sel_genleps_fromele(int pdg) : m_pdg(pdg) {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) != m_pdg) continue;
            if (_selectors_detail::_has_electron_parent(p, in, parents_relation)) result.emplace_back(p);
        }
        return result;
    }
};

// Post-BES, pre-ISR beam e± (depth=1 in the e± chain).
// In Winter2023 wzp6 samples the chain is:
//   depth 0 = nominal beams (no parents, m(ee)=ECM exactly, no BES)
//   depth 1 = e± with an e± parent that itself has no parents — BES applied
//             here (σ_BES ≈ 84 MeV per beam, m(ee) symmetric around ECM)
//   depth≥2 = post-ISR (energy-loss tail, m(ee) ≪ ECM with ISR-photon recoil)
// We pick depth=1 to isolate BES from ISR.
struct sel_beam_electrons {
    sel_beam_electrons() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(2);
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) != 11) continue;
            if (p.parents_begin == p.parents_end) continue;  // skip depth 0
            int parent_e = -1;
            for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
                if (j >= parents_relation.size()) break;
                int idx = parents_relation[j];
                if (idx < 0 || idx >= (int)in.size()) continue;
                if (std::abs(in[idx].PDG) == 11) { parent_e = idx; break; }
            }
            if (parent_e < 0) continue;
            const auto& pa = in[parent_e];
            if (pa.parents_begin != pa.parents_end) continue;  // require depth-0 parent
            result.emplace_back(p);
        }
        return result;
    }
};

// Post-ISR beam e± (depth=2 in the e± chain) — the e± entering the hard
// process, after the ISR photons have been emitted. Identified as e± with an
// e± parent whose own first e± parent has no parents (= depth-0 beams).
// (gen_ee_p4_depth1 − gen_ee_p4_depth2) gives the total ISR 4-momentum.
struct sel_post_isr_electrons {
    sel_post_isr_electrons() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        auto first_e_parent = [&](const edm4hep::MCParticleData& p) -> int {
            for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
                if (j >= parents_relation.size()) break;
                int idx = parents_relation[j];
                if (idx < 0 || idx >= (int)in.size()) continue;
                if (std::abs(in[idx].PDG) == 11) return idx;
            }
            return -1;
        };
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(2);
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) != 11) continue;
            int parent = first_e_parent(p);
            if (parent < 0) continue;
            const auto& pa = in[parent];
            if (pa.parents_begin == pa.parents_end) continue;       // parent is depth-0; we want depth-2
            int grandp = first_e_parent(pa);
            if (grandp < 0) continue;
            if (in[grandp].parents_begin != in[grandp].parents_end) continue;  // grandparent must be depth-0
            result.emplace_back(p);
        }
        return result;
    }
};

// Pythia8 post-ISR (hard-process-incoming) electrons via generatorStatus.
// The depth-2 chain walk in sel_post_isr_electrons is Whizard-specific; p8
// writes a variable-length ISR e-chain (the hard-process-incoming e is at
// depth 1, 2, or 3 depending on how many ISR electrons were stored), so the
// rigid depth-2 requirement silently drops ~40% of p8 events. Pythia status 21
// (= incoming to the hardest subprocess) tags exactly the 2 post-ISR electrons:
// verified 2/2 in 5000/5000 p8_ee_WW_ecm160 events (vs depth-0 beams = status 4).
struct sel_post_isr_electrons_status {
    int status;
    explicit sel_post_isr_electrons_status(int s = 21) : status(s) {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(2);
        for (const auto& p : in)
            if (std::abs(p.PDG) == 11 && p.generatorStatus == status)
                result.emplace_back(p);
        return result;
    }
};

// ── WW → 4q (fully hadronic) gen-truth selection ────────────────────────────
// Unlike the Whizard munumuqq samples (W absent from history → sel_*_fromele),
// the inclusive Pythia8 p8_ee_WW sample KEEPS the W in the MC history. Each
// matrix-element light quark points to its unique decaying-W immediate parent
// (status 22/44 — irrelevant, the parent walk is unique). Verified 2026-06-04:
// 100% of all-hadronic events give exactly 4 such quarks in 2 W groups (2 each).
//
// Returns the 4 quarks W-GROUPED-ORDERED: indices [0,1] are the two quarks of
// one W, [2,3] the two quarks of the other (groups ordered by ascending
// parent-W index → deterministic). The ordering encodes the true jet→W pairing
// once the jets are matched (matchJets4). If the event is not a clean 2×2
// hadronic topology the result is empty (size != 4), and the treemaker filters
// on size()==4.
// Quarks from the two resonances of a VV→4q event, grouped by parent boson.
// boson_pdg = 24 → WW→4q (default, keeps sel_quarks_fromW name working below),
//             23 → ZZ→4q. Returns size 4 ([0,1]=boson A, [2,3]=boson B) only for
// a clean 2-boson × 2-quark topology, else empty.
struct sel_quarks_fromBoson {
    int boson_pdg;
    explicit sel_quarks_fromBoson(int pdg = 24) : boson_pdg(pdg) {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        std::map<int, ROOT::VecOps::RVec<edm4hep::MCParticleData>> groups;
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) > 5 || p.PDG == 0) continue;
            int bparent = -1;
            for (unsigned j = p.parents_begin; j < p.parents_end; ++j) {
                if (j >= parents_relation.size()) break;
                int idx = parents_relation[j];
                if (idx < 0 || idx >= (int)in.size()) continue;
                if (std::abs(in[idx].PDG) == boson_pdg) { bparent = idx; break; }
            }
            if (bparent < 0) continue;
            groups[bparent].emplace_back(p);
        }
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        if (groups.size() != 2) return result;          // not a clean 2-boson topology
        for (auto& kv : groups) {
            if (kv.second.size() != 2) { result.clear(); return result; }
            for (auto& q : kv.second) result.emplace_back(q);
        }
        return result;                                  // size 4, [0,1]=A [2,3]=B
    }
};

// Back-compat alias: WW→4q (boson_pdg=24). Existing call sites use this name.
struct sel_quarks_fromW {
    sel_quarks_fromW() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        return sel_quarks_fromBoson(24)(in, parents_relation);
    }
};

// Light quarks (|PDG|<=5) with e± parent — like FCCAnalyses::MCParticle::sel_lightQuarks_fromele
// but using the robust parents-relation walk.
struct sel_lightQuarks_fromele {
    sel_lightQuarks_fromele() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) > 5 || std::abs(p.PDG) == 0) continue;
            if (_selectors_detail::_has_electron_parent(p, in, parents_relation)) result.emplace_back(p);
        }
        return result;
    }
};

// ── WW → ℓνqq (semileptonic) gen-truth selection (Pythia8 p8_ee_WW) ──────────
// As with sel_quarks_fromBoson, the inclusive p8 sample KEEPS the W in the MC
// history, so the matrix-element W-decay products point to their immediate W
// parent (|PDG|==24). These select the ℓνqq products by that W parent — the p8
// analog of sel_genleps_fromele / sel_lightQuarks_fromele (which use the e±-parent
// proxy needed for the Whizard munumuqq samples where the W is absent).
//
// sel_genleps_fromW(pdg): particles with |PDG|==pdg AND a W parent. Reused for the
//   charged lepton (pdg=11/13) and the neutrino (pdg=12/14), exactly as the fromele
//   version is. In a semileptonic event this returns exactly 1 ℓ and 1 ν.
// sel_lightQuarks_fromW: light quarks (1<=|PDG|<=5) with a W parent → the 2 quarks
//   of the hadronic W (un-grouped, unlike sel_quarks_fromBoson which needs 2 bosons).
// τ→ℓ events are excluded automatically: the e/μ from a τ has a τ (not W) parent.
struct sel_genleps_fromW {
    int m_pdg;
    explicit sel_genleps_fromW(int pdg) : m_pdg(pdg) {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(2);
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) != m_pdg) continue;
            if (_selectors_detail::_has_parent_pdg(p, in, parents_relation, 24)) result.emplace_back(p);
        }
        return result;
    }
};

struct sel_lightQuarks_fromW {
    sel_lightQuarks_fromW() {}
    ROOT::VecOps::RVec<edm4hep::MCParticleData> operator()(
        ROOT::VecOps::RVec<edm4hep::MCParticleData> in,
        const ROOT::VecOps::RVec<int>& parents_relation) const {
        ROOT::VecOps::RVec<edm4hep::MCParticleData> result;
        result.reserve(2);
        for (size_t i = 0; i < in.size(); ++i) {
            const auto& p = in[i];
            if (std::abs(p.PDG) > 5 || p.PDG == 0) continue;
            if (_selectors_detail::_has_parent_pdg(p, in, parents_relation, 24)) result.emplace_back(p);
        }
        return result;
    }
};

// ── jet matching ───────────────────────────────────────────────────────────

template<typename V>
std::pair<V, V>
matchJets2(const V& r1, const V& r2, const V& g1, const V& g2) {
    auto dR = [](const V& a, const V& b) {
        double deta = a.Eta() - b.Eta();
        double dphi = TVector2::Phi_mpi_pi(a.Phi() - b.Phi());
        return sqrt(deta*deta + dphi*dphi);
    };
    double dR_A = dR(r1, g1) + dR(r2, g2);
    double dR_B = dR(r1, g2) + dR(r2, g1);
    return (dR_A < dR_B) ? std::make_pair(g1, g2) : std::make_pair(g2, g1);
}

// 4-jet ↔ 4-quark global assignment. Returns a length-4 permutation: out[i] is
// the gen-quark index (0..3) matched to reco jet i, chosen to minimize the
// total ΔR over all 24 bijections. Used both to build the per-jet response
// (p_reco/p_gen) and — combined with the W-grouped quark ordering from
// sel_quarks_fromW — to derive the true jet→W pairing.
template<typename V>
ROOT::VecOps::RVec<int>
matchJets4(const V& j0, const V& j1, const V& j2, const V& j3,
           const V& q0, const V& q1, const V& q2, const V& q3) {
    auto dR = [](const V& a, const V& b) {
        double deta = a.Eta() - b.Eta();
        double dphi = TVector2::Phi_mpi_pi(a.Phi() - b.Phi());
        return std::sqrt(deta*deta + dphi*dphi);
    };
    const V* jets[4] = {&j0, &j1, &j2, &j3};
    const V* qs[4]   = {&q0, &q1, &q2, &q3};
    double dRm[4][4];
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k) dRm[i][k] = dR(*jets[i], *qs[k]);
    std::array<int, 4> p = {0, 1, 2, 3};
    std::array<int, 4> best = p;
    double best_sum = std::numeric_limits<double>::max();
    do {
        double s = dRm[0][p[0]] + dRm[1][p[1]] + dRm[2][p[2]] + dRm[3][p[3]];
        if (s < best_sum) { best_sum = s; best = p; }
    } while (std::next_permutation(p.begin(), p.end()));
    return ROOT::VecOps::RVec<int>(best.begin(), best.end());
}

// Reco-jet partition index from per-jet W-group labels (each 0 or 1):
//   0: (j0 j1)(j2 j3)   1: (j0 j2)(j1 j3)   2: (j0 j3)(j1 j2)
// Returns -1 if the labels don't split 2-2 into a valid partition.
inline int pairing_index_from_groups(int w0, int w1, int w2, int w3) {
    if (w0 == w1 && w2 == w3) return 0;
    if (w0 == w2 && w1 == w3) return 1;
    if (w0 == w3 && w1 == w2) return 2;
    return -1;
}

// ── 4-vector sums ─────────────────────────────────────────────────────────

TLorentzVector sum_p4(const ROOT::VecOps::RVec<TLorentzVector>& ps) {
    TLorentzVector total;
    for (const auto& p : ps) total += p;
    return total;
}

TLorentzVector sum_p4(std::initializer_list<TLorentzVector> ps) {
    TLorentzVector total;
    for (const auto& p : ps) total += p;
    return total;
}

inline TLorentzVector tlv_setmass(const TLorentzVector& p, double m) {
    double e = std::sqrt(p.P()*p.P() + m*m);
    return TLorentzVector(p.Px(), p.Py(), p.Pz(), e);
}

float deltaM(int nIsolep, int nRecoJets,
             const TLorentzVector& Wlep, const TLorentzVector& Whad) {
    if (nIsolep < 1 || nRecoJets < 2) return -1.0;
    TLorentzVector P_initial(0, 0, 0, ECM);
    return (P_initial - (Wlep + Whad)).M();
}


// ── FSR dressing ─────────────────────────────────────────────────────────────
// Add nearby reco photons (RPs with type==22) to each isolated lepton, returning
// a new RVec of dressed RPs. Photons are absorbed if dR(lep, γ) < dR_max and
// E_γ > E_min. The same-photon-multiple-leptons case is handled by giving the
// photon to the closest lepton only.

inline ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
dress_isoleps(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& Isoleps,
              const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& AllReco,
              double dR_max, double E_min) {
    const std::size_t nL = Isoleps.size();
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out(Isoleps);

    std::vector<TLorentzVector> lep_p4(nL);
    for (std::size_t i = 0; i < nL; ++i) {
        lep_p4[i].SetPxPyPzE(Isoleps[i].momentum.x, Isoleps[i].momentum.y,
                             Isoleps[i].momentum.z, Isoleps[i].energy);
    }

    for (const auto& p : AllReco) {
        if (p.type != 22) continue;
        if (p.energy < E_min) continue;
        TLorentzVector ph;
        ph.SetPxPyPzE(p.momentum.x, p.momentum.y, p.momentum.z, p.energy);

        int best = -1; double best_dr = dR_max;
        for (std::size_t i = 0; i < nL; ++i) {
            const double dr = lep_p4[i].DeltaR(ph);
            if (dr < best_dr) { best_dr = dr; best = (int)i; }
        }
        if (best < 0) continue;
        out[best].momentum.x += p.momentum.x;
        out[best].momentum.y += p.momentum.y;
        out[best].momentum.z += p.momentum.z;
        out[best].energy     += p.energy;
    }
    return out;
}

// Return the photons that were absorbed by dress_isoleps, for removal from the
// jet input collection (avoids double-counting FSR γ in the hadronic side).
inline ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>
dressed_photons(const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& Isoleps,
                const ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData>& AllReco,
                double dR_max, double E_min) {
    ROOT::VecOps::RVec<edm4hep::ReconstructedParticleData> out;

    std::vector<TLorentzVector> lep_p4(Isoleps.size());
    for (std::size_t i = 0; i < Isoleps.size(); ++i) {
        lep_p4[i].SetPxPyPzE(Isoleps[i].momentum.x, Isoleps[i].momentum.y,
                             Isoleps[i].momentum.z, Isoleps[i].energy);
    }

    for (const auto& p : AllReco) {
        if (p.type != 22) continue;
        if (p.energy < E_min) continue;
        TLorentzVector ph;
        ph.SetPxPyPzE(p.momentum.x, p.momentum.y, p.momentum.z, p.energy);
        for (const auto& lp : lep_p4) {
            if (lp.DeltaR(ph) < dR_max) { out.push_back(p); break; }
        }
    }
    return out;
}

}}  // namespace FCCAnalyses::WWFunctions

#endif
