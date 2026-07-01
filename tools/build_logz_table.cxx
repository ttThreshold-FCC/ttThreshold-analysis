// Standalone builder for the kinfit log Z table. Run once outside cling/ROOT
// to produce the binary file mmap'd by WWKinReco.h at kinfit init.
//
// Build:  g++ -O2 -std=c++17 -march=native tools/build_logz_table.cxx -o tools/build_logz_table
// Run:    tools/build_logz_table [out_path]
//   default out_path = kinfit_inputs/logz_table.bin
//
// File format:
//   uint64_t magic = 0x4C5A54414220563031   ("LZTAB V01" packed, easy to grep)
//   int32_t  n_mW, n_gW, n_mWW
//   int32_t  pad (alignment)
//   double   mW_lo, mW_hi, gW_lo, gW_hi, mWW_lo, mWW_hi
//   double[] data, row-major: idx = ((i*n_gW)+k)*n_mWW + j
//
// Total file size = 8 (magic) + 16 (ints) + 48 (bounds) + total*8.
// At default grid (501x21x281), data = 22.6 MB.
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// ── Grid constants — KEEP IN SYNC WITH WWFunctions/WWKinReco.h ────────────
static constexpr double KF_LOGZ_GRID_MW_LO  = 50.0,  KF_LOGZ_GRID_MW_HI  = 100.0;
static constexpr int    KF_LOGZ_GRID_N_MW   = 501;
static constexpr double KF_LOGZ_GRID_GW_LO  = 1.95,  KF_LOGZ_GRID_GW_HI  = 2.15;
static constexpr int    KF_LOGZ_GRID_N_GW   = 21;
static constexpr double KF_LOGZ_GRID_MWW_LO = 100.0, KF_LOGZ_GRID_MWW_HI = 170.0;
static constexpr int    KF_LOGZ_GRID_N_MWW  = 281;
static constexpr uint64_t KF_LOGZ_TABLE_MAGIC = 0x4C5A544256303031ULL; // "LZTBV001" (8 bytes)

// 24-pt Gauss-Legendre nodes / weights — KEEP IN SYNC WITH WWFunctions/WWKinReco.h
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

// log Z(m_WW, mW, gW) — copy of WWKinReco.h::log_Z_bw_phasespace_ontf.
static inline double log_Z_bw_phasespace_ontf(double m_WW, double mW, double gW) {
    const double mwgw  = mW * gW;
    const double mW2   = mW * mW;
    const double s_ww  = m_WW * m_WW;
    const double t_min = std::atan(-mW2 / mwgw);
    const double t_max = std::atan((s_ww - mW2) / mwgw);
    const double half_d = 0.5 * (t_max - t_min);
    const double half_s = 0.5 * (t_max + t_min);

    std::array<double, KF_GL_N> m_node, inv_m_node;
    for (int i = 0; i < KF_GL_N; ++i) {
        const double t  = half_d * KF_GL24_X[i] + half_s;
        const double m2 = mW2 + mwgw * std::tan(t);
        m_node[i]     = std::sqrt(std::max(m2, 1e-12));
        inv_m_node[i] = 1.0 / m_node[i];
    }
    double Z = 0.0;
    for (int i = 0; i < KF_GL_N; ++i) {
        const double m_h  = m_node[i];
        const double w_h  = KF_GL24_W[i];
        const double inv_mh_4sww = inv_m_node[i] / (4.0 * s_ww);
        {
            const double m_l = m_h;
            const double sum2 = 4.0 * m_h * m_h;
            const double lam  = (s_ww - sum2) * s_ww;
            Z += w_h * w_h * std::sqrt(std::max(lam, 0.0)) * inv_mh_4sww * inv_m_node[i];
        }
        for (int j = i + 1; j < KF_GL_N; ++j) {
            const double m_l = m_node[j];
            const double sum2 = (m_h + m_l) * (m_h + m_l);
            const double dif2 = (m_h - m_l) * (m_h - m_l);
            const double lam  = (s_ww - sum2) * (s_ww - dif2);
            Z += 2.0 * w_h * KF_GL24_W[j] * std::sqrt(std::max(lam, 0.0))
                 * inv_mh_4sww * inv_m_node[j];
        }
    }
    Z *= half_d * half_d;
    return std::log(Z > 0.0 ? Z : 1e-300);
}

int main(int argc, char** argv) {
    const char* out = (argc > 1) ? argv[1]
                                 : "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco/kinfit_inputs/logz_table.bin";

    const double dmW  = (KF_LOGZ_GRID_MW_HI  - KF_LOGZ_GRID_MW_LO ) / (KF_LOGZ_GRID_N_MW  - 1);
    const double dgW  = (KF_LOGZ_GRID_GW_HI  - KF_LOGZ_GRID_GW_LO ) / (KF_LOGZ_GRID_N_GW  - 1);
    const double dmWW = (KF_LOGZ_GRID_MWW_HI - KF_LOGZ_GRID_MWW_LO) / (KF_LOGZ_GRID_N_MWW - 1);

    const size_t total = (size_t)KF_LOGZ_GRID_N_MW * KF_LOGZ_GRID_N_GW * KF_LOGZ_GRID_N_MWW;
    std::printf("[build] grid %dx%dx%d = %zu entries (%.1f MB)\n",
                KF_LOGZ_GRID_N_MW, KF_LOGZ_GRID_N_GW, KF_LOGZ_GRID_N_MWW,
                total, total * sizeof(double) / 1024.0 / 1024.0);
    // Write to <out>.tmp then rename — atomic on same filesystem. If the
    // destination is currently mmap'd by a running kinfit worker, the mv
    // swaps the directory entry while the old inode stays alive (with the
    // mmap intact). New readers see the new file; old readers keep mapping
    // the old inode until they unmap.
    std::string tmp = std::string(out) + ".tmp";
    std::printf("[build] writing %s (rename → %s on completion)\n", tmp.c_str(), out);

    std::ofstream f(tmp, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s: %s\n", tmp.c_str(), std::strerror(errno)); return 1; }

    // Header.
    const uint64_t magic = KF_LOGZ_TABLE_MAGIC;
    const int32_t  hdr_ints[4] = { KF_LOGZ_GRID_N_MW, KF_LOGZ_GRID_N_GW, KF_LOGZ_GRID_N_MWW, 0 };
    const double   hdr_d[6] = { KF_LOGZ_GRID_MW_LO, KF_LOGZ_GRID_MW_HI,
                                KF_LOGZ_GRID_GW_LO, KF_LOGZ_GRID_GW_HI,
                                KF_LOGZ_GRID_MWW_LO, KF_LOGZ_GRID_MWW_HI };
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    f.write(reinterpret_cast<const char*>(hdr_ints), sizeof(hdr_ints));
    f.write(reinterpret_cast<const char*>(hdr_d), sizeof(hdr_d));

    // Data — row-major: ((i*n_gW)+k)*n_mWW + j
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> row(KF_LOGZ_GRID_N_MWW);
    for (int i = 0; i < KF_LOGZ_GRID_N_MW; ++i) {
        const double mW_i = KF_LOGZ_GRID_MW_LO + i * dmW;
        for (int k = 0; k < KF_LOGZ_GRID_N_GW; ++k) {
            const double gW_k = KF_LOGZ_GRID_GW_LO + k * dgW;
            for (int j = 0; j < KF_LOGZ_GRID_N_MWW; ++j) {
                const double mWW_j = KF_LOGZ_GRID_MWW_LO + j * dmWW;
                row[j] = log_Z_bw_phasespace_ontf(mWW_j, mW_i, gW_k);
            }
            f.write(reinterpret_cast<const char*>(row.data()),
                    row.size() * sizeof(double));
        }
        if ((i + 1) % 50 == 0)
            std::printf("[build] mW row %d/%d done\n", i + 1, KF_LOGZ_GRID_N_MW);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    // Guard against a silent short write (e.g. ENOSPC): a truncated table with a
    // valid 72-byte header would still pass the loader's magic+dims check and then
    // SIGBUS on mmap reads past EOF. Abort before the rename if the stream faulted.
    f.flush();
    if (!f.good()) {
        std::fprintf(stderr, "[build] write/flush error on %s (disk full?); not renaming\n", tmp.c_str());
        f.close(); std::remove(tmp.c_str());
        return 1;
    }
    f.close();
    if (!f.good()) {
        std::fprintf(stderr, "[build] close/flush error on %s; not renaming\n", tmp.c_str());
        std::remove(tmp.c_str());
        return 1;
    }
    if (std::rename(tmp.c_str(), out) != 0) {
        std::fprintf(stderr, "rename %s → %s failed: %s\n",
                     tmp.c_str(), out, std::strerror(errno));
        return 1;
    }
    std::printf("[build] done in %.1f s, file %.1f MB\n",
                std::chrono::duration<double>(t1 - t0).count(),
                (sizeof(magic) + sizeof(hdr_ints) + sizeof(hdr_d) + total*sizeof(double))
                / 1024.0 / 1024.0);
    return 0;
}
