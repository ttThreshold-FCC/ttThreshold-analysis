// Gen-level validation of the BW chi² term and its proposed log Z normalization.
// Mirrors /tmp/check_bw_gen.py but in C++ for speed and using the production
// helper log_Z_bw_phasespace from WWFunctions/WWKinReco.h.
//
// Build:
//   g++ -O2 -std=c++17 -I${WW_RECO_DIR} -I${WW_RECO_DIR}/WWFunctions \
//       $(root-config --cflags --libs) /tmp/check_bw_gen.cxx -o /tmp/check_bw_gen
// (the script run_check_bw_gen.sh wraps this).
//
// Usage: ./check_bw_gen <ECM>
//   reads /afs/.../step2/semihad/wzp6_ee_munumuqq_noCut_ecm<ECM>.root
//   prints argmin & parabola-fit mW under (UN-normalized, NORMALIZED) variants.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

// Local copies of bw_term and log_Z_bw_phasespace. The latter mirrors the
// production helper added to WWFunctions/WWKinReco.h; kept here to avoid
// pulling in Minuit2 / TLorentzVector / generated-params transitive deps.
static inline double bw_term_unnorm(double m_h, double m_l, double m_WW,
                                    double mW, double gW) {
    const double mwgw = mW * gW;
    const double dh = m_h * m_h - mW * mW;
    const double dl = m_l * m_l - mW * mW;
    const double bw_h = mwgw / (dh * dh + mwgw * mwgw);
    const double bw_l = mwgw / (dl * dl + mwgw * mwgw);
    const double s_ww = m_WW * m_WW;
    const double sum2 = (m_h + m_l) * (m_h + m_l);
    const double dif2 = (m_h - m_l) * (m_h - m_l);
    const double lam_raw = (s_ww - sum2) * (s_ww - dif2);
    const double lam = std::sqrt(lam_raw * lam_raw + 1e-24);
    return -2.0 * (std::log(bw_h) + std::log(bw_l))
           + 4.0 * std::log(M_PI)
           - std::log(lam) + 2.0 * std::log(s_ww);
}

// Runtime Gauss-Legendre node/weight builder. N-th order Legendre polynomial
// evaluated by recurrence; roots found via Newton iteration. Weights =
// 2 / ((1−x²) P'_N(x)²).
static std::vector<double> GL_X_RT, GL_W_RT;
static void build_GL_nodes(int N) {
    GL_X_RT.assign(N, 0.0);
    GL_W_RT.assign(N, 0.0);
    for (int k = 0; k < N; ++k) {
        // Initial guess: zero of Chebyshev approximation
        double x = std::cos(M_PI * (k + 0.75) / (N + 0.5));
        double Pn = 0, dPn = 0;
        for (int it = 0; it < 50; ++it) {
            double Pn1 = 1.0, Pn0 = x;
            for (int n = 2; n <= N; ++n) {
                double Pn_next = ((2*n - 1) * x * Pn0 - (n - 1) * Pn1) / n;
                Pn1 = Pn0; Pn0 = Pn_next;
            }
            Pn  = Pn0;
            dPn = N * (x * Pn0 - Pn1) / (x*x - 1.0);
            double dx = -Pn / dPn;
            x += dx;
            if (std::fabs(dx) < 1e-15) break;
        }
        GL_X_RT[k] = x;
        GL_W_RT[k] = 2.0 / ((1.0 - x*x) * dPn * dPn);
    }
}

// Original — full N×N with two `continue`s. Kept for old-vs-new comparison.
static inline double log_Z_old(double m_WW, double mW, double gW) {
    const double mwgw = mW * gW;
    const double mW2  = mW * mW;
    const double s_ww = m_WW * m_WW;
    const double t_min = std::atan(-mW2 / mwgw);
    const double t_max = std::atan((s_ww - mW2) / mwgw);
    const double half_d = 0.5 * (t_max - t_min);
    const double half_s = 0.5 * (t_max + t_min);
    const int N = (int)GL_X_RT.size();
    std::vector<double> m_node(N);
    for (int i = 0; i < N; ++i) {
        const double t  = half_d * GL_X_RT[i] + half_s;
        const double m2 = mW2 + mwgw * std::tan(t);
        m_node[i] = std::sqrt(m2 > 0.0 ? m2 : 0.0);
    }
    double Z = 0.0;
    for (int i = 0; i < N; ++i) {
        const double m_h = m_node[i];
        for (int j = 0; j < N; ++j) {
            const double m_l = m_node[j];
            if (m_h + m_l >= m_WW) continue;
            const double sum2 = (m_h + m_l) * (m_h + m_l);
            const double dif2 = (m_h - m_l) * (m_h - m_l);
            const double lam  = (s_ww - sum2) * (s_ww - dif2);
            if (lam <= 0.0) continue;
            Z += GL_W_RT[i] * GL_W_RT[j] * std::sqrt(lam) / (4.0 * m_h * m_l * s_ww);
        }
    }
    Z *= half_d * half_d;
    return std::log(Z > 0.0 ? Z : 1e-300);
}

// 3D precompute table: log Z(mW, gW, m_WW). Trilinear interp at runtime.
// Layout: row-major (mW, gW, m_WW). Index = ((i*n_gW)+k)*n_mWW + j.
struct LogZTable {
    double mW_lo, mW_hi, gW_lo, gW_hi, mWW_lo, mWW_hi;
    int n_mW, n_gW, n_mWW;
    double dmW, dgW, dmWW;
    double inv_dmW, inv_dgW, inv_dmWW;
    std::vector<double> data;
};

// Forward decl — table is built using on-the-fly version of log Z below.
static inline double log_Z_bw_phasespace(double m_WW, double mW, double gW);

static LogZTable build_log_Z_table(double mW_lo, double mW_hi, int n_mW,
                                    double gW_lo, double gW_hi, int n_gW,
                                    double mWW_lo, double mWW_hi, int n_mWW) {
    LogZTable T;
    T.mW_lo  = mW_lo;  T.mW_hi  = mW_hi;  T.n_mW  = n_mW;
    T.gW_lo  = gW_lo;  T.gW_hi  = gW_hi;  T.n_gW  = n_gW;
    T.mWW_lo = mWW_lo; T.mWW_hi = mWW_hi; T.n_mWW = n_mWW;
    T.dmW   = (mW_hi  - mW_lo)  / (n_mW  - 1);
    T.dgW   = (gW_hi  - gW_lo)  / (n_gW  - 1);
    T.dmWW  = (mWW_hi - mWW_lo) / (n_mWW - 1);
    T.inv_dmW  = 1.0 / T.dmW;
    T.inv_dgW  = 1.0 / T.dgW;
    T.inv_dmWW = 1.0 / T.dmWW;
    T.data.resize((size_t)n_mW * n_gW * n_mWW);
    for (int i = 0; i < n_mW; ++i) {
        const double mW_i = mW_lo + i * T.dmW;
        for (int k = 0; k < n_gW; ++k) {
            const double gW_k = gW_lo + k * T.dgW;
            for (int j = 0; j < n_mWW; ++j) {
                const double mWW_j = mWW_lo + j * T.dmWW;
                T.data[((size_t)i * n_gW + k) * n_mWW + j] =
                    log_Z_bw_phasespace(mWW_j, mW_i, gW_k);
            }
        }
    }
    return T;
}

// Trilinear interpolation. Out-of-grid clamps to nearest edge.
static inline double log_Z_table_lookup(const LogZTable& T,
                                         double m_WW, double mW, double gW) {
    double fmW  = (mW  - T.mW_lo)  * T.inv_dmW;
    double fgW  = (gW  - T.gW_lo)  * T.inv_dgW;
    double fmWW = (m_WW - T.mWW_lo) * T.inv_dmWW;
    if (fmW  < 0) fmW  = 0; else if (fmW  > T.n_mW  - 1) fmW  = T.n_mW  - 1;
    if (fgW  < 0) fgW  = 0; else if (fgW  > T.n_gW  - 1) fgW  = T.n_gW  - 1;
    if (fmWW < 0) fmWW = 0; else if (fmWW > T.n_mWW - 1) fmWW = T.n_mWW - 1;
    int i = (int)fmW;   if (i >= T.n_mW  - 1) i = T.n_mW  - 2;
    int k = (int)fgW;   if (k >= T.n_gW  - 1) k = T.n_gW  - 2;
    int j = (int)fmWW;  if (j >= T.n_mWW - 1) j = T.n_mWW - 2;
    const double wx = fmW - i, wy = fgW - k, wz = fmWW - j;
    const size_t row = T.n_mWW;          // stride along m_WW
    const size_t plane = (size_t)T.n_gW * T.n_mWW;  // stride along mW
    const double* p = T.data.data() + (size_t)i * plane + (size_t)k * row + j;
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

// On-the-fly log Z — upper-triangle (×2 off-diagonal) + branchless via std::max(lam,0).
static inline double log_Z_bw_phasespace(double m_WW, double mW, double gW) {
    const double mwgw = mW * gW;
    const double mW2  = mW * mW;
    const double s_ww = m_WW * m_WW;
    const double t_min = std::atan(-mW2 / mwgw);
    const double t_max = std::atan((s_ww - mW2) / mwgw);
    const double half_d = 0.5 * (t_max - t_min);
    const double half_s = 0.5 * (t_max + t_min);
    const int N = (int)GL_X_RT.size();

    std::vector<double> m_node(N);
    for (int i = 0; i < N; ++i) {
        const double t  = half_d * GL_X_RT[i] + half_s;
        const double m2 = mW2 + mwgw * std::tan(t);
        m_node[i] = std::sqrt(m2 > 0.0 ? m2 : 0.0);
    }
    double Z = 0.0;
    for (int i = 0; i < N; ++i) {
        const double m_h = m_node[i];
        const double w_h = GL_W_RT[i];
        const double inv_mh_4sww = 1.0 / (4.0 * m_h * s_ww);
        {
            const double m_l = m_h;
            const double sum2 = (m_h + m_l) * (m_h + m_l);
            const double lam  = (s_ww - sum2) * s_ww;     // dif² = 0
            Z += w_h * w_h * std::sqrt(std::max(lam, 0.0)) * inv_mh_4sww / m_l;
        }
        for (int j = i + 1; j < N; ++j) {
            const double m_l = m_node[j];
            const double sum2 = (m_h + m_l) * (m_h + m_l);
            const double dif2 = (m_h - m_l) * (m_h - m_l);
            const double lam  = (s_ww - sum2) * (s_ww - dif2);
            Z += 2.0 * w_h * GL_W_RT[j] * std::sqrt(std::max(lam, 0.0))
                 * inv_mh_4sww / m_l;
        }
    }
    Z *= half_d * half_d;
    return std::log(Z > 0.0 ? Z : 1e-300);
}

int main(int argc, char** argv) {
    int ECM   = (argc > 1) ? std::atoi(argv[1]) : 160;
    int Ngl   = (argc > 2) ? std::atoi(argv[2]) : 24;
    build_GL_nodes(Ngl);
    std::printf("[GL] N=%d\n", Ngl);
    char path[512];
    std::snprintf(path, sizeof(path),
        "/afs/cern.ch/work/m/mdefranc/private/WW/WW_reco/outputs/treemaker/lnuqq/step1/semihad_bwtest/"
        "wzp6_ee_munumuqq_noCut_ecm%d.root", ECM);
    std::printf("[load] ECM=%d  %s\n", ECM, path);

    TFile* f = TFile::Open(path, "READ");
    if (!f || f->IsZombie()) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }
    TTree* t = (TTree*)f->Get("events");
    if (!t) { std::fprintf(stderr, "no 'events' tree\n"); return 1; }

    TTreeReader r(t);
    TTreeReaderValue<double> v_mh (r, "gen_Whad_m");
    TTreeReaderValue<double> v_ml (r, "gen_Wlep_m");
    TTreeReaderValue<double> v_mWW(r, "gen_WW_m");

    std::vector<double> mh, ml, mWW;
    mh.reserve(100000); ml.reserve(100000); mWW.reserve(100000);
    while (r.Next()) {
        mh .push_back(*v_mh);
        ml .push_back(*v_ml);
        mWW.push_back(*v_mWW);
    }
    f->Close();
    std::printf("[load] events=%zu\n", mh.size());

    constexpr double GW_KF = 2.049;
    const int N = 121;
    double mW_grid[N];
    for (int k = 0; k < N; ++k) mW_grid[k] = 78.5 + 3.0 * k / (N - 1);

    // Precompute the 3D log Z table once. Grid covers Migrad's exploration
    // range in (mW, gW) with margin and spans the m_WW range observed in
    // step1 events.
    auto t_build0 = std::chrono::high_resolution_clock::now();
    LogZTable LZT = build_log_Z_table(/*mW*/  50.0, 100.0, 501,
                                      /*gW*/  1.95, 2.15,   21,
                                      /*mWW*/ 100.0, 170.0, 281);
    auto t_build1 = std::chrono::high_resolution_clock::now();
    std::printf("[table] %dx%dx%d build %.1f ms (%zu entries, %.1f MB)\n",
                LZT.n_mW, LZT.n_gW, LZT.n_mWW,
                std::chrono::duration<double, std::milli>(t_build1 - t_build0).count(),
                LZT.data.size(),
                LZT.data.size() * sizeof(double) / 1024.0 / 1024.0);

    enum Mode { UN, NRM_OLD, NRM_NEW, NRM_TAB };
    auto scan = [&](const char* tag, Mode mode) {
        std::vector<double> nll(N, 0.0);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int k = 0; k < N; ++k) {
            const double mW = mW_grid[k];
            double s = 0.0;
            for (size_t e = 0; e < mh.size(); ++e) {
                double t = bw_term_unnorm(mh[e], ml[e], mWW[e], mW, GW_KF);
                if (mode == NRM_OLD) t += 2.0 * log_Z_old          (mWW[e], mW, GW_KF);
                if (mode == NRM_NEW) t += 2.0 * log_Z_bw_phasespace(mWW[e], mW, GW_KF);
                if (mode == NRM_TAB) t += 2.0 * log_Z_table_lookup (LZT, mWW[e], mW, GW_KF);
                if (std::isfinite(t)) s += t;
            }
            nll[k] = s;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        // argmin
        int imin = 0;
        for (int k = 1; k < N; ++k) if (nll[k] < nll[imin]) imin = k;
        // parabola fit on ±5 around min
        int lo = std::max(0, imin - 5), hi = std::min(N, imin + 6);
        int n = hi - lo;
        double Sx = 0, Sx2 = 0, Sx3 = 0, Sx4 = 0, Sy = 0, Sxy = 0, Sx2y = 0;
        for (int k = lo; k < hi; ++k) {
            double x = mW_grid[k], y = nll[k];
            Sx += x; Sx2 += x*x; Sx3 += x*x*x; Sx4 += x*x*x*x;
            Sy += y; Sxy += x*y; Sx2y += x*x*y;
        }
        // Solve normal equations for y = a x² + b x + c
        double M[3][4] = {
            {Sx4, Sx3, Sx2, Sx2y},
            {Sx3, Sx2, Sx,  Sxy },
            {Sx2, Sx,  (double)n, Sy  },
        };
        for (int i = 0; i < 3; ++i) {
            int p = i;
            for (int k = i+1; k < 3; ++k) if (std::fabs(M[k][i]) > std::fabs(M[p][i])) p = k;
            std::swap(M[i], M[p]);
            for (int k = i+1; k < 3; ++k) {
                double f = M[k][i] / M[i][i];
                for (int j = i; j < 4; ++j) M[k][j] -= f * M[i][j];
            }
        }
        double a, b, c;
        c = M[2][3] / M[2][2];
        b = (M[1][3] - M[1][2]*c) / M[1][1];
        a = (M[0][3] - M[0][1]*b - M[0][2]*c) / M[0][0];
        double mW_fit = -b / (2 * a);
        double sig    = (a > 0) ? 1.0 / std::sqrt(a) : -1.0;
        std::printf("[%-7s] argmin %.4f  parabola %.4f  sigma=%.2f MeV  bias=%+.4f GeV  wall=%6.1f ms\n",
                    tag, mW_grid[imin], mW_fit, sig*1000, mW_fit - 80.385, ms);
    };

    scan("UN     ", UN);
    scan("NRM_NEW", NRM_NEW);
    scan("NRM_TAB", NRM_TAB);
    return 0;
}
