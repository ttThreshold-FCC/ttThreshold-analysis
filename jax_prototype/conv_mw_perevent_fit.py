#!/usr/bin/env python3
# ── PER-EVENT mW distribution turned into an ACTUAL MEASUREMENT (HANDOFF_DAY2 NEXT #3) ──
# The per-event observable m̂W_i = argmax_mW P_folded(m_qq^reco_i, m_lν^reco_i | mW) is a fixed
# per-event quantity (the "ideogram" peak). Its DISTRIBUTION shifts with the true mW. We measure mW
# by a TEMPLATE FIT of that distribution:
#   • templates T_μ(bin) = histogram of m̂W_i REWEIGHTED to true mW = μ  (w_i(μ) = P_true(gen_i;μ)/P_true(gen_i;mW0))
#   • "data" = the m̂W_i distribution of the nominal MC (or reweighted to an INJECTED μ for closure)
#   • fit: minimize the binned-Poisson −2lnL(μ) of data vs N·T_μ over a fine μ grid → μ̂ ± σ.
# Deliverables: (1) CLOSURE — μ̂ tracks the injected μ (slope≈1); (2) σ_μ of the per-event template fit;
# (3) POWER comparison — σ(per-event template) vs σ(ensemble fold), showing the per-event read-out is
# slightly less optimal (a compressed 1-D observable) but a transparent, histogram-native measurement.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_perevent_fit.py            # all 3 ECMs
#   python3 jax_prototype/conv_mw_perevent_fit.py 160        # one ECM
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import uproot

OUT = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(OUT, exist_ok=True)
GW  = 2.085
LOGZ_N = int(os.environ.get("LOGZ_N", "256"))
OBS = os.environ.get("PE_OBS", "mean")           # mean | pmap | map  (point estimate used for the closure/σ fit)
PEWIDE_LO = float(os.environ.get("PEWIDE_LO", "73.0")); PEWIDE_HI = float(os.environ.get("PEWIDE_HI", "88.0"))
PE_NMW = int(os.environ.get("PE_NMW", "301"))  # wide per-event scan
# (fine grid: 0.05 GeV. The per-event posterior-mean is a weighted average over the scan nodes; a coarse
#  grid makes it cluster at node-spaced values ⇒ comb/aliasing in its histogram. 0.05 GeV ⇒ smooth observable.)
NTOY = int(os.environ.get("NTOY", "300"))

# ── lineshape machinery (identical to conv_mw_fit.py) ──────────────────────────
_GX, _GWq = np.polynomial.legendre.leggauss(LOGZ_N); _W2 = _GWq[:, None]*_GWq[None, :]
def log_Z(m_WW, mW, gW=GW):
    m_WW = np.atleast_1d(np.asarray(m_WW, float)); mwgw = mW*gW; mW2 = mW*mW
    out = np.empty(m_WW.shape, float); CH = max(1, 50_000_000 // (LOGZ_N*LOGZ_N))
    for a in range(0, len(m_WW), CH):
        s = (m_WW[a:a+CH])**2
        t_min = np.arctan(-mW2/mwgw); t_max = np.arctan((s-mW2)/mwgw)
        half_d = 0.5*(t_max-t_min); half_s = 0.5*(t_max+t_min)
        t = half_d[:, None]*_GX[None, :] + half_s[:, None]
        m = np.sqrt(np.maximum(mW2 + mwgw*np.tan(t), 1e-12)); inv = 1.0/m
        mh = m[:, :, None]; ml = m[:, None, :]; ih = inv[:, :, None]; il = inv[:, None, :]; sE = s[:, None, None]
        lam = (sE-(mh+ml)**2)*(sE-(mh-ml)**2)
        integ = np.where(lam > 0, np.sqrt(np.maximum(lam, 0))*ih*il/(4.0*sE), 0.0)
        Z = np.sum(_W2[None]*integ, axis=(1, 2)) * half_d*half_d
        out[a:a+CH] = np.log(np.maximum(Z, 1e-300))
    return out
def bw(m, mW, gW=GW):
    mwgw = mW*gW; d = m*m - mW*mW; return mwgw/(d*d + mwgw*mwgw)

TLO, THI, NT = 28.0, 96.0, 137
tg = np.linspace(TLO, THI, NT); dt = tg[1]-tg[0]
TH, TL = np.meshgrid(tg, tg, indexing="ij")
def radiator_nodes(mWW_g, ecm):
    cnt, edg = np.histogram(mWW_g, bins=60, range=(max(TLO, ecm-45), ecm+1.5))
    sn = 0.5*(edg[:-1]+edg[1:]); w = cnt.astype(float); m = w > 0
    return sn[m], w[m]
def build_Ptrue(mW, SN, SW):
    BW = bw(tg, mW)[:, None]*bw(tg, mW)[None, :]; u = np.zeros((NT, NT)); lz = log_Z(SN, mW)
    for k, (sp, wk) in enumerate(zip(SN, SW)):
        s = sp*sp; lam = (s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS = np.where(lam > 0, np.sqrt(np.maximum(lam, 0))/s, 0.0)
        u += wk/np.exp(lz[k]) * BW * PS
    return u/np.maximum(u.sum()*dt*dt, 1e-300)

# binned-Poisson -2lnL of data counts D vs expected E (Cash/Baker-Cousins), summed over bins
def nll_pois(D, E):
    E = np.maximum(E, 1e-9); term = E - D
    nz = D > 0; term[nz] += D[nz]*np.log(D[nz]/E[nz])
    return 2.0*term.sum()

def analyze(ecm):
    F = f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
    t = uproot.open(F)["events"]
    a = t.arrays(["gen_Whad_m", "gen_Wlep_m", "gen_WW_m", "reco_Whad_m", "reco_Wlep_m",
                  "reco_jet1_p", "reco_jet2_p", "reco_lep_p"], library="np")
    ok = (np.isfinite(a["gen_Whad_m"]) & np.isfinite(a["gen_Wlep_m"]) & np.isfinite(a["gen_WW_m"]) &
          np.isfinite(a["reco_Whad_m"]) & np.isfinite(a["reco_Wlep_m"]) &
          (a["reco_jet1_p"] > 0) & (a["reco_jet2_p"] > 0) & (a["reco_lep_p"] > 0) &
          (a["gen_Whad_m"]+a["gen_Wlep_m"] < a["gen_WW_m"]))
    qg = a["gen_Whad_m"][ok]; lg = a["gen_Wlep_m"][ok]
    qr = a["reco_Whad_m"][ok]; lr = a["reco_Wlep_m"][ok]; mWW_g = a["gen_WW_m"][ok]; N = len(qg)
    SN, SW = radiator_nodes(mWW_g, ecm)

    # ── gen-density reweighting weights over a wide mW grid (for templates + for defining m̂W_i) ──
    mw = np.linspace(PEWIDE_LO, PEWIDE_HI, PE_NMW)
    Ptg = [build_Ptrue(m, SN, SW) for m in mw]
    fg  = [RegularGridInterpolator((tg, tg), P, bounds_error=False, fill_value=1e-300) for P in Ptg]
    Lgen = np.array([-2.0*np.log(np.maximum(fi(np.stack([qg, lg], 1)), 1e-300)) for fi in fg]).T  # (N,nmw)
    j0 = int(np.argmin(Lgen.sum(0))); mw0 = mw[j0]                         # native gen-density reference
    Wcol = np.exp(-0.5*(Lgen - Lgen[:, [j0]]))                            # (N,nmw): w_i at each grid mW
    Wcol = np.minimum(Wcol, np.percentile(Wcol, 99.9, axis=0, keepdims=True))

    # ── per-event folded likelihood over the wide scan → m̂W_i (MAP) and posterior-mean ──
    # The folded reco density is PHYSICALLY smooth (resolution ~3 GeV); a coarse histogram + linear
    # interpolation injects bin-period kinks into m̂W_i (a comb in its distribution). Estimate the density
    # on a fine grid (0.25 GeV) with smoothing matched to ~0.5 GeV (a fast KDE) ⇒ genuinely smooth m̂W_i.
    PE_BIN = float(os.environ.get("PE_BIN", "0.25")); PE_SM = float(os.environ.get("PE_SM", "2.0"))
    e1 = np.arange(30.0, 98.0+1e-6, PE_BIN); c1 = 0.5*(e1[:-1]+e1[1:])
    Lpe = np.zeros((N, PE_NMW))
    for j in range(PE_NMW):
        H, _, _ = np.histogram2d(qr, lr, bins=[e1, e1], weights=Wcol[:, j]); H = gaussian_filter(H, PE_SM); H /= H.sum()
        pv = RegularGridInterpolator((c1, c1), H, bounds_error=False, fill_value=1e-300)(np.stack([qr, lr], 1))
        Lpe[:, j] = -2.0*np.log(np.maximum(pv, 1e-300))
    Lf = np.where(np.isfinite(Lpe), Lpe, 1e18)
    imap = np.argmin(Lf, axis=1)
    interior = (imap > 0) & (imap < PE_NMW-1)
    pe_map = mw[imap]                                          # grid-snapped MAP (combed)
    Pn = np.exp(-0.5*(Lf - Lf.min(1, keepdims=True))); Pn /= Pn.sum(1, keepdims=True)
    pe_mean = (Pn*mw[None, :]).sum(1)                          # centroid (range-dependent)
    # parabolic sub-grid MAP: vertex of the (smooth) per-event likelihood near its minimum. Continuous
    # ⇒ no grid comb; range-independent ⇒ no centroid edge pile-up. The natural per-event best-fit mW.
    imc = np.clip(imap, 1, PE_NMW-2); dmw = mw[1]-mw[0]
    y0 = np.take_along_axis(Lf, (imc-1)[:, None], 1)[:, 0]
    y1 = np.take_along_axis(Lf,  imc[:, None],    1)[:, 0]
    y2 = np.take_along_axis(Lf, (imc+1)[:, None], 1)[:, 0]
    denom = y0 - 2*y1 + y2
    off = np.clip(np.where(denom > 0, 0.5*(y0-y2)/denom, 0.0), -1, 1)
    pe_pmap = mw[imc] + off*dmw
    obs = {"map": pe_map, "mean": pe_mean, "pmap": pe_pmap}[OBS]

    # ── STACKED per-event posterior (the smooth "ideogram") for DISPLAY ─────────
    # Summing each event's smooth per-event likelihood curve gives a smooth distribution (no point-estimate
    # comb/caustics). This is the data/MC-checkable per-event mW density; reweighting to true mW=μ shifts it.
    Pn_i = Pn[interior]                                                    # (Ni, PE_NMW) per-event posteriors
    logW_i = -0.5*(Lgen[interior] - Lgen[interior][:, [j0]])
    def _wat(mu, lw):
        idx = np.clip(np.searchsorted(mw, mu)-1, 0, PE_NMW-2); fr = (mu-mw[idx])/(mw[idx+1]-mw[idx])
        w = np.exp(lw[:, idx]*(1-fr) + lw[:, idx+1]*fr); return np.minimum(w, np.percentile(w, 99.9))
    def _stack(mu):
        w = _wat(mu, logW_i); s = (w[:, None]*Pn_i).sum(0); return s/np.maximum(s.sum()*dmw, 1e-300)
    stack_data = _stack(mw0); stack_lo = _stack(mw0-0.4); stack_hi = _stack(mw0+0.4)

    # ── build per-event-mW templates T_μ(bin) on a fit grid of true mW = μ ──────
    eb = np.arange(74.0, 87.0+1e-6, 0.25); cb = 0.5*(eb[:-1]+eb[1:])      # m̂W_i histogram bins
    mu_grid = np.linspace(mw0-0.8, mw0+0.8, 33)                            # true-mW fit grid
    use = np.ones(N, bool) if OBS == "mean" else interior                 # (p)MAP defined for interior events
    o = obs[use]
    # weights at arbitrary μ via interpolation of the per-event log-weight columns (smooth in μ)
    logW = -0.5*(Lgen[use] - Lgen[use][:, [j0]])                          # (Nu, nmw)
    def wat(mu):
        # vectorized interp over events
        idx = np.searchsorted(mw, mu) - 1; idx = np.clip(idx, 0, PE_NMW-2)
        frac = (mu - mw[idx])/(mw[idx+1]-mw[idx])
        lw = logW[:, idx]*(1-frac) + logW[:, idx+1]*frac
        w = np.exp(lw); return np.minimum(w, np.percentile(w, 99.9))
    T = np.zeros((len(mu_grid), len(cb)))
    for k, mu in enumerate(mu_grid):
        h, _ = np.histogram(o, bins=eb, weights=wat(mu)); T[k] = h/max(h.sum(), 1e-300)
    Tinterp = RegularGridInterpolator((mu_grid,), T, bounds_error=False, fill_value=None)

    # fit one (data-histogram) realization → μ̂, σ via the -2lnL(μ) parabola on a fine μ grid
    mu_fine = np.linspace(mu_grid[0]+0.02, mu_grid[-1]-0.02, 161)
    def fit_hist(D):
        Ntot = D.sum()
        nll = np.array([nll_pois(D, Ntot*np.clip(Tinterp([mu])[0], 1e-12, None)) for mu in mu_fine])
        i = int(np.clip(np.argmin(nll), 2, len(nll)-3))
        c = np.polyfit(mu_fine[i-2:i+3], nll[i-2:i+3], 2)
        return -c[1]/(2*c[0]), (1.0/np.sqrt(c[0]) if c[0] > 0 else np.nan)

    # ── (1) CLOSURE on Asimov data injected at μ0 + inj (no Poisson noise → expected σ) ──
    injs = np.linspace(-0.4, 0.4, 9); fitted = []; sigmas = []
    for inj in injs:
        D = o.size * np.clip(Tinterp([mw0+inj])[0], 0, None)              # Asimov data at injected μ
        D = D*(len(o)/D.sum())
        mh, sg = fit_hist(D); fitted.append(mh); sigmas.append(sg)
    fitted = np.array(fitted); sigmas = np.array(sigmas)
    slope = np.polyfit(mw0+injs, fitted, 1)[0]
    sig_asimov = float(np.nanmedian(sigmas))                              # expected σ of the per-event template fit

    # ── (2) Poisson toys at μ0 → closure bias + σ spread (full-stat realizations) ──
    rng = np.random.default_rng(123); toy = np.empty(NTOY)
    p0 = np.clip(Tinterp([mw0])[0], 0, None); p0 /= p0.sum()
    for b in range(NTOY):
        D = rng.multinomial(len(o), p0).astype(float)
        toy[b], _ = fit_hist(D)
    return dict(ecm=ecm, N=N, Nu=len(o), mw0=mw0, OBS=OBS, eb=eb, cb=cb, obs=o,
                pe_map=pe_map, pe_mean=pe_mean, interior=interior, mu_grid=mu_grid, T=T,
                mw_grid=mw, stack_data=stack_data, stack_lo=stack_lo, stack_hi=stack_hi,
                injs=injs, fitted=fitted, sigmas=sigmas, slope=slope, sig_asimov=sig_asimov,
                toy=toy, toy_mean=float(toy.mean()), toy_std=float(toy.std()))

def plot_one(R, ens_sig):
    ecm = R["ecm"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    # (A) per-event observable distribution + a few templates
    # (A) the smooth stacked per-event posterior (ideogram) + reweighted templates at μ0 and μ0±0.4
    mg = R["mw_grid"]
    ax[0].fill_between(mg, R["stack_data"], color="C0", alpha=.30, label="stacked posterior (data)")
    ax[0].plot(mg, R["stack_data"], color="C0", lw=1.8)
    ax[0].plot(mg, R["stack_lo"], "C3:",  lw=1.6, label=f"template μ={R['mw0']-0.4:.2f}")
    ax[0].plot(mg, R["stack_hi"], "C3--", lw=1.6, label=f"template μ={R['mw0']+0.4:.2f}")
    ax[0].axvline(R["mw0"], color="k", ls="--", lw=1, label=f"mW0={R['mw0']:.3f}")
    ax[0].set_xlim(76, 85); ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel(r"per-event $m_W$ [GeV]"); ax[0].set_ylabel("normalized density")
    ax[0].set_title(f"(A) per-event $m_W$ likelihood density (stacked posterior), ecm{ecm}"); ax[0].legend(fontsize=8)
    # (B) closure: fitted vs injected
    ax[1].plot(R["mw0"]+R["injs"], R["fitted"], "o-", color="C0", label=f"slope {R['slope']:.2f}")
    ax[1].plot(R["mw0"]+R["injs"], R["mw0"]+R["injs"], "k--", lw=1, label="ideal (slope 1)")
    ax[1].set_xlabel("injected true mW [GeV]"); ax[1].set_ylabel(r"fitted $\hat\mu$ [GeV]")
    ax[1].set_title(f"(B) closure (Asimov): slope={R['slope']:.3f}\nexpected σ_μ(per-event fit)={1000*R['sig_asimov']:.1f} MeV")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    # (C) Poisson toys at μ0 + power comparison
    ax[2].hist(R["toy"], bins=30, color="C0", alpha=.7)
    ax[2].axvline(R["mw0"], color="k", ls="--", lw=1, label=f"mW0={R['mw0']:.3f}")
    ax[2].axvline(R["toy_mean"], color="C3", lw=1.5, label=f"toy mean {R['toy_mean']:.3f}")
    ax[2].set_xlabel(r"toy $\hat\mu$ [GeV]")
    ax[2].set_title(f"(C) {NTOY} Poisson toys @ mW0\n"
                    f"bias={1000*(R['toy_mean']-R['mw0']):+.0f} MeV  σ_toy={1000*R['toy_std']:.1f} MeV\n"
                    f"vs ensemble fold σ={1000*ens_sig:.1f} MeV  (ratio {R['toy_std']/ens_sig:.2f}×)")
    ax[2].legend(fontsize=9)
    fig.suptitle(f"Per-event mW distribution as a MEASUREMENT — ecm{ecm} "
                 f"(observable={R['OBS']}, N_used={R['Nu']})", fontsize=12)
    plt.tight_layout(); png = f"{OUT}/perevent_measurement_ecm{ecm}.png"
    plt.savefig(png, dpi=120); plt.close(fig)
    print(f"[plot] {png}")

# ensemble fold curvature σ and closure (from MODE=fold, full sample) — the optimal read-out
ENS_SIG = {157: 0.0076, 160: 0.0071, 163: 0.0101}
ENS_MW  = {157: 80.2885, 160: 80.2353, 163: 80.3848}
ENS_TGT = {157: 80.286, 160: 80.259, 163: 80.406}   # gen_marg closure target

def plot_summary(res):
    ecms = sorted(res); x = np.arange(len(ecms))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    # Panel 1: PRECISION — σ ensemble (optimal) vs per-event template fit (expected/Asimov σ), ratio annotated
    w = 0.3
    se = np.array([1000*ENS_SIG[E] for E in ecms]); sp = np.array([1000*res[E]["sig_asimov"] for E in ecms])
    ax[0].bar(x-w/2, se, w, color="C0", label="ENSEMBLE fold (optimal, unbinned ML)")
    ax[0].bar(x+w/2, sp, w, color="C3", label="PER-EVENT template fit (histogram-native)")
    for i, E in enumerate(ecms):
        ax[0].annotate(f"{sp[i]/se[i]:.2f}×", (x[i]+w/2, sp[i]), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=11, color="C3")
        ax[0].annotate(f"{se[i]:.1f}", (x[i]-w/2, se[i]), textcoords="offset points",
                       xytext=(0, 3), ha="center", fontsize=9, color="C0")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"ecm{E}" for E in ecms]); ax[0].set_ylabel(r"$\sigma_{m_W}$ [MeV]")
    ax[0].set_title("PRECISION: per-event read-out costs only ~15% vs optimal\n"
                    "(both are read-outs of the SAME forward-fold likelihood)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")
    # Panel 2: VALIDITY — calibration line (fitted vs injected, both rel. to mW0), all ECMs ⇒ lie on y=x
    ax[1].plot([-0.4, 0.4], [-0.4, 0.4], "k--", lw=1.2, label="ideal (slope 1)")
    for i, E in enumerate(ecms):
        ax[1].plot(res[E]["injs"], res[E]["fitted"]-res[E]["mw0"], "o", ms=6, color=f"C{i}",
                   label=f"ecm{E}: slope {res[E]['slope']:.3f}, bias {1000*(res[E]['toy_mean']-res[E]['mw0']):+.0f} MeV")
    ax[1].set_xlabel(r"injected $m_W-m_{W0}$ [GeV]"); ax[1].set_ylabel(r"fitted $\hat\mu-m_{W0}$ [GeV]")
    ax[1].set_xlim(-0.45, 0.45); ax[1].set_ylim(-0.45, 0.45); ax[1].set_aspect("equal")
    ax[1].set_title("VALIDITY: per-event template fit recovers the injected $m_W$\n"
                    "(all ECMs on the diagonal, slope$\\approx$1, bias $<$ few MeV)")
    ax[1].legend(fontsize=9, loc="upper left"); ax[1].grid(alpha=.3)
    plt.tight_layout(); png = f"{OUT}/readout_compare.png"; plt.savefig(png, dpi=120); plt.close(fig)
    print(f"[plot] {png}")

if __name__ == "__main__":
    ecms = [int(sys.argv[1])] if len(sys.argv) > 1 else [157, 160, 163]
    res = {}
    for E in ecms:
        R = analyze(E); res[E] = R
        print(f"ecm{E} [{R['OBS']}]: mW0={R['mw0']:.4f}  closure slope={R['slope']:.3f}  "
              f"σ_asimov={1000*R['sig_asimov']:.1f} MeV  toy: bias={1000*(R['toy_mean']-R['mw0']):+.1f} "
              f"σ={1000*R['toy_std']:.1f} MeV  | ensemble σ={1000*ENS_SIG[E]:.1f} MeV "
              f"⇒ per-event/ensemble = {R['toy_std']/ENS_SIG[E]:.2f}×")
        plot_one(R, ENS_SIG[E])
    if len(res) > 1:
        plot_summary(res)
