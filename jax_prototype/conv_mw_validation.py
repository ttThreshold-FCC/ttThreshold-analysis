#!/usr/bin/env python3
# ── DATA/MC VALIDATION PLOT for the forward-folding 1-D mW fit ─────────────────
# Materializes the "how do I check data/MC?" deliverable (HANDOFF_DAY2 NEXT #1).
#
# The fit is unbinned ML on P(m_qq^reco, m_lν^reco | mW), built from MC by reweighting
# (forward-folding). The companion CHECK is the standard pre-fit / post-fit projection:
#   • DATA  = the reco-mass histograms (here MC-as-data: the nominal MC sample).
#   • MODEL = the folded template = the MC reco masses REWEIGHTED to a trial mW,
#             built OUT-OF-FOLD (k-fold) so it never sees the events it is compared to.
#   • pre-fit  = model at a reference mW (PDG 80.379)  → shows what a wrong mW looks like.
#   • post-fit = model at the best-fit m̂W (ensemble 2-D fold) → should describe the data.
# Bottom panel = the pull (data − model)/√model per bin; a flat post-fit pull = good GoF.
# This is exactly what becomes the MC-vs-DATA plot in a real analysis (data replaces the
# nominal MC; the MODEL machinery is unchanged). One figure per ECM + a 3-ECM summary.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_validation.py            # all 3 ECMs
#   python3 jax_prototype/conv_mw_validation.py 160        # one ECM
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import uproot

OUT  = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(OUT, exist_ok=True)
GW   = 2.085
# The contrast ("pre-fit") curve is the model at a DELIBERATELY MIS-SET mW = m̂W + MIS_DELTA. A fixed
# offset (vs e.g. PDG, which happens to coincide with m̂W at ecm163) makes the sensitivity demonstration
# consistent across ECMs: it shows the pull panel WOULD flag a wrong mW / a data-MC shape disagreement.
MIS_DELTA = float(os.environ.get("MIS_DELTA", "0.5"))
NMW  = int(os.environ.get("NMW", "61")); MWLO, MWHI = 79.0, 81.5
FOLD_K   = int(os.environ.get("FOLD_K", "5"))
FOLD_BIN = float(os.environ.get("FOLD_BIN", "0.5"))   # template build resolution (matches the fit)
FOLD_SM  = float(os.environ.get("FOLD_SM", "0.6"))
DISP_BIN = float(os.environ.get("DISP_BIN", "1.0"))   # display/pull binning
LOGZ_N   = int(os.environ.get("LOGZ_N", "256"))

# ── lineshape machinery (numpy port; identical to conv_mw_fit.py) ──────────────
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

# ── per-ECM analysis ──────────────────────────────────────────────────────────
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

    # gen-density reweighting weights w_i(mW) = P_true(gen_i;mW)/P_true(gen_i;mW0), grid weights
    mw_scan = np.linspace(MWLO, MWHI, NMW)
    Lgen = np.empty((N, NMW))
    for j, mW in enumerate(mw_scan):
        P = build_Ptrue(mW, SN, SW)
        f = RegularGridInterpolator((tg, tg), P, bounds_error=False, fill_value=1e-300)
        Lgen[:, j] = -2.0*np.log(np.maximum(f(np.stack([qg, lg], 1)), 1e-300))
    j0 = int(np.argmin(Lgen.sum(0)))                      # native gen density reference (mW0)

    # k-fold 2-D ensemble fit → m̂W (reproduces conv_mw_fit MODE=fold)
    rng = np.random.default_rng(2024); fold = rng.integers(0, FOLD_K, N)
    e1 = np.arange(30.0, 98.0+1e-6, FOLD_BIN); c1 = 0.5*(e1[:-1]+e1[1:])
    L2 = np.zeros((N, NMW))
    for j in range(NMW):
        wfull = np.exp(-0.5*(Lgen[:, j] - Lgen[:, j0]))
        for k in range(FOLD_K):
            tr = fold != k; te = fold == k
            H, _, _ = np.histogram2d(qr[tr], lr[tr], bins=[e1, e1], weights=wfull[tr])
            H = gaussian_filter(H, FOLD_SM); H = H/np.maximum(H.sum(), 1e-300)
            pv = RegularGridInterpolator((c1, c1), H, bounds_error=False, fill_value=1e-300)(np.stack([qr[te], lr[te]], 1))
            L2[te, j] = -2.0*np.log(np.maximum(pv, 1e-300))
    good = np.all(np.isfinite(L2), axis=1) & (np.ptp(L2, axis=1) > 0)
    ens = L2[good].sum(0); ens -= ens.min()
    i = int(np.clip(np.argmin(ens), 2, NMW-3)); c = np.polyfit(mw_scan[i-2:i+3], ens[i-2:i+3], 2)
    mhat = -c[1]/(2*c[0]); sig = 1.0/np.sqrt(c[0]); mw0 = mw_scan[j0]

    # per-event gen weights at an arbitrary trial mW (exact build_Ptrue, not scan-interpolated)
    L0 = Lgen[:, j0]
    def weights_at(mW):
        P = build_Ptrue(mW, SN, SW)
        f = RegularGridInterpolator((tg, tg), P, bounds_error=False, fill_value=1e-300)
        Lj = -2.0*np.log(np.maximum(f(np.stack([qg, lg], 1)), 1e-300))
        return np.exp(-0.5*(Lj - L0))
    w_pre  = weights_at(mhat + MIS_DELTA)         # deliberately mis-set, for the sensitivity contrast
    w_post = weights_at(mhat)

    # For an HONEST data/MC pull the pseudo-data and the MC template must be DISJOINT samples
    # (otherwise the template is histogrammed from the same events as the data ⇒ trivial χ²≈0).
    # pseudo-data = a held-out 40%; MC template = the other 60%, reweighted to the trial mW and
    # scaled to the data total (mW is a SHAPE measurement; the rate is not part of it). The pull
    # error combines data Poisson + the (weighted) MC statistical error of the scaled template.
    rng2 = np.random.default_rng(7); dmask = rng2.random(N) < 0.40; mmask = ~dmask
    edisp = np.arange(40.0, 95.0+1e-6, DISP_BIN); cdisp = 0.5*(edisp[:-1]+edisp[1:])
    def model_1d(obs, w):
        sw,  _ = np.histogram(obs[mmask], bins=edisp, weights=w[mmask])      # Σw  per bin
        sw2, _ = np.histogram(obs[mmask], bins=edisp, weights=w[mmask]**2)   # Σw² per bin
        return sw, sw2
    out = {"ecm": ecm, "mhat": mhat, "sig": sig, "mw0": mw0, "N": N,
           "edisp": edisp, "cdisp": cdisp, "mw_scan": mw_scan, "ens": ens, "fdata": dmask.mean()}
    for nm, obs in (("qq", qr), ("lv", lr)):
        data, _ = np.histogram(obs[dmask], bins=edisp)
        sw_pre,  sw2_pre  = model_1d(obs, w_pre)
        sw_post, sw2_post = model_1d(obs, w_post)
        sc_pre  = data.sum()/max(sw_pre.sum(),  1e-9); sc_post = data.sum()/max(sw_post.sum(), 1e-9)
        out[nm] = dict(data=data,
                       pre=sc_pre*sw_pre,   pre_err=sc_pre*np.sqrt(sw2_pre),
                       post=sc_post*sw_post, post_err=sc_post*np.sqrt(sw2_post))
    return out

# ── plotting ──────────────────────────────────────────────────────────────────
def plot_one(R):
    ecm = R["ecm"]
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.04, wspace=0.22)
    titles = {"qq": r"hadronic $m_{q\bar q}^{\rm reco}$", "lv": r"leptonic $m_{\ell\nu}^{\rm reco}$"}
    chi2s = {}
    for col, nm in enumerate(("qq", "lv")):
        d = R[nm]; c = R["cdisp"]; data = d["data"]; pre = d["pre"]; post = d["post"]
        derr = np.sqrt(np.maximum(data, 1.0))
        ax = fig.add_subplot(gs[0, col]); axr = fig.add_subplot(gs[1, col], sharex=ax)
        ax.errorbar(c, data, yerr=derr, fmt="ko", ms=4, capsize=0,
                    label=f'pseudo-data ({100*R["fdata"]:.0f}% MC)', zorder=5)
        ax.step(c, pre,  where="mid", color="C3", lw=1.6, ls="--",
                label=f"mis-set model @ m̂W+{MIS_DELTA:.1f}={R['mhat']+MIS_DELTA:.3f}")
        ax.step(c, post, where="mid", color="C0", lw=2.0,
                label=f"post-fit model @ m̂W={R['mhat']:.3f}")
        ax.set_ylabel("events / bin"); ax.set_title(f"{titles[nm]}  (ecm{ecm})")
        ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=.25)
        plt.setp(ax.get_xticklabels(), visible=False)
        # pull = (data - model)/sqrt(data_Poisson + MC_stat^2); χ² over bins with data>0
        def pulls(model, merr):
            sig2 = np.maximum(data, 1.0) + merr**2
            return (data - model)/np.sqrt(sig2)
        pull = pulls(post, d["post_err"]); pull_pre = pulls(pre, d["pre_err"])
        m = data > 0; ndf = int(m.sum()) - 1
        chi2 = float(np.sum(pull[m]**2)); chi2_pre = float(np.sum(pull_pre[m]**2))
        chi2s[nm] = (chi2, ndf, chi2_pre)
        axr.axhline(0, color="k", lw=.8)
        axr.bar(c, pull_pre, width=DISP_BIN*0.9, color="C3", alpha=.35, label="mis-set")
        axr.bar(c, pull,     width=DISP_BIN*0.55, color="C0", alpha=.9, label="post-fit")
        axr.set_ylim(-5, 5); axr.set_ylabel("pull", fontsize=9); axr.set_xlabel("mass [GeV]")
        axr.grid(alpha=.25)
        axr.text(0.02, 0.92, f"post χ²/ndf={chi2:.0f}/{ndf}  (mis-set {chi2_pre:.0f})",
                 transform=axr.transAxes, fontsize=8.5, color="C0", va="top")
        if col == 0: axr.legend(fontsize=8, loc="lower right", ncol=2)
    fig.suptitle(f"Data/MC validation — forward-fold mW fit, ecm{ecm}  "
                 f"(m̂W = {R['mhat']:.3f} ± {1000*R['sig']:.0f} MeV, mW0 = {R['mw0']:.3f})",
                 fontsize=12)
    png = f"{OUT}/validation_prepostfit_ecm{ecm}.png"
    plt.savefig(png, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"[plot] {png}   post-fit χ²/ndf  qq={chi2s['qq'][0]:.0f}/{chi2s['qq'][1]} (pre {chi2s['qq'][2]:.0f})  "
          f"lv={chi2s['lv'][0]:.0f}/{chi2s['lv'][1]} (pre {chi2s['lv'][2]:.0f})")
    return chi2s

if __name__ == "__main__":
    ecms = [int(sys.argv[1])] if len(sys.argv) > 1 else [157, 160, 163]
    for E in ecms:
        R = analyze(E)
        print(f"ecm{E}: m̂W={R['mhat']:.4f} ± {R['sig']:.4f}  mW0={R['mw0']:.3f}  N={R['N']}")
        plot_one(R)
