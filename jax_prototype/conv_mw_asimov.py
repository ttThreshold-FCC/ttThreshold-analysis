#!/usr/bin/env python3
# Asimov / 1-D statistical reach of the forward-fold mW fit: σ_mW ∝ 1/√N.
# Subsamples the ACTUAL fold likelihood @ecm160 to verify the 1/√N law, then projects to 1e6/1e7/1e8.
# Regenerates figs/asimov_scaling.png (DAY3 plot was an inline script, now saved + extended to 1e8).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# python3 jax_prototype/conv_mw_asimov.py [ecm]
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

ECM    = int(sys.argv[1]) if len(sys.argv) > 1 else 160
GW     = float(os.environ.get("GW", "2.085"))
NMW    = int(os.environ.get("NMW", "61"))
MWLO, MWHI = 79.0, 81.5
LOGZ_N = 256
FOLD_K = 5; FOLD_BIN = 0.5; FOLD_SM = 0.6
NDRAW  = int(os.environ.get("NDRAW", "40"))            # random subsamples per N (median for stability)
ROOT = f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
OUT  = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(OUT, exist_ok=True)
FIGS = os.path.join(os.path.dirname(__file__), "..", "slides", "figs")
import uproot

_GX, _GWl = np.polynomial.legendre.leggauss(LOGZ_N); _W2 = _GWl[:, None]*_GWl[None, :]
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
        out[a:a+CH] = np.log(np.maximum(np.sum(_W2[None]*integ, axis=(1, 2)) * half_d*half_d, 1e-300))
    return out
def bw(m, mW, gW=GW):
    mwgw = mW*gW; d = m*m - mW*mW; return mwgw/(d*d + mwgw*mwgw)

t = uproot.open(ROOT)["events"]
br = ["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_Whad_m","reco_Wlep_m","reco_jet1_p","reco_jet2_p","reco_lep_p"]
a = t.arrays(br, library="np")
ok = (np.isfinite(a["gen_Whad_m"]) & np.isfinite(a["gen_Wlep_m"]) & np.isfinite(a["gen_WW_m"]) &
      np.isfinite(a["reco_Whad_m"]) & np.isfinite(a["reco_Wlep_m"]) &
      (a["reco_jet1_p"]>0) & (a["reco_jet2_p"]>0) & (a["reco_lep_p"]>0) &
      (a["gen_Whad_m"]+a["gen_Wlep_m"] < a["gen_WW_m"]))
idx = np.where(ok)[0]; a = {k: v[idx] for k, v in a.items()}; N = len(idx)
mqq_g=a["gen_Whad_m"]; mlv_g=a["gen_Wlep_m"]; mqq_r=a["reco_Whad_m"]; mlv_r=a["reco_Wlep_m"]
mw_scan = np.linspace(MWLO, MWHI, NMW)
print(f"[asimov] ecm{ECM} N={N}")

# ── grid gen lineshape (reweighting weights) ──
TLO, THI, NT = 28.0, 96.0, 137
tg = np.linspace(TLO, THI, NT); dt = tg[1]-tg[0]; TH, TL = np.meshgrid(tg, tg, indexing="ij")
cnt, edg = np.histogram(a["gen_WW_m"], bins=60, range=(max(TLO, ECM-45), ECM+1.5))
SN = 0.5*(edg[:-1]+edg[1:]); SW = cnt.astype(float); mok = SW>0; SN, SW = SN[mok], SW[mok]
def build_Ptrue(mW):
    bwh = bw(tg, mW); BWout = bwh[:,None]*bwh[None,:]; u = np.zeros((NT, NT)); logZ = log_Z(SN, mW)
    for k,(sp,wk) in enumerate(zip(SN,SW)):
        s = sp*sp; lam = (s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS = np.where(lam>0, np.sqrt(np.maximum(lam,0))/s, 0.0)
        u += (wk/np.exp(logZ[k]))*BWout*PS
    return u/np.maximum(u.sum()*dt*dt, 1e-300)
def scan_grid_gen():
    L = np.zeros((N, NMW))
    for j,mW in enumerate(mw_scan):
        P = build_Ptrue(mW)
        L[:, j] = -2.0*np.log(np.maximum(RegularGridInterpolator((tg,tg),P,bounds_error=False,fill_value=1e-300)(np.stack([mqq_g,mlv_g],1)),1e-300))
    return L

# ── fold per-event L matrix (k-fold, grid weights) ──
Lgen = scan_grid_gen(); j0 = int(np.argmin(Lgen.sum(0)))
rng = np.random.default_rng(2024); fold = rng.integers(0, FOLD_K, N)
e1 = np.arange(30.0, 98.0+1e-6, FOLD_BIN); c1 = 0.5*(e1[:-1]+e1[1:]); Lf = np.zeros((N, NMW))
for j in range(NMW):
    wfull = np.exp(-0.5*(Lgen[:, j] - Lgen[:, j0]))
    for k in range(FOLD_K):
        tr = fold!=k; te = fold==k; ii = np.where(te)[0]
        H,_,_ = np.histogram2d(mqq_r[tr], mlv_r[tr], bins=[e1,e1], weights=wfull[tr])
        H = gaussian_filter(H, sigma=FOLD_SM); H = H/np.maximum(H.sum(),1e-300)
        pv = RegularGridInterpolator((c1,c1),H,bounds_error=False,fill_value=1e-300)(np.stack([mqq_r[te],mlv_r[te]],1))
        Lf[ii, j] = -2.0*np.log(np.maximum(pv, 1e-300))
good = np.all(np.isfinite(Lf), axis=1) & (np.ptp(Lf, axis=1) > 0); Lg = Lf[good]; Ng = Lg.shape[0]

def curv_sigma(L):
    e = L.sum(0); e -= e.min(); i = int(np.clip(np.argmin(e), 2, NMW-3))
    c = np.polyfit(mw_scan[i-2:i+3], e[i-2:i+3], 2); return 1.0/np.sqrt(c[0]) if c[0]>0 else np.nan
sig_full = curv_sigma(Lg)
print(f"[asimov] full-sample σ_curv = {1000*sig_full:.2f} MeV  (Ng={Ng})")

# ── subsample to verify 1/√N ──
Nsub = [6000, 12000, 24000, 48000, Ng]
rs = np.random.default_rng(7); meas_N, meas_sig = [], []
for n in Nsub:
    ss = [curv_sigma(Lg[rs.choice(Ng, n, replace=False)]) for _ in range(NDRAW)]
    meas_N.append(n); meas_sig.append(np.nanmedian(ss))
    print(f"  N={n:>7}  σ={1000*meas_sig[-1]:6.2f} MeV   (1/√N pred {1000*sig_full*np.sqrt(Ng/n):6.2f})")
meas_N = np.array(meas_N); meas_sig = np.array(meas_sig)

# ── projections ──
proj_N = np.array([1e6, 1e7, 1e8]); proj_sig = sig_full*np.sqrt(Ng/proj_N)
for n, s in zip(proj_N, proj_sig): print(f"  PROJECT N={n:.0e}  σ≈{1000*s:.2f} MeV")

# ── plot ──
fig, ax = plt.subplots(figsize=(7.6, 5.6))
xx = np.logspace(np.log10(4000), np.log10(2e8), 200)
ax.plot(xx, 1000*sig_full*np.sqrt(Ng/xx), "-", color="gray", lw=1.4, label=r"$1/\sqrt{N}$ law (anchored at full sample)")
ax.plot(meas_N, 1000*meas_sig, "o", color="C0", ms=8, label="MC subsampling (curvature σ)")
ax.plot([Ng], [1000*sig_full], "*", color="C3", ms=16, label=f"full sample: {1000*sig_full:.1f} MeV @ {Ng/1e3:.0f}k")
ax.plot(proj_N, 1000*proj_sig, "D", color="C2", ms=9, mfc="none", mew=1.8, label="projection")
for n, s in zip(proj_N, proj_sig):
    ax.annotate(f"{1000*s:.2f} MeV\n@ {n:.0e}".replace("e+0","e"), (n, 1000*s),
                textcoords="offset points", xytext=(6, 8), fontsize=8.5, color="C2")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N events"); ax.set_ylabel(r"$\sigma_{m_W}$  [MeV]")
ax.set_title(f"Statistical reach (ecm{ECM}): $\\sigma_{{m_W}}\\propto 1/\\sqrt{{N}}$")
ax.grid(which="both", alpha=.3); ax.legend(fontsize=8.5, loc="upper right")
ax.set_xlim(4e3, 2.5e8); ax.set_ylim(0.15, 40)
plt.tight_layout()
for d in (OUT, FIGS):
    p = os.path.join(d, "asimov_scaling.png"); plt.savefig(p, dpi=120); print(f"[plot] {p}")
