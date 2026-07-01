#!/usr/bin/env python3
# ── Convolution-based 1-D mW fit (mW is the ONLY free parameter) ───────────────
# Staged validation (each isolates one piece; see notes/convolution_mw_fit/HANDOFF.md):
#   MODE=gen_pe   Test 0: per-event L(mW)=Π P(m_qq^g,m_lν^g | m_WW^g, mW) using the EXACT per-event
#                 m_WW=gen_WW_m and the kinfit joint pdf (bw·bw·PS / Z, log_Z_ontf). Validates the
#                 LINESHAPE + normalization. No radiator, no convolution. → measures the generator mW.
#   MODE=gen_marg Test 1: same gen masses but m_WW is MARGINALIZED over the ISR radiator ρ(√s')
#                 (RAD=obs: measured gen_WW_m spectrum, ÷Z per node ; RAD=analytic: bare Kuraev-Fadin
#                 R(x), Z cancels). Validates the MARGINALIZATION + radiator (this is what reco must use,
#                 since per-event √s' is unknown at reco).
#   MODE=reco     Test 2: fit reco_Whad_m, reco_Wlep_m with P_reco = P_true ⊗ kernel (empirical 2-D
#                 resolution kernel from conv_mw_kernel.py). Validates the FULL method. Compares σ_mW,
#                 closure, pull to the per-event soft fit (kinfit_mW / kinfit_valid in the tree).
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   MODE=gen_pe MAXN=0 python3 jax_prototype/conv_mw_fit.py [ecm]
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.interpolate import RegularGridInterpolator
np.set_printoptions(linewidth=160, suppress=True)

ECM    = int(sys.argv[1]) if len(sys.argv) > 1 else 160
MODE   = os.environ.get("MODE", "gen_pe")          # gen_pe | gen_marg | reco
RAD    = os.environ.get("RAD", "obs")              # obs | analytic   (marg/reco only)
GW     = float(os.environ.get("GW", "2.085"))      # W width used in the BW lineshape (wzp6 default)
MW_REF = float(os.environ.get("MW_REF", "80.379")) # reference pole for bias reporting (wzp6 default)
MAXN   = int(os.environ.get("MAXN", "0"))
NMW    = int(os.environ.get("NMW", "61"))          # mW scan points
MWLO, MWHI = float(os.environ.get("MWLO","79.0")), float(os.environ.get("MWHI","81.5"))
NBOOT  = int(os.environ.get("NBOOT", "200"))       # bootstrap replicas for stat σ / pull
ROOT = f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
OUT  = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(OUT, exist_ok=True)
KDIR = os.path.join(os.path.dirname(__file__), "conv_kernels")
import uproot

# ── GL-24 nodes for the BW phase-space normalization Z (numpy port of log_Z_ontf) ──
sys.path.insert(0, os.path.dirname(__file__)); import jaxfit_common as J
# High-order Gauss-Legendre for the BW phase-space normalization Z. The shared jaxfit_common.log_Z_ontf
# uses GL-24 to mirror the C++ kinfit, but the √λ kinematic edge makes GL-24 ~5% low with an error that
# SLOPES ~−0.04/GeV in mW ⇒ a spurious ~+40 MeV bias in the gen-level MLE (adversarial-review finding).
# GL-256 cuts that slope ~30× (~1–3 MeV residual). LOGZ_N is env-tunable; we do NOT touch log_Z_ontf.
LOGZ_N = int(os.environ.get("LOGZ_N", "256"))
_GX, _GW = np.polynomial.legendre.leggauss(LOGZ_N); _W2 = _GW[:, None]*_GW[None, :]
def log_Z(m_WW, mW, gW=GW):
    """log Z(m_WW,mW)=∫∫ bw·bw·(√λ/s) dm_h dm_l via the arctan(m²) substitution + GL-LOGZ_N quadrature,
    chunked over m_WW to bound memory. Vectorized over m_WW (array)."""
    m_WW = np.atleast_1d(np.asarray(m_WW, float)); mwgw = mW*gW; mW2 = mW*mW
    out = np.empty(m_WW.shape, float)
    CH = max(1, 50_000_000 // (LOGZ_N*LOGZ_N))                            # ~50M-cell chunks
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

def bw(m, mW, gW=GW):                                   # relativistic BW in m² (Lorentzian)
    mwgw = mW*gW; d = m*m - mW*mW
    return mwgw/(d*d + mwgw*mwgw)

# ── load ──────────────────────────────────────────────────────────────────────
t = uproot.open(ROOT)["events"]
br = ["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_Whad_m","reco_Wlep_m",
      "reco_jet1_p","reco_jet2_p","reco_lep_p","kinfit_mW","kinfit_valid"]
a = t.arrays(br, library="np")
ok = (np.isfinite(a["gen_Whad_m"]) & np.isfinite(a["gen_Wlep_m"]) & np.isfinite(a["gen_WW_m"]) &
      np.isfinite(a["reco_Whad_m"]) & np.isfinite(a["reco_Wlep_m"]) &
      (a["reco_jet1_p"]>0) & (a["reco_jet2_p"]>0) & (a["reco_lep_p"]>0) &
      (a["gen_Whad_m"]+a["gen_Wlep_m"] < a["gen_WW_m"]))     # gen on phase space
idx = np.where(ok)[0]
if MAXN > 0: idx = idx[:MAXN]
a = {k: v[idx] for k, v in a.items()}; N = len(idx)
mqq_g=a["gen_Whad_m"]; mlv_g=a["gen_Wlep_m"]; mWW_g=a["gen_WW_m"]
mqq_r=a["reco_Whad_m"]; mlv_r=a["reco_Wlep_m"]
print(f"\n{'='*78}\n[conv-fit] MODE={MODE} RAD={RAD} ecm{ECM}  N={N}  GW={GW} MW_REF={MW_REF}\n{'='*78}")

mw_scan = np.linspace(MWLO, MWHI, NMW)

# ─────────────────────────────────────────────────────────────────────────────
# Test 0 — per-event exact lineshape  L(mW) = Σ_i  -2 log P(m_h,m_l | m_WW_i, mW)
# ─────────────────────────────────────────────────────────────────────────────
def n2ll_perevent(mh, ml, mWW, mW, logz=None):
    s = mWW*mWW
    lam = (s-(mh+ml)**2)*(s-(mh-ml)**2)
    bad = lam <= 0
    lam = np.where(bad, 1.0, lam)
    lz = log_Z(mWW, mW) if logz is None else logz          # logz: precomputed (tabulated) per-event log_Z
    t = (-2.0*np.log(np.maximum(bw(mh,mW),1e-300)) - 2.0*np.log(np.maximum(bw(ml,mW),1e-300))
         - np.log(lam) + 2.0*np.log(s) + 2.0*lz)
    return np.where(bad, 1e6, t)

# ─────────────────────────────────────────────────────────────────────────────
# Grid model — P_true(m_h,m_l; mW) marginalized over √s' nodes (for gen_marg / reco)
# ─────────────────────────────────────────────────────────────────────────────
TLO, THI, NT = 28.0, 96.0, 137                      # true-mass grid (0.5 GeV)
tg = np.linspace(TLO, THI, NT); dt = tg[1]-tg[0]
TH, TL = np.meshgrid(tg, tg, indexing="ij")         # TH=m_h axis0, TL=m_l axis1

def radiator_nodes():
    """Return (snodes, weights, divide_by_Z). snodes = √s' values; weights ∝ marginalization weight."""
    if RAD == "obs":
        # measured gen_WW_m spectrum = R_bare·σ_WW(s') ⇒ divide model by Z per node to undo σ_WW
        cnt, edg = np.histogram(mWW_g, bins=60, range=(max(TLO, ECM-45), ECM+1.5))
        sn = 0.5*(edg[:-1]+edg[1:]); w = cnt.astype(float)
        m = w > 0; return sn[m], w[m], True
    else:
        # bare Kuraev-Fadin radiator R(x), x=fractional energy lost, s'=s(1-x).
        # NB on softness (review correction): the bare R(x) is ALREADY soft on its own — the integrable
        # x^(β−1) singularity + the −½β(2−x) soft term + the x<0.45 truncation give ⟨x⟩≈0.008 here, NOT
        # the textbook ⟨x⟩=β/(β+1)≈0.10. The √λ/s phase-space factor in the model then applies the
        # σ_WW(s') threshold suppression ON TOP (hardening low-s'), and obs-mode divides the observed
        # spectrum by Z to recover this same bare R — so both modes apply the Z=σ_WW weight exactly once.
        beta = (2*7.2973525693e-3/np.pi)*(np.log(ECM**2/0.000510999**2) - 1.0)
        x = np.linspace(1e-5, 0.45, 400); dx = x[1]-x[0]
        Rx = beta*x**(beta-1.0)*(1.0+3.0/4.0*beta) - 0.5*beta*(2.0-x)   # LL + soft, O(β)
        Rx = np.maximum(Rx, 0.0)
        sp = ECM*np.sqrt(1.0-x)                                          # √s'
        # (caveat) analytic R lacks beam-energy-spread smearing present in the obs spectrum ⇒ ~10-20 MeV
        # obs-vs-analytic mW spread; fold in a Gaussian(σ_BES≈0.12) on √s' if tighter agreement is needed.
        return sp, Rx*dx, False

SN, SW, DIVZ = radiator_nodes()

def build_Ptrue(mW):
    """P_true(m_h,m_l;mW) on the (NT,NT) grid, normalized to ∫∫=1."""
    bwh = bw(tg, mW); bwl = bw(tg, mW)                # (NT,)
    BWout = bwh[:,None]*bwl[None,:]                   # (NT,NT)
    u = np.zeros((NT, NT))
    logZ = log_Z(SN, mW) if DIVZ else None
    for k, (sp, wk) in enumerate(zip(SN, SW)):
        s = sp*sp
        lam = (s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS = np.where(lam > 0, np.sqrt(np.maximum(lam,0))/s, 0.0)        # √λ/s' in dm dm measure
        wgt = wk/np.exp(logZ[k]) if DIVZ else wk
        u += wgt * BWout * PS
    Z = u.sum()*dt*dt
    return u/np.maximum(Z, 1e-300)

# ─────────────────────────────────────────────────────────────────────────────
# Kernel (reco mode): empirical 2-D residual histogram on the SAME dt spacing.
# KBINS>1 ⇒ the kernel is measured in KBINS bins of the TRUE m_qq (a piecewise
# transfer matrix) to capture the non-stationary detector response slope.
# ─────────────────────────────────────────────────────────────────────────────
KBINS = int(os.environ.get("KBINS", "1"))
KQ = np.arange(-34.0, 16.0+dt/2, dt); KL = np.arange(-28.0, 16.0+dt/2, dt)
KQC = 0.5*(KQ[:-1]+KQ[1:]); KLC = 0.5*(KL[:-1]+KL[1:])

def load_kernels():
    """Return (kernels[KBINS] each summing to 1, mqq_edges[KBINS+1]). Raw residual sample
    + its gen m_qq carried in the npz so the kernel can be sliced by true mass."""
    K = np.load(os.path.join(KDIR, f"kernel_ecm{ECM}.npz"))
    dqq, dlv, qg = K["dqq"].astype(float), K["dlv"].astype(float), K["mqq_g"].astype(float)
    edges = np.percentile(qg, np.linspace(0, 100, KBINS+1)); edges[0] -= 1; edges[-1] += 1
    ks = []
    for b in range(KBINS):
        m = (qg >= edges[b]) & (qg < edges[b+1])
        Hk, _, _ = np.histogram2d(dqq[m], dlv[m], bins=[KQ, KL])
        ks.append(Hk/np.maximum(Hk.sum(), 1.0))
    return ks, edges

def build_Preco(mW, kernels, edges):
    """P_reco = Σ_b [P_true restricted to true m_qq∈bin_b] ⊗ kernel_b. Returns (Pr, rq, rl)."""
    Pt = build_Ptrue(mW)
    out = None
    for b, Hk in enumerate(kernels):
        if KBINS == 1:
            blk = Pt
        else:
            rows = (tg >= edges[b]) & (tg < edges[b+1])      # true m_qq axis = rows (TH)
            blk = np.where(rows[:, None], Pt, 0.0)
        c = fftconvolve(blk, Hk, mode="full")
        out = c if out is None else out + c
    rq = tg[0] + KQC[0] + dt*np.arange(out.shape[0])
    rl = tg[0] + KLC[0] + dt*np.arange(out.shape[1])
    out = out/np.maximum(out.sum()*dt*dt, 1e-300)
    return out, rq, rl

# ─────────────────────────────────────────────────────────────────────────────
# Run the scan
# ─────────────────────────────────────────────────────────────────────────────
def scan_perevent():
    L = np.zeros((N, NMW))
    # LOGZ_TAB: tabulate log_Z(m_WW,mW) on a fine m_WW grid per scan mW and interpolate per event
    # (log_Z is smooth in m_WW ⇒ sub-MeV; turns the per-event GL-256 quadrature from N×NMW into
    # LOGZ_TAB_N×NMW evals ⇒ FOLD_W=exact / gen_pe ~as fast as grid). Default OFF (bit-identical path).
    if int(os.environ.get("LOGZ_TAB", "0")):
        wn = np.linspace(mWW_g.min()-0.5, mWW_g.max()+0.5, int(os.environ.get("LOGZ_TAB_N", "400")))
        for j, mW in enumerate(mw_scan):
            lz_ev = np.interp(mWW_g, wn, log_Z(wn, mW))
            L[:, j] = n2ll_perevent(mqq_g, mlv_g, mWW_g, mW, logz=lz_ev)
    else:
        for j, mW in enumerate(mw_scan):
            L[:, j] = n2ll_perevent(mqq_g, mlv_g, mWW_g, mW)
    return L

def scan_grid(x_h, x_l, reco):
    """Per-event -2logP on the scan grid by interpolating the (reco or true) template."""
    L = np.zeros((len(x_h), NMW))
    kernels = edges = None
    if reco: kernels, edges = load_kernels()
    for j, mW in enumerate(mw_scan):
        if reco:
            P, gx, gy = build_Preco(mW, kernels, edges)
        else:
            P = build_Ptrue(mW); gx = gy = tg
        f = RegularGridInterpolator((gx, gy), P, bounds_error=False, fill_value=1e-300)
        pv = f(np.stack([x_h, x_l], axis=1))
        L[:, j] = -2.0*np.log(np.maximum(pv, 1e-300))
    return L

FOLD_TGT  = os.environ.get("FOLD_TGT", "reco")    # reco (default) | gen (identity-smearing machinery test)
FOLD_W    = os.environ.get("FOLD_W", "grid")      # grid (fast, default) | exact per-event log_Z (the PRINCIPLED
# weight: per-event m_WW^gen=√s'). grid is NOT ≡ exact: grid−exact = +11..+70 MeV @163/160/157 (DAY4: the gap is
# the RADIATOR-MARGINALIZATION bias gen_marg−gen_pe; its spread component flips sign across 2mW=160.76 ⇒ threshold-
# driven, NOT a code bug). exact removes it by construction ⇒ adopt FOLD_W=exact + FOLD_BIN≈0.25/FOLD_SM≈0.3 for the
# ABSOLUTE mW; the closure residual is then a calibratable binned-template ESTIMATOR FLOOR (swings ~140 MeV w/ bin,
# targets/gap bin-invariant). For closure-CALIBRATED RELATIVE mW the weight cancels ⇒ grid fine. See HANDOFF_DAY4.md
# + conv_mw_closure_anatomy.py. Speed exact with LOGZ_TAB=1 (~50×, bit-identical); plain exact ≈10min@40k.
FOLD_SM   = float(os.environ.get("FOLD_SM", "0.6"))  # bias-free default; sm≳1 biases the morphing MLE ~−170 MeV
FOLD_K    = int(os.environ.get("FOLD_K", "5"))    # k-fold: ALL N events measured out-of-fold ⇒ full-sample σ
def scan_fold():
    """Forward-folding (full transfer matrix via per-event MC reweighting). The reco template at
    trial mW is the ACTUAL MC reco-mass density reweighted by the gen lineshape ratio
    w_i(mW)=P_true(gen_i;mW)/P_true(gen_i;mW0). Each event carries its REAL gen→reco smearing ⇒ no
    kernel-stationarity assumption, no P_true-vs-gen shape mismatch.
    K-FOLD (FOLD_K folds): for each fold k the template is built from the OTHER folds and evaluated on
    fold k. Every event is thus measured against an out-of-fold template ⇒ no self-use bias AND all N
    events enter the measurement (so σ_mW is the FULL-sample error, not a half-sample one). Closure
    target = the gen-level fit value. FOLD_TGT=gen (identity smearing) must reproduce the gen self-fit
    (machinery unit test). FOLD_W=exact uses the per-event (gen_WW_m) log_Z lineshape weights."""
    from scipy.ndimage import gaussian_filter
    Lgen = scan_perevent() if FOLD_W == "exact" else scan_grid(mqq_g, mlv_g, reco=False)
    j0 = int(np.argmin(Lgen.sum(0)))                          # native gen density = reweighting reference
    th = mqq_g if FOLD_TGT == "gen" else mqq_r                # target ('reco') masses
    tl = mlv_g if FOLD_TGT == "gen" else mlv_r
    DIM = os.environ.get("FOLD_DIM", "2d")                    # 2d (m_qq,m_lν) | qq | lv
    bw_bin = float(os.environ.get("FOLD_BIN", "0.5")); _clip = float(os.environ.get("FOLD_CLIP", "99.9"))
    rng = np.random.default_rng(int(os.environ.get("FOLD_SEED", "2024")))
    fold = rng.integers(0, FOLD_K, N)                         # k-fold assignment
    e1 = np.arange(30.0, 98.0+1e-6, bw_bin); c1 = 0.5*(e1[:-1]+e1[1:])
    L = np.zeros((N, NMW))
    for j in range(NMW):
        wfull = np.exp(-0.5*(Lgen[:, j] - Lgen[:, j0]))
        if _clip < 100: wfull = np.minimum(wfull, np.percentile(wfull, _clip))   # inert in-window; latent for wide scans
        for k in range(FOLD_K):
            tr = fold != k; te = fold == k                    # template from other folds, evaluate this fold
            w = wfull[tr]; idx = np.where(te)[0]
            if DIM == "2d":
                H, _, _ = np.histogram2d(th[tr], tl[tr], bins=[e1, e1], weights=w)
                if FOLD_SM > 0: H = gaussian_filter(H, sigma=FOLD_SM)
                H = H/np.maximum(H.sum(), 1e-300)
                pv = RegularGridInterpolator((c1, c1), H, bounds_error=False, fill_value=1e-300)(np.stack([th[te], tl[te]], 1))
            else:
                obs = th if DIM == "qq" else tl
                H, _ = np.histogram(obs[tr], bins=e1, weights=w)
                if FOLD_SM > 0: H = gaussian_filter(H, sigma=FOLD_SM)
                H = H/np.maximum(H.sum(), 1e-300)
                pv = np.interp(obs[te], c1, H, left=1e-300, right=1e-300)
            L[idx, j] = -2.0*np.log(np.maximum(pv, 1e-300))
    print(f"[fold] TGT={FOLD_TGT} W={FOLD_W} DIM={DIM} K={FOLD_K} bin={bw_bin} sm={FOLD_SM} clip={_clip} "
          f"ref mW0={mw_scan[j0]:.3f}; N={N} (all events measured out-of-fold)")
    return L

SYNTH = int(os.environ.get("SYNTH", "0"))   # reco mode: synthetic reco = gen + INDEPENDENT kernel draw
if SYNTH and MODE == "reco":
    K = np.load(os.path.join(KDIR, f"kernel_ecm{ECM}.npz"))
    dqq, dlv = K["dqq"].astype(float), K["dlv"].astype(float)
    rng0 = np.random.default_rng(7)
    j = rng0.integers(0, len(dqq), N)            # random pairing ⇒ Δ ⟂ gen mass (stationary by construction)
    mqq_r = mqq_g + dqq[j]; mlv_r = mlv_g + dlv[j]
    print(f"[SYNTH] synthetic reco = gen + INDEPENDENT kernel draw (machinery+stationarity unit test)")

if MODE == "gen_pe":
    L = scan_perevent()
elif MODE == "gen_marg":
    L = scan_grid(mqq_g, mlv_g, reco=False)
elif MODE == "reco":
    L = scan_grid(mqq_r, mlv_r, reco=True)
elif MODE == "fold":
    L = scan_fold()
else:
    raise SystemExit(f"unknown MODE {MODE}")
Ntot = L.shape[0]

# PER-EVENT mW from the folded-BW likelihood: m̂W_i = argmax_mW P_folded(masses_i|mW). No nuisance fit
# ⇒ no flat direction; defined for ~100% of events; its distribution is a data/MC-checkable observable.
if os.environ.get("DUMP_PE"):
    Lf = np.where(np.isfinite(L), L, 1e18)
    pe_mw = mw_scan[np.argmin(Lf, axis=1)]                         # per-event MAP mW
    Pn = np.exp(-0.5*(Lf - Lf.min(1, keepdims=True))); Pn /= Pn.sum(1, keepdims=True)
    pe_mean = (Pn*mw_scan[None, :]).sum(1)                         # per-event posterior-mean mW (smoother)
    interior = (np.argmin(Lf, axis=1) > 0) & (np.argmin(Lf, axis=1) < NMW-1)
    np.savez(os.environ["DUMP_PE"], pe_mw=pe_mw, pe_mean=pe_mean, interior=interior,
             mqq=mqq_r if MODE in ("reco","fold") else mqq_g, mlv=mlv_r if MODE in ("reco","fold") else mlv_g,
             mw_scan=mw_scan, ecm=ECM)
    print(f"[per-event mW] MAP: mean={pe_mw[interior].mean():.3f} std={pe_mw[interior].std():.3f} "
          f"median={np.median(pe_mw[interior]):.3f} interior={100*interior.mean():.0f}% | "
          f"posterior-mean: mean={pe_mean.mean():.3f} std={pe_mean.std():.3f}")

# drop events that are off-grid / infinite at all scan points
good = np.all(np.isfinite(L), axis=1) & (np.ptp(L, axis=1) > 0)
Lg = L[good]; Ng = Lg.shape[0]
ens = Lg.sum(0); ens -= ens.min()

def parab_min(xs, ys):
    i = int(np.clip(np.argmin(ys), 2, len(ys)-3))
    c = np.polyfit(xs[i-2:i+3], ys[i-2:i+3], 2)
    xm = -c[1]/(2*c[0]); sig = 1.0/np.sqrt(c[0]) if c[0] > 0 else np.nan
    return xm, sig, c[0]
mhat, sig_ens, curv = parab_min(mw_scan, ens)
print(f"\n[fit] usable events {Ng}/{Ntot} ({100*Ng/Ntot:.1f}%)")
print(f"  ENSEMBLE  m̂W = {mhat:.4f} GeV   σ_mW(curv) = {sig_ens:.4f}   bias vs {MW_REF} = {mhat-MW_REF:+.4f}")

# bootstrap statistical σ + pull (resample events, refit min)
rng = np.random.default_rng(12345)
boots = np.empty(NBOOT)
for b in range(NBOOT):
    sel = rng.integers(0, Ng, Ng)
    eb = Lg[sel].sum(0); eb -= eb.min()
    boots[b], _, _ = parab_min(mw_scan, eb)
print(f"  BOOTSTRAP m̂W = {boots.mean():.4f} ± {boots.std():.4f} (stat, N={Ng})   pull(curv vs boot σ) = {(boots.std()/sig_ens):.2f}")
if os.environ.get("DUMP"):
    np.savez(os.environ["DUMP"], mw_scan=mw_scan, ens=ens, boots=boots, mhat=mhat, sig=sig_ens, ecm=ECM)

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
ax[0].plot(mw_scan, ens, "o-", ms=3, color="C0")
ax[0].axvline(mhat, color="C0", ls="-", lw=1, label=f"m̂W={mhat:.3f}")
ax[0].axvline(MW_REF, color="k", ls="--", lw=1, label=f"ref {MW_REF}")
ax[0].set_xlabel("trial mW [GeV]"); ax[0].set_ylabel("Δ(−2 ln L) ensemble")
ax[0].set_ylim(0, np.percentile(ens, 95)+5); ax[0].set_title(f"{MODE}/{RAD} ecm{ECM}: ensemble L(mW)")
ax[0].legend()
ax[1].hist(boots, bins=40, color="C0", alpha=.7)
ax[1].axvline(MW_REF, color="k", ls="--", lw=1)
ax[1].set_xlabel(r"bootstrap $\hat m_W$ [GeV]"); ax[1].set_title(f"bootstrap (σ={boots.std():.3f})")
plt.tight_layout(); png = f"{OUT}/fit_{MODE}_{RAD}_ecm{ECM}.png"; plt.savefig(png, dpi=110)
print(f"  [plot] {png}")

# reco/fold: compare to the per-event soft fit in the tree
if MODE in ("reco", "fold"):
    vs = a["kinfit_valid"].astype(bool); cm = a["kinfit_mW"]
    v = vs & np.isfinite(cm)
    print(f"\n[compare] PER-EVENT SOFT fit (tree, valid={100*vs.mean():.1f}%): "
          f"median={np.median(cm[v]):.3f} mean={cm[v].mean():.3f} std={cm[v].std():.3f} "
          f"σ_mW(ens, valid only)≈{cm[v].std()/np.sqrt(v.sum()):.4f}")
    print(f"[compare] {'FORWARD-FOLD' if MODE=='fold' else 'CONVOLUTION'} fit (this, usable={100*Ng/Ntot:.1f}%): "
          f"m̂W={mhat:.3f} σ_mW={sig_ens:.4f}")
