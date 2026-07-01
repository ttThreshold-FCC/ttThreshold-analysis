#!/usr/bin/env python3
# ── WW → 4q (fully hadronic) forward-fold mW estimator — STAGE 1 (HANDOFF_DAY10) ──────────────────
# The μνqq conv_mw forward-fold, adapted to TWO HADRONIC W's. The observable is the 2-D pair of dijet
# masses (m_W1, m_W2); the lineshape P_true(m1,m2;mW) = BW(m1)·BW(m2)·PS(λ)/Z is SYMMETRIC in the two
# W's, so log_Z / n2ll_perevent / scan_perevent / fold_fit are reused VERBATIM from
# conv_mw_closure_anatomy.py.  The only 4q-specific work:
#   (a) reco dijet masses from the 4 raw (massless) reco jets, for a chosen jet→W PAIRING;
#   (b) the pairing choice itself (3 partitions of 4 jets into 2 pairs).
#
# PAIRING modes (env PAIRING):
#   true   — use gen_pairing_true (truth; isolates the fold machinery from pairing confusion)
#   bw     — no-fit pole-referenced Breit-Wigner pick over the 3 partitions (data-applicable, ~82%)
#   kinfit — use the kinfit4q_pairing branch (the in-tree fit's choice, ~69%)
#   all    — run all three and print a comparison table (DEFAULT)
# Pairing-index convention (matches WWFunctions::pairing_index_from_groups / gen_pairing_true):
#   0:(j1 j2)(j3 j4)   1:(j1 j3)(j2 j4)   2:(j1 j4)(j2 j3)
#
# Asimov closure logic (inherited): template & pseudo-data share the gen↔reco response, so radiation /
# jet smearing CANCEL; the residual closure (reco_fit − gen_pe) = binning floor + (for p8-on-p8) ~0
# lineshape bias.  WRONG pairing decouples the per-event gen-mass weight from the reco observable ⇒ it
# is the one genuine closure-breaker.  This script's deliverable: the gen-true-pairing closure CLOSES
# (machinery validation), and the bw/kinfit closure quantifies the pairing-confusion systematic.
#
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# PAIRING=all python3 jax_prototype/conv_mw_4q.py [ecm]
import sys, os, json
# ── force single-thread numpy/BLAS so each worker process owns exactly one core (set BEFORE numpy) ──
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import multiprocessing as mp
NWORKERS = int(os.environ.get("NWORKERS", "48"))   # use up to 48 of the 64 cores
_FORK = mp.get_context("fork")
def _pmap(worker, chunks):
    """fork-based parallel map; children inherit current module globals (no big-array pickling)."""
    nw = max(1, min(NWORKERS, len(chunks)))
    if nw == 1:
        return [worker(c) for c in chunks]
    with _FORK.Pool(nw) as pool:
        return pool.map(worker, chunks)
np.set_printoptions(linewidth=160, suppress=True)

ECM    = int(sys.argv[1]) if len(sys.argv) > 1 else 160
GW     = float(os.environ.get("GW", "2.085"))     # p8 default W width
MW_REF = float(os.environ.get("MW_REF", "80.4"))  # p8 default-ish W pole
BW_RUN = int(os.environ.get("BW_RUN", "0"))       # 0=fixed-width BW (default); 1=running-width
DECAY_P = float(os.environ.get("DECAY_P", "3"))   # W→qq decay phase-space numerator m^p (DAY8 const×m³ baseline)
MAXN   = int(os.environ.get("MAXN", "0"))
NMW    = int(os.environ.get("NMW", "121"))
MWLO, MWHI = float(os.environ.get("MWLO","79.0")), float(os.environ.get("MWHI","81.5"))
LOGZ_N = int(os.environ.get("LOGZ_N", "256"))
LOGZ_TAB_N = int(os.environ.get("LOGZ_TAB_N", "400"))
FOLD_K = int(os.environ.get("FOLD_K", "5"))
FOLD_BIN = float(os.environ.get("FOLD_BIN", "0.5"))
FOLD_SM  = float(os.environ.get("FOLD_SM", "0.6"))
FOLD_CLIP = float(os.environ.get("FOLD_CLIP", "100.0"))
EST       = os.environ.get("EST", "hist")
KDE_H     = float(os.environ.get("KDE_H", "0.6"))
KDE_CH    = int(os.environ.get("KDE_CH", "2000"))
FINE_J0   = int(os.environ.get("FINE_J0", "1"))
PAIRING   = os.environ.get("PAIRING", "all")      # true | bw | kinfit | all
CLEAN     = int(os.environ.get("CLEAN", "0"))     # 1 ⇒ restrict to events whose CHOSEN pairing == gen_pairing_true
ROOT = os.environ.get("ROOT",
        f"outputs/treemaker/4q/step2/had_scap_ctrl_s1/p8_ee_WW_ecm{ECM}.root")
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)
import uproot

# ══ BW + log_Z + per-event n2ll — copied VERBATIM from conv_mw_closure_anatomy.py (symmetric) ══════
_GX, _GWl = np.polynomial.legendre.leggauss(LOGZ_N); _W2 = _GWl[:, None]*_GWl[None, :]
def log_Z(m_WW, mW, gW=GW):
    m_WW = np.atleast_1d(np.asarray(m_WW, float)); mwgw = mW*gW; mW2 = mW*mW
    out = np.empty(m_WW.shape, float)
    CH = max(1, 50_000_000 // (LOGZ_N*LOGZ_N))
    for a in range(0, len(m_WW), CH):
        s = (m_WW[a:a+CH])**2
        t_min = np.arctan(-mW2/mwgw); t_max = np.arctan((s-mW2)/mwgw)
        half_d = 0.5*(t_max-t_min); half_s = 0.5*(t_max+t_min)
        t = half_d[:, None]*_GX[None, :] + half_s[:, None]
        m = np.sqrt(np.maximum(mW2 + mwgw*np.tan(t), 1e-12)); inv = 1.0/m
        if BW_RUN:
            dd = m*m - mW2; wm = gW*m*m/mW; rf = (m*m/mW2)*(dd*dd + mwgw*mwgw)/(dd*dd + wm*wm)
        else:
            rf = np.ones_like(m)
        if DECAY_P:
            rf = rf * m**DECAY_P
        mh = m[:, :, None]; ml = m[:, None, :]; ih = inv[:, :, None]; il = inv[:, None, :]; sE = s[:, None, None]
        rh = rf[:, :, None]; rl = rf[:, None, :]
        lam = (sE-(mh+ml)**2)*(sE-(mh-ml)**2)
        integ = np.where(lam > 0, np.sqrt(np.maximum(lam, 0))*ih*il*rh*rl/(4.0*sE), 0.0)
        Z = np.sum(_W2[None]*integ, axis=(1, 2)) * half_d*half_d
        out[a:a+CH] = np.log(np.maximum(Z, 1e-300))
    return out

def bw_fixed(m, mW, gW=GW):
    mwgw = mW*gW; d = m*m - mW*mW
    return mwgw/(d*d + mwgw*mwgw)
def bw_run(m, mW, gW=GW):
    d = m*m - mW*mW; wm = gW*m*m/mW
    return wm/(d*d + wm*wm)
def bw(m, mW, gW=GW):
    return bw_run(m, mW, gW) if BW_RUN else bw_fixed(m, mW, gW)

def n2ll_perevent(mh, ml, mWW, mW, logz):
    s = mWW*mWW
    lam = (s-(mh+ml)**2)*(s-(mh-ml)**2)
    bad = lam <= 0; lam = np.where(bad, 1.0, lam)
    tv = (-2.0*np.log(np.maximum(bw(mh,mW),1e-300)) - 2.0*np.log(np.maximum(bw(ml,mW),1e-300))
          - np.log(lam) + 2.0*np.log(s) + 2.0*logz)
    if DECAY_P:
        tv = tv - 2.0*DECAY_P*(np.log(np.maximum(mh,1e-9)) + np.log(np.maximum(ml,1e-9)))
    return np.where(bad, 1e6, tv)

def parab_min(xs, ys):
    i = int(np.clip(np.argmin(ys), 2, len(ys)-3))
    c = np.polyfit(xs[i-2:i+3], ys[i-2:i+3], 2)
    xm = -c[1]/(2*c[0]); sig = 1.0/np.sqrt(c[0]) if c[0] > 0 else np.nan
    return xm, sig

# ══ load the 4q tree + build reco dijet masses for the 3 jet→W partitions ═════════════════════════
t = uproot.open(ROOT)["events"]
_avail = set(t.keys())
HAVE_KINFIT = ("kinfit4q_pairing" in _avail) and ("kinfit4q_valid" in _avail)  # FF/genqk trees have no kinfit
HAVE_GENQW  = all(f"gen_qW{i}_px" in _avail for i in range(4))                  # only the genqk trees carry these
need = ["gen_W1_m","gen_W2_m","gen_WW_m","gen_pairing_true"]
if HAVE_KINFIT: need += ["kinfit4q_pairing","kinfit4q_valid"]
if HAVE_GENQW:  need += [f"gen_qW{i}_{c}" for i in range(4) for c in ("px","py","pz","e")]
for i in (1,2,3,4):
    need += [f"reco_jet{i}_p", f"reco_jet{i}_theta", f"reco_jet{i}_phi"]
a = t.arrays(need, library="np")
Nraw = len(a["gen_W1_m"])

def jet_vec(i):                                  # massless reco-jet 4-vector (px,py,pz,E)
    p = a[f"reco_jet{i}_p"]; th = a[f"reco_jet{i}_theta"]; ph = a[f"reco_jet{i}_phi"]
    st = np.sin(th)
    return np.stack([p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th), p], 1)
J = {i: jet_vec(i) for i in (1,2,3,4)}
def dimass(u, v):
    s = u + v; m2 = s[:,3]**2 - s[:,0]**2 - s[:,1]**2 - s[:,2]**2
    return np.sqrt(np.maximum(m2, 0.0))
PORDER = [((1,2),(3,4)), ((1,3),(2,4)), ((1,4),(2,3))]   # pairing 0,1,2
mA = np.stack([dimass(J[p[0][0]], J[p[0][1]]) for p in PORDER], 1)   # (Nraw,3)
mB = np.stack([dimass(J[p[1][0]], J[p[1][1]]) for p in PORDER], 1)   # (Nraw,3)

gpt = a["gen_pairing_true"].astype(int); kfp = a["kinfit4q_pairing"].astype(int) if HAVE_KINFIT else None
# ── (level 1) no-fit pole-referenced BARE BW pick (the original): −logBW(mA)−logBW(mB), fixed-width ──
def _negbw(m):
    d = m*m - MW_REF*MW_REF; mwgw = MW_REF*GW
    return -np.log(np.maximum(mwgw/(d*d + mwgw*mwgw), 1e-300))
bwp = np.argmin(_negbw(mA) + _negbw(mB), 1).astype(int)

# reco 4-jet invariant mass = the per-event ISR/BES-reduced √s' proxy (PAIRING-INVARIANT: same for all 3
# partitions, so it only enters the discriminant through the √λ(√s', mA, mB) phase-space term).
_Jsum = J[1] + J[2] + J[3] + J[4]
mWW_reco = np.sqrt(np.maximum(_Jsum[:,3]**2 - _Jsum[:,0]**2 - _Jsum[:,1]**2 - _Jsum[:,2]**2, 0.0))

# ── (level 2) "forward-folded, ISR-convoluted BW" pick = the estimator's OWN per-event likelihood used
#    as the pairing discriminant.  Uses the SAME lineshape as the fold: bw() (const×m³ via DECAY_P) and,
#    with PAIR_PS=1, the phase-space √λ evaluated at the per-event ISR-reduced √s' (= the ISR convolution).
#    Pairing-invariant n2ll terms (2 log s + 2 log_Z) cancel in the argmin and are dropped.
g1 = a["gen_W1_m"]; g2 = a["gen_W2_m"]; mWW_g = a["gen_WW_m"]
PAIR_PS    = int(os.environ.get("PAIR_PS", "1"))      # 1=include √λ(√s') phase-space term (ISR-convolution)
PAIR_DECAY = int(os.environ.get("PAIR_DECAY", "1"))   # 1=include the m^p decay factor (lineshape-consistent)
PAIR_SQRTS = os.environ.get("PAIR_SQRTS", "reco4j")   # gen | reco4j | ecm — √s' for the √λ term
def pll_pick(ps=None, decay=None, sqrts=None):
    ps = PAIR_PS if ps is None else ps; decay = PAIR_DECAY if decay is None else decay
    sqrts = PAIR_SQRTS if sqrts is None else sqrts
    sp = mWW_g if sqrts == "gen" else (np.full(Nraw, float(ECM)) if sqrts == "ecm" else mWW_reco)
    s = (sp*sp)[:, None]                                                   # (Nraw,1)
    t = -2.0*np.log(np.maximum(bw(mA, MW_REF), 1e-300)) - 2.0*np.log(np.maximum(bw(mB, MW_REF), 1e-300))
    if decay and DECAY_P:
        t = t - 2.0*DECAY_P*(np.log(np.maximum(mA, 1e-9)) + np.log(np.maximum(mB, 1e-9)))
    if ps:
        lam = (s - (mA+mB)**2)*(s - (mA-mB)**2)
        t = np.where(lam > 0, t - np.log(np.maximum(lam, 1e-300)), 1e6)
    return np.argmin(t, 1).astype(int)

# sorted dijet masses for ALL 3 partitions (Nraw,3) — used by foldpick + the marginalisation modes
hi3 = np.maximum(mA, mB); lo3 = np.minimum(mA, mB)
# pairing prior w_p for MARGINALISATION (Σ_p w_p = 1 per event): flat = 1/3, soft = softmax of the pllnops
# lineshape score (const×m³ BW, NO √λ — the discriminant we adopted). Computed once at MW_REF.
def pair_weights(kind):
    if kind == "flat":
        return np.full((Nraw, 3), 1.0/3.0)
    sc = -2.0*np.log(np.maximum(bw(mA, MW_REF), 1e-300)) - 2.0*np.log(np.maximum(bw(mB, MW_REF), 1e-300))
    if DECAY_P:
        sc = sc - 2.0*DECAY_P*(np.log(np.maximum(mA, 1e-9)) + np.log(np.maximum(mB, 1e-9)))
    T = float(os.environ.get("MARG_TEMP", "1.0"))                    # softmax temperature: →0 = hard pick, ∞ = flat
    w = np.exp(-0.5*(sc - sc.min(1, keepdims=True))/max(T, 1e-6))    # stable softmax of −½·score/T
    return w/np.maximum(w.sum(1, keepdims=True), 1e-300)

# gen observable (sorted, symmetric); reco observable depends on the chosen pairing
gen_hi = np.maximum(g1, g2); gen_lo = np.minimum(g1, g2)

ok_base = (np.isfinite(g1) & np.isfinite(g2) & np.isfinite(mWW_g) &
           (gpt >= 0) & (gpt <= 2) & np.all(np.isfinite(mA), 1) & np.all(np.isfinite(mB), 1) &
           ((g1 + g2) < mWW_g))

def reco_obs(pair_idx):
    rows = np.arange(Nraw)
    rA = mA[rows, pair_idx]; rB = mB[rows, pair_idx]
    return np.maximum(rA, rB), np.minimum(rA, rB)

mw_scan = np.linspace(MWLO, MWHI, NMW)

# ══ per-event gen_pe −2lnL (exact per-event √s', tabulated log_Z) — PARALLEL over the mW grid ═══════
#    The log_Z tabulation dominates; it is embarrassingly parallel across mW values.  Workers inherit
#    the (forked) _S_* arrays + mw_scan ⇒ no big-array pickling, one core each.
_S_mh = _S_ml = _S_sq = _S_wn = None
def _scan_worker(jidx):
    out = np.empty((len(_S_mh), len(jidx)))
    for c, j in enumerate(jidx):
        mW = mw_scan[j]
        lz = np.interp(_S_sq, _S_wn, log_Z(_S_wn, mW))
        out[:, c] = n2ll_perevent(_S_mh, _S_ml, _S_sq, mW, lz)
    return jidx, out
def scan_perevent(mh, ml, sq, idx):
    global _S_mh, _S_ml, _S_sq, _S_wn
    _S_sq = sq[idx]; _S_mh = mh[idx]; _S_ml = ml[idx]
    _S_wn = np.linspace(float(np.min(_S_sq))-0.5, float(np.max(_S_sq))+0.5, LOGZ_TAB_N)
    n = len(idx); L = np.empty((n, NMW))
    chunks = [c for c in np.array_split(np.arange(NMW), min(NWORKERS, NMW)) if len(c)]
    for jidx, out in _pmap(_scan_worker, chunks):
        L[:, jidx] = out
    return L

def fit_sub(L, mask=None):
    e = L[mask].sum(0) if mask is not None else L.sum(0); e = e - e.min()
    return parab_min(mw_scan, e)

# ══ forward fold (k-fold), ported VERBATIM from conv_mw_closure_anatomy.py ═════════════════════════
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
_F_Lgen = _F_Lref = _F_fold = _F_th = _F_tl = _F_e1 = _F_c1 = None
def _fold_hist_worker(jidx):                      # one mW-chunk → its L columns (parallel over the grid)
    out = np.zeros((len(_F_th), len(jidx)))
    for c, j in enumerate(jidx):
        wfull = np.exp(-0.5*(_F_Lgen[:, j] - _F_Lref))
        if FOLD_CLIP < 100.0: wfull = np.minimum(wfull, np.percentile(wfull, FOLD_CLIP))
        col = np.zeros(len(_F_th))
        for k in range(FOLD_K):
            tr = _F_fold != k; te = _F_fold == k; w = wfull[tr]; ii = np.where(te)[0]
            H, _, _ = np.histogram2d(_F_th[tr], _F_tl[tr], bins=[_F_e1, _F_e1], weights=w)
            if FOLD_SM > 0: H = gaussian_filter(H, sigma=FOLD_SM)
            H = H/np.maximum(H.sum(), 1e-300)
            pv = RegularGridInterpolator((_F_c1, _F_c1), H, bounds_error=False, fill_value=1e-300)(np.stack([_F_th[te], _F_tl[te]], 1))
            col[ii] = -2.0*np.log(np.maximum(pv, 1e-300))
        out[:, c] = col
    return jidx, out
def _fold_kde_worker(k):                           # one fold → its test-event L rows (parallel over folds)
    pts = np.stack([_F_th, _F_tl], 1); inv2h2 = 1.0/(2.0*KDE_H*KDE_H)
    tr = np.where(_F_fold != k)[0]; te = np.where(_F_fold == k)[0]; Xtr = pts[tr]
    W = np.exp(-0.5*(_F_Lgen[tr] - _F_Lref[tr][:, None]))
    if FOLD_CLIP < 100.0: W = np.minimum(W, np.percentile(W, FOLD_CLIP, axis=0)[None, :])
    norm = W.sum(0); res = np.empty((len(te), NMW))
    for a in range(0, len(te), KDE_CH):
        Xte = pts[te[a:a+KDE_CH]]
        d2 = (Xte[:, 0, None]-Xtr[None, :, 0])**2 + (Xte[:, 1, None]-Xtr[None, :, 1])**2
        res[a:a+KDE_CH] = -2.0*np.log(np.maximum((np.exp(-d2*inv2h2) @ W)/np.maximum(norm[None, :], 1e-300), 1e-300))
    return te, res
def fold_fit(Lgen, th, tl):
    global _F_Lgen, _F_Lref, _F_fold, _F_th, _F_tl, _F_e1, _F_c1
    n = Lgen.shape[0]
    if FINE_J0:
        eg = Lgen.sum(0); eg = eg - eg.min(); mwref = parab_min(mw_scan, eg)[0]
        k0 = int(np.clip(np.searchsorted(mw_scan, mwref)-1, 0, NMW-2))
        fr = (mwref-mw_scan[k0])/(mw_scan[k0+1]-mw_scan[k0]); Lref = Lgen[:, k0]*(1-fr)+Lgen[:, k0+1]*fr
    else:
        Lref = Lgen[:, int(np.argmin(Lgen.sum(0)))]
    rng = np.random.default_rng(int(os.environ.get("FOLD_SEED", "2024")))
    _F_Lgen, _F_Lref, _F_fold, _F_th, _F_tl = Lgen, Lref, rng.integers(0, FOLD_K, n), th, tl
    L = np.zeros((n, NMW))
    if EST == "kde":
        for te, res in _pmap(_fold_kde_worker, list(range(FOLD_K))):
            L[te] = res
    else:
        _F_e1 = np.arange(30.0, 98.0+1e-6, FOLD_BIN); _F_c1 = 0.5*(_F_e1[:-1]+_F_e1[1:])
        chunks = [c for c in np.array_split(np.arange(NMW), min(NWORKERS, NMW)) if len(c)]
        for jidx, out in _pmap(_fold_hist_worker, chunks):
            L[:, jidx] = out
    good = np.all(np.isfinite(L), axis=1) & (np.ptp(L, axis=1) > 0)
    return parab_min(mw_scan, (L[good].sum(0) - L[good].sum(0).min()))

# ══ MARGINALISED fold: L_i(mW) = −2 ln Σ_p w_p · P(masses^p_i|mW).  The template P is the CONDITIONAL-on-
#    correct-pairing reco density (built from a CLEAN pairing tmpl_hi/tmpl_lo — true for closure, pllnops for
#    data) — NOT the marginal mix.  Wrong partitions then auto-down-weight because their masses fall in P's
#    low-density tails.  (Building P from the all-partition mix injects the combinatorial background and biases
#    the fit catastrophically low — verified.)  Σ_p w_p=1.  hist EST only. ════════════════════════════════════
_M_Lgen = _M_Lref = _M_fold = _M_thi = _M_tlo = _M_hi3 = _M_lo3 = _M_w = _M_e1 = _M_c1 = None
def _marg_hist_worker(jidx):
    out = np.zeros((_M_hi3.shape[0], len(jidx)))
    for c, j in enumerate(jidx):
        wfull = np.exp(-0.5*(_M_Lgen[:, j] - _M_Lref))
        if FOLD_CLIP < 100.0: wfull = np.minimum(wfull, np.percentile(wfull, FOLD_CLIP))
        col = np.zeros(_M_hi3.shape[0])
        for k in range(FOLD_K):
            tr = _M_fold != k; te = _M_fold == k; ii = np.where(te)[0]
            H, _, _ = np.histogram2d(_M_thi[tr], _M_tlo[tr], bins=[_M_e1, _M_e1], weights=wfull[tr])
            if FOLD_SM > 0: H = gaussian_filter(H, sigma=FOLD_SM)
            H = H/np.maximum(H.sum(), 1e-300)
            interp = RegularGridInterpolator((_M_c1, _M_c1), H, bounds_error=False, fill_value=1e-300)
            dens = np.zeros(len(ii))
            for p in range(3):
                dens += _M_w[te, p]*interp(np.stack([_M_hi3[te, p], _M_lo3[te, p]], 1))
            col[ii] = -2.0*np.log(np.maximum(dens, 1e-300))
        out[:, c] = col
    return jidx, out
def fold_fit_marg(Lgen, tmpl_hi, tmpl_lo, hi3a, lo3a, w3a):
    global _M_Lgen, _M_Lref, _M_fold, _M_thi, _M_tlo, _M_hi3, _M_lo3, _M_w, _M_e1, _M_c1
    n = Lgen.shape[0]
    if FINE_J0:
        eg = Lgen.sum(0); eg = eg - eg.min(); mwref = parab_min(mw_scan, eg)[0]
        k0 = int(np.clip(np.searchsorted(mw_scan, mwref)-1, 0, NMW-2))
        fr = (mwref-mw_scan[k0])/(mw_scan[k0+1]-mw_scan[k0]); Lref = Lgen[:, k0]*(1-fr)+Lgen[:, k0+1]*fr
    else:
        Lref = Lgen[:, int(np.argmin(Lgen.sum(0)))]
    rng = np.random.default_rng(int(os.environ.get("FOLD_SEED", "2024")))
    _M_Lgen, _M_Lref, _M_fold = Lgen, Lref, rng.integers(0, FOLD_K, n)
    _M_thi, _M_tlo = tmpl_hi, tmpl_lo
    _M_hi3, _M_lo3, _M_w = hi3a, lo3a, w3a
    _M_e1 = np.arange(30.0, 98.0+1e-6, FOLD_BIN); _M_c1 = 0.5*(_M_e1[:-1]+_M_e1[1:])
    L = np.zeros((n, NMW))
    chunks = [c for c in np.array_split(np.arange(NMW), min(NWORKERS, NMW)) if len(c)]
    for jidx, out in _pmap(_marg_hist_worker, chunks):
        L[:, jidx] = out
    good = np.all(np.isfinite(L), axis=1) & (np.ptp(L, axis=1) > 0)
    return parab_min(mw_scan, (L[good].sum(0) - L[good].sum(0).min()))

# ══ per-event gen_pe likelihood is PAIRING-INDEPENDENT (gen masses only) ⇒ compute ONCE ════════════
idx_base = np.where(ok_base)[0]
if MAXN > 0: idx_base = idx_base[:MAXN]
L_pe_base = scan_perevent(gen_hi, gen_lo, mWW_g, idx_base)     # (Nbase, NMW)  — shared by all modes
gpt_b = gpt[idx_base]; gen_hi_b = gen_hi[idx_base]; gen_lo_b = gen_lo[idx_base]

# ══ run one pairing mode (reuses L_pe_base; CLEAN row-masks to the correct-pairing subset) ═════════
def run_mode(name, pair_idx):
    pid_b = pair_idx[idx_base]
    sub = (pid_b == gpt_b) if CLEAN else np.ones(len(idx_base), bool)
    n = int(sub.sum())
    eff = float((pid_b == gpt_b).mean())
    reco_hi, reco_lo = reco_obs(pair_idx)
    L_pe = L_pe_base[sub]
    genpe, s_pe = fit_sub(L_pe)
    # folds: identity (gen observable → floor) and reco (data-applicable)
    g_hi_i, g_lo_i = gen_hi_b[sub], gen_lo_b[sub]
    r_hi_i, r_lo_i = reco_hi[idx_base][sub], reco_lo[idx_base][sub]
    fit_id, _      = fold_fit(L_pe, g_hi_i, g_lo_i)
    fit_rec, sig_r = fold_fit(L_pe, r_hi_i, r_lo_i)
    d = dict(name=name, N=n, eff=eff, gen_pe=genpe, sig_pe=s_pe, identity=fit_id, reco=fit_rec, sig_rec=sig_r,
             closure=fit_rec-genpe, floor=fit_id-genpe, smear=fit_rec-fit_id)
    print(f"  {name:8s} N={n:6d} eff={100*eff:5.1f}%  gen_pe={genpe:.4f}  reco={fit_rec:.4f}  |  "
          f"closure={1000*d['closure']:+6.1f}  floor={1000*d['floor']:+6.1f}  smear={1000*d['smear']:+6.1f}  "
          f"σ_mW={1000*sig_r:5.1f} MeV")
    return d

# ══ run a MARGINALISATION mode (sum over the 3 partitions; no hard pick) — reports σ_mW for the bias/variance trade
#    MARG_TMPL = which CLEAN pairing builds the conditional template: true (MC-truth, closure-ideal) | pllnops | bw
MARG_TMPL = os.environ.get("MARG_TMPL", "pllnops")
_tmpl_pid = {"true": gpt, "pllnops": pll_pick(ps=0, decay=1), "bw": bwp}[MARG_TMPL]
_tmpl_hi, _tmpl_lo = reco_obs(_tmpl_pid)
def run_marg(name, w3):
    L_pe = L_pe_base
    genpe, s_pe = fit_sub(L_pe)
    fit_id, _      = fold_fit(L_pe, gen_hi_b, gen_lo_b)                 # gen has no pairing ambiguity ⇒ shared floor
    fit_rec, sig_r = fold_fit_marg(L_pe, _tmpl_hi[idx_base], _tmpl_lo[idx_base],
                                   hi3[idx_base], lo3[idx_base], w3[idx_base])
    # effective pairing efficiency of the soft weights = mean weight placed on the TRUE partition
    eff = float(w3[idx_base][np.arange(len(idx_base)), gpt_b].mean())
    d = dict(name=name, N=len(idx_base), eff=eff, gen_pe=genpe, sig_pe=s_pe, identity=fit_id, reco=fit_rec,
             sig_rec=sig_r, closure=fit_rec-genpe, floor=fit_id-genpe, smear=fit_rec-fit_id)
    print(f"  {name:8s} N={len(idx_base):6d} w_true={100*eff:4.1f}%  gen_pe={genpe:.4f}  reco={fit_rec:.4f}  |  "
          f"closure={1000*d['closure']:+6.1f}  floor={1000*d['floor']:+6.1f}  smear={1000*d['smear']:+6.1f}  "
          f"σ_mW={1000*sig_r:5.1f} MeV")
    return d

# ── (level 3) FORWARD-FOLD TEMPLATE pick = the literal "forward-folded" discriminant.  Build the reco-mass
#    density P_fold(m_hi,m_lo|mW_ref) (= smoothed 2-D histogram of a seed pairing's reco masses on idx_base;
#    at the reference column the reweighting weights ≈ 1) ⇒ this FOLDS IN the detector resolution.  Score
#    each of the 3 partitions by that density and pick the max.  Resolution-aware, unlike the analytic picks.
def foldpick(seed):
    e1 = np.arange(30.0, 98.0+1e-6, FOLD_BIN); c1 = 0.5*(e1[:-1]+e1[1:])
    sh, sl = reco_obs(seed)
    H, _, _ = np.histogram2d(sh[idx_base], sl[idx_base], bins=[e1, e1])
    if FOLD_SM > 0: H = gaussian_filter(H, sigma=FOLD_SM)
    H = np.maximum(H/np.maximum(H.sum(), 1e-300), 1e-300)
    interp = RegularGridInterpolator((c1, c1), H, bounds_error=False, fill_value=1e-300)
    # hi3/lo3 are the module-level sorted dijet masses (defined at top); no need to recompute
    dens = np.stack([interp(np.stack([hi3[:, p], lo3[:, p]], 1)) for p in range(3)], 1)
    return np.argmax(dens, 1).astype(int)

# ── (level 4) TEMPLATE LIKELIHOOD-RATIO pick (DAY12 NEXT-1) — the analytic, non-ML beat-pllnops discriminant.
#   From MC truth (gen_pairing_true) build P_true(m_hi,m_lo) (density of the TRUE partition's reco masses) and
#   P_wrong(m_hi,m_lo) (density of the two WRONG partitions'); pick argmax_p [log P_true − log P_wrong].
#   Train/test-HONEST via K folds (templates from the complementary folds).  Shape-only (self-normalised).
#   ★ The mass-template RANGE MUST cover the wrong-pairing high-mass tail (>98 GeV for 74%/94% of wrong pairs
#   at 240/365) — that tail is THE discriminator; clipping it collapses the LR BELOW pllnops.  Recovers the
#   GBM's +10pp at 240/365 analytically (eff 82.6/90.1% vs pllnops 72.7/80.8%).  use_angle adds the within-W
#   reco-jet opening angle (4-D template) — a small threshold-only refinement (+3pp @160).
LR_K     = int(os.environ.get("LR_K", "4"))
LR_BIN   = float(os.environ.get("LR_BIN", "2.0"))
LR_SM    = float(os.environ.get("LR_SM", "1.0"))
LR_M_LO  = float(os.environ.get("LR_M_LO", "20.0"))
LR_M_HI  = float(os.environ.get("LR_M_HI", "340.0"))
LR_A_BIN = float(os.environ.get("LR_A_BIN", "6.0"))
def _reco_ang3():                                  # within-W reco-jet opening angle (Nraw,3): hi/lo per partition
    def ang(u, v):
        c = (u[:, :3]*v[:, :3]).sum(1)/(np.linalg.norm(u[:, :3], axis=1)*np.linalg.norm(v[:, :3], axis=1) + 1e-9)
        return np.degrees(np.arccos(np.clip(c, -1, 1)))
    tA = np.stack([ang(J[p[0][0]], J[p[0][1]]) for p in PORDER], 1)
    tB = np.stack([ang(J[p[1][0]], J[p[1][1]]) for p in PORDER], 1)
    return np.maximum(tA, tB), np.minimum(tA, tB)
def tmpl_lr_pick(use_angle=False):
    me = np.arange(LR_M_LO, LR_M_HI+1e-6, LR_BIN); ae = np.arange(0.0, 180.0+1e-6, LR_A_BIN)
    edges = [me, me] + ([ae, ae] if use_angle else [])
    thi3, tlo3 = _reco_ang3() if use_angle else (None, None)
    def cols(rows, p):
        f = [hi3[rows, p], lo3[rows, p]]
        if use_angle: f += [thi3[rows, p], tlo3[rows, p]]
        return np.stack(f, 1)
    def cols_true(rows):
        f = [hi3[rows, gpt[rows]], lo3[rows, gpt[rows]]]
        if use_angle: f += [thi3[rows, gpt[rows]], tlo3[rows, gpt[rows]]]
        return np.stack(f, 1)
    def cols_wrong(rows):
        wm = np.ones((len(rows), 3), bool); wm[np.arange(len(rows)), gpt[rows]] = False
        f = [hi3[rows][wm], lo3[rows][wm]]
        if use_angle: f += [thi3[rows][wm], tlo3[rows][wm]]
        return np.stack(f, 1)
    def dens(X):
        H, _ = np.histogramdd(X, bins=edges)
        if LR_SM > 0: H = gaussian_filter(H, sigma=LR_SM)
        H = np.maximum(H/np.maximum(H.sum(), 1e-300), 1e-300)
        return RegularGridInterpolator(tuple(0.5*(e[:-1]+e[1:]) for e in edges),
                                       H, bounds_error=False, fill_value=1e-300)
    rng = np.random.default_rng(int(os.environ.get("LR_SEED", "7")))
    valid = (gpt >= 0) & (gpt <= 2); fold = rng.integers(0, LR_K, Nraw)
    logLR = np.full((Nraw, 3), -np.inf)
    for k in range(LR_K):
        tri = np.where(valid & (fold != k))[0]; tei = np.where(fold == k)[0]
        Pt = dens(cols_true(tri)); Pw = dens(cols_wrong(tri))
        for p in range(3):
            X = cols(tei, p); logLR[tei, p] = np.log(Pt(X)) - np.log(Pw(X))
    return np.argmax(logLR, 1).astype(int)

# ── (level 5) SMOOTH / phase-space-MC template-LR (DAY13 NEXT-1) — the analytic+ISR pairing discriminant.
#   P_true, P_wrong are built NOT from data MC-truth folds (tmplLR) but from an INDEPENDENT physics MC: WW→4
#   massless partons generated at the data s' spectrum (gen_WW_m ⇒ ISR baked in; C5-validated), folded to reco
#   by a per-DIJET multiplicative response calibrated on the data true-pairing (reco-dijet / gen-W, angle-matched
#   — beats per-jet, which over-inflates the true ridge @240/365; see prototype_pairing_ps_mc_reco.py).
#   Wins vs tmplLR: (i) SMOOTH (MC stat negligible) ⇒ NO K-fold, NO fold/LR-seed dependence; (ii) physically
#   parametrised ⇒ the pairing systematic is lineshape+resolution, not template MC-stat; (iii) ISR-CONDITIONABLE
#   on the per-event RECO 4-jet mass (cond=1).
#   DAY13 result (5 seeds): smooth(mass+angle, tmplLRsma) REPRODUCES the histogram tmplLRa's pairing bias at all √s
#   (paired Δ ≈ +0.4/+0.6/+0.4 MeV, ≤2σ) at ~1pp lower eff — a genuine Bayes-optimal ceiling (eff saturates below the
#   truth histogram); ISR-conditioning (tmplLRsmac) shows NO resolvable margin (hint of mild +0.8 MeV degradation @365).
#   ⚠ LEADING MODELING RESIDUAL: the within-W angle response is approximated as ADDITIVE/angle-INDEPENDENT (_dijet_calib
#   builds one Δθ pool; sang adds it), but Δθ=θ_reco−θ_gen actually swings +1°→−8/−25/−32° with the gen angle @160/240/365
#   — calibrated on large-angle true W decays, extrapolated to small-angle cross-W wrong pairs; a hazard growing with √s.
#   REFINE = bin Δθ by gen angle (angle-conditioned smear) before propagating a smooth-model pairing systematic.
SM_NMC      = int(os.environ.get("SM_NMC", "2000000"))
SM_SEED     = int(os.environ.get("SM_SEED", "11"))
SM_COND     = int(os.environ.get("SM_COND", "0"))     # 1 ⇒ ISR-condition on the per-event reco 4-jet mass
SM_NSP      = int(os.environ.get("SM_NSP", "12"))     # # of reco-s' quantile bins when conditioning
SM_MBIN     = float(os.environ.get("SM_MBIN", "1.5")) # mass-template bin [GeV] (2-D mass-only template)
SM_MBIN_A   = float(os.environ.get("SM_MBIN_A","2.0"))# mass bin [GeV] for the 4-D mass+angle template (= incumbent
                                                     # LR_BIN; DAY13 ceiling test: eff saturates here, coarser 3.0 cost ~1pp)
SM_ABIN     = float(os.environ.get("SM_ABIN", "6.0")) # within-W angle bin [deg]
SM_SM       = float(os.environ.get("SM_SM", "1.0"))   # gaussian smoothing sigma [bins]
_LOGFLOOR   = float(np.log(1e-300))
def _opang_deg(u, v):                                  # opening angle [deg] between two 3-momenta
    c = (u[:,:3]*v[:,:3]).sum(1)/(np.linalg.norm(u[:,:3],axis=1)*np.linalg.norm(v[:,:3],axis=1)+1e-9)
    return np.degrees(np.arccos(np.clip(c,-1,1)))
def _dijet_calib():
    """data-calibrated per-dijet smearing pools, reco dijet↔gen W matched by angle (true pairing):
       R = m_reco/m_gen_W (multiplicative mass)  and  Δθ = θ_reco − θ_gen_W [deg] (within-W opening angle)."""
    Q = {i: np.stack([a[f"gen_qW{i}_px"],a[f"gen_qW{i}_py"],a[f"gen_qW{i}_pz"],a[f"gen_qW{i}_e"]],1) for i in range(4)}
    W1 = Q[0]+Q[1]; W2 = Q[2]+Q[3]
    DA = np.zeros((Nraw,4)); DB = np.zeros((Nraw,4)); thA = np.zeros(Nraw); thB = np.zeros(Nraw)
    for pi,p in enumerate(PORDER):
        sel = gpt==pi
        DA[sel] = (J[p[0][0]]+J[p[0][1]])[sel]; DB[sel] = (J[p[1][0]]+J[p[1][1]])[sel]
        thA[sel] = _opang_deg(J[p[0][0]],J[p[0][1]])[sel]; thB[sel] = _opang_deg(J[p[1][0]],J[p[1][1]])[sel]
    swap = (_opang_deg(DA,W2)+_opang_deg(DB,W1)) < (_opang_deg(DA,W1)+_opang_deg(DB,W2))
    rW1 = np.where(swap[:,None],DB,DA); rW2 = np.where(swap[:,None],DA,DB)
    thW1 = np.where(swap,thB,thA);      thW2 = np.where(swap,thA,thB)
    def m(v): return np.sqrt(np.maximum(v[:,3]**2-v[:,0]**2-v[:,1]**2-v[:,2]**2,0))
    okp = (gpt>=0)&(gpt<=2)
    R = np.concatenate([(m(rW1)/np.maximum(g1,1e-6))[okp], (m(rW2)/np.maximum(g2,1e-6))[okp]])
    dth = np.concatenate([(thW1-_opang_deg(Q[0],Q[1]))[okp], (thW2-_opang_deg(Q[2],Q[3]))[okp]])
    keepR = (R>0.2)&(R<3.0)
    return R[keepR], dth[keepR]   # filter BOTH pools consistently (angle pool was leaking mass-outlier events)
def _ps_mc(rng):
    """WW→4 partons at s' sampled from data gen_WW_m (ISR in); dijet-multiplicative mass smear + additive
    within-W angle smear to reco.  Returns mass-sorted (m_hi,m_lo) AND angle-sorted (θ_hi,θ_lo) for the TRUE
    partition and the two WRONG partitions, + phase-space weight + per-MC-event gen s' (for conditioning)."""
    sp = mWW_g[np.isfinite(mWW_g) & (mWW_g>90.0)]
    sqrts = sp[rng.integers(0,len(sp),SM_NMC)]; s = sqrts**2; mwgw = MW_REF*GW
    mg = np.linspace(45.,125.,6000); dd = mg*mg-MW_REF*MW_REF
    cdf = np.cumsum((mwgw/(dd*dd+mwgw*mwgw))*mg**DECAY_P); cdf /= cdf[-1]
    m1 = np.interp(rng.random(SM_NMC),cdf,mg); m2 = np.interp(rng.random(SM_NMC),cdf,mg)
    lam = (s-(m1+m2)**2)*(s-(m1-m2)**2)
    w = np.where((m1+m2<sqrts)&(lam>0), np.sqrt(np.maximum(lam,0)), 0.0)
    E1 = (s+m1**2-m2**2)/(2*sqrts); E2 = sqrts-E1; pst = np.sqrt(np.maximum(E1**2-m1**2,0))
    bb1 = pst/np.maximum(E1,1e-9); gg1 = E1/np.maximum(m1,1e-9)
    bb2 = pst/np.maximum(E2,1e-9); gg2 = E2/np.maximum(m2,1e-9)
    def parton(mw,g,b,cosA,phi,zs):
        sA = np.sqrt(np.maximum(1-cosA**2,0)); e = mw/2.; bz = zs*b
        px = e*sA*np.cos(phi); py = e*sA*np.sin(phi); pz = e*cosA
        return np.stack([px,py,g*(pz+bz*e),g*(e+bz*pz)],1)
    c1 = 2*rng.random(SM_NMC)-1; p1 = 2*np.pi*rng.random(SM_NMC)
    c2 = 2*rng.random(SM_NMC)-1; p2 = 2*np.pi*rng.random(SM_NMC)
    A1 = parton(m1,gg1,bb1,c1,p1,+1); A2 = parton(m1,gg1,bb1,-c1,p1+np.pi,+1)
    B1 = parton(m2,gg2,bb2,c2,p2,-1); B2 = parton(m2,gg2,bb2,-c2,p2+np.pi,-1)
    def mm4(v): return np.sqrt(np.maximum(v[:,3]**2-v[:,0]**2-v[:,1]**2-v[:,2]**2,0))
    Rp, Ap = _dijet_calib()
    def smass(x): return x*Rp[rng.integers(0,len(Rp),len(x))]                 # dijet multiplicative mass smear
    def sang(x):  return np.clip(x+Ap[rng.integers(0,len(Ap),len(x))],0,180)  # within-W additive angle smear
    # TRUE partition: dijets (A1,A2) and (B1,B2)
    mt1 = smass(mm4(A1+A2)); mt2 = smass(mm4(B1+B2))
    at1 = sang(_opang_deg(A1,A2)); at2 = sang(_opang_deg(B1,B2))
    # WRONG partition 1: (A1,B1)(A2,B2) ; WRONG partition 2: (A1,B2)(A2,B1)
    w11 = smass(mm4(A1+B1)); w12 = smass(mm4(A2+B2)); w21 = smass(mm4(A1+B2)); w22 = smass(mm4(A2+B1))
    aw11 = sang(_opang_deg(A1,B1)); aw12 = sang(_opang_deg(A2,B2)); aw21 = sang(_opang_deg(A1,B2)); aw22 = sang(_opang_deg(A2,B1))
    cat = np.concatenate
    return dict(
        t_mhi=np.maximum(mt1,mt2), t_mlo=np.minimum(mt1,mt2),                  # mass-sorted (true)
        t_ahi=np.maximum(at1,at2), t_alo=np.minimum(at1,at2),                  # angle-sorted (true, independent)
        tw=w, tsq=sqrts,
        w_mhi=cat([np.maximum(w11,w12), np.maximum(w21,w22)]),
        w_mlo=cat([np.minimum(w11,w12), np.minimum(w21,w22)]),
        w_ahi=cat([np.maximum(aw11,aw12), np.maximum(aw21,aw22)]),
        w_alo=cat([np.minimum(aw11,aw12), np.minimum(aw21,aw22)]),
        ww=cat([w,w]), wsq=cat([sqrts,sqrts]))
_SMOOTH_MC = None
def _get_smooth_mc():
    global _SMOOTH_MC
    if _SMOOTH_MC is None: _SMOOTH_MC = _ps_mc(np.random.default_rng(SM_SEED))
    return _SMOOTH_MC
_thi3a = _tlo3a = None
def _reco_angles_sorted():                              # data within-W reco angles, sorted hi/lo per partition
    global _thi3a, _tlo3a
    if _thi3a is None: _thi3a, _tlo3a = _reco_ang3()
    return _thi3a, _tlo3a
def tmpl_lr_smooth_pick(cond=None, use_angle=False):
    cond = SM_COND if cond is None else cond
    mc = _get_smooth_mc()
    if use_angle:
        me = np.arange(LR_M_LO, LR_M_HI+1e-6, SM_MBIN_A); ae = np.arange(0.0, 180.0+1e-6, SM_ABIN)
        edg = [me, me, ae, ae]; ctr = tuple(0.5*(e[:-1]+e[1:]) for e in edg)
        thi3a, tlo3a = _reco_angles_sorted()
        Xt = (mc["t_mhi"], mc["t_mlo"], mc["t_ahi"], mc["t_alo"])
        Xw = (mc["w_mhi"], mc["w_mlo"], mc["w_ahi"], mc["w_alo"])
        data_cols = [np.stack([hi3[:,p], lo3[:,p], thi3a[:,p], tlo3a[:,p]], 1) for p in range(3)]
    else:
        me = np.arange(LR_M_LO, LR_M_HI+1e-6, SM_MBIN); edg = [me, me]; ctr = (0.5*(me[:-1]+me[1:]),)*2
        Xt = (mc["t_mhi"], mc["t_mlo"]); Xw = (mc["w_mhi"], mc["w_mlo"])
        data_cols = [np.stack([hi3[:,p], lo3[:,p]], 1) for p in range(3)]
    def dens(samples, sel, wt):
        H,_ = np.histogramdd([x[sel] for x in samples], bins=edg, weights=wt[sel])
        if SM_SM>0: H = gaussian_filter(H, sigma=SM_SM)
        H = np.maximum(H/np.maximum(H.sum(),1e-300), 1e-300)
        return RegularGridInterpolator(ctr, np.log(H), bounds_error=False, fill_value=_LOGFLOOR)
    if not cond:
        allt = np.ones(len(mc["tw"]),bool); allw = np.ones(len(mc["ww"]),bool)
        lPt = dens(Xt, allt, mc["tw"]); lPw = dens(Xw, allw, mc["ww"])
        return np.argmax(np.stack([lPt(c)-lPw(c) for c in data_cols],1), 1).astype(int)
    # ISR-conditioned: bin by reco 4-jet mass; MC gets a reco-s' proxy = gen s' + sampled (reco−gen) s' residual.
    rng = np.random.default_rng(SM_SEED+1)
    okm = (gpt>=0)&(gpt<=2)&np.isfinite(mWW_g)&np.isfinite(mWW_reco)
    dpool = (mWW_reco - mWW_g)[okm]
    edges = np.quantile(mWW_reco[okm], np.linspace(0,1,SM_NSP+1)); edges[0] = -1e9; edges[-1] = 1e9
    mc_t = mc["tsq"] + dpool[rng.integers(0,len(dpool),len(mc["tsq"]))]
    mc_w = mc["wsq"] + dpool[rng.integers(0,len(dpool),len(mc["wsq"]))]
    binof = lambda x: np.clip(np.searchsorted(edges, x, side="right")-1, 0, SM_NSP-1)
    dbin = binof(mWW_reco); tbin = binof(mc_t); wbin = binof(mc_w)
    logLR = np.zeros((Nraw,3))
    for b in range(SM_NSP):
        drows = np.where(dbin==b)[0]
        if len(drows)==0: continue
        lPt = dens(Xt, tbin==b, mc["tw"]); lPw = dens(Xw, wbin==b, mc["ww"])
        for p in range(3):
            logLR[drows,p] = lPt(data_cols[p][drows]) - lPw(data_cols[p][drows])
    return np.argmax(logLR,1).astype(int)

modes = {"true": gpt, "bw": bwp,
         "pll":      pll_pick(ps=1, decay=1),   # ISR-convoluted per-event likelihood (phase-space √λ ON)
         "pllnops":  pll_pick(ps=0, decay=1),   # same lineshape (const×m³) but NO threshold √λ term
         "foldpick": foldpick(bwp),             # full forward-fold template (resolution-aware)
         "tmplLR":   tmpl_lr_pick(False),       # DAY12: analytic template-LR on the dijet masses (beats pllnops)
         "tmplLRa":  tmpl_lr_pick(True)}        #        + within-W angle = ADOPTED uniform pick (best/tied all √s; needed @160)
if HAVE_GENQW:
    modes["tmplLRsm"]   = tmpl_lr_smooth_pick(cond=False, use_angle=False)  # smooth MC-LR, mass-only, integrated
    modes["tmplLRsmc"]  = tmpl_lr_smooth_pick(cond=True,  use_angle=False)  #   + ISR-conditioned on reco 4-jet mass
    modes["tmplLRsma"]  = tmpl_lr_smooth_pick(cond=False, use_angle=True)   #   mass+within-W angle, integrated
    modes["tmplLRsmac"] = tmpl_lr_smooth_pick(cond=True,  use_angle=True)   #   mass+angle + ISR-conditioned (full)
if HAVE_KINFIT: modes["kinfit"] = kfp
margmodes = {"marg_flat": "flat", "marg_soft": "soft"}   # marginalise over 3 partitions (no hard pick)
ALLSEL = list(modes.keys()) + list(margmodes.keys())
sel = ALLSEL if PAIRING == "all" else [m for m in PAIRING.split(",") if m in ALLSEL]
print(f"\n{'='*100}\n[conv_mw_4q] ecm{ECM}  Nraw={Nraw}  N_ok={int(ok_base.sum())}  "
      f"GW={GW} MW_REF={MW_REF} BW_RUN={BW_RUN} DECAY_P={DECAY_P} NMW={NMW} EST={EST}"
      f"{' bin='+str(FOLD_BIN)+' sm='+str(FOLD_SM) if EST=='hist' else ' KDE_H='+str(KDE_H)}"
      f"  CLEAN={CLEAN}\n  <gen_W>={np.r_[g1[ok_base],g2[ok_base]].mean():.3f} "
      f"<sqrts'>={mWW_g[ok_base].mean():.3f}  PAIRING={PAIRING}\n{'='*100}")
print("  closure = reco_fit − gen_pe   (floor = identity_fold − gen_pe ; smear = reco_fit − identity_fold)")
_wcache = {}
def _run(m):
    if m in margmodes:
        if margmodes[m] not in _wcache: _wcache[margmodes[m]] = pair_weights(margmodes[m])
        return run_marg(m, _wcache[margmodes[m]])
    return run_mode(m, modes[m])
results = [_run(m) for m in sel]

RDIR = os.path.join(os.path.dirname(__file__), "anatomy_results"); os.makedirs(RDIR, exist_ok=True)
TAG = os.environ.get("TAG", "")
jpath = os.path.join(RDIR, f"conv_mw_4q_ecm{ECM}{'_n'+str(MAXN) if MAXN else ''}{TAG}.json")
with open(jpath, "w") as fh:
    json.dump(dict(ecm=ECM, Nraw=Nraw, N_ok=int(ok_base.sum()), gw=GW, mw_ref=MW_REF,
                   bw_run=BW_RUN, decay_p=DECAY_P, nmw=NMW, est=EST, fold_bin=FOLD_BIN,
                   fold_sm=FOLD_SM, kde_h=KDE_H, clean=CLEAN,
                   pair_ps=PAIR_PS, pair_decay=PAIR_DECAY, pair_sqrts=PAIR_SQRTS,
                   results=results), fh, indent=2)
print(f"\n[json] {jpath}")
