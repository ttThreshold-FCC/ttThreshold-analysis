#!/usr/bin/env python3
# ── Closure anatomy: WHY does the 1-D mW closure scatter ECM-dependently? ──────────────────────────
# HANDOFF_DAY3 NEXT #1.  Two questions, answered with measurements (no guessing):
#
#   (Q1) Decompose the matched-target closure  fold(reco) − gen_target  into:
#          estimator-floor  = identity_fold(gen) − gen_target   (binned-template MLE bias; calibratable)
#          smearing residual= fold(reco) − identity_fold(gen)    (real gen→reco smearing × template)
#        for BOTH weight schemes (grid → gen_marg target ; exact → gen_pe target).
#        ⇒ shows how much of the ECM-dependent scatter (grid +3/−24/−21, exact −9/−11/−49) is a
#          binning artifact vs a genuine modeling effect.
#
#   (Q2) The TARGET gap  gen_marg − gen_pe  (+70/+54/+11 MeV @157/160/163) is the radiator-marginalization
#        bias that decides grid-vs-exact.  Test the hypothesis "it tracks the WW threshold":
#          (a) RADIATOR-WIDTH LADDER: gen_marg with the radiator √s' spread scaled by f∈[0,1].
#              f=1 = full radiator (published gap); f=0 = delta at <√s'>.  Decomposes the gap into
#                gap_width = gen_marg(f=1) − gen_marg(f=0)   (Jensen/phase-space-curvature over the spread)
#                gap_pe    = gen_marg(f=0) − gen_pe          (single-global vs per-event √s')
#              Steeper ladder at low ECM ⇒ threshold-driven.
#          (b) PHASE-SPACE-MARGIN bins: gap vs δ = √s' − (m_qq+m_lν) (distance from the kinematic edge).
#              Hypothesis: gap blows up as δ→0 (the √λ cusp), shrinks far above threshold.
#          (c) THRESHOLD SPLIT: gap for √s'<2mW vs √s'>2mW within each ECM.
#
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# MAXN=0 python3 jax_prototype/conv_mw_closure_anatomy.py [ecm]
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
np.set_printoptions(linewidth=160, suppress=True)

ECM    = int(sys.argv[1]) if len(sys.argv) > 1 else 160
GW     = float(os.environ.get("GW", "2.085"))
MW_REF = float(os.environ.get("MW_REF", "80.379"))
BW_RUN = int(os.environ.get("BW_RUN", "0"))   # 0=fixed-width BW (default); 1=running-width Γ_W(m)=gW·m²/mW²
DECAY_P = float(os.environ.get("DECAY_P", "0"))  # W→qq decay phase-space factor: lineshape f(m)=m^p·BW(m).
#   DAY8: WHIZARD propagator is CONSTANT-width (ground-truth), but the OBSERVED m_qq density = |prop|²×(m-dep
#   decay factor).  DECAY_P>0 with BW_RUN=0 keeps WHIZARD's const-width denominator + adds the physical decay
#   numerator m^p (p≈2 mimics the running-width BW, p≈3 fits even better). Default 0 = bit-identical baseline.
#   DAY7 test: PDG mW is the running-width convention; M(run)=M(pole)+Γ²/(2mW)≈+27 MeV. A running-width BW with
#   parameter mW peaks ~27 MeV BELOW mW, so a fit recovers mW ~27 MeV ABOVE the fixed-width fit. Expect gen_pe
#   to shift ~+27 MeV (toward PDG) when BW_RUN=1.
MAXN   = int(os.environ.get("MAXN", "0"))
NMW    = int(os.environ.get("NMW", "61"))
MWLO, MWHI = float(os.environ.get("MWLO","79.0")), float(os.environ.get("MWHI","81.5"))
LOGZ_N = int(os.environ.get("LOGZ_N", "256"))
LOGZ_TAB_N = int(os.environ.get("LOGZ_TAB_N", "400"))
FOLD_K = int(os.environ.get("FOLD_K", "5"))
FOLD_BIN = float(os.environ.get("FOLD_BIN", "0.5"))
FOLD_SM  = float(os.environ.get("FOLD_SM", "0.6"))
# ── floor-reduction knobs (DAY5; default OFF ⇒ hist path bit-identical to DAY4) ──────────────────────
EST       = os.environ.get("EST", "hist")           # hist (binned, bit-identical) | kde (unbinned Gaussian KDE)
KDE_H     = float(os.environ.get("KDE_H", "0.6"))   # KDE bandwidth [GeV]: the floor's principled smoothing knob (no grid)
KDE_CH    = int(os.environ.get("KDE_CH", "2000"))   # test-point chunk size for the pairwise KDE (memory knob)
FOLD_CLIP = float(os.environ.get("FOLD_CLIP", "100.0"))  # per-mW weight clip percentile (tames ISR-tail spiky weights; 100=off)
FLOOR_ONLY= int(os.environ.get("FLOOR_ONLY", "0"))  # 1 ⇒ skip the radiator ladder + δ/threshold splits (fast floor-only iteration)
ROOT = f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
OUT  = "/eos/user/m/mdefranc/www/mW/kinfit_correlations"
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW, exist_ok=True)
import uproot

# ── BW + log_Z (GL-LOGZ_N), copied verbatim from conv_mw_fit.py for bit-compatibility ───────────────
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
        if BW_RUN:    # reweight the fixed-width arctan importance sampling → running-width BW (exact, no new grid)
            dd = m*m - mW2; wm = gW*m*m/mW; rf = (m*m/mW2)*(dd*dd + mwgw*mwgw)/(dd*dd + wm*wm)
        else:
            rf = np.ones_like(m)
        if DECAY_P:   # W→qq decay phase-space numerator m^p on each W ⇒ integrand carries (m_h·m_l)^p
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
def bw_run(m, mW, gW=GW):      # running width: width term ∝ m² ⇒ BW peaks ~Γ²/2mW (≈27 MeV) below mW
    d = m*m - mW*mW; wm = gW*m*m/mW
    return wm/(d*d + wm*wm)
def bw(m, mW, gW=GW):
    return bw_run(m, mW, gW) if BW_RUN else bw_fixed(m, mW, gW)

# ── load ────────────────────────────────────────────────────────────────────────────────────────
t = uproot.open(ROOT)["events"]
br = ["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_Whad_m","reco_Wlep_m",
      "reco_jet1_p","reco_jet2_p","reco_lep_p","reco_WW_m","kinfit_WW_m","reco_WW_pz","reco_met_p",
      "kinfit_chi2","kinfit_chi2_ndof"]
a = t.arrays(br, library="np")
SQRTS_MIN = float(os.environ.get("SQRTS_MIN", "-1e9"))  # DAY5 diagnostic: drop events with gen √s' below this
MQQ_MIN   = float(os.environ.get("MQQ_MIN", "-1e9"))    # DAY9: drop events with gen_Whad_m below this (off-shell
#   low-m_qq tail handle — the data-undershoot firm-up: tests whether the 4f off-shell-tail under-population
#   that biases the GEN self-fit also moves the data-applicable RECO closure (it should largely cancel).
# ── DAY6: DATA-APPLICABLE reco-proxy tail handle (the data analog of the gen SQRTS_MIN cut) ─────────
#   PROXY = reco branch to cut on (e.g. reco_WW_m); PROXY_DIR=low|high (which tail to REMOVE);
#   PROXY_FRAC = fraction of the ok-sample to remove (threshold = that percentile of the proxy).
#   Applied in the selection like SQRTS_MIN ⇒ gen targets + floor + closure all recompute on the kept set.
PROXY     = os.environ.get("PROXY", "")
PROXY_DIR = os.environ.get("PROXY_DIR", "low")          # low ⇒ remove smallest PROXY_FRAC (reco_WW_m tail)
PROXY_FRAC= float(os.environ.get("PROXY_FRAC", "0.0"))
ok = (np.isfinite(a["gen_Whad_m"]) & np.isfinite(a["gen_Wlep_m"]) & np.isfinite(a["gen_WW_m"]) &
      np.isfinite(a["reco_Whad_m"]) & np.isfinite(a["reco_Wlep_m"]) &
      (a["reco_jet1_p"]>0) & (a["reco_jet2_p"]>0) & (a["reco_lep_p"]>0) &
      (a["gen_WW_m"] >= SQRTS_MIN) & (a["gen_Whad_m"] >= MQQ_MIN) &
      (a["gen_Whad_m"]+a["gen_Wlep_m"] < a["gen_WW_m"]))
if PROXY and PROXY_FRAC > 0:
    pv = a[PROXY]; okv = ok & np.isfinite(pv)
    q = np.quantile(pv[okv], PROXY_FRAC if PROXY_DIR == "low" else 1.0-PROXY_FRAC)
    keep = (pv >= q) if PROXY_DIR == "low" else (pv <= q)
    ok = ok & np.isfinite(pv) & keep
    print(f"[proxy-cut] {PROXY} dir={PROXY_DIR} frac={PROXY_FRAC}  thr={q:.3f}  "
          f"kept {int((ok).sum())} of {int(okv.sum())}")
idx = np.where(ok)[0]
if MAXN > 0: idx = idx[:MAXN]
a = {k: v[idx] for k, v in a.items()}; N = len(idx)
mqq_g=a["gen_Whad_m"]; mlv_g=a["gen_Wlep_m"]; mWW_g=a["gen_WW_m"]
mqq_r=a["reco_Whad_m"]; mlv_r=a["reco_Wlep_m"]
# ── DAY6: per-event √s' used in the exact lineshape.  "gen" = gen_WW_m (truth, default/target).
#   Any reco branch (e.g. reco_WW_m) = the data-applicable per-event collision energy (handoff option b).
PEV_SQRTS = os.environ.get("PEV_SQRTS", "gen")
sqrts_pev = mWW_g if PEV_SQRTS == "gen" else np.asarray(a[PEV_SQRTS], float)
mw_scan = np.linspace(MWLO, MWHI, NMW)
print(f"\n{'='*86}\n[anatomy] ecm{ECM}  N={N}  GW={GW}  2mW_PDG={2*MW_REF:.3f}  "
      f"<√s'>={mWW_g.mean():.3f}  frac(√s'<2mW)={100*(mWW_g<2*MW_REF).mean():.1f}%\n{'='*86}")

# ── per-event gen_pe -2lnL matrix (exact per-event √s', tabulated log_Z) ─────────────────────────
def n2ll_perevent(mh, ml, mWW, mW, logz):
    s = mWW*mWW
    lam = (s-(mh+ml)**2)*(s-(mh-ml)**2)
    bad = lam <= 0; lam = np.where(bad, 1.0, lam)
    tv = (-2.0*np.log(np.maximum(bw(mh,mW),1e-300)) - 2.0*np.log(np.maximum(bw(ml,mW),1e-300))
          - np.log(lam) + 2.0*np.log(s) + 2.0*logz)
    if DECAY_P:   # lineshape f(m)=m^p·BW(m): add the decay numerator (mW-independent ⇒ no effect on argmin, kept for consistency)
        tv = tv - 2.0*DECAY_P*(np.log(np.maximum(mh,1e-9)) + np.log(np.maximum(ml,1e-9)))
    return np.where(bad, 1e6, tv)

def scan_perevent(sqrts=None):
    # √s' used in the per-event lineshape: gen (target/default) or a reco proxy (data-applicable, option b).
    sq = mWW_g if sqrts is None else np.asarray(sqrts, float)
    L = np.zeros((N, NMW))
    wn = np.linspace(float(np.min(sq))-0.5, float(np.max(sq))+0.5, LOGZ_TAB_N)
    for j, mW in enumerate(mw_scan):
        lz_ev = np.interp(sq, wn, log_Z(wn, mW))
        L[:, j] = n2ll_perevent(mqq_g, mlv_g, sq, mW, lz_ev)
    return L

# ── grid model: P_true(m_h,m_l;mW) marginalized over a radiator (sn,sw,divz) ─────────────────────
TLO, THI, NT = 28.0, 96.0, 137
tg = np.linspace(TLO, THI, NT); dt = tg[1]-tg[0]
TH, TL = np.meshgrid(tg, tg, indexing="ij")

def radiator_obs(width_scale=1.0):
    """Measured gen_WW_m spectrum (R_bare·σ_WW) ⇒ divide model by Z per node. width_scale shrinks the
    spread of √s' about its (count-weighted) mean: 1=full, 0=delta at <√s'>."""
    cnt, edg = np.histogram(mWW_g, bins=60, range=(max(TLO, ECM-45), ECM+1.5))
    sn = 0.5*(edg[:-1]+edg[1:]); w = cnt.astype(float)
    m = w > 0; sn, w = sn[m], w[m]
    if width_scale != 1.0:
        mu = np.average(sn, weights=w); sn = mu + width_scale*(sn - mu)
    return sn, w, True

def build_Ptrue(mW, rad):
    sn, sw, divz = rad
    bwh = bw(tg, mW)
    if DECAY_P: bwh = bwh * tg**DECAY_P              # lineshape f(m)=m^p·BW(m), consistent with log_Z
    BWout = bwh[:,None]*bwh[None,:]
    u = np.zeros((NT, NT)); logZ = log_Z(sn, mW) if divz else None
    for k, (sp, wk) in enumerate(zip(sn, sw)):
        s = sp*sp
        lam = (s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS = np.where(lam > 0, np.sqrt(np.maximum(lam,0))/s, 0.0)
        wgt = wk/np.exp(logZ[k]) if divz else wk
        u += wgt * BWout * PS
    Z = u.sum()*dt*dt
    return u/np.maximum(Z, 1e-300)

def scan_grid_gen(rad):
    """gen_marg per-event -2lnL on (mqq_g,mlv_g) for the given radiator."""
    L = np.zeros((N, NMW))
    for j, mW in enumerate(mw_scan):
        P = build_Ptrue(mW, rad)
        f = RegularGridInterpolator((tg, tg), P, bounds_error=False, fill_value=1e-300)
        L[:, j] = -2.0*np.log(np.maximum(f(np.stack([mqq_g, mlv_g], 1)), 1e-300))
    return L

def parab_min(xs, ys):
    i = int(np.clip(np.argmin(ys), 2, len(ys)-3))
    c = np.polyfit(xs[i-2:i+3], ys[i-2:i+3], 2)
    xm = -c[1]/(2*c[0]); sig = 1.0/np.sqrt(c[0]) if c[0] > 0 else np.nan
    return xm, sig
def fit_sub(L, mask=None):
    e = L[mask].sum(0) if mask is not None else L.sum(0); e = e - e.min()
    return parab_min(mw_scan, e)

# ── forward fold (k-fold), ported from conv_mw_fit.py scan_fold ───────────────────────────────────
def fold_fit(Lgen, th, tl):
    # Importance-reweighting reference (the sample's "generated mW").  DAY6: j0 = grid-snapped argmin is
    # mis-snapped by ~½·grid-spacing (~20 MeV at NMW=61) ⇒ biases the template/floor.  FINE_J0=1 uses a
    # parabola-interpolated, per-event reference column ⇒ removes the NMW dependence at fixed grid.
    if int(os.environ.get("FINE_J0", "0")):
        eg = Lgen.sum(0); eg = eg - eg.min(); mwref = parab_min(mw_scan, eg)[0]
        k0 = int(np.clip(np.searchsorted(mw_scan, mwref)-1, 0, NMW-2))
        fr = (mwref-mw_scan[k0])/(mw_scan[k0+1]-mw_scan[k0]); Lref = Lgen[:, k0]*(1-fr)+Lgen[:, k0+1]*fr
    else:
        Lref = Lgen[:, int(np.argmin(Lgen.sum(0)))]
    rng = np.random.default_rng(int(os.environ.get("FOLD_SEED", "2024")))
    fold = rng.integers(0, FOLD_K, N)
    L = np.zeros((N, NMW))
    if EST == "kde":
        # UNBINNED Gaussian-KDE template:  P(x|mW) = Σ_tr w_tr(mW)·N(x; x_tr, h·I) / Σ_tr w_tr(mW).
        # The pairwise kernel K[te,tr]=exp(-|x_te-x_tr|²/2h²) is mW-INDEPENDENT ⇒ precompute per fold, then
        # the density at every scan mW is one BLAS matmul  dens = K @ W(mW).  No grid ⇒ NO discretization
        # (FOLD_BIN) bias — only the smooth bandwidth h, the floor's principled knob.  k-fold = out-of-sample.
        pts = np.stack([th, tl], 1); inv2h2 = 1.0/(2.0*KDE_H*KDE_H)
        for k in range(FOLD_K):
            tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]; Xtr = pts[tr]
            W = np.exp(-0.5*(Lgen[tr] - Lref[tr][:, None]))                           # (Ntr,NMW)
            if FOLD_CLIP < 100.0: W = np.minimum(W, np.percentile(W, FOLD_CLIP, axis=0)[None, :])
            norm = W.sum(0)                                                           # (NMW,)
            for a in range(0, len(te), KDE_CH):
                tei = te[a:a+KDE_CH]; Xte = pts[tei]
                d2 = (Xte[:, 0, None]-Xtr[None, :, 0])**2 + (Xte[:, 1, None]-Xtr[None, :, 1])**2  # (c,Ntr)
                dens = (np.exp(-d2*inv2h2) @ W) / np.maximum(norm[None, :], 1e-300)   # (c,NMW)
                L[tei] = -2.0*np.log(np.maximum(dens, 1e-300))
    else:
        e1 = np.arange(30.0, 98.0+1e-6, FOLD_BIN); c1 = 0.5*(e1[:-1]+e1[1:])
        for j in range(NMW):
            wfull = np.exp(-0.5*(Lgen[:, j] - Lref))
            if FOLD_CLIP < 100.0: wfull = np.minimum(wfull, np.percentile(wfull, FOLD_CLIP))
            for k in range(FOLD_K):
                tr = fold != k; te = fold == k; w = wfull[tr]; ii = np.where(te)[0]
                H, _, _ = np.histogram2d(th[tr], tl[tr], bins=[e1, e1], weights=w)
                if FOLD_SM > 0: H = gaussian_filter(H, sigma=FOLD_SM)
                H = H/np.maximum(H.sum(), 1e-300)
                pv = RegularGridInterpolator((c1, c1), H, bounds_error=False, fill_value=1e-300)(np.stack([th[te], tl[te]], 1))
                L[ii, j] = -2.0*np.log(np.maximum(pv, 1e-300))
    good = np.all(np.isfinite(L), axis=1) & (np.ptp(L, axis=1) > 0)
    return parab_min(mw_scan, (L[good].sum(0) - L[good].sum(0).min()))

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# (0) gen targets + (1) closure decomposition
# ══════════════════════════════════════════════════════════════════════════════════════════════════
SCHEMES = os.environ.get("SCHEMES", "both")   # both (grid+exact) | exact (skip grid fits + L_marg: fast KDE sweeps)
L_pe   = scan_perevent()                      # gen √s' ⇒ the TARGET gen_pe (truth, always gen)
genpe, s_pe = fit_sub(L_pe)
# fold TEMPLATE weights: gen √s' (default) or reco proxy √s' (PEV_SQRTS=reco_WW_m, the data-applicable lineshape)
L_tmpl = L_pe if PEV_SQRTS == "gen" else scan_perevent(sqrts_pev)
dec = {}
if SCHEMES == "both":
    L_marg = scan_grid_gen(radiator_obs(1.0)); genmarg, s_marg = fit_sub(L_marg)
    grid_id  = fold_fit(L_marg, mqq_g, mlv_g)[0]; grid_rec = fold_fit(L_marg, mqq_r, mlv_r)[0]
    dec["grid"] = dict(target=genmarg, identity=grid_id, reco=grid_rec,
                       closure=grid_rec-genmarg, floor=grid_id-genmarg, smear=grid_rec-grid_id)
else:
    genmarg = s_marg = float("nan")
print(f"\n[targets] gen_pe={genpe:.4f}±{s_pe:.4f}   gen_marg={genmarg:.4f}   "
      f"gen_marg−gen_pe = {1000*(genmarg-genpe):+.1f} MeV")
if int(os.environ.get("GENPE_ONLY", "0")):   # DAY8: skip the (expensive) fold estimator when only gen_pe is wanted
    sys.stdout.flush(); sys.exit(0)
exa_id   = fold_fit(L_tmpl, mqq_g, mlv_g)[0]
exa_rec  = fold_fit(L_tmpl, mqq_r, mlv_r)[0]
dec["exact"] = dict(target=genpe, identity=exa_id, reco=exa_rec,
                    closure=exa_rec-genpe, floor=exa_id-genpe, smear=exa_rec-exa_id)
print(f"\n[Q1] CLOSURE DECOMPOSITION  (closure = floor + smearing)  "
      f"EST={EST}" + (f" KDE_H={KDE_H}" if EST == "kde" else f" bin={FOLD_BIN} sm={FOLD_SM}")
      + (f" clip={FOLD_CLIP}" if FOLD_CLIP < 100 else "") + ":")
for sch, d in dec.items():
    print(f"  {sch:5s}: target={d['target']:.4f} identity={d['identity']:.4f} reco={d['reco']:.4f}  "
          f"|  closure={1000*d['closure']:+6.1f}  floor={1000*d['floor']:+6.1f}  smear={1000*d['smear']:+6.1f}  MeV")

# ══════════════════════════════════════════════════════════════════════════════════════════════════
# (2) radiator ladder + δ/threshold splits — DAY4's Q2 (gen-only, EST-independent).  Skipped for the
#     DAY5 floor-reduction sweeps (FLOOR_ONLY=1): the targets/gap are bit-invariant, so re-deriving them
#     each estimator run is wasted compute.
# ══════════════════════════════════════════════════════════════════════════════════════════════════
gap_width = gap_pe = float("nan"); fL = ladder = np.array([]); mb_d = mb_gap = np.array([]); mb_n = []; split = {}
if not FLOOR_ONLY:
    # (2a) radiator-width ladder: gen_marg(f) for f in [0,1]
    fL = np.array([0.0, 0.125, 0.25, 0.5, 0.75, 1.0])
    ladder = np.array([fit_sub(scan_grid_gen(radiator_obs(f)))[0] for f in fL])
    gap_width = ladder[-1] - ladder[0]      # full − delta(mean)  : marginalization-spread (Jensen) piece
    gap_pe    = ladder[0]  - genpe          # delta(mean) − per-event : single-global vs per-event √s'
    print(f"\n[Q2a] RADIATOR-WIDTH LADDER gen_marg(f) [GeV]:")
    for f, m in zip(fL, ladder): print(f"   f={f:5.3f}  mW={m:.4f}   (−gen_pe = {1000*(m-genpe):+6.1f} MeV)")
    print(f"   gap_total={1000*(ladder[-1]-genpe):+.1f}  = gap_width(spread)={1000*gap_width:+.1f} "
          f"+ gap_pe(per-event)={1000*gap_pe:+.1f}  MeV")

    # (2b) phase-space-margin bins:  gap = gen_marg − gen_pe  vs  δ = √s' − (m_qq+m_lν)
    delta = mWW_g - (mqq_g + mlv_g)
    qedges = np.quantile(delta, np.linspace(0, 1, 7))
    qedges[0]-=1e-6; qedges[-1]+=1e-6
    mb_d, mb_gap, mb_n = [], [], []
    for b in range(len(qedges)-1):
        m = (delta>=qedges[b]) & (delta<qedges[b+1])
        if m.sum() < 200: continue
        gpe = fit_sub(L_pe, m)[0]; gmg = fit_sub(L_marg, m)[0]
        mb_d.append(delta[m].mean()); mb_gap.append(gmg-gpe); mb_n.append(int(m.sum()))
    mb_d, mb_gap = np.array(mb_d), np.array(mb_gap)
    print(f"\n[Q2b] GAP vs phase-space margin δ=√s'−(m_qq+m_lν):")
    for d_, g_, n_ in zip(mb_d, mb_gap, mb_n): print(f"   <δ>={d_:6.2f} GeV  gap={1000*g_:+7.1f} MeV  (N={n_})")

    # (2c) threshold split (√s' below / above 2mW)
    below = mWW_g < 2*MW_REF; above = ~below
    for nm, m in (("below_2mW", below), ("above_2mW", above)):
        if m.sum() < 200: split[nm] = None; continue
        gpe = fit_sub(L_pe, m)[0]; gmg = fit_sub(L_marg, m)[0]
        split[nm] = dict(frac=float(m.mean()), gap=gmg-gpe, gen_pe=gpe, gen_marg=gmg, n=int(m.sum()))
    print(f"\n[Q2c] THRESHOLD SPLIT (gap = gen_marg − gen_pe):")
    for nm, d in split.items():
        if d: print(f"   {nm:10s} frac={100*d['frac']:4.1f}%  gap={1000*d['gap']:+7.1f} MeV  (N={d['n']})")

# ── dump JSON ─────────────────────────────────────────────────────────────────────────────────────
res = dict(ecm=ECM, N=N, mean_sqrts=float(mWW_g.mean()), frac_below_2mW=float((mWW_g<2*MW_REF).mean()),
           gen_pe=genpe, gen_marg=genmarg, sig_pe=s_pe, sig_marg=s_marg, decomp=dec,
           est=EST, kde_h=KDE_H, fold_clip=FOLD_CLIP, fold_bin=FOLD_BIN, fold_sm=FOLD_SM,
           proxy=PROXY, proxy_dir=PROXY_DIR, proxy_frac=PROXY_FRAC, pev_sqrts=PEV_SQRTS,
           ladder_f=fL.tolist(), ladder_mW=ladder.tolist(), gap_width=float(gap_width), gap_pe=float(gap_pe),
           margin_delta=mb_d.tolist(), margin_gap=mb_gap.tolist(), margin_n=mb_n, split=split,
           thr_deficit=float(2*MW_REF-ECM),
           mean_margin=float((mWW_g-(mqq_g+mlv_g)).mean()))
RDIR = os.path.join(os.path.dirname(__file__), "anatomy_results"); os.makedirs(RDIR, exist_ok=True)
TAG = os.environ.get("TAG", "")
jpath = os.path.join(RDIR, f"anatomy_ecm{ECM}{'_n'+str(MAXN) if MAXN else ''}{TAG}.json")
with open(jpath, "w") as fh: json.dump(res, fh, indent=2)
print(f"\n[json] {jpath}")
