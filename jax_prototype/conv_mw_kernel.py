#!/usr/bin/env python3
# ── STEP 1/2/3 of the convolution-based 1-D mW fit: measure the 2-D resolution KERNEL ──
# The convolution fit needs the experimental response r(Δm_qq, Δm_lν) = (reco − gen) for the two
# W masses, INCLUDING their correlation and tails (handoff: notes/convolution_mw_fit/HANDOFF.md).
# Observables (already in the tree, verified bit-identical to hand-built):
#   m_qq  = reco_Whad_m  (massless dijet)              ; truth = gen_Whad_m
#   m_lν  = reco_Wlep_m  (lep + ν, ν=MET 3-mom, m=0)   ; truth = gen_Wlep_m
# This script:
#   (1) measures the 2-D residual kernel: mean vector, covariance, correlation, IQR, tail shape;
#   (2) tests STATIONARITY  — does r depend on the true mass? (if yes → need a transfer matrix);
#   (3) tests ISR dependence — does the tail widen with E_ISR? (the dominant systematic);
#   (4) ν energy-vs-momentum cross-check (handoff step 1c, repeat at 157/163);
#   (5) saves the empirical kernel (fine 2-D histogram + raw residual sample) to npz for the fit;
#   (6) publishes diagnostic plots to EOS www.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   MAXN=0 python3 jax_prototype/conv_mw_kernel.py [root] [ecm]
import sys, os
import numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.set_printoptions(linewidth=160, suppress=True)
MU = 0.1056583745
ECM = int(sys.argv[2]) if len(sys.argv) > 2 else 160
ROOT = sys.argv[1] if len(sys.argv) > 1 else \
  f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
MAXN = int(os.environ.get("MAXN", "0"))
OUT = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(OUT, exist_ok=True)
KDIR = os.path.join(os.path.dirname(__file__), "conv_kernels"); os.makedirs(KDIR, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
t = uproot.open(ROOT)["events"]
br = ["reco_Whad_m", "reco_Wlep_m", "gen_Whad_m", "gen_Wlep_m",
      "reco_jet1_p", "reco_jet1_theta", "reco_jet1_phi",
      "reco_jet2_p", "reco_jet2_theta", "reco_jet2_phi",
      "reco_lep_p", "reco_lep_theta", "reco_lep_phi",
      "gen_isr_px", "gen_isr_py", "gen_isr_pz",
      "kinfit_mW", "kinfit_valid"]
a = t.arrays(br, library="np")
ok = (np.isfinite(a["reco_Whad_m"]) & np.isfinite(a["reco_Wlep_m"]) &
      np.isfinite(a["gen_Whad_m"]) & np.isfinite(a["gen_Wlep_m"]) &
      (a["reco_jet1_p"] > 0) & (a["reco_jet2_p"] > 0) & (a["reco_lep_p"] > 0))
idx = np.where(ok)[0]
if MAXN > 0: idx = idx[:MAXN]
a = {k: v[idx] for k, v in a.items()}
N = len(idx)

mqq_r = a["reco_Whad_m"]; mlv_r = a["reco_Wlep_m"]
mqq_g = a["gen_Whad_m"];  mlv_g = a["gen_Wlep_m"]
dqq = mqq_r - mqq_g       # Δm_qq  (the hadronic residual)
dlv = mlv_r - mlv_g       # Δm_lν  (the leptonic residual)
E_isr = np.sqrt(a["gen_isr_px"]**2 + a["gen_isr_py"]**2 + a["gen_isr_pz"]**2)   # collinear ⇒ E≈|p|

def stats(x): return dict(mean=x.mean(), std=x.std(), med=np.median(x),
                          iqr=np.subtract(*np.percentile(x, [75, 25]))/1.349)  # robust σ
print(f"\n{'='*78}\n[conv-kernel] ecm{ECM}  N={N}  file={os.path.basename(ROOT)}\n{'='*78}")
print(f"GEN mass distribution (lineshape target):")
print(f"  m_qq(gen): mean={mqq_g.mean():.3f} std={mqq_g.std():.3f}   m_lν(gen): mean={mlv_g.mean():.3f} std={mlv_g.std():.3f}")
print(f"  sum m_qq+m_lν: mean={(mqq_g+mlv_g).mean():.2f} (√s={ECM}) ⇒ phase-space boundary {'active' if (mqq_g+mlv_g).mean()>ECM-5 else 'rarely active'}")

# ── (1) KERNEL: mean vector, covariance, correlation ──────────────────────────
sq, sl = stats(dqq), stats(dlv)
cov = np.cov(dqq, dlv); rho = cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])
print(f"\n[1] 2-D RESOLUTION KERNEL r(Δm_qq, Δm_lν)  (reco − gen):")
print(f"  Δm_qq: mean={sq['mean']:+.3f} std={sq['std']:.3f} med={sq['med']:+.3f} robustσ(IQR)={sq['iqr']:.3f}")
print(f"  Δm_lν: mean={sl['mean']:+.3f} std={sl['std']:.3f} med={sl['med']:+.3f} robustσ(IQR)={sl['iqr']:.3f}")
print(f"  corr(Δm_qq, Δm_lν) = {rho:+.3f}   cov=[[{cov[0,0]:.2f},{cov[0,1]:.2f}],[{cov[1,0]:.2f},{cov[1,1]:.2f}]]")
from scipy.stats import skew, kurtosis
print(f"  tails: skew(Δqq)={skew(dqq):+.2f} kurt(Δqq)={kurtosis(dqq):+.2f} | skew(Δlν)={skew(dlv):+.2f} kurt(Δlν)={kurtosis(dlv):+.2f}  (0=Gaussian)")

# ── (2) STATIONARITY: does the kernel depend on the TRUE mass? ─────────────────
print(f"\n[2] STATIONARITY — kernel vs TRUE mass (if mean/std drift strongly ⇒ need transfer matrix):")
def bin_report(key, val, edges, dx, dy):
    print(f"  binned by {key}:")
    for i in range(len(edges)-1):
        m = (val >= edges[i]) & (val < edges[i+1])
        if m.sum() < 50: continue
        r = np.corrcoef(dx[m], dy[m])[0, 1]
        print(f"    [{edges[i]:5.1f},{edges[i+1]:5.1f}) n={m.sum():6d}  Δqq μ={dx[m].mean():+.2f} σ={dx[m].std():.2f} | "
              f"Δlν μ={dy[m].mean():+.2f} σ={dy[m].std():.2f} | ρ={r:+.2f}")
qedges = np.percentile(mqq_g, [0, 20, 40, 60, 80, 100])
ledges = np.percentile(mlv_g, [0, 20, 40, 60, 80, 100])
bin_report("m_qq(gen)", mqq_g, qedges, dqq, dlv)
bin_report("m_lν(gen)", mlv_g, ledges, dqq, dlv)

# ── (3) ISR dependence of the kernel tail ─────────────────────────────────────
print(f"\n[3] ISR dependence — E_ISR≈|gen_isr_3| (radiator should remove this from 'resolution'):")
iedges = [0, 0.5, 2.0, 5.0, 1e9]
for i in range(len(iedges)-1):
    m = (E_isr >= iedges[i]) & (E_isr < iedges[i+1])
    if m.sum() < 30: continue
    lab = f"[{iedges[i]:.1f},{iedges[i+1]:.1f})" if iedges[i+1] < 1e8 else f">{iedges[i]:.1f}"
    print(f"  E_ISR {lab:>10s}  frac={100*m.mean():5.1f}%  Δqq σ={dqq[m].std():.2f} μ={dqq[m].mean():+.2f} | "
          f"Δlν σ={dlv[m].std():.2f} μ={dlv[m].mean():+.2f}")
print(f"  E_ISR spectrum: median={np.median(E_isr):.3f} mean={E_isr.mean():.3f} p90={np.percentile(E_isr,90):.2f} "
      f"p99={np.percentile(E_isr,99):.2f} max={E_isr.max():.1f}  (>1GeV: {100*(E_isr>1).mean():.1f}%)")

# ── (4) ν energy-vs-momentum cross-check ──────────────────────────────────────
def vec(p, th, ph, m): st = np.sin(th); return np.stack([np.sqrt(p*p+m*m), p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th)], 0)
def M(v): return np.sqrt(np.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2), 0))
J1 = vec(a["reco_jet1_p"], a["reco_jet1_theta"], a["reco_jet1_phi"], 0.0)
J2 = vec(a["reco_jet2_p"], a["reco_jet2_theta"], a["reco_jet2_phi"], 0.0)
L  = vec(a["reco_lep_p"],  a["reco_lep_theta"],  a["reco_lep_phi"],  MU)
Vis = J1 + J2 + L
nux = -Vis[1]; nuy = -Vis[2]                      # transverse from momentum balance (same for both)
Enu_E = ECM - Vis[0]; disc = Enu_E**2 - (nux**2 + nuy**2)
nuz_E = np.where(-Vis[3] >= 0, np.sqrt(np.maximum(disc, 0)), -np.sqrt(np.maximum(disc, 0)))
NU_E = np.stack([np.maximum(Enu_E, 0), nux, nuy, nuz_E], 0)
mlv_energy = M(L + NU_E)
de = mlv_energy - mlv_g
print(f"\n[4] ν construction: 3-MOMENTUM (MET) vs ENERGY (√s−E_vis):")
print(f"  3-mom (reco_Wlep_m): std={dlv.std():.3f} mean={dlv.mean():+.3f}   ← adopted")
print(f"  energy             : std={de.std():.3f} mean={de.mean():+.3f}   (disc<0 in {100*(disc<0).mean():.1f}% ⇒ NaN-prone)")

# ── (5) save empirical kernel (fine 2-D histogram, normalized) + raw sample ────
# kernel grid: symmetric-ish, covers the bias + ~5σ tails
qlo, qhi = sq['mean']-6*sq['iqr'], sq['mean']+6*sq['iqr']
llo, lhi = sl['mean']-6*sl['iqr'], sl['mean']+6*sl['iqr']
NB = 121
qe = np.linspace(qlo, qhi, NB+1); le = np.linspace(llo, lhi, NB+1)
Hk, _, _ = np.histogram2d(dqq, dlv, bins=[qe, le])
Hk = Hk / (Hk.sum() * (qe[1]-qe[0]) * (le[1]-le[0]))   # normalized 2-D pdf (per GeV²)
qc = 0.5*(qe[:-1]+qe[1:]); lc = 0.5*(le[:-1]+le[1:])
kpath = os.path.join(KDIR, f"kernel_ecm{ECM}.npz")
np.savez(kpath, Hk=Hk, qc=qc, lc=lc, qe=qe, le=le,
         mean=np.array([sq['mean'], sl['mean']]), cov=cov, rho=rho,
         dqq=dqq.astype(np.float32), dlv=dlv.astype(np.float32),
         mqq_g=mqq_g.astype(np.float32), mlv_g=mlv_g.astype(np.float32),
         mqq_r=mqq_r.astype(np.float32), mlv_r=mlv_r.astype(np.float32),
         E_isr=E_isr.astype(np.float32))
print(f"\n[5] saved kernel → {kpath}  (grid {NB}², per-GeV² pdf; raw residual sample incl.)")

# ── (6) plots ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(18, 11))
ax[0,0].hexbin(dqq, dlv, gridsize=60, extent=(qlo, qhi, llo, lhi), cmap="viridis", mincnt=1)
ax[0,0].axhline(0, color="w", lw=.6); ax[0,0].axvline(0, color="w", lw=.6)
ax[0,0].set_xlabel("Δm_qq = reco−gen [GeV]"); ax[0,0].set_ylabel("Δm_lν = reco−gen [GeV]")
ax[0,0].set_title(f"2-D kernel  ρ={rho:+.2f}  (ecm{ECM}, N={N})")
ax[0,1].hist(dqq, bins=120, range=(qlo, qhi), histtype="step", lw=2, color="C0", density=True,
             label=f"Δm_qq μ={sq['mean']:+.2f} σ={sq['std']:.2f}")
ax[0,1].hist(dlv, bins=120, range=(llo, lhi), histtype="step", lw=2, color="C3", density=True,
             label=f"Δm_lν μ={sl['mean']:+.2f} σ={sl['std']:.2f}")
ax[0,1].axvline(0, color="k", ls="--", lw=1); ax[0,1].set_xlabel("residual [GeV]"); ax[0,1].legend(); ax[0,1].set_yscale("log")
ax[0,1].set_title("marginal residuals (log — see tails)")
# gen mass distributions (lineshape target)
ax[0,2].hist(mqq_g, bins=100, range=(40, 100), histtype="step", lw=2, color="C0", density=True, label="m_qq(gen)")
ax[0,2].hist(mlv_g, bins=100, range=(40, 100), histtype="step", lw=2, color="C3", density=True, label="m_lν(gen)")
ax[0,2].axvline(80.385, color="k", ls="--", lw=1, label="pole"); ax[0,2].set_xlabel("true W mass [GeV]")
ax[0,2].legend(); ax[0,2].set_title("GEN mass dists (lineshape must reproduce)")
# stationarity: residual mean vs true mass
for v, d, c, lab in [(mqq_g, dqq, "C0", "Δqq vs m_qq"), (mlv_g, dlv, "C3", "Δlν vs m_lν")]:
    be = np.percentile(v, np.linspace(0, 100, 11)); bc = 0.5*(be[:-1]+be[1:])
    mu = [d[(v >= be[i]) & (v < be[i+1])].mean() for i in range(10)]
    ax[1,0].plot(bc, mu, "o-", color=c, label=lab)
ax[1,0].axhline(0, color="k", ls=":"); ax[1,0].set_xlabel("true mass [GeV]"); ax[1,0].set_ylabel("⟨residual⟩ [GeV]")
ax[1,0].legend(); ax[1,0].set_title("[2] stationarity: ⟨Δ⟩ vs true mass")
# ISR dependence
for d, c, lab in [(dqq, "C0", "Δqq"), (dlv, "C3", "Δlν")]:
    be = [0, 0.5, 1, 2, 5, 20]; bc = [0.25, 0.75, 1.5, 3.5, 12]
    sd = [d[(E_isr >= be[i]) & (E_isr < be[i+1])].std() for i in range(5)]
    ax[1,1].plot(bc, sd, "o-", color=c, label=lab)
ax[1,1].set_xlabel("E_ISR [GeV]"); ax[1,1].set_ylabel("σ(residual) [GeV]"); ax[1,1].set_xscale("log")
ax[1,1].legend(); ax[1,1].set_title("[3] ISR widens the tail")
# empirical kernel image
im = ax[1,2].imshow(Hk.T, origin="lower", aspect="auto", extent=(qlo, qhi, llo, lhi), cmap="viridis")
ax[1,2].set_xlabel("Δm_qq [GeV]"); ax[1,2].set_ylabel("Δm_lν [GeV]"); ax[1,2].set_title("saved empirical kernel (pdf)")
plt.colorbar(im, ax=ax[1,2])
plt.tight_layout(); png = f"{OUT}/kernel_diag_ecm{ECM}.png"; plt.savefig(png, dpi=100)
print(f"[6] plot → {png}")
