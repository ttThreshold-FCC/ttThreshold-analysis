#!/usr/bin/env python3
# ── NAME the lnuqq kfit flat direction (handoff step 1) + status-3 physics regime (step 2) ──
# Reconstructs the EXACT production minimum y* from the stored kinfit_* params in the
# step2 tree (no re-fit, no solver ambiguity), computes the OBJECTIVE Hessian via
# jax.hessian at y*, drops the fixed gW row/col (15 free params), eigendecomposes, and
# reports the soft eigenvector composition for valid vs status-3 events. Then correlates
# kinfit_status with gen-level physics regime (ISR energy, jet resolution, W boost, forward
# topology, nu_pz) using branches already in the tree.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_ISR_MODE=kfit KF_SIGMA_R=1.5 python3 jax_prototype/name_flat_direction.py <root> [ecm]
import sys, os
os.environ.setdefault("KF_ISR_MODE", "kfit")     # production lnuqq kfit objective
os.environ.setdefault("KF_SIGMA_R", "1.5")       # = C++ KF_ISR_SYS_SIGMA (reco closure width)
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, jax, jax.numpy as jnp, uproot
sys.path.insert(0, os.path.dirname(__file__))
import jaxfit_common as J
import kinfit_lnuqq_jax as K

np.set_printoptions(linewidth=160, suppress=True)

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM = int(sys.argv[2]) if len(sys.argv) > 2 else 160
MAXN = int(os.environ.get("MAXN", "0"))   # 0 = all
print(f"[flat-dir] {ROOT}  ecm{ECM}  KF_ISR_MODE={K.KF_ISR_MODE} SIGMA_R={K.SIGMA_R}")
assert K.KF_ISR_MODE == "kfit", "set KF_ISR_MODE=kfit"

# ── load tree ────────────────────────────────────────────────────────────────
RECO = ["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta",
        "reco_jet2_phi","reco_lep_p","reco_lep_theta","reco_lep_phi",
        "reco_met_p","reco_met_theta","reco_met_phi"]
KF = ["kinfit_mW","kinfit_gW","kinfit_s1","kinfit_s2","kinfit_sl","kinfit_sn","kinfit_t1",
      "kinfit_t2","kinfit_tn","kinfit_tl","kinfit_p1","kinfit_p2","kinfit_pn","kinfit_pl",
      "kinfit_bes_m_minus_ecm","kinfit_bes_pz","kinfit_status","kinfit_valid",
      "kinfit_valid_loose","kinfit_edm","kinfit_chi2","kinfit_priors_swapped","kinfit_winner_pass"]
GEN = ["gen_isr_px","gen_isr_py","gen_isr_pz","gen_WW_p_imbalance_tot","gen_ee_pz",
       "gen_Whad_p","gen_Wlep_p","gen_Whad_costheta","gen_Wlep_costheta","gen_lep_costheta",
       "gen_nu_p","gen_nu_costheta","gen_Whad_m","gen_Wlep_m",
       "jet1_p_resp","jet2_p_resp","lep_p_resp","jet1_matched_q_dR","jet2_matched_q_dR",
       "jet1_theta_resol","jet2_theta_resol"]
t = uproot.open(ROOT)["events"]
have = set(t.keys())
GEN = [g for g in GEN if g in have]
a = t.arrays(RECO + KF + GEN, library="np")
D = np.stack([a[c] for c in RECO], axis=1).astype(np.float64)
ok = np.all(np.isfinite(D), axis=1) & (D[:,0]>0) & (D[:,3]>0) & (D[:,6]>0) & (D[:,9]>0)
idx = np.where(ok)[0]
if MAXN > 0 and len(idx) > MAXN: idx = idx[:MAXN]
D = D[idx]; a = {k: v[idx] for k, v in a.items()}
N = len(D)
print(f"  N={N}  swapped(any)={int(a['kinfit_priors_swapped'].sum())} (pool => expect 0)")

# ── build pooled prior packs (same family C++ used) ──────────────────────────
PR = K.build_prior_pack_pool(ECM, D)             # (N,93)
P = J.load_params(K.HEADER)
bm = P[f"GAUSS_GEN_EE_M_MINUS_ECM_{ECM}"]; bz = P[f"GAUSS_GEN_EE_PZ_{ECM}"]

def mu_sig(slot):
    o = K._OFF[slot][0]
    return PR[:, o], PR[:, o+1]

# ── reconstruct y* (N,16) from stored params ─────────────────────────────────
y = np.zeros((N, 16))
y[:,0]  = (a["kinfit_mW"] - K.KF_MW_INIT) / K.KF_MW_PHYS_SIGMA
y[:,1]  = 0.0                                              # gW fixed
mu,sg = mu_sig("j1p"); y[:,2]  = (a["kinfit_s1"] - mu)/sg
mu,sg = mu_sig("j2p"); y[:,3]  = (a["kinfit_s2"] - mu)/sg
mu,sg = mu_sig("lp");  y[:,4]  = (a["kinfit_sl"] - mu)/sg
y[:,5]  = a["kinfit_sn"] / K.ISR_SX                        # pgx = ISR_SX*y5
mu,sg = mu_sig("j1t"); y[:,6]  = (a["kinfit_t1"] - mu)/sg
mu,sg = mu_sig("j2t"); y[:,7]  = (a["kinfit_t2"] - mu)/sg
y[:,8]  = a["kinfit_tn"] / K.ISR_SY                        # pgy = ISR_SY*y8
mu,sg = mu_sig("j1q"); y[:,9]  = (a["kinfit_p1"] - mu)/sg
mu,sg = mu_sig("j2q"); y[:,10] = (a["kinfit_p2"] - mu)/sg
y[:,11] = a["kinfit_pn"] / K.ISR_SZ                        # kz = ISR_SZ*y11
mu,sg = mu_sig("lt");  y[:,12] = (a["kinfit_tl"] - mu)/sg
mu,sg = mu_sig("lq");  y[:,13] = (a["kinfit_pl"] - mu)/sg
y[:,14] = (a["kinfit_bes_m_minus_ecm"] - bm[0]) / bm[1]
y[:,15] = (a["kinfit_bes_pz"]          - bz[0]) / bz[1]
y = y.astype(np.float64)

# ── JAX objective handles ────────────────────────────────────────────────────
chi2 = K.build_chi2(ECM)                          # captures kfit + SIGMA_R from env
val_fn  = jax.jit(jax.vmap(lambda yy,DD,pp: chi2(yy,DD,pp)))
grad_fn = jax.jit(jax.vmap(lambda yy,DD,pp: jax.grad(chi2)(yy,DD,pp)))
hess_fn = jax.jit(jax.vmap(lambda yy,DD,pp: jax.hessian(chi2)(yy,DD,pp)))

def chunked(fn, *arrs, bs=4000):
    out = []
    for s in range(0, N, bs):
        out.append(np.asarray(fn(*[jnp.asarray(x[s:s+bs]) for x in arrs])))
    return np.concatenate(out, axis=0)

# ── VALIDATION: JAX chi2(y*) vs tree kinfit_chi2 ─────────────────────────────
c_jax = chunked(val_fn, y, D, PR)
c_tree = a["kinfit_chi2"]
valid = (a["kinfit_status"] == 0) | (a["kinfit_status"] == 1)
st3   = (a["kinfit_status"] == 3)
fin = np.isfinite(c_jax) & np.isfinite(c_tree)
dc = (c_jax - c_tree)[fin & valid]
print("\n=== VALIDATION: JAX chi2(y*) reproduces tree kinfit_chi2 (valid evts) ===")
print(f"  median|Δchi2|={np.median(np.abs(dc)):.4f}  p95|Δ|={np.percentile(np.abs(dc),95):.4f}  "
      f"mean Δ={np.mean(dc):+.4f}  (N={dc.size})")
print(f"  status counts: valid(0/1)={valid.sum()} ({100*valid.mean():.1f}%)  "
      f"status3={st3.sum()} ({100*st3.mean():.1f}%)  other={N-valid.sum()-st3.sum()}")

# ── gradient + Hessian at y*, drop fixed gW (index 1) -> 15 free params ───────
KEEP = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
LBL  = ["mW","s1","s2","sl","ISRpx","t1","t2","ISRpy","p1","p2","ISRkz","tl","pl","besm","bespz"]
g = chunked(grad_fn, y, D, PR)[:, KEEP]                      # (N,15)
H = chunked(hess_fn, y, D, PR)[:, KEEP][:, :, KEEP]          # (N,15,15)
gnorm = np.linalg.norm(g, axis=1)

# symmetrize + eigendecompose (ascending eigenvalues)
Hs = 0.5*(H + np.transpose(H, (0,2,1)))
w, V = np.linalg.eigh(Hs)                                    # w:(N,15) asc, V:(N,15,15)
lam_min = w[:, 0]; lam_max = w[:, -1]                        # algebraic min/max
n_neg = np.sum(w < -1e-6, axis=1)                            # # negative-curvature dirs
i_flat = np.argmin(np.abs(w), axis=1)                        # FLATTEST = min |eigenvalue|
ar = np.arange(len(w))
lam_flat = w[ar, i_flat]                                     # flat-direction curvature
vflat = V[ar, :, i_flat]                                     # flat eigenvector
vneg = V[:, :, 0]                                            # most-negative-curvature eigenvector
lam_pos = np.where(w > 1e-9, w, np.inf).min(axis=1)          # smallest POSITIVE eigenvalue
cond = lam_max / np.where(lam_pos > 0, lam_pos, np.nan)
hess_finite = np.all(np.isfinite(w), axis=1)

def comp_str(vecs, m, top=7):
    comp = np.mean(np.abs(vecs[m]), axis=0)
    order = np.argsort(comp)[::-1]
    return "  ".join(f"{LBL[i]}={comp[i]:.2f}" for i in order[:top]), comp

def grp_stats(mask, tag):
    m = mask & hess_finite
    n = m.sum()
    print(f"\n--- {tag}  (N={n}) ---")
    print(f"  |grad|        : median={np.median(gnorm[m]):.3e}  p90={np.percentile(gnorm[m],90):.3e}")
    print(f"  #neg eigvals  : median={np.median(n_neg[m]):.0f}  mean={np.mean(n_neg[m]):.2f}  "
          f"frac(any neg)={100*np.mean(n_neg[m]>0):.1f}%")
    print(f"  lam_flat(|λ|min): median={np.median(np.abs(lam_flat[m])):.4f}  "
          f"frac(<0.1)={100*np.mean(np.abs(lam_flat[m])<0.1):.1f}%  frac(<0.5)={100*np.mean(np.abs(lam_flat[m])<0.5):.1f}%")
    print(f"  lam_min(algeb): median={np.median(lam_min[m]):+.4f}  p10={np.percentile(lam_min[m],10):.3e}")
    print(f"  lam_pos(min>0): median={np.median(lam_pos[m]):.4f}")
    print(f"  cond(λmax/λposmin): median={np.nanmedian(cond[m]):.2e}")
    s_flat,_ = comp_str(vflat, m); s_neg,_ = comp_str(vneg, m)
    print(f"  FLAT  eigvec |comp|: {s_flat}")
    print(f"  NEG   eigvec |comp|: {s_neg}")
    return None
vmin = vflat   # downstream overlap uses the FLAT direction

print("\n========== STEP 1: NAME THE FLAT DIRECTION ==========")
c_valid = grp_stats(valid, "VALID (status 0/1)")
c_st3   = grp_stats(st3,   "STATUS-3 (non-converged)")
# split status3 by edm
edm = a["kinfit_edm"]
grp_stats(st3 & np.isfinite(edm) & (edm < 1.0), "STATUS-3 / edm<1 (near-converged sliver)")
grp_stats(st3 & ~(np.isfinite(edm) & (edm < 1.0)), "STATUS-3 / edm>=1 or nonfinite (stranded)")

# ── overlap of soft eigenvector with named physics hypotheses ────────────────
def unit(v):
    v = np.array(v, float); return v/np.linalg.norm(v)
HYP = {
    "common-visible-scale (s1+s2+sl)": unit([0,1,1,1, 0,0,0,0, 0,0,0,0,0, 0,0]),
    "all-momenta-scale (mW+s1+s2+sl)": unit([1,1,1,1, 0,0,0,0, 0,0,0,0,0, 0,0]),
    "mW-only":                         unit([1,0,0,0, 0,0,0,0, 0,0,0,0,0, 0,0]),
    "longitudinal (ISRkz+bespz)":      unit([0,0,0,0, 0,0,0,0, 0,0,1,0,0, 0,1]),
    "scale-vs-mW (s1+s2+sl - mW)":     unit([-1,1,1,1, 0,0,0,0, 0,0,0,0,0, 0,0]),
}
print("\n--- soft eigenvector overlap with named hypotheses (mean |cosθ| over events) ---")
for tag, mask in [("VALID", valid), ("STATUS-3", st3)]:
    m = mask & hess_finite
    print(f"  [{tag}]")
    for name, h in HYP.items():
        ov = np.abs(vmin[m] @ h)
        print(f"     {name:38s}: mean|cos|={ov.mean():.3f}  p50={np.median(ov):.3f}")

# ── STEP 2: status-3 vs physics regime ───────────────────────────────────────
print("\n========== STEP 2: STATUS-3 vs PHYSICS REGIME ==========")
isr_p  = np.sqrt(a["gen_isr_px"]**2 + a["gen_isr_py"]**2 + a["gen_isr_pz"]**2)
nu_pz  = a["gen_nu_p"]*a["gen_nu_costheta"] if "gen_nu_costheta" in a else None
feat = {
    "gen_isr_p":            isr_p,
    "|gen_isr_pz|":         np.abs(a["gen_isr_pz"]),
    "gen_WW_p_imbalance":   a.get("gen_WW_p_imbalance_tot"),
    "gen_Whad_p":           a.get("gen_Whad_p"),
    "gen_Wlep_p":           a.get("gen_Wlep_p"),
    "|gen_Whad_costheta|":  np.abs(a["gen_Whad_costheta"]) if "gen_Whad_costheta" in a else None,
    "|gen_lep_costheta|":   np.abs(a["gen_lep_costheta"]) if "gen_lep_costheta" in a else None,
    "gen_nu_p":             a.get("gen_nu_p"),
    "|gen_nu_pz|":          np.abs(nu_pz) if nu_pz is not None else None,
    "|jet1_p_resp|":        np.abs(a["jet1_p_resp"]) if "jet1_p_resp" in a else None,
    "|jet2_p_resp|":        np.abs(a["jet2_p_resp"]) if "jet2_p_resp" in a else None,
    "jet1_matched_q_dR":    a.get("jet1_matched_q_dR"),
    "jet2_matched_q_dR":    a.get("jet2_matched_q_dR"),
    "|jet1_theta_resol|":   np.abs(a["jet1_theta_resol"]) if "jet1_theta_resol" in a else None,
}
print(f"  {'feature':22s} {'mean(valid)':>12s} {'mean(st3)':>12s} {'Δ/σ':>8s} {'P(st3|topdecile)':>16s} {'P(st3|botdecile)':>16s}")
base = st3.mean()
rows = []
for name, x in feat.items():
    if x is None: continue
    x = np.asarray(x, float); fm = np.isfinite(x)
    mv = x[fm & valid].mean(); ms = x[fm & st3].mean()
    sd = x[fm].std() + 1e-12
    dsig = (ms - mv)/sd
    q90 = np.percentile(x[fm], 90); q10 = np.percentile(x[fm], 10)
    top = fm & (x >= q90); bot = fm & (x <= q10)
    p_top = st3[top].mean(); p_bot = st3[bot].mean()
    rows.append((abs(dsig), name, mv, ms, dsig, p_top, p_bot))
for _, name, mv, ms, dsig, p_top, p_bot in sorted(rows, reverse=True):
    print(f"  {name:22s} {mv:12.4f} {ms:12.4f} {dsig:8.2f} {p_top:16.3f} {p_bot:16.3f}")
print(f"  [baseline P(status-3) = {base:.3f}]")

# correlation of fitted Whad/Wlep with valid vs st3 (the low-W-mass stranding check)
for nm in ["kinfit_Whad_m","kinfit_Wlep_m"]:
    if nm in have:
        x = t.arrays([nm], library="np")[nm][idx]
        print(f"  {nm}: valid mean={x[valid].mean():.3f}  st3 mean={x[st3].mean():.3f}")

np.savez(os.path.join(os.path.dirname(__file__), "flatdir_out.npz"),
         lam_min=lam_min, lam_max=lam_max, lam_flat=lam_flat, lam_pos=lam_pos,
         n_neg=n_neg, vflat=vflat, vneg=vneg, gnorm=gnorm,
         status=a["kinfit_status"], edm=edm, valid=valid, st3=st3,
         y=y, isr_p=isr_p, LBL=np.array(LBL))
print("\n[saved jax_prototype/flatdir_out.npz]")
