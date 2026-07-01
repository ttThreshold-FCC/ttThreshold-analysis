#!/usr/bin/env python3
# ── DAY6 NEXT#1: a DATA-APPLICABLE handle for the deep-ISR / sub-threshold (√s'<157) tail at ecm163 ──
# DAY5 established: exact@163 closure −27 MeV is carried ENTIRELY by the gen √s'<157 tail; a gen-√s'
# LOWER CUT moves the fit +38..41 MeV (closure −27→+6).  But √s'=gen_WW_m is NOT observable in data.
# This script asks: which RECO observable tags that gen tail well enough to reproduce the cut?
#
# Candidates (all reco, all in the tree):
#   reco_WW_m              — reco analog of gen_WW_m (visible+MET invariant mass), ISR lowers it     [low→tail]
#   reco_WW_m_minus_ecm    — same, deficit form                                                       [low→tail]
#   |reco_WW_pz|           — net longitudinal momentum: hard ISR recoils the WW system along beam    [high→tail]
#   reco_WW_p_imbalance_tot— total reco momentum imbalance                                            [high→tail]
#   acol(Whad,Wlep)        — 180°−openingAngle: ISR boost breaks the back-to-back topology           [high→tail]
#   reco_met_p             — MET (neutrino + ISR pT)                                                  [high→tail]
#   kinfit_WW_m            — kinfit WW mass (likely pinned at ECM by the energy constraint → useless) [low→tail]
#
# For each: AUC for tagging y=(gen_WW_m<TAIL), and purity/efficiency/mean-gen-√s' at the operating point
# that removes the SAME fraction as the gen cut.  Also confirm the mechanism (gen ISR energy vs gen_WW_m).
#
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# OPENBLAS_NUM_THREADS=4 python3 jax_prototype/conv_mw_isr_proxy.py [ecm]
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.set_printoptions(linewidth=160, suppress=True)

ECM      = int(sys.argv[1]) if len(sys.argv) > 1 else 163
TAIL     = float(os.environ.get("TAIL", "157.0"))    # the DAY5 diagnostic gen-√s' cut
MW_REF   = float(os.environ.get("MW_REF", "80.379"))
ROOT = f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW, exist_ok=True)
import uproot

br = ["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_Whad_m","reco_Wlep_m",
      "reco_jet1_p","reco_jet2_p","reco_lep_p",
      "reco_WW_m","reco_WW_m_minus_ecm","reco_WW_pz","reco_WW_px","reco_WW_py","reco_WW_p_imbalance_tot",
      "reco_met_p",
      "reco_Whad_costheta","reco_Whad_phi","reco_Wlep_costheta","reco_Wlep_phi",
      "kinfit_WW_m","kinfit_valid",
      "gen_isr_px","gen_isr_py","gen_isr_pz"]
t = uproot.open(ROOT)["events"]
a = t.arrays(br, library="np")
ok = (np.isfinite(a["gen_Whad_m"]) & np.isfinite(a["gen_Wlep_m"]) & np.isfinite(a["gen_WW_m"]) &
      np.isfinite(a["reco_Whad_m"]) & np.isfinite(a["reco_Wlep_m"]) &
      (a["reco_jet1_p"]>0) & (a["reco_jet2_p"]>0) & (a["reco_lep_p"]>0) &
      (a["gen_Whad_m"]+a["gen_Wlep_m"] < a["gen_WW_m"]))
a = {k: v[ok] for k, v in a.items()}; N = len(a["gen_WW_m"])

gwm = a["gen_WW_m"]
tail = gwm < TAIL
frac = tail.mean()
isr_E = np.sqrt(a["gen_isr_px"]**2 + a["gen_isr_py"]**2 + a["gen_isr_pz"]**2)   # |p_ISR| ≈ E_ISR (massless γ)

# acolinearity between the two reco W's: 180° − opening angle (0 = back-to-back, grows with ISR boost)
def vec(ct, phi):
    st = np.sqrt(np.maximum(1.0-ct*ct, 0.0))
    return np.stack([st*np.cos(phi), st*np.sin(phi), ct], 1)
v1 = vec(a["reco_Whad_costheta"], a["reco_Whad_phi"]); v2 = vec(a["reco_Wlep_costheta"], a["reco_Wlep_phi"])
cos_open = np.clip((v1*v2).sum(1), -1, 1)
acol = 180.0 - np.degrees(np.arccos(cos_open))     # deg; 0=back-to-back

print(f"\n{'='*92}\n[isr-proxy] ecm{ECM}  N={N}  TAIL=gen_WW_m<{TAIL}  frac_tail={100*frac:.2f}%  "
      f"<gen√s'>={gwm.mean():.3f}\n{'='*92}")
print(f"  gen ISR |p| : tail<{TAIL} mean={isr_E[tail].mean():.2f} GeV  vs  rest mean={isr_E[~tail].mean():.2f} GeV")

# ── proxies: (name, value, orientation higher-means-more-tail) ─────────────────────────────────────
proxies = [
    ("reco_WW_m",            a["reco_WW_m"],               -1),
    ("reco_WW_m_minus_ecm",  a["reco_WW_m_minus_ecm"],     -1),
    ("abs_reco_WW_pz",       np.abs(a["reco_WW_pz"]),      +1),
    ("reco_WW_p_imbal",      a["reco_WW_p_imbalance_tot"], +1),
    ("acol_Whad_Wlep",       acol,                         +1),
    ("reco_met_p",           a["reco_met_p"],              +1),
    ("kinfit_WW_m",          a["kinfit_WW_m"],             -1),
]

def auc(score, y):
    # higher score = more positive(tail).  rank-based AUC.
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score)+1)
    # average ranks for ties
    s_sorted = score[order]; i = 0
    while i < len(s_sorted):
        j = i
        while j+1 < len(s_sorted) and s_sorted[j+1] == s_sorted[i]: j += 1
        if j > i: r[order[i:j+1]] = 0.5*(i+1 + j+1)
        i = j+1
    npos = y.sum(); nneg = len(y)-npos
    return (r[y].sum() - npos*(npos+1)/2.0) / (npos*nneg)

results = {}
nrm = int(round(frac*N))   # remove the same NUMBER as the gen cut
print(f"\n  operating point = remove top {nrm} events (= {100*frac:.2f}%, matching the gen cut)\n")
print(f"  {'proxy':22s} {'AUC':>6s} {'purity':>7s} {'effic':>7s} {'<g√s'+chr(39)+'>rm':>9s} {'<g√s'+chr(39)+'>keep':>11s} {'corr':>7s}")
for name, val, ori in proxies:
    s = ori*val
    A = auc(s, tail)
    thr = np.sort(s)[::-1][nrm-1]
    removed = s >= thr
    # trim to exactly nrm if ties overshoot
    if removed.sum() > nrm:
        idx = np.where(removed)[0]; removed = np.zeros(N, bool); removed[idx[np.argsort(-s[idx])][:nrm]] = True
    purity = tail[removed].mean()                       # of removed, frac truly in tail
    effic  = removed[tail].mean()                       # of tail, frac removed
    corr   = np.corrcoef(val, gwm)[0,1]
    results[name] = dict(auc=float(A), purity=float(purity), effic=float(effic),
                         mean_gsqrts_removed=float(gwm[removed].mean()),
                         mean_gsqrts_kept=float(gwm[~removed].mean()), corr=float(corr), ori=ori)
    print(f"  {name:22s} {A:6.3f} {100*purity:6.1f}% {100*effic:6.1f}% {gwm[removed].mean():9.2f} "
          f"{gwm[~removed].mean():11.3f} {corr:+7.3f}")

# ── plots ───────────────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 3, figsize=(17, 10))
# (a) gen ISR energy vs gen √s'
ax[0,0].hexbin(gwm, isr_E, gridsize=60, mincnt=1, bins="log", cmap="viridis")
ax[0,0].axvline(TAIL, color="r", ls="--"); ax[0,0].axvline(2*MW_REF, color="orange", ls=":")
ax[0,0].set_xlabel("gen √s' = gen_WW_m [GeV]"); ax[0,0].set_ylabel("gen |p_ISR| [GeV]")
ax[0,0].set_title(f"ecm{ECM}: tail (√s'<{TAIL}) IS hard ISR")
# (b) reco_WW_m vs gen_WW_m (the headline proxy)
ax[0,1].hexbin(gwm, a["reco_WW_m"], gridsize=60, mincnt=1, bins="log", cmap="viridis")
ax[0,1].axvline(TAIL, color="r", ls="--"); ax[0,1].plot([140,170],[140,170],"w-",lw=0.6)
ax[0,1].set_xlabel("gen √s' [GeV]"); ax[0,1].set_ylabel("reco_WW_m [GeV]"); ax[0,1].set_title("reco_WW_m vs gen √s'")
# (c) |reco_WW_pz| vs gen √s'
ax[0,2].hexbin(gwm, np.abs(a["reco_WW_pz"]), gridsize=60, mincnt=1, bins="log", cmap="viridis")
ax[0,2].axvline(TAIL, color="r", ls="--")
ax[0,2].set_xlabel("gen √s' [GeV]"); ax[0,2].set_ylabel("|reco_WW_pz| [GeV]"); ax[0,2].set_title("ISR longitudinal recoil")
# (d) ROC curves
for name, val, ori in proxies:
    s = ori*val; thr_grid = np.quantile(s, np.linspace(0,1,200))
    tpr = [ (s>=th)[tail].mean() for th in thr_grid ]
    fpr = [ (s>=th)[~tail].mean() for th in thr_grid ]
    ax[1,0].plot(fpr, tpr, label=f"{name} ({results[name]['auc']:.3f})", lw=1.3)
ax[1,0].plot([0,1],[0,1],"k:",lw=0.6); ax[1,0].set_xlabel("FPR"); ax[1,0].set_ylabel("TPR (tail eff)")
ax[1,0].set_title(f"ROC: tag gen √s'<{TAIL}"); ax[1,0].legend(fontsize=7, loc="lower right")
# (e) reco_WW_m distributions tail vs rest
bins = np.linspace(a["reco_WW_m"].min(), a["reco_WW_m"].max(), 80)
ax[1,1].hist(a["reco_WW_m"][~tail], bins=bins, density=True, histtype="step", label="rest", color="C0")
ax[1,1].hist(a["reco_WW_m"][tail],  bins=bins, density=True, histtype="step", label=f"tail √s'<{TAIL}", color="C3")
ax[1,1].set_xlabel("reco_WW_m [GeV]"); ax[1,1].set_ylabel("norm"); ax[1,1].legend(fontsize=8); ax[1,1].set_title("separation")
# (f) kinfit_WW_m (is it pinned at ECM?)
ax[1,2].hist(a["kinfit_WW_m"][np.isfinite(a["kinfit_WW_m"])], bins=80, histtype="step", color="C2")
ax[1,2].axvline(ECM, color="orange", ls=":")
ax[1,2].set_xlabel("kinfit_WW_m [GeV]"); ax[1,2].set_ylabel("count"); ax[1,2].set_title("kinfit_WW_m (pinned?)")
fig.suptitle(f"DAY6 reco ISR-tail proxies — ecm{ECM}", fontsize=13)
fig.tight_layout()
png = f"{EOSW}/isr_proxy_ecm{ECM}.png"; fig.savefig(png, dpi=110); print(f"\n[plot] {png}")

with open(f"jax_prototype/anatomy_results/isr_proxy_ecm{ECM}.json","w") as fh:
    json.dump(dict(ecm=ECM, N=N, tail_cut=TAIL, frac_tail=float(frac),
                   isr_tail=float(isr_E[tail].mean()), isr_rest=float(isr_E[~tail].mean()),
                   results=results), fh, indent=2)
print(f"[json] jax_prototype/anatomy_results/isr_proxy_ecm{ECM}.json")
