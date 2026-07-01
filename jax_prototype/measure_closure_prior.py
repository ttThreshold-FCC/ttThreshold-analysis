#!/usr/bin/env python3
# Measure the kfit ENERGY-CLOSURE prior from gen, to replace the tuned σ_R.
#
# kfit closure at truth:  r = (ECM+besm) − E_vis − E_ν − E_γ
#   With momentum conserved (ν derived) and a massless single ISR photon E_γ=|p_γ|,
#   this collapses to  r = E_ISR − |p_ISR| = the ISR-system INVARIANT MASS, where
#   E_ISR = E_ee − E_WW (energy conservation).  So r is a property of the ISR photons
#   ONLY — BES is carried separately by besm/besz and should NOT appear in r.
#
# This script:
#   1. computes r_gen = E_ISR − |p_ISR| (the NEW closure prior, at truth)
#   2. compares to the OLD closure gen_WW_m_minus_m_ee (= m_WW − m_ee, which BUNDLED BES+ISR)
#   3. checks BES bookkeeping: corr(r, BES) — should be ~0 for the new, nonzero for old
#   4. a reco proxy (reco jets/lep + TRUE ISR/BES) to see how much reco resolution
#      broadens the closure (the source of the σ_R≈1.5 the fit wanted)
#   5. characterizes the lineshape (one-sided? spike+tail?) and overlays σ_R=0.45/1.5
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/measure_closure_prior.py [root] [ecm]
import sys, os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT=sys.argv[1] if len(sys.argv)>1 else "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160
t=uproot.open(ROOT)["events"]
br=["gen_ee_m_minus_ecm","gen_ee_pz","gen_WW_m","gen_WW_px","gen_WW_py","gen_WW_pz",
    "gen_WW_m_minus_m_ee","gen_isr_px","gen_isr_py","gen_isr_pz",
    "reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
    "reco_lep_p","reco_lep_theta","reco_lep_phi"]
a=t.arrays(br,library="np")
N=len(a["gen_WW_m"])

# ── initial ee state (with BES) ──────────────────────────────────────────────
bes = a["gen_ee_m_minus_ecm"]                 # BES energy fluctuation (= m_ee − ECM)
m_ee = ECM + bes
ee_pz = a["gen_ee_pz"]
E_ee = np.sqrt(m_ee**2 + ee_pz**2)            # lab energy of the ee system

# ── WW system (visible + neutrino) at gen ────────────────────────────────────
WWp2 = a["gen_WW_px"]**2 + a["gen_WW_py"]**2 + a["gen_WW_pz"]**2
E_WW = np.sqrt(a["gen_WW_m"]**2 + WWp2)

# ── ISR system ───────────────────────────────────────────────────────────────
isr_p = np.sqrt(a["gen_isr_px"]**2 + a["gen_isr_py"]**2 + a["gen_isr_pz"]**2)
E_ISR = E_ee - E_WW                            # energy conservation
r_gen = E_ISR - isr_p                          # = ISR-system invariant mass = NEW closure

# consistency: is gen_isr the recoil p_ee − p_WW?
dpx=(-a["gen_WW_px"])-a["gen_isr_px"]; dpy=(-a["gen_WW_py"])-a["gen_isr_py"]; dpz=(ee_pz-a["gen_WW_pz"])-a["gen_isr_pz"]
print(f"[lnuqq ecm{ECM}] N={N}")
print(f"momentum-recoil consistency |p_ee−p_WW − gen_isr|: med px/py/pz = "
      f"{np.median(np.abs(dpx)):.2e}/{np.median(np.abs(dpy)):.2e}/{np.median(np.abs(dpz)):.2e} GeV")

# ── OLD closure variable (mloss prior) ───────────────────────────────────────
r_old = a["gen_WW_m_minus_m_ee"]              # m_WW − m_ee (bundled BES+ISR mass deficit)

def stats(x,name):
    fin=np.isfinite(x); x=x[fin]
    print(f"  [{name}] mean={np.mean(x):+.3f} med={np.median(x):+.3f} std={np.std(x):.3f} "
          f"RMS={np.sqrt(np.mean(x**2)):.3f}  p1/p50/p90/p99={np.percentile(x,[1,50,90,99])}")
    print(f"        frac |x|<0.1={100*np.mean(np.abs(x)<0.1):.1f}%  <0.5={100*np.mean(np.abs(x)<0.5):.1f}%  "
          f"x<0={100*np.mean(x<0):.1f}%  x>2={100*np.mean(x>2):.1f}%")
    return x

print("\n=== NEW closure  r_gen = E_ISR − |p_ISR|  (ISR-system mass; the suitable prior) ===")
rg=stats(r_gen,"r_gen")
print("=== OLD closure  gen_WW_m_minus_m_ee  (= m_WW − m_ee, bundled BES+ISR) ===")
ro=stats(r_old,"r_old")

# ── BES bookkeeping: correlation of each closure with BES ─────────────────────
def corr(x,y):
    f=np.isfinite(x)&np.isfinite(y); return np.corrcoef(x[f],y[f])[0,1]
print("\n=== BES bookkeeping (should be ~0 for r_gen if BES is fully in besm) ===")
print(f"  corr(r_gen, BES)={corr(r_gen,bes):+.3f}   corr(r_old, BES)={corr(r_old,bes):+.3f}")
print(f"  corr(r_gen, |p_ISR|)={corr(r_gen,isr_p):+.3f}  (mass grows with harder/multi-photon ISR)")

# ── reco proxy: reco jets/lep + TRUE ISR & BES → how reco resolution broadens r ──
def vec(p,th,ph):
    st=np.sin(th); return np.stack([p, p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th)],axis=0)
J1=vec(a["reco_jet1_p"],a["reco_jet1_theta"],a["reco_jet1_phi"])
J2=vec(a["reco_jet2_p"],a["reco_jet2_theta"],a["reco_jet2_phi"])
L =vec(a["reco_lep_p"], a["reco_lep_theta"], a["reco_lep_phi"])
Vis=J1+J2+L
pgx,pgy,pgz=a["gen_isr_px"],a["gen_isr_py"],a["gen_isr_pz"]; Eg=isr_p
nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=ee_pz-Vis[3]-pgz
Enu=np.sqrt(nux**2+nuy**2+nuz**2)
r_reco=(ECM+bes)-Vis[0]-Enu-Eg
ok=np.isfinite(r_reco)&(np.abs(r_reco)<200)
print("\n=== reco proxy  (reco jets/lep, TRUE ISR & BES; isolates reco-resolution leakage) ===")
rr=stats(r_reco[ok],"r_reco")

# ── plot ─────────────────────────────────────────────────────────────────────
fig,ax=plt.subplots(1,3,figsize=(17,5))
ax[0].hist(np.clip(rg,-1,3),bins=120,range=(-1,3),histtype="step",lw=2,color="C0",density=True,label=f"r_gen (std={np.std(rg):.2f})")
xs=np.linspace(-1,3,400)
for sr,c in [(0.45,"C3"),(1.5,"C2")]:
    ax[0].plot(xs,np.exp(-0.5*(xs/sr)**2)/(sr*np.sqrt(2*np.pi)),c,ls="--",lw=1.3,label=f"Gauss σ={sr}")
ax[0].set_yscale("log"); ax[0].set_xlabel("r_gen = E_ISR − |p_ISR|  [GeV]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_title("NEW closure prior (truth) — one-sided ISR-system mass")
ax[1].hist(np.clip(rg,-2,6),bins=120,range=(-2,6),histtype="step",lw=2,color="C0",density=True,label=f"r_gen std={np.std(rg):.2f}")
ax[1].hist(np.clip(rr,-2,6),bins=120,range=(-2,6),histtype="step",lw=2,color="C1",density=True,label=f"r_reco std={np.std(rr):.2f}")
ax[1].hist(np.clip(ro,-2,6),bins=120,range=(-2,6),histtype="step",lw=2,color="C4",density=True,label=f"r_old (m_WW−m_ee) std={np.std(ro):.2f}")
ax[1].set_yscale("log"); ax[1].set_xlabel("closure [GeV]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
ax[1].set_title("gen vs reco-proxy vs OLD closure")
ax[2].scatter(bes[:4000],r_gen[:4000],s=3,alpha=.3,label=f"r_gen (corr={corr(r_gen,bes):+.2f})")
ax[2].scatter(bes[:4000],r_old[:4000],s=3,alpha=.3,color="C4",label=f"r_old (corr={corr(r_old,bes):+.2f})")
ax[2].set_xlabel("BES = m_ee − ECM [GeV]"); ax[2].set_ylabel("closure [GeV]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
ax[2].set_title("BES bookkeeping: closure vs BES")
fig.suptitle(f"lnuqq ecm{ECM}: kfit energy-closure prior  |  r_gen std={np.std(rg):.2f} (truth, one-sided)  "
             f"r_reco std={np.std(rr):.2f}  |  corr(r_gen,BES)={corr(r_gen,bes):+.2f}",fontsize=11)
fig.tight_layout(rect=[0,0,1,0.95])
out="/eos/user/m/mdefranc/www/mW/jax_lnuqq"; os.makedirs(out,exist_ok=True)
fig.savefig(f"{out}/closure_prior_ecm{ECM}.png",dpi=110)
print(f"\nwrote {out}/closure_prior_ecm{ECM}.png")
