#!/usr/bin/env python3
# ── HARD-CONSTRAINT gen fit: remove ISR/BES, fix ν by 3-momentum conservation, fit mW ──
# (per user: "implement the hard constraint and remove the ISR/BES in the gen level fit,
#  plot the mW distribution, check convergence".)
#
# Exact objects: hadronic W = the two gen quarks (fixed), lepton = gen muon (fixed). NO ISR, NO BES.
# Hard 3-momentum conservation at (√s,0,0,0): neutrino is FULLY DETERMINED, p_ν = −p_visible,
# E_ν = |p_ν| (massless). => m_had and m_lep are BOTH fixed per event, NO nuisances at all. The
# ONLY free parameter is mW, pinned by the double Breit-Wigner (+ phase-space normalization log_Z).
# This is the hard-constraint analog of the soft ISR/BES fit: removing the soft freedom removes the
# flat direction by construction. We scan the 1-D mW likelihood per event, take m̂W = argmin, and
# check (a) convergence = clean interior minimum, (b) the m̂W distribution (bias/resolution vs pole).
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   MAXN=20000 python3 jax_prototype/kinfit_gen_hard.py [root] [ecm]
import sys, os
os.environ.setdefault("XLA_FLAGS","--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, jax, jax.numpy as jnp, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import jaxfit_common as J
jax.config.update("jax_enable_x64", True); np.set_printoptions(linewidth=160, suppress=True)
MW_INIT=80.419; GW=2.049; MU=0.1056583745
ROOT=sys.argv[1] if len(sys.argv)>1 else \
  "outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160; MAXN=int(os.environ.get("MAXN","20000"))
OUT="/eos/user/m/mdefranc/www/mW/kinfit_correlations"

t=uproot.open(ROOT)["events"]
br=["gen_quark1_p","gen_quark1_theta","gen_quark1_phi","gen_quark2_p","gen_quark2_theta","gen_quark2_phi",
    "gen_lep_p","gen_lep_theta","gen_lep_phi","gen_Whad_m","gen_Wlep_m",
    "gen_isr_px","gen_isr_py","gen_isr_pz"]
a=t.arrays(br,library="np"); ok=(a["gen_quark1_p"]>0)&(a["gen_quark2_p"]>0)&(a["gen_lep_p"]>0)
idx=np.where(ok)[0];
if MAXN>0: idx=idx[:MAXN]
a={k:v[idx] for k,v in a.items()}; N=len(idx)

def vec(p,th,ph,m=0.0):
    st=np.sin(th); E=np.sqrt(p*p+m*m)
    return np.stack([E,p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th)],axis=1)
# quarks massless (like reco jets) so m_had uses directions+momenta; gen_Whad_m available for ref
Q1=vec(a["gen_quark1_p"],a["gen_quark1_theta"],a["gen_quark1_phi"])
Q2=vec(a["gen_quark2_p"],a["gen_quark2_theta"],a["gen_quark2_phi"])
L =vec(a["gen_lep_p"],a["gen_lep_theta"],a["gen_lep_phi"],MU)
Wh=Q1+Q2
# HARD 3-momentum conservation, NO ISR/BES: ν = -(Wh+L) 3-mom, E_ν=|p_ν| (massless)
Vis=Wh+L
nu3=-Vis[:,1:4]; Enu=np.linalg.norm(nu3,axis=1)
Nu=np.concatenate([Enu[:,None],nu3],axis=1)
Wl=L+Nu; WW=Wh+Wl
def M(v): return np.sqrt(np.maximum(v[:,0]**2-(v[:,1]**2+v[:,2]**2+v[:,3]**2),1e-12))
mhad=M(Wh); mlep=M(Wl); mWW=M(WW)
isr_p=np.sqrt(a["gen_isr_px"]**2+a["gen_isr_py"]**2+a["gen_isr_pz"]**2)
print(f"[gen-hard] N={N} ecm{ECM}  m_had mean={mhad.mean():.3f}({mhad.std():.2f})  "
      f"m_lep(hard) mean={mlep.mean():.3f}({mlep.std():.2f})  gen_Wlep_m mean={a['gen_Wlep_m'].mean():.3f}")

# 1-D mW likelihood per event: -2log bw_h -2log bw_l + 2 log_Z(m_WW,mW,gW). (phase-space lam/s_ww
# terms are mW-independent here -> dropped.)  m_had, m_lep, m_WW fixed.
logZ=jax.jit(jax.vmap(lambda mww,mw: J.log_Z_ontf(mww,mw,GW)))
grid=np.linspace(74.0,86.0,241)
def chi2_mW(mw):
    mwgw=mw*GW
    dh=mhad*mhad-mw*mw; dl=mlep*mlep-mw*mw
    bwh=mwgw/(dh*dh+mwgw*mwgw); bwl=mwgw/(dl*dl+mwgw*mwgw)
    lz=np.asarray(logZ(jnp.asarray(mWW),jnp.full(N,mw)))
    return -2.0*(np.log(bwh)+np.log(bwl))+2.0*lz
C=np.stack([chi2_mW(mw) for mw in grid],axis=1)     # (N, ngrid)
fin=np.all(np.isfinite(C),axis=1); C=C[fin]; Nf=fin.sum()
imin=np.argmin(C,axis=1)
interior=(imin>0)&(imin<len(grid)-1)
# parabolic refine + convexity at min
def refine(row,i):
    i=int(np.clip(i,1,len(grid)-2)); x=grid[i-1:i+2]; y=row[i-1:i+2]
    c=np.polyfit(x,y,2);
    return (-c[1]/(2*c[0]) if c[0]>0 else grid[i]), c[0]
mhat=np.empty(Nf); curv=np.empty(Nf)
for k in range(Nf): mhat[k],curv[k]=refine(C[k],imin[k])
conv=interior&(curv>0)
mhat=np.clip(mhat,grid[0],grid[-1])
print(f"\n=== HARD-CONSTRAINT (no ISR/BES) 1-param mW fit ===")
print(f"  CONVERGENCE: interior-min={100*interior.mean():.2f}%  convex-at-min={100*np.mean(curv>0):.2f}%  "
      f"clean(both)={100*conv.mean():.2f}%")
print(f"  m̂W (conv): mean={mhat[conv].mean():.3f}  median={np.median(mhat[conv]):.3f}  std={mhat[conv].std():.3f}  "
      f"  bias vs 80.385(true pole)={mhat[conv].mean()-80.385:+.3f}")
print(f"  per-event σ_mW (1/sqrt curv, conv): median={np.median(1/np.sqrt(curv[conv])):.3f} GeV")
# ISR dependence of bias
isrf=isr_p[fin]
for lab,sel in [("ISR<0.2",isrf<0.2),("0.2<ISR<1",(isrf>=0.2)&(isrf<1)),("ISR>1",isrf>=1)]:
    s=sel&conv
    print(f"    {lab:11s} N={s.sum():5d}  m̂W mean={mhat[s].mean():.3f} std={mhat[s].std():.3f}")

# ── plot mW distribution ──
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].hist(mhat[conv],bins=120,range=(74,86),histtype="step",lw=2,color="C0",
           label=f"hard-constraint, no ISR/BES\nmean={mhat[conv].mean():.2f}, std={mhat[conv].std():.2f}")
ax[0].axvline(80.385,color="k",ls="--",lw=1,label="gen pole 80.385")
ax[0].set_xlabel(r"per-event $\hat m_W$ [GeV]"); ax[0].set_ylabel("events")
ax[0].set_title(f"Gen hard-constraint mW (ecm{ECM}, N={conv.sum()})\nconvergence={100*conv.mean():.1f}%")
ax[0].legend(fontsize=9)
# m_had vs m_lep(hard) to show the inputs
ax[1].hist(mhad[fin][conv],bins=100,range=(60,95),histtype="step",lw=2,label=f"m_had (dijet) {mhad[fin][conv].mean():.1f}")
ax[1].hist(mlep[fin][conv],bins=100,range=(60,95),histtype="step",lw=2,label=f"m_lep (hard ν) {mlep[fin][conv].mean():.1f}")
ax[1].axvline(80.385,color="k",ls="--",lw=1); ax[1].set_xlabel("mass [GeV]"); ax[1].set_ylabel("events")
ax[1].set_title("Fixed W masses fed to the mW fit"); ax[1].legend(fontsize=9)
plt.tight_layout(); png=f"{OUT}/mW_hard_gen_ecm{ECM}.png"; plt.savefig(png,dpi=110)
print(f"\n[plot] {png}")

# ── mW distribution in BINS of gen ISR energy ──
POLE=80.385
edges=[0.0,0.1,0.3,0.7,1.5,3.0,1e9]
labels=["ISR<0.1","0.1–0.3","0.3–0.7","0.7–1.5","1.5–3","ISR>3"]
isrc=isr_p[fin]                          # align ISR to the finite-fit events (mhat/conv arrays)
fig,ax=plt.subplots(1,2,figsize=(14,5.2))
cmap=plt.cm.viridis(np.linspace(0,0.92,len(labels)))
means=[]; stds=[]; meds=[]; cents=[]; ns=[]
for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
    sel=conv&(isrc>=lo)&(isrc<hi); n=int(sel.sum())
    if n<30: means.append(np.nan); stds.append(np.nan); meds.append(np.nan); cents.append(np.nan); ns.append(n); continue
    mw=mhat[sel]; m=mw.mean(); s=mw.std(); md=np.median(mw)
    means.append(m); stds.append(s); meds.append(md); ns.append(n)
    cents.append(np.median(isrc[sel]))
    ax[0].hist(mw,bins=80,range=(76,85),density=True,histtype="step",lw=2,color=cmap[k],
               label=f"{labels[k]} GeV  (N={n})  μ={m:.2f} σ={s:.2f}")
ax[0].axvline(POLE,color="k",ls="--",lw=1,label=f"pole {POLE}")
ax[0].set_xlabel(r"per-event $\hat m_W$ [GeV]"); ax[0].set_ylabel("normalized"); ax[0].set_xlim(76,85)
ax[0].set_title(f"Gen hard-constraint $\\hat m_W$ in ISR bins (ecm{ECM})"); ax[0].legend(fontsize=8.5)
# bias & resolution vs ISR
means=np.array(means); stds=np.array(stds); cents=np.array(cents); meds=np.array(meds)
g=np.isfinite(means)
ax[1].errorbar(cents[g],means[g]-POLE,yerr=stds[g]/np.sqrt(np.array(ns)[g]),fmt="o-",color="C3",capsize=3,label="mean − pole (±stat err)")
ax[1].plot(cents[g],meds[g]-POLE,"s--",color="C0",label="median − pole")
ax2=ax[1].twinx(); ax2.plot(cents[g],stds[g],"^:",color="C2",label="std (resolution)")
ax[1].axhline(0,color="k",lw=0.8); ax[1].set_xscale("symlog",linthresh=0.1)
ax[1].set_xlabel("gen ISR |p| [GeV] (bin median, symlog)"); ax[1].set_ylabel(r"$\hat m_W$ bias [GeV]",color="C3")
ax2.set_ylabel(r"$\hat m_W$ std [GeV]",color="C2"); ax[1].set_title("mW bias & resolution vs ISR")
ax[1].legend(fontsize=9,loc="lower left"); ax2.legend(fontsize=9,loc="upper right")
plt.tight_layout(); png2=f"{OUT}/mW_hard_gen_isrbins_ecm{ECM}.png"; plt.savefig(png2,dpi=110)
print(f"[plot] {png2}")
print("\n  ISR bin       N     mean     median    std")
for k in range(len(labels)):
    if np.isfinite(means[k]): print(f"  {labels[k]:9s} {ns[k]:6d}  {means[k]:7.3f}  {meds[k]:7.3f}  {stds[k]:6.3f}")
