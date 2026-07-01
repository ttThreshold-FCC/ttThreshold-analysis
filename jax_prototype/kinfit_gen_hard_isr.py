#!/usr/bin/env python3
# ── ISR MITIGATION for the hard-constraint gen fit: collinear ISR photon ──────
# The no-ISR hard constraint (kinfit_gen_hard.py) fixes ν by 3-momentum conservation assuming
# initial (√s,0,0,0); it degrades resolution for high-ISR events (σ 1.45→2.5 GeV). Mitigation:
# allow ONE collinear ISR photon along ±z (the physical ISR direction) and USE energy conservation
# (which the no-ISR version dropped) to recover it. Then ν is fixed by full 4-momentum conservation:
#   ν_x=-Vis_x, ν_y=-Vis_y ;  ν_z=-Vis_z-g ;  E_ν=S-|g|  with  E_ν=|ν|  (massless) ,  S=√s-E_vis.
# The massless condition (S-|g|)²=ν_x²+ν_y²+(Vis_z+g)² is LINEAR in g (k² cancels) → two branches:
#   g_+ = num/(2(S+Vis_z))  [photon +z, valid if g_+≥0]   g_- = num/(2(Vis_z-S)) [photon -z, g_-<0]
#   num = S² - (Vis_x²+Vis_y²) - Vis_z².
# Ambiguity (±z) resolved two ways: (BEST) branch closest to true gen_isr_pz; (REAL) smaller |g|.
# Compare m̂W bias & resolution in ISR bins vs the no-ISR baseline. Convergence reported per variant.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   MAXN=0 python3 jax_prototype/kinfit_gen_hard_isr.py [root] [ecm]
import sys, os
os.environ.setdefault("XLA_FLAGS","--xla_force_host_platform_device_count=1"); os.environ.setdefault("OMP_NUM_THREADS","8")
import numpy as np, jax, jax.numpy as jnp, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); import jaxfit_common as J
jax.config.update("jax_enable_x64", True); np.set_printoptions(linewidth=160, suppress=True)
GW=2.049; MU=0.1056583745; POLE=80.385
ROOT=sys.argv[1] if len(sys.argv)>1 else \
  "outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=float(sys.argv[2]) if len(sys.argv)>2 else 160.0; MAXN=int(os.environ.get("MAXN","0"))
OUT="/eos/user/m/mdefranc/www/mW/kinfit_correlations"

t=uproot.open(ROOT)["events"]
br=["gen_quark1_p","gen_quark1_theta","gen_quark1_phi","gen_quark2_p","gen_quark2_theta","gen_quark2_phi",
    "gen_lep_p","gen_lep_theta","gen_lep_phi","gen_isr_px","gen_isr_py","gen_isr_pz"]
a=t.arrays(br,library="np"); ok=(a["gen_quark1_p"]>0)&(a["gen_quark2_p"]>0)&(a["gen_lep_p"]>0)
idx=np.where(ok)[0];
if MAXN>0: idx=idx[:MAXN]
a={k:v[idx] for k,v in a.items()}; N=len(idx)
def vec(p,th,ph,m=0.0):
    st=np.sin(th); return np.stack([np.sqrt(p*p+m*m),p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th)],1)
Wh=vec(a["gen_quark1_p"],a["gen_quark1_theta"],a["gen_quark1_phi"])+vec(a["gen_quark2_p"],a["gen_quark2_theta"],a["gen_quark2_phi"])
L =vec(a["gen_lep_p"],a["gen_lep_theta"],a["gen_lep_phi"],MU)
Vis=Wh+L; Evis=Vis[:,0]; pxv=Vis[:,1]; pyv=Vis[:,2]; pzv=Vis[:,3]
isr_pz=a["gen_isr_pz"]; isr_p=np.sqrt(a["gen_isr_px"]**2+a["gen_isr_py"]**2+a["gen_isr_pz"]**2)
def Mass(E,px,py,pz): return np.sqrt(np.maximum(E*E-(px*px+py*py+pz*pz),1e-12))

def masses(g):                      # given collinear ISR pz=g, build ν and return (m_had,m_lep,m_WW)
    nuz=-pzv-g; nux=-pxv; nuy=-pyv; Enu=np.sqrt(nux*nux+nuy*nuy+nuz*nuz)
    Elp=L[:,0]; mlep=Mass(Elp+Enu, L[:,1]+nux, L[:,2]+nuy, L[:,3]+nuz)
    mWW =Mass(Evis+Enu, pxv+nux, pyv+nuy, pzv+nuz)
    mhad=Mass(Wh[:,0],Wh[:,1],Wh[:,2],Wh[:,3])
    return mhad,mlep,mWW

# collinear ISR solution
S=ECM-Evis; num=S*S-(pxv*pxv+pyv*pyv)-pzv*pzv
gp=num/(2.0*(S+pzv)); gm=num/(2.0*(pzv-S))
vp=gp>=0; vm=gm<0
# BEST: among valid branches, pick g closest to true gen_isr_pz
def pick_best():
    cand=[];
    g=np.zeros(N); best=np.full(N,np.inf)
    for gc,val in [(gp,vp),(gm,vm),(np.zeros(N),np.ones(N,bool))]:   # include no-ISR fallback
        d=np.where(val,np.abs(gc-isr_pz),np.inf)
        take=d<best; g=np.where(take,gc,g); best=np.where(take,d,best)
    return g
def pick_real():   # smallest |g| among valid branches (most probable ISR), else 0
    g=np.zeros(N); best=np.full(N,np.inf)
    for gc,val in [(gp,vp),(gm,vm)]:
        d=np.where(val,np.abs(gc),np.inf); take=d<best; g=np.where(take,gc,g); best=np.where(take,d,best)
    return g
g_best=pick_best(); g_real=pick_real()
print(f"[gen-hard-isr] N={N} ecm{ECM:.0f}  branches valid: +z {100*vp.mean():.0f}%  -z {100*vm.mean():.0f}%  "
      f"num<0 {100*np.mean(num<0):.1f}%")
print(f"  g_best vs gen_isr_pz: corr={np.corrcoef(g_best,isr_pz)[0,1]:.3f}  g_real |g| median={np.median(np.abs(g_real)):.3f}")

# 1-param mW fit (double BW + logZ), shared across variants
logZ=jax.jit(jax.vmap(lambda mww,mw: J.log_Z_ontf(mww,mw,GW)))
grid=np.linspace(74.0,86.0,241)
def fit_mW(mhad,mlep,mWW):
    C=np.empty((len(mhad),len(grid)))
    for j,mw in enumerate(grid):
        mwgw=mw*GW; dh=mhad*mhad-mw*mw; dl=mlep*mlep-mw*mw
        bwh=mwgw/(dh*dh+mwgw*mwgw); bwl=mwgw/(dl*dl+mwgw*mwgw)
        C[:,j]=-2.0*(np.log(bwh)+np.log(bwl))+2.0*np.asarray(logZ(jnp.asarray(mWW),jnp.full(len(mhad),mw)))
    fin=np.all(np.isfinite(C),axis=1); imin=np.argmin(C,axis=1); interior=(imin>0)&(imin<len(grid)-1)
    mhat=np.full(len(mhad),np.nan); curv=np.zeros(len(mhad))
    for k in np.where(fin)[0]:
        i=int(np.clip(imin[k],1,len(grid)-2)); c=np.polyfit(grid[i-1:i+2],C[k,i-1:i+2],2)
        mhat[k]=(-c[1]/(2*c[0]) if c[0]>0 else grid[i]); curv[k]=c[0]
    conv=fin&interior&(curv>0)
    return np.clip(mhat,grid[0],grid[-1]),conv

variants={"no-ISR":np.zeros(N),"collinear-ISR (best)":g_best,"collinear-ISR (real)":g_real}
res={}
edges=[0.0,0.1,0.3,0.7,1.5,3.0,1e9]; labels=["<0.1","0.1–0.3","0.3–0.7","0.7–1.5","1.5–3",">3"]
for name,g in variants.items():
    mh,ml,mww=masses(g); mhat,conv=fit_mW(mh,ml,mww); res[name]=(mhat,conv)
    print(f"\n=== {name} ===  convergence={100*conv.mean():.2f}%  m̂W mean={mhat[conv].mean():.3f} "
          f"median={np.median(mhat[conv]):.3f} std={mhat[conv].std():.3f} bias={mhat[conv].mean()-POLE:+.3f}")
    print("   ISR bin     N     mean   median   std")
    for lo,hi,lb in zip(edges[:-1],edges[1:],labels):
        s=conv&(isr_p>=lo)&(isr_p<hi)
        if s.sum()>30: print(f"   {lb:9s} {s.sum():6d}  {mhat[s].mean():7.3f} {np.median(mhat[s]):7.3f} {mhat[s].std():6.3f}")

# ── plot: bias & resolution vs ISR for the 3 variants ──
cents=[np.median(isr_p[(isr_p>=lo)&(isr_p<hi)]) for lo,hi in zip(edges[:-1],edges[1:])]
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
col={"no-ISR":"C3","collinear-ISR (best)":"C2","collinear-ISR (real)":"C0"}
for name,(mhat,conv) in res.items():
    bias=[]; std=[]; cc=[]
    for lo,hi,c in zip(edges[:-1],edges[1:],cents):
        s=conv&(isr_p>=lo)&(isr_p<hi)
        if s.sum()>30: bias.append(np.median(mhat[s])-POLE); std.append(mhat[s].std()); cc.append(c)
    ax[0].plot(cc,bias,"o-",color=col[name],label=name)
    ax[1].plot(cc,std,"o-",color=col[name],label=name)
for x in (ax[0],ax[1]): x.set_xscale("symlog",linthresh=0.1); x.set_xlabel("gen ISR |p| [GeV] (symlog)")
ax[0].axhline(0,color="k",lw=0.8); ax[0].set_ylabel(r"median $\hat m_W$ − pole [GeV]"); ax[0].set_title(f"mW bias vs ISR (ecm{ECM:.0f})"); ax[0].legend(fontsize=9)
ax[1].set_ylabel(r"$\hat m_W$ std [GeV]"); ax[1].set_title("mW resolution vs ISR"); ax[1].legend(fontsize=9)
plt.tight_layout(); png=f"{OUT}/mW_hard_isr_mitig_ecm{ECM:.0f}.png"; plt.savefig(png,dpi=110)
print(f"\n[plot] {png}")

# ── mW DISTRIBUTION for the ISR-mitigation variants (overlaid) ──
fig,ax=plt.subplots(1,2,figsize=(14,5.4))
for name,(mhat,conv) in res.items():
    ax[0].hist(mhat[conv],bins=90,range=(76,85),histtype="step",lw=2,color=col[name],density=True,
               label=f"{name}\nmed={np.median(mhat[conv]):.2f} std={mhat[conv].std():.2f}")
ax[0].axvline(POLE,color="k",ls="--",lw=1,label=f"pole {POLE}")
ax[0].set_xlabel(r"per-event $\hat m_W$ [GeV]"); ax[0].set_ylabel("normalized"); ax[0].set_xlim(76,85)
ax[0].set_title(f"All events (ecm{ECM:.0f}, N={int(N)})"); ax[0].legend(fontsize=8.5)
hi=isr_p>1.0
for name,(mhat,conv) in res.items():
    s=conv&hi
    ax[1].hist(mhat[s],bins=70,range=(76,85),histtype="step",lw=2,color=col[name],density=True,
               label=f"{name}\nmed={np.median(mhat[s]):.2f} std={mhat[s].std():.2f}")
ax[1].axvline(POLE,color="k",ls="--",lw=1)
ax[1].set_xlabel(r"$\hat m_W$ [GeV]"); ax[1].set_ylabel("normalized"); ax[1].set_xlim(76,85)
ax[1].set_title(f"HIGH-ISR only (gen ISR>1 GeV, N={int((hi).sum())})"); ax[1].legend(fontsize=8.5)
plt.tight_layout(); png2=f"{OUT}/mW_hard_isr_mitig_dist_ecm{ECM:.0f}.png"; plt.savefig(png2,dpi=110)
print(f"[plot] {png2}")
