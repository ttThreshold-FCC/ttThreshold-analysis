#!/usr/bin/env python3
# ── Does the DOUBLE-BW constraint pin mW per-event?  PROFILE likelihood L(mW) ──
# User's objection: floating ISR lets the LEPTONIC m_lep follow any mW, but the HADRONIC BW is
# anchored to the fixed dijet mass m_had, so it should still pin mW. Test it properly: build the
# per-event PROFILE likelihood L(mW) = min_{ISR,BES} chi2(mW, ISR, BES) on the reduced gen model
# (visible = 2 gen quarks + gen muon, FIXED). Then decompose the ENSEMBLE L(mW) into its terms
# (hadronic BW, leptonic BW, log_Z normalization, ISR, closure) to see which one is curved vs flat
# in mW — i.e. whether the hadronic BW pins mW or log_Z cancels it.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_SIGMA_R=1.5 MAXN=4000 NDEV=32 python3 jax_prototype/mw_profile_scan.py [root] [ecm]
import sys, os
NDEV=int(os.environ.get("NDEV","32"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={NDEV}")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, jax, jax.numpy as jnp, uproot
sys.path.insert(0, os.path.dirname(__file__))
import jaxfit_common as J
jax.config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=160, suppress=True)

MW_INIT=80.419; GW=2.049; ISR_SX=0.4; ISR_SY=0.4; ISR_SZ=2.0
SIG_R=float(os.environ.get("KF_SIGMA_R","1.5")); SIG_R_LN=np.log(2*np.pi*SIG_R**2)
MU=0.1056583745
HEADER=os.path.join(os.path.dirname(__file__),"..","kinfit_inputs","dcb_params.h"); P=J.load_params(HEADER)
ROOT=sys.argv[1] if len(sys.argv)>1 else \
  "outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160; MAXN=int(os.environ.get("MAXN","4000"))
ix=P[f"SDCBG_GEN_ISR_PX_{ECM}"]; iy=P[f"SDCBG_GEN_ISR_PY_{ECM}"]; iz=P[f"SDCBG_GEN_ISR_PZ_{ECM}"]
bm=P[f"GAUSS_GEN_EE_M_MINUS_ECM_{ECM}"]; bz=P[f"GAUSS_GEN_EE_PZ_{ECM}"]

def terms(mw, yr, Wh, Lp):
    # yr = [gx,gy,gz, besm_y, besz_y]; returns (total, [bw_h, bw_l, logZ, isr, bes, eclos])
    pgx=ISR_SX*yr[0]; pgy=ISR_SY*yr[1]; kz=ISR_SZ*yr[2]
    besm=bm[0]+bm[1]*yr[3]; besz=bz[0]+bz[1]*yr[4]; Eg=jnp.sqrt(pgx*pgx+pgy*pgy+kz*kz+1e-12)
    Vis=Wh+Lp; nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=besz-Vis[3]-kz
    Enu=jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12); Nu=jnp.array([Enu,nux,nuy,nuz])
    Wl=Lp+Nu; WW=Wh+Wl
    def M(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))
    mh=M(Wh); ml=M(Wl); mwgw=mw*GW; dh=mh*mh-mw*mw; dl=ml*ml-mw*mw
    bw_h=mwgw/(dh*dh+mwgw*mwgw); bw_l=mwgw/(dl*dl+mwgw*mwgw)
    s_ww=WW[0]**2-(WW[1]**2+WW[2]**2+WW[3]**2)
    lam=(s_ww-(mh+ml)**2)*(s_ww-(mh-ml)**2); lam=jnp.sqrt(lam*lam+1e-24); m_WW=jnp.sqrt(jnp.maximum(s_ww,1e-12))
    t_bwh=-2.0*jnp.log(bw_h); t_bwl=-2.0*jnp.log(bw_l)
    t_logZ=4.0*jnp.log(jnp.pi)-jnp.log(lam)+2.0*jnp.log(s_ww)+2.0*J.log_Z_ontf(m_WW,mw,GW)
    t_isr=J.spike_dcb_n2ll(pgx,ix)+J.spike_dcb_n2ll(pgy,iy)+J.spike_dcb_n2ll(kz,iz)
    t_bes=J.gauss_n2ll(besm,bm)+J.gauss_n2ll(besz,bz)
    r=(float(ECM)+besm)-WW[0]-Eg; t_ecl=(r/SIG_R)**2+SIG_R_LN
    tot=t_bwh+t_bwl+t_logZ+t_isr+t_bes+t_ecl
    return tot, jnp.array([t_bwh,t_bwl,t_logZ,t_isr,t_bes,t_ecl])
TLBL=["bw_had","bw_lep","logZ+ps","isr","bes","closure"]
def chi2r(mw,yr,Wh,Lp): return terms(mw,yr,Wh,Lp)[0]

# profile: minimize the 5 nuisances at fixed mW (PD-floored Newton + backtracking), warm-started
def make_prof(niter=50, nls=20, ftol=1e-3, c1=1e-4):
    g=jax.grad(chi2r,argnums=1); h=jax.hessian(chi2r,argnums=1)
    def prof(mw, yr0, Wh, Lp):
        def body(k,carry):
            yr,c=carry; gg=g(mw,yr,Wh,Lp); H=h(mw,yr,Wh,Lp); H=0.5*(H+H.T)
            w,V=jnp.linalg.eigh(H); wf=jnp.maximum(w,ftol*jnp.max(jnp.abs(w)))
            d=-(V@((V.T@gg)/wf)); gd=jnp.dot(gg,d)
            def ls(j,st):
                a,f,cb,yb=st; yt=yr+a*d; ct=chi2r(mw,yt,Wh,Lp)
                ok=jnp.isfinite(ct)&(ct<=c+c1*a*gd)&(~f)
                return (a*0.5,f|ok,jnp.where(ok,ct,cb),jnp.where(ok,yt,yb))
            _,_,c,yr=jax.lax.fori_loop(0,nls,ls,(1.0,False,c,yr)); return (yr,c)
        yr,c=jax.lax.fori_loop(0,niter,body,(yr0,chi2r(mw,yr0,Wh,Lp)))
        return yr, terms(mw,yr,Wh,Lp)[1]
    return prof
prof=jax.jit(jax.vmap(make_prof(), in_axes=(None,0,0,0)))   # over events at one mw

# load fixed gen visible + truth init
t=uproot.open(ROOT)["events"]
br=["gen_Whad_p","gen_Whad_costheta","gen_Whad_phi","gen_Whad_m","gen_lep_p","gen_lep_theta",
    "gen_lep_phi","gen_isr_px","gen_isr_py","gen_isr_pz","gen_ee_pz","gen_ee_m_minus_ecm"]
a=t.arrays(br,library="np"); ok=(a["gen_Whad_p"]>0)&(a["gen_lep_p"]>0)&np.isfinite(a["gen_Whad_m"])
idx=np.where(ok)[0][:MAXN]; a={k:v[idx] for k,v in a.items()}; N=len(idx)
sth=np.sqrt(1-a["gen_Whad_costheta"]**2); Whp=a["gen_Whad_p"]; Whm=a["gen_Whad_m"]
Whad=np.stack([np.sqrt(Whp**2+Whm**2),Whp*sth*np.cos(a["gen_Whad_phi"]),Whp*sth*np.sin(a["gen_Whad_phi"]),Whp*a["gen_Whad_costheta"]],1).astype(np.float64)
lp=a["gen_lep_p"]; lth=a["gen_lep_theta"]
Lep=np.stack([np.sqrt(lp**2+MU**2),lp*np.sin(lth)*np.cos(a["gen_lep_phi"]),lp*np.sin(lth)*np.sin(a["gen_lep_phi"]),lp*np.cos(lth)],1).astype(np.float64)
yr0=np.zeros((N,5)); yr0[:,0]=a["gen_isr_px"]/ISR_SX; yr0[:,1]=a["gen_isr_py"]/ISR_SY; yr0[:,2]=a["gen_isr_pz"]/ISR_SZ
yr0[:,3]=(a["gen_ee_m_minus_ecm"]-bm[0])/bm[1]; yr0[:,4]=(a["gen_ee_pz"]-bz[0])/bz[1]
mhad=np.sqrt(np.maximum(Whad[:,0]**2-(Whad[:,1]**2+Whad[:,2]**2+Whad[:,3]**2),0))
print(f"[mw-profile] N={N} ecm{ECM} σ_R={SIG_R}  m_had(gen) mean={mhad.mean():.3f} std={mhad.std():.3f}")

grid=np.linspace(76.0,84.5,26)
Lprof=np.zeros((N,len(grid))); TermAvg=np.zeros((len(grid),6)); yr=jnp.asarray(yr0)
for j,mw in enumerate(grid):                  # warm-start each grid point from previous
    yrn, tt = prof(float(mw), yr, jnp.asarray(Whad), jnp.asarray(Lep))
    yr=yrn; tt=np.asarray(tt); Lprof[:,j]=tt.sum(1); TermAvg[j]=tt.mean(0)
# reference each curve to its own min (so we compare SHAPES/curvature, not offsets)
fin=np.all(np.isfinite(Lprof),axis=1); L=Lprof[fin]
imin=np.argmin(L,axis=1)
interior=np.mean((imin>0)&(imin<len(grid)-1))
# per-event curvature near its min -> sigma_mW
def curv(row):
    i=int(np.clip(np.argmin(row),1,len(grid)-2))
    c=np.polyfit(grid[i-1:i+2],row[i-1:i+2],2)[0]; return c
cs=np.array([curv(r) for r in L]); pos=cs>0
sig=np.where(pos,1.0/np.sqrt(np.abs(cs)),np.nan)
print(f"\n[PROFILE L(mW)=min_(ISR,BES) chi2]  per-event:")
print(f"  interior-minimum fraction = {100*interior:.1f}%   convex-at-min (c>0) = {100*np.mean(pos):.1f}%")
print(f"  median σ_mW (where convex) = {np.nanmedian(sig):.3f} GeV   median |argmin-m_had| = {np.median(np.abs(grid[imin]-mhad[fin])):.3f} GeV")
ens=L.sum(0); ens-=ens.min(); i0=int(np.clip(np.argmin(ens),1,len(grid)-2))
ce=np.polyfit(grid[i0-1:i0+2],ens[i0-1:i0+2],2)[0]
print(f"  ENSEMBLE profiled L(mW): min@{grid[i0]:.2f} GeV  σ_ens(N={fin.sum()})≈{1/np.sqrt(ce):.3f}  per-evt≈{np.sqrt(fin.sum())/np.sqrt(ce):.1f} GeV")

# DECOMPOSITION: ensemble-mean of each term vs mW, referenced to its own value at mW=80.4
j80=int(np.argmin(np.abs(grid-MW_INIT)))
print(f"\n[ENSEMBLE term shapes vs mW]  (Δ from mW=80.4; +curved-up = pulls/pins mW, −down = anti-pins)")
print("  mW   " + " ".join(f"{l:>9s}" for l in TLBL) + f"  {'TOTAL':>9s}")
for j in [0,5,10,j80,15,20,25]:
    d=TermAvg[j]-TermAvg[j80]
    print(f"  {grid[j]:4.1f} " + " ".join(f"{x:9.3f}" for x in d) + f"  {d.sum():9.3f}")
# curvature of each ensemble term near 80.4
def tc(col):
    return np.polyfit(grid[j80-2:j80+3], TermAvg[j80-2:j80+3,col],2)[0]
print("\n  per-term ENSEMBLE curvature d²/dmW² near 80.4 (positive=pins mW):")
print("   " + "  ".join(f"{TLBL[k]}={tc(k):+.3f}" for k in range(6)) + f"   TOTAL={sum(tc(k) for k in range(6)):+.3f}")
print("\n  => compare bw_had (hadronic anchor) vs logZ+ps (normalization) curvatures above.")
