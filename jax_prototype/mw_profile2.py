#!/usr/bin/env python3
# ── Localize the flat direction: profile L(mW) with jet-scale freedom ADDED BACK ──
# Reduced fit (visible FIXED) pins mW (profile convex 97%, σ_mW~0.76). The full fit fails ~25%.
# The difference is the jet/lepton SCALE freedom. Test directly: same reduced gen model, but now
# the two gen quarks carry FREE scales s1,s2 (with their DCB priors) so m_had FLOATS — then profile
# L(mW)=min_{s1,s2,ISR,BES} chi2 and compare convexity to the visible-fixed baseline.
# Config via env MODE = reduced | jetscale | jetlepscale.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_SIGMA_R=1.5 MAXN=2500 MODE=jetscale NDEV=32 python3 jax_prototype/mw_profile2.py
import sys, os
NDEV=int(os.environ.get("NDEV","32"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={NDEV}")
os.environ.setdefault("OMP_NUM_THREADS","1")
import numpy as np, jax, jax.numpy as jnp, uproot
sys.path.insert(0, os.path.dirname(__file__))
import jaxfit_common as J
jax.config.update("jax_enable_x64", True); np.set_printoptions(linewidth=160, suppress=True)

MW_INIT=80.419; GW=2.049; ISR_SX=0.4; ISR_SY=0.4; ISR_SZ=2.0
SIG_R=float(os.environ.get("KF_SIGMA_R","1.5")); SIG_R_LN=np.log(2*np.pi*SIG_R**2); MU=0.1056583745
MODE=os.environ.get("MODE","jetscale")
HEADER=os.path.join(os.path.dirname(__file__),"..","kinfit_inputs","dcb_params.h"); P=J.load_params(HEADER)
ROOT=sys.argv[1] if len(sys.argv)>1 else \
  "outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm160.root"
ECM=int(sys.argv[2]) if len(sys.argv)>2 else 160; MAXN=int(os.environ.get("MAXN","2500"))
ix=P[f"SDCBG_GEN_ISR_PX_{ECM}"]; iy=P[f"SDCBG_GEN_ISR_PY_{ECM}"]; iz=P[f"SDCBG_GEN_ISR_PZ_{ECM}"]
bm=P[f"GAUSS_GEN_EE_M_MINUS_ECM_{ECM}"]; bz=P[f"GAUSS_GEN_EE_PZ_{ECM}"]
# nuisance layout: [s1,s2 (jetscale only), sl (jetlepscale only), gx,gy,gz, besm,besz]
NJ = {"reduced":0,"jetscale":2,"jetlepscale":3}[MODE]
NPAR = NJ + 5
def vec(p,th,ph):
    st=jnp.sin(th); return jnp.array([p,p*st*jnp.cos(ph),p*st*jnp.sin(ph),p*jnp.cos(th)])
def Mass(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))

def chi2(mw, nz, D):
    # D packs: q1(p,th,ph)=0:3, q2=3:6, lep(p,th,ph)=6:9, prj1(10)=9:19, prj2(10)=19:29, prl(10)=29:39
    q1=D[0:3]; q2=D[3:6]; lp=D[6:9]
    if NJ>=2:
        s1=nz[0]; s2=nz[1]
        J1=vec(q1[0]/s1,q1[1],q1[2]); J2=vec(q2[0]/s2,q2[1],q2[2])
        sp=J.dcb_gauss_n2ll(s1,D[9:19])+J.dcb_gauss_n2ll(s2,D[19:29])
    else:
        J1=vec(q1[0],q1[1],q1[2]); J2=vec(q2[0],q2[1],q2[2]); sp=0.0
    if NJ>=3:
        sl=nz[2]; L=vec(lp[0]/sl,lp[1],lp[2]); sp=sp+J.dcb_gauss_n2ll(sl,D[29:39])
    else:
        L=jnp.array([jnp.sqrt(lp[0]**2+MU**2),lp[0]*jnp.sin(lp[1])*jnp.cos(lp[2]),
                     lp[0]*jnp.sin(lp[1])*jnp.sin(lp[2]),lp[0]*jnp.cos(lp[1])])
    gx,gy,gz,bmy,bzy=nz[NJ],nz[NJ+1],nz[NJ+2],nz[NJ+3],nz[NJ+4]
    pgx=ISR_SX*gx; pgy=ISR_SY*gy; kz=ISR_SZ*gz; besm=bm[0]+bm[1]*bmy; besz=bz[0]+bz[1]*bzy
    Eg=jnp.sqrt(pgx*pgx+pgy*pgy+kz*kz+1e-12); Wh=J1+J2; Vis=Wh+L
    nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=besz-Vis[3]-kz
    Nu=jnp.array([jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12),nux,nuy,nuz]); Wl=L+Nu; WW=Wh+Wl
    mh=Mass(Wh); ml=Mass(Wl); mwgw=mw*GW; dh=mh*mh-mw*mw; dl=ml*ml-mw*mw
    bwh=mwgw/(dh*dh+mwgw*mwgw); bwl=mwgw/(dl*dl+mwgw*mwgw)
    s_ww=WW[0]**2-(WW[1]**2+WW[2]**2+WW[3]**2)
    lam=(s_ww-(mh+ml)**2)*(s_ww-(mh-ml)**2); lam=jnp.sqrt(lam*lam+1e-24); m_WW=jnp.sqrt(jnp.maximum(s_ww,1e-12))
    bw=-2.0*(jnp.log(bwh)+jnp.log(bwl))+4.0*jnp.log(jnp.pi)-jnp.log(lam)+2.0*jnp.log(s_ww)+2.0*J.log_Z_ontf(m_WW,mw,GW)
    isr=J.spike_dcb_n2ll(pgx,ix)+J.spike_dcb_n2ll(pgy,iy)+J.spike_dcb_n2ll(kz,iz)
    bes=J.gauss_n2ll(besm,bm)+J.gauss_n2ll(besz,bz)
    r=(float(ECM)+besm)-WW[0]-Eg
    return bw+isr+bes+sp+(r/SIG_R)**2+SIG_R_LN

# profile: minimize nuisances at fixed mW (PD-floored damped Newton + trust cap; no line search)
def make_prof(niter=45, ftol=1e-3, trust=3.0):
    g=jax.grad(chi2,argnums=1); h=jax.hessian(chi2,argnums=1)
    def prof(mw, nz0, D):
        def body(k,nz):
            gg=g(mw,nz,D); H=h(mw,nz,D); H=0.5*(H+H.T)
            w,V=jnp.linalg.eigh(H); wf=jnp.maximum(w,ftol*jnp.max(jnp.abs(w)))
            d=-(V@((V.T@gg)/wf)); dn=jnp.linalg.norm(d)
            return nz+d*jnp.minimum(1.0,trust/jnp.maximum(dn,1e-9))
        nz=jax.lax.fori_loop(0,niter,body,nz0); return chi2(mw,nz,D)
    return jax.jit(jax.vmap(make_prof(), in_axes=(None,0,0)))
prof=make_prof()

t=uproot.open(ROOT)["events"]
br=["gen_quark1_p","gen_quark1_theta","gen_quark1_phi","gen_quark2_p","gen_quark2_theta","gen_quark2_phi",
    "gen_lep_p","gen_lep_theta","gen_lep_phi","gen_Whad_m","gen_isr_px","gen_isr_py","gen_isr_pz",
    "gen_ee_pz","gen_ee_m_minus_ecm"]
a=t.arrays(br,library="np"); ok=(a["gen_quark1_p"]>0)&(a["gen_quark2_p"]>0)&(a["gen_lep_p"]>0)
idx=np.where(ok)[0][:MAXN]; a={k:v[idx] for k,v in a.items()}; N=len(idx)
# pooled jet-scale prior at each quark's p
JP,JPe=J.get_bins(P,"DCBG_JET_P_RESP",ECM); LP,LPe=J.get_bins(P,"DCBER3G_LEP_P_RESP",ECM)
pb=J.pick_bin_idx
prj1=JP[pb(a["gen_quark1_p"],JPe)]; prj2=JP[pb(a["gen_quark2_p"],JPe)]; prl=LP[pb(a["gen_lep_p"],LPe)][:,:10]
D=np.concatenate([np.stack([a["gen_quark1_p"],a["gen_quark1_theta"],a["gen_quark1_phi"],
                            a["gen_quark2_p"],a["gen_quark2_theta"],a["gen_quark2_phi"],
                            a["gen_lep_p"],a["gen_lep_theta"],a["gen_lep_phi"]],1),
                  prj1,prj2,prl],axis=1).astype(np.float64)
nz0=np.zeros((N,NPAR))
if NJ>=2: nz0[:,0]=1.0; nz0[:,1]=1.0
if NJ>=3: nz0[:,2]=1.0
nz0[:,NJ]=a["gen_isr_px"]/ISR_SX; nz0[:,NJ+1]=a["gen_isr_py"]/ISR_SY; nz0[:,NJ+2]=a["gen_isr_pz"]/ISR_SZ
nz0[:,NJ+3]=(a["gen_ee_m_minus_ecm"]-bm[0])/bm[1]; nz0[:,NJ+4]=(a["gen_ee_pz"]-bz[0])/bz[1]
print(f"[mw-profile2] MODE={MODE} NPAR_nuis={NPAR} N={N} ecm{ECM} σ_R={SIG_R}")

grid=np.linspace(76.0,84.5,18); L=np.zeros((N,len(grid))); nz=jnp.asarray(nz0)
import time; t0=time.time()
for j,mw in enumerate(grid):
    nzt=nz  # warm start from prev grid point
    # one profile minimization per grid point
    c=prof(float(mw),nzt,jnp.asarray(D)); L[:,j]=np.asarray(c)
print(f"  scan {len(grid)} pts x {N} evts in {time.time()-t0:.1f}s")
fin=np.all(np.isfinite(L),axis=1); Lf=L[fin]; imin=np.argmin(Lf,axis=1)
interior=np.mean((imin>0)&(imin<len(grid)-1))
def curv(row):
    i=int(np.clip(np.argmin(row),1,len(grid)-2)); return np.polyfit(grid[i-1:i+2],row[i-1:i+2],2)[0]
cs=np.array([curv(r) for r in Lf]); pos=cs>0; sig=np.where(pos,1/np.sqrt(np.abs(cs)),np.nan)
ens=Lf.sum(0); ens-=ens.min(); i0=int(np.clip(np.argmin(ens),1,len(grid)-2))
ce=np.polyfit(grid[i0-1:i0+2],ens[i0-1:i0+2],2)[0]
print(f"  PROFILE convex-at-min={100*np.mean(pos):.1f}%  interior-min={100*interior:.1f}%  "
      f"median σ_mW={np.nanmedian(sig):.3f} GeV")
print(f"  ENSEMBLE L(mW): min@{grid[i0]:.2f}  σ_ens(N={fin.sum()})={1/np.sqrt(ce):.3f}  per-evt≈{np.sqrt(fin.sum())/np.sqrt(ce):.2f} GeV")
