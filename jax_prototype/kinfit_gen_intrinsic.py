#!/usr/bin/env python3
# ── INTRINSIC vs DETECTOR limitation of the lnuqq kfit (user's gen-level idea) ──
# Evaluates the SAME production kfit objective (build_chi2, kfit mode, σ_R=1.5) under
# three scenarios and reports the per-event conditioning — especially how well mW is
# determined (Schur-complement profile curvature) and the non-convex fraction:
#
#   (1) RECO @ min    : real reco data, at the production minimum y* (from tree).
#   (2) RECO @ truth  : real reco data, at the TRUE detector-correction params
#                       (s=reco/gen, angle=reco-gen, ISR=gen, BES=gen, mW=pole)  -> closure.
#   (3) GEN  @ truth  : PERFECT visible (gen quarks+lepton as the measurement), s=1,angle=0,
#                       ISR=gen, BES=gen, mW=pole  -> the INTRINSIC degeneracy with no detector.
#
# If (3) pins mW (small σ_mW, convex) -> non-convergence is DETECTOR/resolution-driven (mitigable
# by resolution/constraint). If (3) is STILL flat/non-convex in mW -> intrinsic kinematic
# degeneracy (mW<->scale<->ISR-longitudinal), mitigation must be a hard constraint / ensemble.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   KF_ISR_MODE=kfit KF_SIGMA_R=1.5 MAXN=6000 python3 jax_prototype/kinfit_gen_intrinsic.py [root] [ecm]
import sys, os
os.environ.setdefault("KF_ISR_MODE", "kfit")
os.environ.setdefault("KF_SIGMA_R", "1.5")
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
MAXN = int(os.environ.get("MAXN", "6000"))
print(f"[gen-intrinsic] {ROOT} ecm{ECM} kfit σ_R={K.SIGMA_R} MAXN={MAXN}")

RECO = ["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta",
        "reco_jet2_phi","reco_lep_p","reco_lep_theta","reco_lep_phi",
        "reco_met_p","reco_met_theta","reco_met_phi"]
GENV = ["gen_quark1_p","gen_quark1_theta","gen_quark1_phi","gen_quark2_p","gen_quark2_theta",
        "gen_quark2_phi","gen_lep_p","gen_lep_theta","gen_lep_phi","gen_nu_p","gen_nu_theta","gen_nu_phi"]
KFP  = ["kinfit_mW","kinfit_gW","kinfit_s1","kinfit_s2","kinfit_sl","kinfit_sn","kinfit_t1",
        "kinfit_t2","kinfit_tn","kinfit_tl","kinfit_p1","kinfit_p2","kinfit_pn","kinfit_pl",
        "kinfit_bes_m_minus_ecm","kinfit_bes_pz","kinfit_status","kinfit_chi2"]
GENX = ["gen_isr_px","gen_isr_py","gen_isr_pz","gen_ee_pz","gen_ee_m_minus_ecm"]
t = uproot.open(ROOT)["events"]
a = t.arrays(RECO+GENV+KFP+GENX, library="np")
Dreco = np.stack([a[c] for c in RECO], axis=1).astype(np.float64)
ok = np.all(np.isfinite(Dreco),axis=1)&(Dreco[:,0]>0)&(Dreco[:,3]>0)&(Dreco[:,6]>0)&(Dreco[:,9]>0)
ok &= (a["gen_quark1_p"]>0)&(a["gen_quark2_p"]>0)&(a["gen_lep_p"]>0)
idx = np.where(ok)[0][:MAXN]
a = {k:v[idx] for k,v in a.items()}; Dreco = Dreco[idx]; N=len(idx)
status = a["kinfit_status"]; valid=(status==0)|(status==1); st3=(status==3)
print(f"  N={N}  valid={100*valid.mean():.1f}%  status3={100*st3.mean():.1f}%")

# GEN visible "measurement" (cols 9-11 = MET slot, unused in kfit -> fill with gen nu)
Dgen = np.stack([a["gen_quark1_p"],a["gen_quark1_theta"],a["gen_quark1_phi"],
                 a["gen_quark2_p"],a["gen_quark2_theta"],a["gen_quark2_phi"],
                 a["gen_lep_p"],a["gen_lep_theta"],a["gen_lep_phi"],
                 a["gen_nu_p"],a["gen_nu_theta"],a["gen_nu_phi"]],axis=1).astype(np.float64)

P = J.load_params(K.HEADER)
bm = P[f"GAUSS_GEN_EE_M_MINUS_ECM_{ECM}"]; bz = P[f"GAUSS_GEN_EE_PZ_{ECM}"]
chi2 = K.build_chi2(ECM)
grad_fn = jax.jit(jax.vmap(lambda y,D,p: jax.grad(chi2)(y,D,p)))
hess_fn = jax.jit(jax.vmap(lambda y,D,p: jax.hessian(chi2)(y,D,p)))
val_fn  = jax.jit(jax.vmap(lambda y,D,p: chi2(y,D,p)))
def chunk(fn,*arr,bs=3000):
    return np.concatenate([np.asarray(fn(*[jnp.asarray(x[s:s+bs]) for x in arr]))
                           for s in range(0,N,bs)],axis=0)

def PR_of(D):  return K.build_prior_pack_pool(ECM, D)
def musig(PR,slot):
    o=K._OFF[slot][0]; return PR[:,o], PR[:,o+1]

def y_from_min(PR):                       # scenario 1: reconstruct y* from stored params
    y=np.zeros((N,16))
    y[:,0]=(a["kinfit_mW"]-K.KF_MW_INIT)/K.KF_MW_PHYS_SIGMA
    for j,(slot,br) in enumerate([("j1p","kinfit_s1"),("j2p","kinfit_s2"),("lp","kinfit_sl")],start=2):
        mu,sg=musig(PR,slot); y[:,j]=(a[br]-mu)/sg
    y[:,5]=a["kinfit_sn"]/K.ISR_SX; y[:,8]=a["kinfit_tn"]/K.ISR_SY; y[:,11]=a["kinfit_pn"]/K.ISR_SZ
    for j,(slot,br) in enumerate([("j1t","kinfit_t1"),("j2t","kinfit_t2")],start=6):
        mu,sg=musig(PR,slot); y[:,j]=(a[br]-mu)/sg
    for j,(slot,br) in enumerate([("j1q","kinfit_p1"),("j2q","kinfit_p2")],start=9):
        mu,sg=musig(PR,slot); y[:,j]=(a[br]-mu)/sg
    mu,sg=musig(PR,"lt"); y[:,12]=(a["kinfit_tl"]-mu)/sg
    mu,sg=musig(PR,"lq"); y[:,13]=(a["kinfit_pl"]-mu)/sg
    y[:,14]=(a["kinfit_bes_m_minus_ecm"]-bm[0])/bm[1]; y[:,15]=(a["kinfit_bes_pz"]-bz[0])/bz[1]
    return y

def y_truth(PR, scales, t_shift, p_shift):
    # scales=(s1,s2,sl), t_shift/p_shift=(j1,j2,lep) angle offsets reco-gen ; ISR/BES = gen truth
    y=np.zeros((N,16)); y[:,0]=0.0
    for j,slot,s in [(2,"j1p",scales[0]),(3,"j2p",scales[1]),(4,"lp",scales[2])]:
        mu,sg=musig(PR,slot); y[:,j]=(s-mu)/sg
    for j,slot,sh in [(6,"j1t",t_shift[0]),(7,"j2t",t_shift[1]),(12,"lt",t_shift[2])]:
        mu,sg=musig(PR,slot); y[:,j]=(sh-mu)/sg
    for j,slot,sh in [(9,"j1q",p_shift[0]),(10,"j2q",p_shift[1]),(13,"lq",p_shift[2])]:
        mu,sg=musig(PR,slot); y[:,j]=(sh-mu)/sg
    y[:,5]=a["gen_isr_px"]/K.ISR_SX; y[:,8]=a["gen_isr_py"]/K.ISR_SY; y[:,11]=a["gen_isr_pz"]/K.ISR_SZ
    y[:,14]=(a["gen_ee_m_minus_ecm"]-bm[0])/bm[1]; y[:,15]=(a["gen_ee_pz"]-bz[0])/bz[1]
    return y

KEEP=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
LBL =["mW","s1","s2","sl","ISRpx","t1","t2","ISRpy","p1","p2","ISRkz","tl","pl","besm","bespz"]

def analyze(tag, D, y):
    PR=PR_of(D)
    g=chunk(grad_fn,y,D,PR)[:,KEEP]; H=chunk(hess_fn,y,D,PR)[:,KEEP][:,:,KEEP]
    c=chunk(val_fn,y,D,PR)
    Hs=0.5*(H+np.transpose(H,(0,2,1))); w,_=np.linalg.eigh(Hs)
    fin=np.all(np.isfinite(w),axis=1)
    gnorm=np.linalg.norm(g,axis=1)
    n_neg=np.sum(w<-1e-6,axis=1)
    lam_flat=w[np.arange(len(w)),np.argmin(np.abs(w),axis=1)]
    # Schur complement of mW (index 0) against the other 14 params -> profile curvature c_mW.
    # c_mW>0 => mW locally determined (σ_mW = 2/sqrt(c_mW) GeV); c_mW<=0 => mW non-convex (runaway).
    H00=Hs[:,0,0]; Hr0=Hs[:,1:,0]; Hrr=Hs[:,1:,1:]
    c_mW=np.full(N,np.nan); sig_mW=np.full(N,np.nan)
    for i in range(N):
        if not fin[i]: continue
        try:
            sol=np.linalg.solve(Hrr[i],Hr0[i])
        except np.linalg.LinAlgError:
            continue
        cm=H00[i]-Hr0[i]@sol; c_mW[i]=cm
        if cm>0: sig_mW[i]=2.0/np.sqrt(cm)        # GeV (mW=80.4+2*y0)
    return dict(tag=tag,c=c,g=gnorm,n_neg=n_neg,lam_flat=lam_flat,c_mW=c_mW,sig_mW=sig_mW,fin=fin,w=w)

# true detector-correction params for RECO@truth
s1t=a["reco_jet1_p"]/a["gen_quark1_p"]; s2t=a["reco_jet2_p"]/a["gen_quark2_p"]; slt=a["reco_lep_p"]/a["gen_lep_p"]
t1t=a["reco_jet1_theta"]-a["gen_quark1_theta"]; t2t=a["reco_jet2_theta"]-a["gen_quark2_theta"]; tlt=a["reco_lep_theta"]-a["gen_lep_theta"]
def dphi(x,y):
    d=x-y; return (d+np.pi)%(2*np.pi)-np.pi
p1t=dphi(a["reco_jet1_phi"],a["gen_quark1_phi"]); p2t=dphi(a["reco_jet2_phi"],a["gen_quark2_phi"]); plt=dphi(a["reco_lep_phi"],a["gen_lep_phi"])

PRreco=PR_of(Dreco); PRgen=PR_of(Dgen)
R = [
  analyze("(1) RECO @ min",   Dreco, y_from_min(PRreco)),
  analyze("(2) RECO @ truth", Dreco, y_truth(PRreco,(s1t,s2t,slt),(t1t,t2t,tlt),(p1t,p2t,plt))),
  analyze("(3) GEN  @ truth", Dgen,  y_truth(PRgen,(np.ones(N),np.ones(N),np.ones(N)),
                                              (np.zeros(N),np.zeros(N),np.zeros(N)),
                                              (np.zeros(N),np.zeros(N),np.zeros(N)))),
]

print(f"\n{'scenario':18s} {'|grad|med':>10s} {'#neg med':>9s} {'frac cvx(mW)':>13s} {'σ_mW med[GeV]':>14s} {'σ_mW p90':>10s} {'λflat med':>10s}")
for r in R:
    m=r["fin"]
    cvx=np.mean(r["c_mW"][m]>0)
    smw=r["sig_mW"][np.isfinite(r["sig_mW"])]
    print(f"  {r['tag']:16s} {np.median(r['g'][m]):10.2e} {np.median(r['n_neg'][m]):9.0f} "
          f"{cvx:13.3f} {np.median(smw):14.3f} {np.percentile(smw,90):10.3f} {np.median(np.abs(r['lam_flat'][m])):10.3f}")

# Scenario-3 split by gen ISR energy (is the intrinsic degeneracy ISR-driven?)
isr_p=np.sqrt(a["gen_isr_px"]**2+a["gen_isr_py"]**2+a["gen_isr_pz"]**2)
r3=R[2]; m=r3["fin"]
print("\n[GEN@truth] mW conditioning vs gen ISR energy (ISR≈0 bin + nonzero terciles):")
soft=m&(isr_p<0.05)
cvx=np.mean(r3["c_mW"][soft]>0); smw=r3["sig_mW"][soft&np.isfinite(r3["sig_mW"])]
print(f"  ISR_p<0.05 (~no ISR) N={soft.sum():4d}  frac_cvx(mW)={cvx:.3f}  σ_mW med={np.median(smw):.3f}")
hard=m&(isr_p>=0.05)
q=np.quantile(isr_p[hard],[0,1/3,2/3,1.0])
for lo,hi in zip(q[:-1],q[1:]):
    sel=hard&(isr_p>=lo)&(isr_p<=hi)
    cvx=np.mean(r3["c_mW"][sel]>0); smw=r3["sig_mW"][sel&np.isfinite(r3["sig_mW"])]
    print(f"  ISR_p∈[{lo:.2f},{hi:.2f}] N={sel.sum():4d}  frac_cvx(mW)={cvx:.3f}  σ_mW med={np.median(smw):.3f}")

# ── DIRECT TEST: actually MINIMIZE the fit on gen vs reco (PD-floored Newton from truth) ──
def make_polish(niter=60, ftol=1e-3, trust=4.0):
    grad=jax.grad(chi2); hess=jax.hessian(chi2)
    def body(k,y,D,PR):
        g=grad(y,D,PR); H=0.5*(hess(y,D,PR)+hess(y,D,PR).T)
        w,V=jnp.linalg.eigh(H); wf=jnp.maximum(w, ftol*jnp.max(jnp.abs(w)))
        d=-(V@((V.T@g)/wf)); dn=jnp.linalg.norm(d)
        return y+d*jnp.minimum(1.0, trust/jnp.maximum(dn,1e-9))
    def fit(y0,D,PR):
        yf=jax.lax.fori_loop(0,niter,lambda k,y: body(k,y,D,PR),y0)
        return yf, jnp.linalg.norm(grad(yf,D,PR))
    return jax.jit(jax.vmap(fit))
polish=make_polish()
def run_min(tag, D, y0):
    PR=PR_of(D); yf=[]; gf=[]
    for s in range(0,N,3000):
        sl=slice(s,s+3000)
        yy,gg=polish(jnp.asarray(y0[sl]),jnp.asarray(D[sl]),jnp.asarray(PR[sl]))
        yf.append(np.asarray(yy)); gf.append(np.asarray(gg))
    yf=np.concatenate(yf); gf=np.concatenate(gf)
    mW=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*yf[:,0]
    # post-fit mW convexity (Schur at the polished point)
    H=chunk(hess_fn,yf,D,PR)[:,KEEP][:,:,KEEP]; Hs=0.5*(H+np.transpose(H,(0,2,1)))
    H00=Hs[:,0,0]; Hr0=Hs[:,1:,0]; Hrr=Hs[:,1:,1:]; cvx=np.zeros(N,bool)
    for i in range(N):
        try: cvx[i]=(H00[i]-Hr0[i]@np.linalg.solve(Hrr[i],Hr0[i]))>0
        except np.linalg.LinAlgError: pass
    conv=gf<1e-1
    run=np.abs(mW-K.KF_MW_INIT)>5.0
    print(f"  {tag:18s} converged(|g|<0.1)={100*conv.mean():4.1f}%  mW mean={mW.mean():7.3f} std={mW.std():.3f}  "
          f"runaway(|ΔmW|>5)={100*run.mean():4.1f}%  postfit frac_cvx(mW)={cvx.mean():.3f}")
    return mW
print("\n[DIRECT MINIMIZATION from gen-truth init — does the fit converge & pin mW?]")
y_tr_reco=y_truth(PRreco,(s1t,s2t,slt),(t1t,t2t,tlt),(p1t,p2t,plt))
y_tr_gen =y_truth(PRgen,(np.ones(N),)*3,(np.zeros(N),)*3,(np.zeros(N),)*3)
run_min("RECO (from truth)", Dreco, y_tr_reco)
run_min("GEN  (from truth)", Dgen,  y_tr_gen)

# How status-3 (reco) splits on the GEN@truth mW-convexity (is the failing-event degeneracy intrinsic?)
print("\n[link] reco status-3 vs GEN@truth mW-convexity:")
cvx3=R[2]["c_mW"]>0
print(f"  frac_cvx(mW) | reco valid = {np.mean(cvx3[valid&m]):.3f}    | reco status-3 = {np.mean(cvx3[st3&m]):.3f}")
np.savez(os.path.join(os.path.dirname(__file__),"gen_intrinsic_out.npz"),
         **{f"{k}_{i}":R[i][k] for i in range(3) for k in ["c_mW","sig_mW","n_neg","lam_flat","g"]},
         valid=valid,st3=st3,isr_p=isr_p)
print("\n[saved jax_prototype/gen_intrinsic_out.npz]")
