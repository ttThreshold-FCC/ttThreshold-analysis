#!/usr/bin/env python3
# Forward-fold mW: per-channel σ_mW (statistical) + linearity (slope vs injected mW), for the FORWARD-FOLD
# observable being either the RAW reco W masses or the KINEMATIC-FIT-constrained W masses ("kin reco").
# CHANNEL=4q|lnuqq  OBS=raw|kinfit.  Mirrors the conv_mw asimov σ; adds a mass-injection linearity test.
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# NWORKERS=48 CHANNEL=4q OBS=raw python3 jax_prototype/conv_mw_method_compare.py [ecm]
import sys, os, json
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, multiprocessing as mp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import uproot
NWORKERS=int(os.environ.get("NWORKERS","48")); _FORK=mp.get_context("fork")
def _pmap(w,ch):
    nw=max(1,min(NWORKERS,len(ch)))
    return [w(c) for c in ch] if nw==1 else _FORK.Pool(nw).map(w,ch)

ECM=int(sys.argv[1]) if len(sys.argv)>1 else 160
CHANNEL=os.environ.get("CHANNEL","4q"); OBS=os.environ.get("OBS","raw")
GW=2.085; MW_REF=80.4; DECAY_P=3.0; NMW=int(os.environ.get("NMW","121")); MWLO,MWHI=79.0,81.5
LOGZ_N=256; LOGZ_TAB_N=400; FOLD_K=5; FOLD_BIN=float(os.environ.get("FOLD_BIN","0.25")); FOLD_SM=0.6
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW,exist_ok=True)

_GX,_GWl=np.polynomial.legendre.leggauss(LOGZ_N); _W2=_GWl[:,None]*_GWl[None,:]
def log_Z(m_WW,mW,gW=GW):
    m_WW=np.atleast_1d(np.asarray(m_WW,float)); mwgw=mW*gW; mW2=mW*mW; out=np.empty(m_WW.shape,float)
    CH=max(1,50_000_000//(LOGZ_N*LOGZ_N))
    for a in range(0,len(m_WW),CH):
        s=(m_WW[a:a+CH])**2; t_min=np.arctan(-mW2/mwgw); t_max=np.arctan((s-mW2)/mwgw)
        half_d=0.5*(t_max-t_min); half_s=0.5*(t_max+t_min); t=half_d[:,None]*_GX[None,:]+half_s[:,None]
        m=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1.0/m; rf=m**DECAY_P if DECAY_P else np.ones_like(m)
        mh=m[:,:,None]; ml=m[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=s[:,None,None]; rh=rf[:,:,None]; rl=rf[:,None,:]
        lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2); integ=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il*rh*rl/(4.0*sE),0.0)
        out[a:a+CH]=np.log(np.maximum(np.sum(_W2[None]*integ,axis=(1,2))*half_d*half_d,1e-300))
    return out
def bw(m,mW,gW=GW):
    mwgw=mW*gW; d=m*m-mW*mW; return mwgw/(d*d+mwgw*mwgw)
def n2ll(mh,ml,mWW,mW,logz):
    s=mWW*mWW; lam=(s-(mh+ml)**2)*(s-(mh-ml)**2); bad=lam<=0; lam=np.where(bad,1.0,lam)
    tv=(-2.0*np.log(np.maximum(bw(mh,mW),1e-300))-2.0*np.log(np.maximum(bw(ml,mW),1e-300))-np.log(lam)+2.0*np.log(s)+2.0*logz)
    if DECAY_P: tv=tv-2.0*DECAY_P*(np.log(np.maximum(mh,1e-9))+np.log(np.maximum(ml,1e-9)))
    return np.where(bad,1e6,tv)
def parab_min(xs,ys):
    i=int(np.clip(np.argmin(ys),2,len(ys)-3)); c=np.polyfit(xs[i-2:i+3],ys[i-2:i+3],2)
    return -c[1]/(2*c[0]), (1.0/np.sqrt(c[0]) if c[0]>0 else np.nan)

# ── load channel-specific gen masses + reco observable (raw or kinfit) ──
if CHANNEL=="4q":
    ROOT=os.environ.get("ROOT",f"outputs/treemaker/4q/step2/had_scap_ctrl_s1/p8_ee_WW_ecm{ECM}.root")
    need=["gen_W1_m","gen_W2_m","gen_WW_m"]
    if OBS=="kinfit": need+=["kinfit4q_Wa_m","kinfit4q_Wb_m","kinfit4q_valid"]
    for i in (1,2,3,4): need+=[f"reco_jet{i}_p",f"reco_jet{i}_theta",f"reco_jet{i}_phi"]
    a=uproot.open(ROOT)["events"].arrays(need,library="np"); Nr=len(a["gen_W1_m"])
    g1,g2,mWW=a["gen_W1_m"],a["gen_W2_m"],a["gen_WW_m"]
    if OBS=="gen":   # SANITY: reco≡gen (no smearing) ⇒ slope MUST be exactly 1
        rA,rB=g1,g2
    elif OBS=="raw":   # pllnops pairing on raw reco dijet masses
        def jv(i):
            p=a[f"reco_jet{i}_p"]; th=a[f"reco_jet{i}_theta"]; ph=a[f"reco_jet{i}_phi"]; st=np.sin(th)
            return np.stack([p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th),p],1)
        J={i:jv(i) for i in (1,2,3,4)}
        def dm(u,v): s=u+v; return np.sqrt(np.maximum(s[:,3]**2-s[:,0]**2-s[:,1]**2-s[:,2]**2,0))
        PO=[((1,2),(3,4)),((1,3),(2,4)),((1,4),(2,3))]
        mA=np.stack([dm(J[p[0][0]],J[p[0][1]]) for p in PO],1); mB=np.stack([dm(J[p[1][0]],J[p[1][1]]) for p in PO],1)
        sc=-2*np.log(np.maximum(bw(mA,MW_REF),1e-300))-2*np.log(np.maximum(bw(mB,MW_REF),1e-300))-2*DECAY_P*(np.log(np.maximum(mA,1e-9))+np.log(np.maximum(mB,1e-9)))
        pid=np.argmin(sc,1); r=np.arange(Nr); rA,rB=mA[r,pid],mB[r,pid]
    else:            # kinfit-constrained winner-pairing dijet masses
        rA,rB=a["kinfit4q_Wa_m"],a["kinfit4q_Wb_m"]
    rhi,rlo=np.maximum(rA,rB),np.minimum(rA,rB); ghi,glo=np.maximum(g1,g2),np.minimum(g1,g2)
    okv=(np.isfinite(g1)&np.isfinite(g2)&np.isfinite(mWW)&np.isfinite(rhi)&np.isfinite(rlo)&(rhi>0)&(rlo>0)&((g1+g2)<mWW))
    if OBS=="kinfit": okv=okv&(a["kinfit4q_valid"]==1)
else:  # lnuqq
    ROOT=os.environ.get("ROOT",f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root")
    need=["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_Whad_m","reco_Wlep_m","reco_jet1_p","reco_jet2_p","reco_lep_p"]
    if OBS=="kinfit": need+=["kinfit_Whad_m","kinfit_Wlep_m","kinfit_chi2"]
    a=uproot.open(ROOT)["events"].arrays(need,library="np"); Nr=len(a["gen_Whad_m"])
    ghi,glo,mWW=a["gen_Whad_m"],a["gen_Wlep_m"],a["gen_WW_m"]
    if OBS=="gen":   rhi,rlo=a["gen_Whad_m"],a["gen_Wlep_m"]    # SANITY: slope must be 1
    elif OBS=="raw": rhi,rlo=a["reco_Whad_m"],a["reco_Wlep_m"]
    else:            rhi,rlo=a["kinfit_Whad_m"],a["kinfit_Wlep_m"]
    okv=(np.isfinite(ghi)&np.isfinite(glo)&np.isfinite(mWW)&np.isfinite(rhi)&np.isfinite(rlo)&(rhi>0)&(rlo>0)&(a["reco_jet1_p"]>0)&(a["reco_jet2_p"]>0)&(a["reco_lep_p"]>0)&((ghi+glo)<mWW))
    if OBS=="kinfit": okv=okv&np.isfinite(a["kinfit_chi2"])

idx=np.where(okv)[0]; N=len(idx)
ghi,glo,sq=ghi[idx],glo[idx],mWW[idx]; rhi,rlo=rhi[idx],rlo[idx]
mw_scan=np.linspace(MWLO,MWHI,NMW)
print(f"[method] CHANNEL={CHANNEL} OBS={OBS} ecm{ECM} N={N} bin={FOLD_BIN}")
print(f"  reco obs: <hi>={rhi.mean():.2f} std={rhi.std():.2f}  <lo>={rlo.mean():.2f} std={rlo.std():.2f}")

# ── gen reweighting matrix (parallel over mW) ──
_wn=np.linspace(float(sq.min())-0.5,float(sq.max())+0.5,LOGZ_TAB_N)
def _sw(jc):
    o=np.empty((N,len(jc)))
    for c,j in enumerate(jc): o[:,c]=n2ll(ghi,glo,sq,mw_scan[j],np.interp(sq,_wn,log_Z(_wn,mw_scan[j])))
    return jc,o
Lgen=np.empty((N,NMW))
for jc,o in _pmap(_sw,[c for c in np.array_split(np.arange(NMW),min(NWORKERS,NMW)) if len(c)]): Lgen[:,jc]=o
egen=Lgen.sum(0); egen-=egen.min(); mw0=parab_min(mw_scan,egen)[0]
def Lgen_col(mu):  # interpolated gen column at mW=mu (for injection weights)
    k=int(np.clip(np.searchsorted(mw_scan,mu)-1,0,NMW-2)); f=(mu-mw_scan[k])/(mw_scan[k+1]-mw_scan[k])
    return Lgen[:,k]*(1-f)+Lgen[:,k+1]*f
Lref=Lgen_col(mw0)

# ── reco fold per-event L matrix (k-fold), parallel over mW ──
rng=np.random.default_rng(2024); fold=rng.integers(0,FOLD_K,N)
e1=np.arange(30.0,98.0+1e-6,FOLD_BIN); c1=0.5*(e1[:-1]+e1[1:])
def _fw(jc):
    out=np.zeros((N,len(jc)))
    for c,j in enumerate(jc):
        wfull=np.exp(-0.5*(Lgen[:,j]-Lref)); col=np.zeros(N)
        for k in range(FOLD_K):
            tr=fold!=k; te=fold==k; ii=np.where(te)[0]
            H,_,_=np.histogram2d(rhi[tr],rlo[tr],bins=[e1,e1],weights=wfull[tr])
            if FOLD_SM>0: H=gaussian_filter(H,sigma=FOLD_SM)
            H=H/np.maximum(H.sum(),1e-300)
            pv=RegularGridInterpolator((c1,c1),H,bounds_error=False,fill_value=1e-300)(np.stack([rhi[te],rlo[te]],1))
            col[ii]=-2.0*np.log(np.maximum(pv,1e-300))
        out[:,c]=col
    return jc,out
Lf=np.zeros((N,NMW))
for jc,o in _pmap(_fw,[c for c in np.array_split(np.arange(NMW),min(NWORKERS,NMW)) if len(c)]): Lf[:,jc]=o
good=np.all(np.isfinite(Lf),axis=1)&(np.ptp(Lf,axis=1)>0); Lg=Lf[good]; Lgg=Lgen[good]; Ng=Lg.shape[0]
def gen_pe(): return parab_min(mw_scan,(Lgg.sum(0)-Lgg.sum(0).min()))[0]
gpe=gen_pe()
reco_fit,sig_full=parab_min(mw_scan,(Lg.sum(0)-Lg.sum(0).min()))
print(f"  gen_pe={gpe:.4f}  reco_fit={reco_fit:.4f}  closure={1000*(reco_fit-gpe):+.1f} MeV  σ_full={1000*sig_full:.2f} MeV (Ng={Ng})")

# ── LINEARITY: inject true mW = mw0+Δ (reweight gen), refit the reco fold, measure slope ──
inj=np.array([-0.30,-0.20,-0.10,0.0,0.10,0.20,0.30]); fitted=[]
Lref_g=Lgen_col(mw0)[good]
for d in inj:
    v=np.exp(-0.5*(Lgen_col(mw0+d)[good]-Lref_g))           # reweight sample to injected mW
    mu=parab_min(mw_scan,(v[:,None]*Lg).sum(0)-((v[:,None]*Lg).sum(0)).min())[0]
    fitted.append(mu)
fitted=np.array(fitted); inj_abs=mw0+inj
A=np.polyfit(inj_abs,fitted,1); slope=A[0]; resid=fitted-np.polyval(A,inj_abs)
print(f"  LINEARITY: slope={slope:.4f}  max|resid|={1000*np.max(np.abs(resid)):.2f} MeV  (inj ±0.3 GeV around mw0={mw0:.3f})")

TAG=f"{CHANNEL}_{OBS}_ecm{ECM}"
res=dict(channel=CHANNEL,obs=OBS,ecm=ECM,N=int(N),Ng=int(Ng),bin=FOLD_BIN,gen_pe=gpe,reco_fit=reco_fit,
         closure_MeV=1000*(reco_fit-gpe),sig_full_MeV=1000*sig_full,sig1ev=1000*sig_full*np.sqrt(Ng),
         slope=slope,max_resid_MeV=1000*float(np.max(np.abs(resid))),mw0=mw0,
         inj=inj_abs.tolist(),fitted=fitted.tolist())
RD="jax_prototype/anatomy_results"; os.makedirs(RD,exist_ok=True)
json.dump(res,open(f"{RD}/method_{TAG}.json","w"),indent=2)
# linearity plot
fig,ax=plt.subplots(figsize=(6.2,5.4))
ax.plot(inj_abs,fitted,"o",ms=8,color="C0"); ax.plot(inj_abs,np.polyval(A,inj_abs),"-",color="C3",label=f"slope={slope:.3f}")
ax.plot(inj_abs,inj_abs,":",color="grey",label="ideal slope 1")
ax.set_xlabel("injected true mW [GeV]"); ax.set_ylabel("fitted mW [GeV]")
ax.set_title(f"Linearity {CHANNEL} {OBS} ecm{ECM}\nσ_full={1000*sig_full:.1f} MeV @ N={Ng}  closure={1000*(reco_fit-gpe):+.1f} MeV")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
p=f"{EOSW}/linearity_{TAG}.png"; plt.savefig(p,dpi=110); print(f"[plot] {p}  [json] method_{TAG}.json")
