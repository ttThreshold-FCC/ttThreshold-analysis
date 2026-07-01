#!/usr/bin/env python3
# Statistical reach of the WW→4q forward-fold mW fit: σ_mW ∝ 1/√N, MIRRORING conv_mw_asimov.py (μνqq).
# Builds the adopted pllnops-pairing reco fold likelihood, verifies 1/√N by subsampling the ACTUAL fold,
# anchors at the full-sample σ, projects to large N, and maps N→integrated luminosity (FCC-ee WW threshold).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# NWORKERS=48 python3 jax_prototype/conv_mw_4q_asimov.py [ecm]
import sys, os
for _v in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_v]="1"
import numpy as np, multiprocessing as mp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import uproot
NWORKERS=int(os.environ.get("NWORKERS","48")); _FORK=mp.get_context("fork")
def _pmap(w,ch):
    nw=max(1,min(NWORKERS,len(ch)))
    if nw==1: return [w(c) for c in ch]
    with _FORK.Pool(nw) as p: return p.map(w,ch)

ECM=int(sys.argv[1]) if len(sys.argv)>1 else 160
GW=2.085; MW_REF=80.4; DECAY_P=3.0; NMW=121; MWLO,MWHI=79.0,81.5; LOGZ_N=256; LOGZ_TAB_N=400
FOLD_K=5; FOLD_BIN=float(os.environ.get("FOLD_BIN","0.25")); FOLD_SM=0.6; NDRAW=int(os.environ.get("NDRAW","40"))
ROOT=os.environ.get("ROOT",f"outputs/treemaker/4q/step2/had_scap_ctrl_s1/p8_ee_WW_ecm{ECM}.root")
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW,exist_ok=True)

_GX,_GWl=np.polynomial.legendre.leggauss(LOGZ_N); _W2=_GWl[:,None]*_GWl[None,:]
def log_Z(m_WW,mW,gW=GW):
    m_WW=np.atleast_1d(np.asarray(m_WW,float)); mwgw=mW*gW; mW2=mW*mW; out=np.empty(m_WW.shape,float)
    CH=max(1,50_000_000//(LOGZ_N*LOGZ_N))
    for a in range(0,len(m_WW),CH):
        s=(m_WW[a:a+CH])**2; t_min=np.arctan(-mW2/mwgw); t_max=np.arctan((s-mW2)/mwgw)
        half_d=0.5*(t_max-t_min); half_s=0.5*(t_max+t_min); t=half_d[:,None]*_GX[None,:]+half_s[:,None]
        m=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1.0/m
        rf=m**DECAY_P if DECAY_P else np.ones_like(m)
        mh=m[:,:,None]; ml=m[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=s[:,None,None]; rh=rf[:,:,None]; rl=rf[:,None,:]
        lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2)
        integ=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il*rh*rl/(4.0*sE),0.0)
        out[a:a+CH]=np.log(np.maximum(np.sum(_W2[None]*integ,axis=(1,2))*half_d*half_d,1e-300))
    return out
def bw(m,mW,gW=GW):
    mwgw=mW*gW; d=m*m-mW*mW; return mwgw/(d*d+mwgw*mwgw)
def n2ll(mh,ml,mWW,mW,logz):
    s=mWW*mWW; lam=(s-(mh+ml)**2)*(s-(mh-ml)**2); bad=lam<=0; lam=np.where(bad,1.0,lam)
    tv=(-2.0*np.log(np.maximum(bw(mh,mW),1e-300))-2.0*np.log(np.maximum(bw(ml,mW),1e-300))-np.log(lam)+2.0*np.log(s)+2.0*logz)
    if DECAY_P: tv=tv-2.0*DECAY_P*(np.log(np.maximum(mh,1e-9))+np.log(np.maximum(ml,1e-9)))
    return np.where(bad,1e6,tv)

# ── load + pllnops pairing reco masses ──
t=uproot.open(ROOT)["events"]; need=["gen_W1_m","gen_W2_m","gen_WW_m","gen_pairing_true"]
for i in (1,2,3,4): need+=[f"reco_jet{i}_p",f"reco_jet{i}_theta",f"reco_jet{i}_phi"]
a=t.arrays(need,library="np"); Nr=len(a["gen_W1_m"])
def jv(i):
    p=a[f"reco_jet{i}_p"]; th=a[f"reco_jet{i}_theta"]; ph=a[f"reco_jet{i}_phi"]; st=np.sin(th)
    return np.stack([p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th),p],1)
J={i:jv(i) for i in (1,2,3,4)}
def dm(u,v): s=u+v; return np.sqrt(np.maximum(s[:,3]**2-s[:,0]**2-s[:,1]**2-s[:,2]**2,0))
PO=[((1,2),(3,4)),((1,3),(2,4)),((1,4),(2,3))]
mA=np.stack([dm(J[p[0][0]],J[p[0][1]]) for p in PO],1); mB=np.stack([dm(J[p[1][0]],J[p[1][1]]) for p in PO],1)
# pllnops pick: const×m³ BW score, NO √λ
sc=-2.0*np.log(np.maximum(bw(mA,MW_REF),1e-300))-2.0*np.log(np.maximum(bw(mB,MW_REF),1e-300))
sc=sc-2.0*DECAY_P*(np.log(np.maximum(mA,1e-9))+np.log(np.maximum(mB,1e-9)))
pid=np.argmin(sc,1); rows=np.arange(Nr)
rA=mA[rows,pid]; rB=mB[rows,pid]; reco_hi=np.maximum(rA,rB); reco_lo=np.minimum(rA,rB)
g1,g2,mWWg=a["gen_W1_m"],a["gen_W2_m"],a["gen_WW_m"]; gen_hi=np.maximum(g1,g2); gen_lo=np.minimum(g1,g2)
ok=(np.isfinite(g1)&np.isfinite(g2)&np.isfinite(mWWg)&np.all(np.isfinite(mA),1)&np.all(np.isfinite(mB),1)&((g1+g2)<mWWg))
idx=np.where(ok)[0]; N=len(idx)
mqq_g,mlv_g,sq=gen_hi[idx],gen_lo[idx],mWWg[idx]; rhi,rlo=reco_hi[idx],reco_lo[idx]
mw_scan=np.linspace(MWLO,MWHI,NMW)
print(f"[asimov-4q] ecm{ECM} N={N} pairing=pllnops bin={FOLD_BIN}")

# ── gen per-event L (reweighting weights), parallel over mW ──
_wn=np.linspace(float(sq.min())-0.5,float(sq.max())+0.5,LOGZ_TAB_N)
def _sw(jc):
    out=np.empty((N,len(jc)))
    for c,j in enumerate(jc):
        lz=np.interp(sq,_wn,log_Z(_wn,mw_scan[j])); out[:,c]=n2ll(mqq_g,mlv_g,sq,mw_scan[j],lz)
    return jc,out
Lgen=np.empty((N,NMW))
for jc,o in _pmap(_sw,[c for c in np.array_split(np.arange(NMW),min(NWORKERS,NMW)) if len(c)]): Lgen[:,jc]=o
j0=int(np.argmin(Lgen.sum(0)))

# ── reco fold per-event L matrix (k-fold, hist), parallel over mW ──
rng=np.random.default_rng(2024); fold=rng.integers(0,FOLD_K,N)
e1=np.arange(30.0,98.0+1e-6,FOLD_BIN); c1=0.5*(e1[:-1]+e1[1:])
def _fw(jc):
    out=np.zeros((N,len(jc)))
    for c,j in enumerate(jc):
        wfull=np.exp(-0.5*(Lgen[:,j]-Lgen[:,j0])); col=np.zeros(N)
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
good=np.all(np.isfinite(Lf),axis=1)&(np.ptp(Lf,axis=1)>0); Lg=Lf[good]; Ng=Lg.shape[0]

def curv_sigma(L):
    e=L.sum(0); e=e-e.min(); i=int(np.clip(np.argmin(e),2,NMW-3))
    c=np.polyfit(mw_scan[i-2:i+3],e[i-2:i+3],2); return 1.0/np.sqrt(c[0]) if c[0]>0 else np.nan
sig_full=curv_sigma(Lg)
print(f"[asimov-4q] full-sample σ_curv = {1000*sig_full:.2f} MeV  (Ng={Ng})")

# ── subsample to verify 1/√N ──
Nsub=[3000,6000,12000,Ng]; rs=np.random.default_rng(7); mN,mS=[],[]
for n in Nsub:
    ss=[curv_sigma(Lg[rs.choice(Ng,n,replace=False)]) for _ in range(NDRAW)]
    mN.append(n); mS.append(np.nanmedian(ss))
    print(f"  N={n:>7}  σ={1000*mS[-1]:6.2f} MeV   (1/√N pred {1000*sig_full*np.sqrt(Ng/n):6.2f})")
mN,mS=np.array(mN),np.array(mS)
proj_N=np.array([1e6,1e7,1e8]); proj_S=sig_full*np.sqrt(Ng/proj_N)
for n,s in zip(proj_N,proj_S): print(f"  PROJECT N={n:.0e}  σ≈{1000*s:.2f} MeV")

# ── N → integrated luminosity (FCC-ee WW threshold). ASSUMPTIONS (stated): σ_WW, BR(4q), ε_sel ──
SIG_WW=float(os.environ.get("SIG_WW","4.0"))      # pb, representative e+e-→WW at the threshold scan (√s-dep: ~1.5@157,~5@162.5)
BR4Q=0.457                                        # BR(WW→qqqq)=(0.676)²
EPS=float(os.environ.get("EPS","0.43"))           # 4q selection eff incl. genuine-4-jet √d45<7 (keeps ~45% of gen-4q)
ev_per_pbinv=SIG_WW*BR4Q*EPS                      # selected 4q events per pb⁻¹
def N_to_L_abinv(n): return n/ev_per_pbinv/1e6    # pb⁻¹ → ab⁻¹  (1 ab⁻¹ = 1e6 pb⁻¹)
def L_to_N(L_ab): return L_ab*1e6*ev_per_pbinv
L_FCC=10.0                                          # ab⁻¹, FCC-ee WW threshold (4-IP baseline ~10; 2-IP ~half)
N_FCC=L_to_N(L_FCC); sig_FCC=sig_full*np.sqrt(Ng/N_FCC)
print(f"\n[lumi] σ_WW={SIG_WW}pb BR4q={BR4Q} ε={EPS} ⇒ {ev_per_pbinv:.3f} sel-4q/pb⁻¹ ; sample Ng={Ng} ≈ {N_to_L_abinv(Ng)*1000:.2f} fb⁻¹")
print(f"[lumi] FCC-ee WW threshold L={L_FCC} ab⁻¹ ⇒ N_4q≈{N_FCC:.2e}  ⇒  σ_mW(stat,4q) ≈ {1000*sig_FCC:.2f} MeV")

# ── plot: σ vs N (bottom axis) + luminosity (top axis) ──
fig,ax=plt.subplots(figsize=(8.4,5.8))
xx=np.logspace(np.log10(2000),np.log10(2e8),200)
ax.plot(xx,1000*sig_full*np.sqrt(Ng/xx),"-",color="gray",lw=1.4,label=r"$1/\sqrt{N}$ law (anchored at full sample)")
ax.plot(mN,1000*mS,"o",color="C0",ms=8,label="MC subsampling (curvature σ)")
ax.plot([Ng],[1000*sig_full],"*",color="C3",ms=17,label=f"this sample: {1000*sig_full:.0f} MeV @ {Ng/1e3:.1f}k")
ax.plot(proj_N,1000*proj_S,"D",color="C2",ms=9,mfc="none",mew=1.8,label="projection")
ax.axvline(N_FCC,color="C4",ls=":",lw=1.6); ax.plot([N_FCC],[1000*sig_FCC],"P",color="C4",ms=13)
ax.annotate(f"FCC-ee WW thr\n~{L_FCC:.0f} ab⁻¹\nσ≈{1000*sig_FCC:.1f} MeV",(N_FCC,1000*sig_FCC),
            textcoords="offset points",xytext=(-95,18),fontsize=9,color="C4",
            arrowprops=dict(arrowstyle="->",color="C4"))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N selected 4q events")
ax.set_ylabel(r"$\sigma_{m_W}$ (statistical) [MeV]")
ax.set_title(f"WW→4q statistical reach (ecm{ECM}, pllnops): $\\sigma_{{m_W}}\\propto 1/\\sqrt{{N}}$")
ax.grid(which="both",alpha=.3); ax.set_xlim(2e3,2e8); ax.set_ylim(0.15,60)
secx=ax.secondary_xaxis('top',functions=(lambda n: N_to_L_abinv(np.maximum(n,1e-9))*1000, lambda L: L_to_N(L/1000)))
secx.set_xlabel(f"integrated luminosity [fb⁻¹]   (σ_WW={SIG_WW}pb × BR(4q)={BR4Q} × ε={EPS})")
ax.legend(fontsize=8.5,loc="upper right")
plt.tight_layout(); p=os.path.join(EOSW,f"asimov_scaling_4q_ecm{ECM}.png"); plt.savefig(p,dpi=120); print(f"[plot] {p}")
