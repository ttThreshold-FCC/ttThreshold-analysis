#!/usr/bin/env python3
# DAY13 NEXT-1 step (a): extend the C5 phase-space-MC (prototype_pairing_ps_mc.py) to RECO level.
# C5 validated the GEN-level wrong/true dijet-mass spectra from a tiny phase-space MC (ISR via the data s'
# spectrum).  To use the densities as a RECO discriminant we must fold in the gen->reco jet response.  This
# script:
#   1. calibrates the jet response from the data (reco jet <-> gen quark matched by min total angle):
#        - per-JET   : momentum response R=p_reco/p_gen + angular kick dtheta  (model "jet")
#        - per-DIJET : multiplicative mass factor m_reco/m_gen on the angle-matched true W dijets (model "dijet")
#   2. generates the phase-space MC partons at the data s' spectrum, applies each smearing model,
#   3. validates the SMEARED MC reco true/wrong (m_hi,m_lo) spectra against the DATA reco spectra (gpt truth).
# If blue MC tracks green data at reco level, the smooth/parametrised template route is viable -> step (b)
# wires it into conv_mw_4q.py as the `tmplLRsmooth` pairing mode.
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# SMEAR=jet|dijet python3 jax_prototype/prototype_pairing_ps_mc_reco.py
import os, numpy as np, uproot
from itertools import permutations
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)
TREES = {160:"had_ff160_genqk",240:"had_ff240_genqk",365:"had_ff365_genqk"}
GW, MW, DECAY_P = 2.085, 80.4, 3.0
GREEN, BLUE = "#2e9e2e", "#1f6fc0"
PORDER = [((1,2),(3,4)),((1,3),(2,4)),((1,4),(2,3))]
SMEAR  = os.environ.get("SMEAR", "dijet")        # dijet | jet  -- dijet WINS: nails the true ridge at all sqrt(s)
                                                 # (per-jet over-inflates true mass @240/365 via the R high-tail)
NMC    = int(os.environ.get("NMC", "1500000"))   # MC partons per ecm (each event -> 1 true-set + 2 wrong-sets)
MATCH_ANG_CUT = float(os.environ.get("MATCH_ANG_CUT", "0.4"))  # rad: drop gross mis-matches in response calib
PERMS = list(permutations(range(4)))

def bw_line(m):
    mwgw=MW*GW; d=m*m-MW*MW; return (mwgw/(d*d+mwgw*mwgw))*m**DECAY_P
def sample_m(n, rng, lo=45.0, hi=125.0, ng=6000):
    mg=np.linspace(lo,hi,ng); cdf=np.cumsum(bw_line(mg)); cdf/=cdf[-1]; return np.interp(rng.random(n),cdf,mg)
def mass4(v):
    return np.sqrt(np.maximum(v[:,3]**2-v[:,0]**2-v[:,1]**2-v[:,2]**2,0))

def ps_partons(sqrts, rng):
    """phase-space WW->4 massless partons at per-event sqrt(s') (ISR baked in via sqrts sampling).
    returns lab-frame 4-vectors a1,a2 (W1) and b1,b2 (W2) and the WW phase-space weight w."""
    n=len(sqrts); s=sqrts**2
    m1=sample_m(n,rng); m2=sample_m(n,rng)
    lam=(s-(m1+m2)**2)*(s-(m1-m2)**2)
    w=np.where((m1+m2<sqrts)&(lam>0),np.sqrt(np.maximum(lam,0)),0.0)
    E1=(s+m1**2-m2**2)/(2*sqrts); E2=sqrts-E1
    pst=np.sqrt(np.maximum(E1**2-m1**2,0))
    b1=pst/np.maximum(E1,1e-9); g1=E1/np.maximum(m1,1e-9)
    b2=pst/np.maximum(E2,1e-9); g2=E2/np.maximum(m2,1e-9)
    def parton(mw,g,b,cosA,phi,zs):
        sA=np.sqrt(np.maximum(1-cosA**2,0)); e=mw/2.0; bz=zs*b
        px=e*sA*np.cos(phi); py=e*sA*np.sin(phi); pz=e*cosA
        E=g*(e+bz*pz); Pz=g*(pz+bz*e); return np.stack([px,py,Pz,E],1)
    c1=2*rng.random(n)-1; p1=2*np.pi*rng.random(n)
    c2=2*rng.random(n)-1; p2=2*np.pi*rng.random(n)
    a1=parton(m1,g1,b1,c1,p1,+1); a2=parton(m1,g1,b1,-c1,p1+np.pi,+1)
    b1v=parton(m2,g2,b2,c2,p2,-1); b2v=parton(m2,g2,b2,-c2,p2+np.pi,-1)
    return a1,a2,b1v,b2v,w

def smear_jet(part, rng, Rpool, Apool):
    """per-jet response: scale |p| by R~Rpool, rotate dir by dtheta~Apool in random azimuth (massless)."""
    p3=part[:,:3]; pmag=np.linalg.norm(p3,axis=1); d=p3/np.maximum(pmag[:,None],1e-9)
    R=Rpool[rng.integers(0,len(Rpool),len(pmag))]; dth=Apool[rng.integers(0,len(Apool),len(pmag))]
    # orthonormal basis perp to d
    ref=np.tile([0.,0.,1.],(len(d),1)); near=np.abs((d*ref).sum(1))>0.9
    ref[near]=[1.,0.,0.]
    e1=np.cross(d,ref); e1/=np.maximum(np.linalg.norm(e1,axis=1)[:,None],1e-9)
    e2=np.cross(d,e1)
    az=2*np.pi*rng.random(len(d))
    dnew=(np.cos(dth)[:,None]*d + np.sin(dth)[:,None]*(np.cos(az)[:,None]*e1+np.sin(az)[:,None]*e2))
    pnew=(pmag*R)[:,None]*dnew
    return np.concatenate([pnew, np.linalg.norm(pnew,axis=1,keepdims=True)],1)

def calib_jet(a, rng):
    """response pool R=p_reco/p_gen and angular pool dtheta from min-angle reco<->gen-quark assignment."""
    N=len(a["gen_pairing_true"])
    RP=np.stack([a[f"reco_jet{i}_p"] for i in (1,2,3,4)],1)
    def jdir(i):
        th=a[f"reco_jet{i}_theta"]; ph=a[f"reco_jet{i}_phi"]; st=np.sin(th)
        return np.stack([st*np.cos(ph),st*np.sin(ph),np.cos(th)],1)
    RD=np.stack([jdir(i) for i in (1,2,3,4)],1)
    GV=np.stack([np.stack([a[f"gen_qW{j}_px"],a[f"gen_qW{j}_py"],a[f"gen_qW{j}_pz"]],1) for j in range(4)],1)
    GP=np.linalg.norm(GV,axis=2); GD=GV/np.maximum(GP[:,:,None],1e-9)
    angM=np.arccos(np.clip(np.einsum("nik,njk->nij",RD,GD),-1,1))
    best=np.full(N,1e9); bestperm=np.zeros((N,4),int)
    for perm in PERMS:
        tot=sum(angM[:,i,perm[i]] for i in range(4)); upd=tot<best; best=np.where(upd,tot,best)
        for i in range(4): bestperm[upd,i]=perm[i]
    Rs=[]; As=[]
    for i in range(4):
        j=bestperm[:,i]; gp=GP[np.arange(N),j]; ang=angM[np.arange(N),i,j]
        keep=ang<MATCH_ANG_CUT
        Rs.append((RP[:,i]/np.maximum(gp,1e-9))[keep]); As.append(ang[keep])
    return np.concatenate(Rs), np.concatenate(As)

def data_reco(a):
    """data reco (m_hi,m_lo) for TRUE partition and pooled WRONG partitions (gpt truth)."""
    def jv(i):
        p=a[f"reco_jet{i}_p"]; th=a[f"reco_jet{i}_theta"]; ph=a[f"reco_jet{i}_phi"]; st=np.sin(th)
        return np.stack([p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th),p],1)
    J={i:jv(i) for i in (1,2,3,4)}
    def dm(u,v):
        s=u+v; return np.sqrt(np.maximum(s[:,3]**2-s[:,0]**2-s[:,1]**2-s[:,2]**2,0))
    mA=np.stack([dm(J[p[0][0]],J[p[0][1]]) for p in PORDER],1)
    mB=np.stack([dm(J[p[1][0]],J[p[1][1]]) for p in PORDER],1)
    hi=np.maximum(mA,mB); lo=np.minimum(mA,mB); gpt=a["gen_pairing_true"].astype(int)
    N=len(gpt); rows=np.arange(N); ok=(gpt>=0)&(gpt<=2)
    thi=hi[rows,gpt][ok]; tlo=lo[rows,gpt][ok]
    wm=np.ones((N,3),bool); wm[rows,gpt]=False; wm&=ok[:,None]
    whi=hi[wm]; wlo=lo[wm]
    return thi,tlo,whi,wlo, a["gen_W1_m"][ok], a["gen_W2_m"][ok], mA, mB, gpt, ok

def dijet_ratio_pool(a, rng):
    """multiplicative pool R=m_reco_true/m_gen_W, reco dijet<->gen W matched by angle (per-dijet model)."""
    def jv(i):
        p=a[f"reco_jet{i}_p"]; th=a[f"reco_jet{i}_theta"]; ph=a[f"reco_jet{i}_phi"]; st=np.sin(th)
        return np.stack([p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th),p],1)
    J={i:jv(i) for i in (1,2,3,4)}; N=len(a["gen_pairing_true"])
    Q={i:np.stack([a[f"gen_qW{i}_px"],a[f"gen_qW{i}_py"],a[f"gen_qW{i}_pz"],a[f"gen_qW{i}_e"]],1) for i in range(4)}
    W1=Q[0]+Q[1]; W2=Q[2]+Q[3]; gpt=a["gen_pairing_true"].astype(int)
    DA=np.zeros((N,4)); DB=np.zeros((N,4))
    for pi,p in enumerate(PORDER):
        sel=gpt==pi; DA[sel]=(J[p[0][0]]+J[p[0][1]])[sel]; DB[sel]=(J[p[1][0]]+J[p[1][1]])[sel]
    def ang(u,v):
        c=(u[:,:3]*v[:,:3]).sum(1)/(np.linalg.norm(u[:,:3],axis=1)*np.linalg.norm(v[:,:3],axis=1)+1e-9)
        return np.arccos(np.clip(c,-1,1))
    swap=(ang(DA,W2)+ang(DB,W1))<(ang(DA,W1)+ang(DB,W2))
    rW1=np.where(swap[:,None],DB,DA); rW2=np.where(swap[:,None],DA,DB)
    def m(v): return np.sqrt(np.maximum(v[:,3]**2-v[:,0]**2-v[:,1]**2-v[:,2]**2,0))
    ok=(gpt>=0)&(gpt<=2)
    r=np.concatenate([(m(rW1)/np.maximum(a["gen_W1_m"],1e-6))[ok],(m(rW2)/np.maximum(a["gen_W2_m"],1e-6))[ok]])
    return r[(r>0.2)&(r<3.0)]

def mc_reco_masses(a1,a2,b1,b2,w, ecm, rng, Rpool,Apool,Dpool):
    if SMEAR=="jet":
        s=lambda P:smear_jet(P,rng,Rpool,Apool)
        a1,a2,b1,b2=s(a1),s(a2),s(b1),s(b2)
        mt1=mass4(a1+a2); mt2=mass4(b1+b2)
        wA1=mass4(a1+b1); wA2=mass4(a2+b2); wB1=mass4(a1+b2); wB2=mass4(a2+b1)
    else:  # dijet multiplicative
        mt1=mass4(a1+a2); mt2=mass4(b1+b2)
        wA1=mass4(a1+b1); wA2=mass4(a2+b2); wB1=mass4(a1+b2); wB2=mass4(a2+b1)
        def sc(x): return x*Dpool[rng.integers(0,len(Dpool),len(x))]
        mt1,mt2,wA1,wA2,wB1,wB2=map(sc,(mt1,mt2,wA1,wA2,wB1,wB2))
    thi=np.maximum(mt1,mt2); tlo=np.minimum(mt1,mt2)
    whi=np.concatenate([np.maximum(wA1,wA2),np.maximum(wB1,wB2)]); ww=np.concatenate([w,w])
    wlo=np.concatenate([np.minimum(wA1,wA2),np.minimum(wB1,wB2)])
    return thi,tlo,w, whi,wlo,ww

rng=np.random.default_rng(1)
fig,axs=plt.subplots(3,4,figsize=(17,10.5))
COLS=[("true m$_{hi}$ [GeV]",(55,105)),("true m$_{lo}$ [GeV]",(45,100)),
      ("wrong m$_{hi}$ [GeV]",(20,320)),("wrong m$_{lo}$ [GeV]",(10,140))]
for ie,ecm in enumerate((160,240,365)):
    F=f"outputs/treemaker/4q/step2_ff/{TREES[ecm]}/p8_ee_WW_ecm{ecm}.root"
    br=["gen_W1_m","gen_W2_m","gen_WW_m","gen_pairing_true"]+[f"reco_jet{i}_{c}" for i in (1,2,3,4) for c in ("p","theta","phi")]+[f"gen_qW{i}_{c}" for i in range(4) for c in ("px","py","pz","e")]
    a=uproot.open(F)["events"].arrays(br,library="np")
    thi_d,tlo_d,whi_d,wlo_d,_,_,_,_,_,_=data_reco(a)
    Rpool,Apool=calib_jet(a,rng); Dpool=dijet_ratio_pool(a,rng)
    sp=a["gen_WW_m"]; sp=sp[np.isfinite(sp)&(sp>90.0)]
    mcsq=sp[rng.integers(0,len(sp),NMC)]
    a1,a2,b1,b2,w=ps_partons(mcsq,rng)
    thi_m,tlo_m,tw,whi_m,wlo_m,ww=mc_reco_masses(a1,a2,b1,b2,w,ecm,rng,Rpool,Apool,Dpool)
    series=[(thi_d,thi_m,tw),(tlo_d,tlo_m,tw),(whi_d,whi_m,ww),(wlo_d,wlo_m,ww)]
    for ic,(xlab,rg) in enumerate(COLS):
        ax=axs[ie,ic]; dat,mcv,mcw=series[ic]; bins=np.linspace(*rg,80)
        ax.hist(dat,bins=bins,density=True,histtype="stepfilled",color=GREEN,alpha=0.40,label="data reco (gpt truth)")
        ax.hist(mcv,bins=bins,weights=mcw,density=True,histtype="step",color=BLUE,lw=1.9,label=f"smeared MC ({SMEAR})")
        if ic==0: ax.set_ylabel(f"√s = {ecm} GeV\nnormalised",fontsize=11)
        if ie==0: ax.set_title(xlab,fontsize=11)
        if ie==2: ax.set_xlabel(xlab,fontsize=10)
        ax.grid(alpha=.25)
        if ie==0 and ic==0: ax.legend(fontsize=8.5,loc="upper left")
    # numeric: true m_hi median + wrong m_hi quartiles
    def q(x,wt,p):
        i=np.argsort(x); xs=x[i]; cw=np.cumsum(wt[i]); cw/=cw[-1]; return np.interp(p,cw,xs)
    dt=[np.percentile(thi_d,p) for p in (25,50,75)]; mt=[q(thi_m,tw,p/100) for p in (25,50,75)]
    dw=[np.percentile(whi_d,p) for p in (25,50,75)]; mw_=[q(whi_m,ww,p/100) for p in (25,50,75)]
    print(f"ecm{ecm} [{SMEAR}] true m_hi q data{np.round(dt,1)} MC{np.round(mt,1)} | wrong m_hi q data{np.round(dw,1)} MC{np.round(mw_,1)}")
fig.suptitle(f"RECO-level pairing densities: smeared phase-space MC ({SMEAR}) vs DATA reco (gpt truth)",fontsize=13,y=0.998)
plt.tight_layout(rect=[0,0,1,0.985])
p=os.path.join(EOSW,f"pairing_ps_mc_reco_{SMEAR}.png"); plt.savefig(p,dpi=115); print(f"[plot] {p}")
