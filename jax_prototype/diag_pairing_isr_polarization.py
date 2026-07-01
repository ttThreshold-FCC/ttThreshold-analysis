#!/usr/bin/env python3
# DAY14 check (user question): the gen-level phase-space MC (pairing_ps_mc_validation.png) deviates from gen_qW truth
# in the WRONG-pairing mass at 240/365. Is it ISR, or the isotropic-decay approximation (W polarization)?
#   A wrong-pairing invariant mass m(a+b) is Lorentz-INVARIANT -> the ISR boost of the WW system cannot change it, and
#   s'=gen_WW_m is sampled directly (ISR energy loss already in). So the suspect is the W decay angular distribution.
# This script:
#   (1) measures the TRUTH W helicity decay angle cosθ* (from gen_qW) vs the isotropic assumption, per √s;
#   (2) ISR cross-check: splits events into low/high ISR (gen_WW_m percentile) and compares MC-vs-truth in each;
#   (3) regenerates the MC sampling cosθ* from the TRUTH distribution and checks whether the wrong-mass gap closes.
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh ; python3 jax_prototype/diag_pairing_isr_polarization.py
import os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW,exist_ok=True)
TREES={160:"had_ff160_genqk",240:"had_ff240_genqk",365:"had_ff365_genqk"}
GW,MW,DECAY_P=2.085,80.4,3.0; GREEN,BLUE,ORANGE="#2e9e2e","#1f6fc0","#e8820c"
def mass(v): return np.sqrt(np.maximum(v[:,3]**2-v[:,0]**2-v[:,1]**2-v[:,2]**2,0))
def boost_to_rest(p,P):                      # boost 4-vec p into the rest frame of parent 4-vec P
    beta=P[:,:3]/P[:,3:4]; b2=(beta**2).sum(1); b2s=np.where(b2>1e-12,b2,1.0)
    g=1.0/np.sqrt(np.maximum(1-b2,1e-12)); bp=(beta*p[:,:3]).sum(1); E=p[:,3]
    fac=((g-1)/b2s)*bp - g*E
    sp=p[:,:3]+fac[:,None]*beta; En=g*(E-bp)
    return np.concatenate([sp,En[:,None]],1)
def helicity_cos(q,W,WW):                     # cosθ* : daughter q dir in W rest frame · W dir in WW rest frame
    Wcm=boost_to_rest(W,WW); dW=Wcm[:,:3]/np.maximum(np.linalg.norm(Wcm[:,:3],axis=1,keepdims=True),1e-9)
    qr=boost_to_rest(q,W);   dq=qr[:,:3]/np.maximum(np.linalg.norm(qr[:,:3],axis=1,keepdims=True),1e-9)
    return (dW*dq).sum(1)
def load(ecm):
    F=f"outputs/treemaker/4q/step2_ff/{TREES[ecm]}/p8_ee_WW_ecm{ecm}.root"
    br=["gen_WW_m","gen_W1_m","gen_W2_m"]+[f"gen_qW{i}_{c}" for i in range(4) for c in ("px","py","pz","e")]
    a=uproot.open(F)["events"].arrays(br,library="np")
    Q={i:np.stack([a[f"gen_qW{i}_px"],a[f"gen_qW{i}_py"],a[f"gen_qW{i}_pz"],a[f"gen_qW{i}_e"]],1) for i in range(4)}
    return a,Q
def bw_sample_m(n,rng,lo=45,hi=125,ng=6000):
    mg=np.linspace(lo,hi,ng); mwgw=MW*GW; d=mg*mg-MW*MW
    cdf=np.cumsum((mwgw/(d*d+mwgw*mwgw))*mg**DECAY_P); cdf/=cdf[-1]; return np.interp(rng.random(n),cdf,mg)
def ps_mc_wrong(sqrts,rng,cos_sampler):       # returns wrong m_hi, m_lo + weight, with a custom cosθ* sampler
    n=len(sqrts); s=sqrts**2; m1=bw_sample_m(n,rng); m2=bw_sample_m(n,rng)
    lam=(s-(m1+m2)**2)*(s-(m1-m2)**2); w=np.where((m1+m2<sqrts)&(lam>0),np.sqrt(np.maximum(lam,0)),0.)
    E1=(s+m1**2-m2**2)/(2*sqrts);E2=sqrts-E1;pst=np.sqrt(np.maximum(E1**2-m1**2,0))
    b1=pst/np.maximum(E1,1e-9);g1=E1/np.maximum(m1,1e-9);b2=pst/np.maximum(E2,1e-9);g2=E2/np.maximum(m2,1e-9)
    def part(mw,g,b,c,phi,zs):
        sA=np.sqrt(np.maximum(1-c**2,0)); e=mw/2.; bz=zs*b; pz=e*c
        return np.stack([e*sA*np.cos(phi),e*sA*np.sin(phi),g*(pz+bz*e),g*(e+bz*pz)],1)
    c1=cos_sampler(n,rng);p1=2*np.pi*rng.random(n);c2=cos_sampler(n,rng);p2=2*np.pi*rng.random(n)
    A1=part(m1,g1,b1,c1,p1,+1);A2=part(m1,g1,b1,-c1,p1+np.pi,+1);B1=part(m2,g2,b2,c2,p2,-1);B2=part(m2,g2,b2,-c2,p2+np.pi,-1)
    wA1=mass(A1+B1);wA2=mass(A2+B2);wB1=mass(A1+B2);wB2=mass(A2+B1)
    return np.concatenate([np.maximum(wA1,wA2),np.maximum(wB1,wB2)]), np.concatenate([w,w])

rng=np.random.default_rng(3)
fig,axs=plt.subplots(2,3,figsize=(15.5,8.6))
def wq(x,w,p): i=np.argsort(x);xs=x[i];cw=np.cumsum(w[i]);cw/=cw[-1];return np.interp(p,cw,xs)
for ie,ecm in enumerate((160,240,365)):
    a,Q=load(ecm); WW_m=a["gen_WW_m"]; ok=np.isfinite(WW_m)&(WW_m>90)
    W1=Q[0]+Q[1];W2=Q[2]+Q[3];WW=W1+W2
    # truth wrong-pairing masses
    wA1=mass(Q[0]+Q[2]);wA2=mass(Q[1]+Q[3]);wB1=mass(Q[0]+Q[3]);wB2=mass(Q[1]+Q[2])
    twhi=np.concatenate([np.maximum(wA1,wA2),np.maximum(wB1,wB2)])[np.concatenate([ok,ok])]
    # truth helicity cosθ* (both W's)
    cs=np.concatenate([helicity_cos(Q[0],W1,WW)[ok], helicity_cos(Q[2],W2,WW)[ok]])
    # (panel row 1) truth cosθ* vs isotropic
    ax=axs[0,ie]; bins=np.linspace(-1,1,41)
    ax.hist(cs,bins=bins,density=True,histtype="stepfilled",color=GREEN,alpha=.4,label="truth gen_qW")
    ax.axhline(0.5,color="k",ls="--",lw=1.4,label="isotropic")
    ax.set_title(f"√s={ecm}: W decay cosθ* (helicity)"); ax.set_xlabel("cosθ*"); ax.set_ylim(0,None); ax.grid(alpha=.25)
    if ie==0: ax.set_ylabel("normalised"); ax.legend(fontsize=9)
    # build a truth-cosθ* sampler (inverse-CDF of the measured distribution)
    hc,he=np.histogram(cs,bins=np.linspace(-1,1,81),density=True); ce=0.5*(he[:-1]+he[1:])
    cdf=np.cumsum(hc); cdf/=cdf[-1]
    truth_cos=lambda n,r: np.interp(r.random(n),cdf,ce)
    iso_cos  =lambda n,r: 2*r.random(n)-1
    # MC at the data s' spectrum, two angular models
    sp=WW_m[ok]; sqrts=sp[rng.integers(0,len(sp),1_500_000)]
    whi_iso,w_iso=ps_mc_wrong(sqrts,rng,iso_cos)
    whi_pol,w_pol=ps_mc_wrong(sqrts,rng,truth_cos)
    # (panel row 2) wrong m_hi: truth vs isotropic-MC vs truth-cosθ*-MC
    ax=axs[1,ie]; rg=(20,320) if ecm>200 else (20,160); b2=np.linspace(*rg,80)
    ax.hist(twhi,bins=b2,density=True,histtype="stepfilled",color=GREEN,alpha=.4,label="truth gen_qW")
    ax.hist(whi_iso,bins=b2,weights=w_iso,density=True,histtype="step",color=BLUE,lw=1.9,label="MC isotropic (current)")
    ax.hist(whi_pol,bins=b2,weights=w_pol,density=True,histtype="step",color=ORANGE,lw=1.9,label="MC truth-cosθ* (polariz.)")
    ax.set_title(f"√s={ecm}: wrong m_hi"); ax.set_xlabel("wrong-pairing m$_{hi}$ [GeV]"); ax.grid(alpha=.25)
    if ie==0: ax.set_ylabel("normalised"); ax.legend(fontsize=9)
    # numbers: quartiles + ISR split test
    print(f"ecm{ecm} wrong m_hi quartiles  TRUTH {np.round([np.percentile(twhi,p) for p in (25,50,75)],1)} "
          f" iso-MC {np.round([wq(whi_iso,w_iso,p/100) for p in (25,50,75)],1)} "
          f" pol-MC {np.round([wq(whi_pol,w_pol,p/100) for p in (25,50,75)],1)}")
print()
# proper ISR split (per √s): compare iso-MC vs truth in low/high-ISR halves
for ecm in (240,365):
    a,Q=load(ecm); WW_m=a["gen_WW_m"]; ok=np.isfinite(WW_m)&(WW_m>90)
    wA1=mass(Q[0]+Q[2]);wA2=mass(Q[1]+Q[3]); whi_ev=np.maximum(wA1,wA2)  # one wrong mass per event (W1a+W2a grouping)
    med=np.median(WW_m[ok])
    for lab,emsk in (("highISR s'<med",ok&(WW_m<med)),("lowISR s'>=med",ok&(WW_m>=med))):
        tw=whi_ev[emsk]; sp=WW_m[emsk]; sq=sp[rng.integers(0,len(sp),400_000)]
        wi,wwi=ps_mc_wrong(sq,rng,lambda n,r:2*r.random(n)-1)
        dq=[np.percentile(tw,p) for p in (25,50,75)]; mq=[wq(wi,wwi,p/100) for p in (25,50,75)]
        print(f"ecm{ecm} {lab:16s}: truth m_hi q {np.round(dq,1)}  iso-MC q {np.round(mq,1)}  Δmedian={mq[1]-dq[1]:+.1f}")
fig.suptitle("Is the wrong-pairing deviation ISR or W-polarization? (top: truth decay cosθ* vs isotropic; bottom: MC isotropic vs truth-cosθ*)",fontsize=12,y=1.0)
plt.tight_layout(); p=os.path.join(EOSW,"pairing_isr_vs_polarization.png"); fig.savefig(p,dpi=118); print("\n[plot]",p)
