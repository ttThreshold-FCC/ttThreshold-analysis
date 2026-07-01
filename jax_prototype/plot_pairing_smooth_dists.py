#!/usr/bin/env python3
# DAY13 NEXT-1 distributions for the SMOOTH phase-space-MC pairing discriminant (tmplLRsma):
#   FIG 1 pairing_smooth_angle_validation.png — within-W opening angle (true & wrong, hi/lo): smeared MC vs DATA reco.
#         The angle analog of pairing_ps_mc_reco_dijet.png (mass).  Also overlays the Δθ-vs-angle smear bias the
#         additive model misses (the leading modeling residual flagged in review).
#   FIG 2 pairing_smooth_logLR.png — the per-partition smooth log-LR = logP_true − logP_wrong for the TRUE vs the WRONG
#         data partitions, mass-only (dashed) vs mass+angle (solid), per √s.  Shows the discriminant separation and how
#         the within-W angle pulls true partitions up / wrong partitions down (the threshold fix).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh ; python3 jax_prototype/plot_pairing_smooth_dists.py
import os, numpy as np, uproot
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)
TREES={160:"had_ff160_genqk",240:"had_ff240_genqk",365:"had_ff365_genqk"}
GW,MW,DECAY_P=2.085,80.4,3.0; GREEN,RED,BLUE="#2e9e2e","#d23b3b","#1f6fc0"
PORDER=[((1,2),(3,4)),((1,3),(2,4)),((1,4),(2,3))]
NMC=2_000_000; MLO,MHI,MBIN,ABIN,SM=20.,340.,2.0,6.0,1.0
def jet4(a,i):
    p=a[f"reco_jet{i}_p"];th=a[f"reco_jet{i}_theta"];ph=a[f"reco_jet{i}_phi"];st=np.sin(th)
    return np.stack([p*st*np.cos(ph),p*st*np.sin(ph),p*np.cos(th),p],1)
def mass4(v): return np.sqrt(np.maximum(v[:,3]**2-v[:,0]**2-v[:,1]**2-v[:,2]**2,0))
def opang(u,v):
    c=(u[:,:3]*v[:,:3]).sum(1)/(np.linalg.norm(u[:,:3],axis=1)*np.linalg.norm(v[:,:3],axis=1)+1e-9)
    return np.degrees(np.arccos(np.clip(c,-1,1)))
def load(ecm):
    F=f"outputs/treemaker/4q/step2_ff/{TREES[ecm]}/p8_ee_WW_ecm{ecm}.root"
    br=["gen_W1_m","gen_W2_m","gen_WW_m","gen_pairing_true"]+[f"reco_jet{i}_{c}" for i in (1,2,3,4) for c in ("p","theta","phi")]+[f"gen_qW{i}_{c}" for i in range(4) for c in ("px","py","pz","e")]
    return uproot.open(F)["events"].arrays(br,library="np")
def calib(a):                                # R=m_reco/m_gen, Δθ=θ_reco−θ_gen pools (true pairing, angle-matched)
    N=len(a["gen_pairing_true"]); J={i:jet4(a,i) for i in (1,2,3,4)}
    Q={i:np.stack([a[f"gen_qW{i}_px"],a[f"gen_qW{i}_py"],a[f"gen_qW{i}_pz"],a[f"gen_qW{i}_e"]],1) for i in range(4)}
    W1=Q[0]+Q[1];W2=Q[2]+Q[3];gpt=a["gen_pairing_true"].astype(int)
    DA=np.zeros((N,4));DB=np.zeros((N,4));tA=np.zeros(N);tB=np.zeros(N)
    for pi,p in enumerate(PORDER):
        s=gpt==pi; DA[s]=(J[p[0][0]]+J[p[0][1]])[s]; DB[s]=(J[p[1][0]]+J[p[1][1]])[s]
        tA[s]=opang(J[p[0][0]],J[p[0][1]])[s]; tB[s]=opang(J[p[1][0]],J[p[1][1]])[s]
    sw=(opang(DA,W2)+opang(DB,W1))<(opang(DA,W1)+opang(DB,W2))
    rW1=np.where(sw[:,None],DB,DA);rW2=np.where(sw[:,None],DA,DB);tW1=np.where(sw,tB,tA);tW2=np.where(sw,tA,tB)
    ok=(gpt>=0)&(gpt<=2)
    R=np.concatenate([(mass4(rW1)/np.maximum(a["gen_W1_m"],1e-6))[ok],(mass4(rW2)/np.maximum(a["gen_W2_m"],1e-6))[ok]])
    gth=np.concatenate([opang(Q[0],Q[1])[ok],opang(Q[2],Q[3])[ok]])
    dth=np.concatenate([(tW1-opang(Q[0],Q[1]))[ok],(tW2-opang(Q[2],Q[3]))[ok]])
    return R[(R>0.2)&(R<3.0)], dth, gth                       # also return gen-angle to show Δθ(angle)
def ps_mc(a,rng):                            # smeared-MC true/wrong masses + within-W angles (sorted hi/lo)
    sp=a["gen_WW_m"]; sp=sp[np.isfinite(sp)&(sp>90)]; sqrts=sp[rng.integers(0,len(sp),NMC)]; s=sqrts**2; mwgw=MW*GW
    mg=np.linspace(45,125,6000); dd=mg*mg-MW*MW; cdf=np.cumsum((mwgw/(dd*dd+mwgw*mwgw))*mg**DECAY_P); cdf/=cdf[-1]
    m1=np.interp(rng.random(NMC),cdf,mg); m2=np.interp(rng.random(NMC),cdf,mg)
    lam=(s-(m1+m2)**2)*(s-(m1-m2)**2); w=np.where((m1+m2<sqrts)&(lam>0),np.sqrt(np.maximum(lam,0)),0.)
    E1=(s+m1**2-m2**2)/(2*sqrts);E2=sqrts-E1;pst=np.sqrt(np.maximum(E1**2-m1**2,0))
    b1=pst/np.maximum(E1,1e-9);g1=E1/np.maximum(m1,1e-9);b2=pst/np.maximum(E2,1e-9);g2=E2/np.maximum(m2,1e-9)
    def part(mw,g,b,c,phi,zs):
        sA=np.sqrt(np.maximum(1-c**2,0)); e=mw/2.; bz=zs*b; pz=e*c
        return np.stack([e*sA*np.cos(phi),e*sA*np.sin(phi),g*(pz+bz*e),g*(e+bz*pz)],1)
    c1=2*rng.random(NMC)-1;p1=2*np.pi*rng.random(NMC);c2=2*rng.random(NMC)-1;p2=2*np.pi*rng.random(NMC)
    A1=part(m1,g1,b1,c1,p1,+1);A2=part(m1,g1,b1,-c1,p1+np.pi,+1);B1=part(m2,g2,b2,c2,p2,-1);B2=part(m2,g2,b2,-c2,p2+np.pi,-1)
    R,dth,_=calib(a)
    sm=lambda x:x*R[rng.integers(0,len(R),len(x))]; sa=lambda x:np.clip(x+dth[rng.integers(0,len(dth),len(x))],0,180)
    mt1=sm(mass4(A1+A2));mt2=sm(mass4(B1+B2)); at1=sa(opang(A1,A2));at2=sa(opang(B1,B2))
    w11=sm(mass4(A1+B1));w12=sm(mass4(A2+B2));w21=sm(mass4(A1+B2));w22=sm(mass4(A2+B1))
    aw11=sa(opang(A1,B1));aw12=sa(opang(A2,B2));aw21=sa(opang(A1,B2));aw22=sa(opang(A2,B1))
    cat=np.concatenate
    return dict(t_mhi=np.maximum(mt1,mt2),t_mlo=np.minimum(mt1,mt2),t_ahi=np.maximum(at1,at2),t_alo=np.minimum(at1,at2),tw=w,
                w_mhi=cat([np.maximum(w11,w12),np.maximum(w21,w22)]),w_mlo=cat([np.minimum(w11,w12),np.minimum(w21,w22)]),
                w_ahi=cat([np.maximum(aw11,aw12),np.maximum(aw21,aw22)]),w_alo=cat([np.minimum(aw11,aw12),np.minimum(aw21,aw22)]),
                ww=cat([w,w]))
def data_obs(a):                             # data reco masses + angles, sorted hi/lo per partition; + gpt mask
    J={i:jet4(a,i) for i in (1,2,3,4)};gpt=a["gen_pairing_true"].astype(int);N=len(gpt);rows=np.arange(N)
    mA=np.stack([mass4(J[p[0][0]]+J[p[0][1]]) for p in PORDER],1);mB=np.stack([mass4(J[p[1][0]]+J[p[1][1]]) for p in PORDER],1)
    aA=np.stack([opang(J[p[0][0]],J[p[0][1]]) for p in PORDER],1);aB=np.stack([opang(J[p[1][0]],J[p[1][1]]) for p in PORDER],1)
    hi=np.maximum(mA,mB);lo=np.minimum(mA,mB);ahi=np.maximum(aA,aB);alo=np.minimum(aA,aB)
    ok=(gpt>=0)&(gpt<=2)
    return hi,lo,ahi,alo,gpt,ok
def dens(samples,wt,edg):
    H,_=np.histogramdd(samples,bins=edg,weights=wt)
    if SM>0:H=gaussian_filter(H,sigma=SM)
    H=np.maximum(H/np.maximum(H.sum(),1e-300),1e-300)
    return RegularGridInterpolator(tuple(0.5*(e[:-1]+e[1:]) for e in edg),np.log(H),bounds_error=False,fill_value=float(np.log(1e-300)))

rng=np.random.default_rng(11)
# ───────── FIG 1: within-W angle validation (smeared MC vs data) ─────────
fig1,ax1=plt.subplots(3,4,figsize=(17,10.5))
COLS=[("true θ$_{hi}$ [deg]","t_ahi","tw",(0,180)),("true θ$_{lo}$ [deg]","t_alo","tw",(0,180)),
      ("wrong θ$_{hi}$ [deg]","w_ahi","ww",(0,180)),("wrong θ$_{lo}$ [deg]","w_alo","ww",(0,180))]
DCOLS=[("ahi","t"),("alo","t"),("ahi","w"),("alo","w")]
for ie,ecm in enumerate((160,240,365)):
    a=load(ecm); mc=ps_mc(a,rng); hi,lo,ahi,alo,gpt,ok=data_obs(a); rows=np.arange(len(gpt))
    tahi=ahi[rows,gpt][ok];talo=alo[rows,gpt][ok]
    wm=np.ones((len(gpt),3),bool);wm[rows,gpt]=False;wm&=ok[:,None]
    wahi=ahi[wm];walo=alo[wm]
    dat={("ahi","t"):tahi,("alo","t"):talo,("ahi","w"):wahi,("alo","w"):walo}
    for ic,(xlab,mk,wk,rg) in enumerate(COLS):
        ax=ax1[ie,ic];bins=np.linspace(*rg,60)
        ax.hist(dat[DCOLS[ic]],bins=bins,density=True,histtype="stepfilled",color=GREEN,alpha=0.40,label="data reco (gpt truth)")
        ax.hist(mc[mk],bins=bins,weights=mc[wk],density=True,histtype="step",color=BLUE,lw=1.9,label="smeared MC (additive Δθ)")
        if ic==0:ax.set_ylabel(f"√s = {ecm} GeV\nnormalised",fontsize=11)
        if ie==0:ax.set_title(xlab,fontsize=11)
        if ie==2:ax.set_xlabel(xlab,fontsize=10)
        ax.grid(alpha=.25)
        if ie==0 and ic==0:ax.legend(fontsize=8.5,loc="upper left")
fig1.suptitle("Within-W opening-angle template: smeared phase-space MC vs DATA reco (gpt truth)",fontsize=13,y=0.998)
plt.tight_layout(rect=[0,0,1,0.985]); p1=os.path.join(EOSW,"pairing_smooth_angle_validation.png"); fig1.savefig(p1,dpi=112); print("[plot]",p1)

# ───────── FIG 2: smooth log-LR separation, mass-only vs mass+angle ─────────
fig2,ax2=plt.subplots(1,3,figsize=(16,4.8))
me=np.arange(MLO,MHI+1e-6,MBIN); ae=np.arange(0,180+1e-6,ABIN)
for ie,ecm in enumerate((160,240,365)):
    a=load(ecm); mc=ps_mc(a,rng); hi,lo,ahi,alo,gpt,ok=data_obs(a); N=len(gpt);rows=np.arange(N)
    # 2D (mass) and 4D (mass+angle) smooth templates
    lPt2=dens([mc["t_mhi"],mc["t_mlo"]],mc["tw"],[me,me]); lPw2=dens([mc["w_mhi"],mc["w_mlo"]],mc["ww"],[me,me])
    lPt4=dens([mc["t_mhi"],mc["t_mlo"],mc["t_ahi"],mc["t_alo"]],mc["tw"],[me,me,ae,ae])
    lPw4=dens([mc["w_mhi"],mc["w_mlo"],mc["w_ahi"],mc["w_alo"]],mc["ww"],[me,me,ae,ae])
    def llr(use_ang):
        out=np.full((N,3),0.0)
        for p in range(3):
            if use_ang:
                X=np.stack([hi[:,p],lo[:,p],ahi[:,p],alo[:,p]],1); out[:,p]=lPt4(X)-lPw4(X)
            else:
                X=np.stack([hi[:,p],lo[:,p]],1); out[:,p]=lPt2(X)-lPw2(X)
        return out
    L2=llr(False); L4=llr(True)
    # split each partition's log-LR into TRUE vs WRONG (per gpt)
    istrue=np.zeros((N,3),bool); istrue[rows,gpt]=True; istrue&=ok[:,None]; iswrong=(~istrue)&ok[:,None]
    ax=ax2[ie]; bins=np.linspace(-12,12,61)
    for L,ls,lab in [(L2,"--","mass-only"),(L4,"-","mass+angle")]:
        ax.hist(L[istrue],bins=bins,density=True,histtype="step",color=GREEN,lw=1.8,ls=ls,label=f"TRUE part. ({lab})")
        ax.hist(L[iswrong],bins=bins,density=True,histtype="step",color=RED,lw=1.8,ls=ls,label=f"WRONG part. ({lab})")
    ax.axvline(0,color="grey",lw=.8,ls=":"); ax.set_title(f"√s = {ecm} GeV",fontsize=12)
    ax.set_xlabel("smooth log-LR  =  logP$_{true}$ − logP$_{wrong}$  (per partition)"); ax.grid(alpha=.25)
    if ie==0: ax.set_ylabel("normalised"); ax.legend(fontsize=7.6,loc="upper left")
fig2.suptitle("Smooth phase-space-MC discriminant: per-partition log-LR, TRUE (green) vs WRONG (red) — dashed mass-only, solid mass+angle",fontsize=12,y=1.0)
plt.tight_layout(); p2=os.path.join(EOSW,"pairing_smooth_logLR.png"); fig2.savefig(p2,dpi=120); print("[plot]",p2)

# ───────── FIG 3: the Δθ(angle) smear bias the additive model misses (review caveat, visualised) ─────────
fig3,ax3=plt.subplots(1,1,figsize=(7.2,5.0))
for ecm,col in zip((160,240,365),(GREEN,BLUE,RED)):
    a=load(ecm); R,dth,gth=calib(a)
    centres=np.arange(35,180,15); prof=[]
    for c in centres:
        sel=np.abs(gth-c)<7.5; prof.append(np.median(dth[sel]) if sel.sum()>50 else np.nan)
    ax3.plot(centres,prof,"o-",color=col,label=f"√s={ecm}")
ax3.axhline(0,color="grey",lw=.8,ls=":"); ax3.set_xlabel("gen within-W opening angle [deg]")
ax3.set_ylabel("median Δθ = θ$_{reco}$ − θ$_{gen}$ [deg]")
ax3.set_title("Angle response Δθ is angle-dependent — the additive smear approximates it as flat\n(leading √s-growing modeling residual of the smooth template)", fontsize=10.5)
ax3.grid(alpha=.3); ax3.legend()
plt.tight_layout(); p3=os.path.join(EOSW,"pairing_smooth_angle_response.png"); fig3.savefig(p3,dpi=120); print("[plot]",p3)
