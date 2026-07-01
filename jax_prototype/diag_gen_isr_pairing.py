#!/usr/bin/env python3
# Gen-level: how WW->4q kinematics depend on ISR, and how the WRONG pairing distorts them.
# m_qq, within-W opening angle, W-W opening angle, m_qqqq  vs |ISR| ; true vs wrong pairing.
import uproot, numpy as np, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ECM=int(sys.argv[1]) if len(sys.argv)>1 else 160
_tag={160:"had_ff160_genqk",240:"had_ff240_genqk",365:"had_ff365_genqk"}[ECM]
F=f"outputs/treemaker/4q/step2_ff/{_tag}/p8_ee_WW_ecm{ECM}.root"
EOSW="/eos/user/m/mdefranc/www/mW/kinfit_correlations"; import os; os.makedirs(EOSW,exist_ok=True)
q=[ "gen_qW%d_%s"%(i,c) for i in range(4) for c in ("px","py","pz","e") ]
a=uproot.open(F)["events"].arrays(q+["gen_isr_px","gen_isr_py","gen_isr_pz","gen_WW_m"],library="np")
def Q(i): return np.stack([a["gen_qW%d_px"%i],a["gen_qW%d_py"%i],a["gen_qW%d_pz"%i],a["gen_qW%d_e"%i]],1)
Qs=[Q(i) for i in range(4)]
isr=np.sqrt(a["gen_isr_px"]**2+a["gen_isr_py"]**2+a["gen_isr_pz"]**2)
N=len(isr); print("N=",N,"  |ISR| median=%.2f p90=%.2f max=%.2f"%(np.median(isr),np.percentile(isr,90),isr.max()))
def mass(v):
    s=v[:,3]**2-v[:,0]**2-v[:,1]**2-v[:,2]**2; return np.sqrt(np.maximum(s,0))
def opang(u,v):  # 3D opening angle [deg] between two 4-vec momenta
    du=u[:,:3]; dv=v[:,:3]
    c=(du*dv).sum(1)/(np.linalg.norm(du,axis=1)*np.linalg.norm(dv,axis=1)+1e-12)
    return np.degrees(np.arccos(np.clip(c,-1,1)))
# pairings of the 4 W-grouped quarks (0,1,2,3): TRUE=(01)(23); WRONG=(02)(13),(03)(12)
PAIRINGS={"true":[(0,1),(2,3)], "wrongA":[(0,2),(1,3)], "wrongB":[(0,3),(1,2)]}
def pair_vars(pr):
    (a1,a2),(b1,b2)=pr
    WA=Qs[a1]+Qs[a2]; WB=Qs[b1]+Qs[b2]
    mqq=np.concatenate([mass(WA),mass(WB)])                 # both dijet masses
    dwithin=np.concatenate([opang(Qs[a1],Qs[a2]),opang(Qs[b1],Qs[b2])])  # within-W qq angle
    dWW=opang(WA,WB)                                        # W-W opening angle (per event)
    return mqq,dwithin,dWW
V={k:pair_vars(p) for k,p in PAIRINGS.items()}
# x bins in |ISR| (percentile-based, skewed)
edges=np.unique(np.percentile(isr,np.linspace(0,97,9))); ctr=0.5*(edges[:-1]+edges[1:])
def prof(y,x):
    m=[]; s=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        sel=(x>=lo)&(x<hi); m.append(np.median(y[sel]) if sel.sum() else np.nan); s.append(y[sel].std() if sel.sum() else np.nan)
    return np.array(m),np.array(s)
isr2=np.concatenate([isr,isr])  # for the doubled (per-W) arrays
fig,ax=plt.subplots(2,2,figsize=(12,9))
def panel(axi,key,title,ylab,doubled):
    for name,c,ls in [("true","C0","-"),("wrongA","C3","--"),("wrongB","C1",":")]:
        y=V[name][key]; x=isr2 if doubled else isr
        m,s=prof(y,x); axi.plot(ctr,m,ls,color=c,lw=2,marker="o",ms=4,label=name)
        axi.fill_between(ctr,m-s,m+s,color=c,alpha=0.08)
    axi.set_xlabel("|ISR| [GeV]"); axi.set_ylabel(ylab); axi.set_title(title); axi.grid(alpha=.3); axi.legend(fontsize=8)
panel(ax[0,0],0,"(1) dijet mass m_qq vs ISR","m_qq [GeV] (median ± std)",True)
panel(ax[0,1],1,"(2) within-W opening angle q-q vs ISR","θ(q,q) same W [deg]",True)
panel(ax[1,0],2,"(3) W-W opening angle vs ISR","θ(W,W) [deg]  (180=back-to-back)",False)
# panel 4: m_qqqq (pairing independent) vs ISR
m4,s4=prof(a["gen_WW_m"],isr); ax[1,1].plot(ctr,m4,"-",color="k",lw=2,marker="o",label="m_qqqq (=√s', pairing-indep)")
ax[1,1].fill_between(ctr,m4-s4,m4+s4,color="k",alpha=0.1)
ax[1,1].set_xlabel("|ISR| [GeV]"); ax[1,1].set_ylabel("m_qqqq [GeV]"); ax[1,1].set_title("(4) m_qqqq vs ISR"); ax[1,1].grid(alpha=.3); ax[1,1].legend(fontsize=8)
fig.suptitle("WW→4q gen-level: kinematics vs ISR — true vs wrong pairing (ecm%d, N=%d)"%(ECM,N),fontsize=13)
plt.tight_layout(); p=EOSW+"/gen_isr_pairing_ecm%d.png"%ECM; plt.savefig(p,dpi=120); print("[plot]",p)
# also print the true-vs-wrong separation numerically (the pairing information)
for key,nm,dbl in [(0,"m_qq",True),(1,"theta_within",True),(2,"theta_WW",False)]:
    yt=V["true"][key]; yw=np.concatenate([V["wrongA"][key],V["wrongB"][key]])
    print(f"  {nm:14s}: true median={np.median(yt):7.2f}  wrong median={np.median(yw):7.2f}  Δ={np.median(yt)-np.median(yw):+7.2f}")
