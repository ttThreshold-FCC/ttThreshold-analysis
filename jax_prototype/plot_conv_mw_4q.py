#!/usr/bin/env python3
# Stage-1 4q forward-fold closure plots: (a) the 2-D (m_hi,m_lo) observable gen vs reco (true pairing),
# (b) closure decomposition bar chart across pairing modes (from the conv_mw_4q JSONs).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import uproot

ECM = 160
ROOT = f"outputs/treemaker/4q/step2/had_scap_ctrl_s1/p8_ee_WW_ecm{ECM}.root"
RDIR = "jax_prototype/anatomy_results"
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)

# ── (a) 2-D observable: gen (W1,W2) vs reco dijet masses on the gen-true pairing ───────────────────
t = uproot.open(ROOT)["events"]
need = ["gen_W1_m","gen_W2_m","gen_WW_m","gen_pairing_true"]
for i in (1,2,3,4): need += [f"reco_jet{i}_p", f"reco_jet{i}_theta", f"reco_jet{i}_phi"]
a = t.arrays(need, library="np"); N = len(a["gen_W1_m"])
def jv(i):
    p=a[f"reco_jet{i}_p"]; th=a[f"reco_jet{i}_theta"]; ph=a[f"reco_jet{i}_phi"]; st=np.sin(th)
    return np.stack([p*st*np.cos(ph), p*st*np.sin(ph), p*np.cos(th), p],1)
J={i:jv(i) for i in (1,2,3,4)}
def dm(u,v):
    s=u+v; return np.sqrt(np.maximum(s[:,3]**2-s[:,0]**2-s[:,1]**2-s[:,2]**2,0))
PO=[((1,2),(3,4)),((1,3),(2,4)),((1,4),(2,3))]
mA=np.stack([dm(J[p[0][0]],J[p[0][1]]) for p in PO],1); mB=np.stack([dm(J[p[1][0]],J[p[1][1]]) for p in PO],1)
gpt=a["gen_pairing_true"].astype(int); rows=np.arange(N)
rA=mA[rows,gpt]; rB=mB[rows,gpt]
g1,g2=a["gen_W1_m"],a["gen_W2_m"]
gen_hi=np.maximum(g1,g2); gen_lo=np.minimum(g1,g2)
rec_hi=np.maximum(rA,rB); rec_lo=np.minimum(rA,rB)
ok=np.isfinite(gen_hi)&np.isfinite(rec_hi)&((g1+g2)<a["gen_WW_m"])

fig,ax=plt.subplots(1,2,figsize=(12,5.2))
rng=[[40,100],[20,95]]
for k,(hi,lo,ttl) in enumerate([(gen_hi[ok],gen_lo[ok],"gen (W1,W2)"),(rec_hi[ok],rec_lo[ok],"reco dijet (true pairing)")]):
    h=ax[k].hist2d(hi,lo,bins=80,range=rng,cmap="viridis",cmin=1)
    ax[k].axvline(80.4,color="w",ls=":",lw=0.8); ax[k].axhline(80.4,color="w",ls=":",lw=0.8)
    ax[k].set_xlabel("m_hi [GeV]"); ax[k].set_ylabel("m_lo [GeV]")
    ax[k].set_title(f"{ttl}   <hi>={hi.mean():.1f} <lo>={lo.mean():.1f}")
    fig.colorbar(h[3],ax=ax[k])
fig.suptitle(f"4q forward-fold observable  ecm{ECM}  N={int(ok.sum())}  (dotted = mW=80.4)")
fig.tight_layout(); fig.savefig(f"{EOSW}/obs_2d_ecm{ECM}.png",dpi=110); plt.close(fig)
print(f"[plot] {EOSW}/obs_2d_ecm{ECM}.png")

# 1-D projections of the (sorted) masses, gen vs reco
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
for k,(g,r,ttl) in enumerate([(gen_hi[ok],rec_hi[ok],"m_hi"),(gen_lo[ok],rec_lo[ok],"m_lo")]):
    ax[k].hist(g,bins=80,range=(20,100),histtype="step",lw=1.6,label=f"gen <{g.mean():.1f}>")
    ax[k].hist(r,bins=80,range=(20,100),histtype="step",lw=1.6,label=f"reco <{r.mean():.1f}>")
    ax[k].axvline(80.4,color="k",ls=":",lw=0.8); ax[k].set_xlabel(f"{ttl} [GeV]"); ax[k].legend(); ax[k].set_title(ttl)
fig.suptitle(f"4q sorted dijet masses gen vs reco (true pairing) ecm{ECM}")
fig.tight_layout(); fig.savefig(f"{EOSW}/obs_1d_ecm{ECM}.png",dpi=110); plt.close(fig)
print(f"[plot] {EOSW}/obs_1d_ecm{ECM}.png")

# ── (b) closure decomposition bar chart across configs/pairing modes ───────────────────────────────
cfgs=[("conv_mw_4q_ecm160_full_b05.json","bin0.5"),("conv_mw_4q_ecm160_full_b025.json","bin0.25"),
      ("conv_mw_4q_ecm160_full_kde.json","KDE0.6")]
fig,ax=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
for ci,(fn,lbl) in enumerate(cfgs):
    p=os.path.join(RDIR,fn)
    if not os.path.exists(p):
        ax[ci].set_title(f"{lbl} (missing)"); continue
    d=json.load(open(p))["results"]
    names=[r["name"] for r in d]; x=np.arange(len(names)); w=0.27
    clo=[1000*r["closure"] for r in d]; flo=[1000*r["floor"] for r in d]; sme=[1000*r["smear"] for r in d]
    ax[ci].bar(x-w,clo,w,label="closure",color="C3")
    ax[ci].bar(x,flo,w,label="floor",color="C0")
    ax[ci].bar(x+w,sme,w,label="smear",color="C1")
    for xi,r in zip(x,d): ax[ci].text(xi-w,clo[xi]+(2 if clo[xi]>=0 else -2),f"{clo[xi]:+.0f}",ha="center",fontsize=8)
    ax[ci].axhline(0,color="k",lw=0.6); ax[ci].set_xticks(x); ax[ci].set_xticklabels([f"{n}\n{100*r['eff']:.0f}%" for n,r in zip(names,d)])
    ax[ci].set_title(f"{lbl}"); ax[ci].grid(axis="y",alpha=0.3)
ax[0].set_ylabel("MeV"); ax[0].legend()
fig.suptitle(f"4q Stage-1 closure  (reco_fit − gen_pe) = floor + smear   ecm{ECM}")
fig.tight_layout(); fig.savefig(f"{EOSW}/closure_decomp_ecm{ECM}.png",dpi=110); plt.close(fig)
print(f"[plot] {EOSW}/closure_decomp_ecm{ECM}.png")
