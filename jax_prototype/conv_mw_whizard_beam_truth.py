#!/usr/bin/env python3
# ── Visualize the WHIZARD beam-chain TRUTH of the FCC-ee winter2023 wzp6_ee_munumuqq samples ─────────
# Each event's MCParticle record stores the e± beam in stages (status 2):
#   pair0 = nominal beam (exact)  ;  pair1 = after BES (Gaussian beam-energy spread)  ;
#   pair2 = after ISR (structure function).  We histogram the per-pair √s (= sum of |E| of the e+/e- pair)
#   at all three ECMs to expose: (1) BES is a SYMMETRIC Gaussian σ≈0.118 GeV; (2) ISR adds the asymmetric
#   low tail (mean loss ~0.5 GeV, RMS ~2 GeV); (3) no separate beamstrahlung stage.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_whizard_beam_truth.py
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import uproot
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
SRC="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
MAXN=int(os.environ.get("MAXN","40000"))
ECMS=[157,160,163]
plt.rcParams.update({"font.size":12})
fig,AX=plt.subplots(1,3,figsize=(19,5.4))
print(f"{'ECM':>4} {'stage':10} {'mean':>9} {'std':>7} {'meanloss':>9}")
for j,ECM in enumerate(ECMS):
    F=sorted(glob.glob(f"{SRC}/wzp6_ee_munumuqq_noCut_ecm{ECM}/*.root"))[0]
    a=uproot.open(F)["events"].arrays(["Particle.PDG","Particle.momentum.x","Particle.momentum.y","Particle.momentum.z","Particle.mass"],entry_stop=MAXN,library="np")
    S0=[];S1=[];S2=[]
    for i in range(len(a["Particle.PDG"])):
        pdg=a["Particle.PDG"][i]; px=a["Particle.momentum.x"][i]; py=a["Particle.momentum.y"][i]; pz=a["Particle.momentum.z"][i]; m=a["Particle.mass"][i]
        eidx=[k for k in range(len(pdg)) if abs(pdg[k])==11]
        if len(eidx)<6: continue
        E=lambda k: float(np.sqrt(px[k]**2+py[k]**2+pz[k]**2+m[k]**2))
        S0.append(E(eidx[0])+E(eidx[1])); S1.append(E(eidx[2])+E(eidx[3])); S2.append(E(eidx[4])+E(eidx[5]))
    S0=np.array(S0);S1=np.array(S1);S2=np.array(S2)
    for nm,S in [("nominal",S0),("BES (depth1)",S1),("ISR (depth2)",S2)]:
        print(f"{ECM:>4} {nm:10} {S.mean():9.4f} {S.std():7.4f} {ECM-S.mean():9.4f}")
    ax=AX[j]; lo,hi=ECM-12,ECM+1.5
    ax.hist(S1,bins=120,range=(lo,hi),histtype="step",color="C0",lw=2,density=True,label=f"after BES  (σ={S1.std():.3f})")
    ax.hist(S2,bins=120,range=(lo,hi),histtype="step",color="C3",lw=2,density=True,label=f"after ISR  (loss {ECM-S2.mean():.2f}, rms {S2.std():.2f})")
    ax.axvline(ECM,color="0.5",ls=":",lw=1.3,label=f"nominal √s={ECM} (σ={S0.std():.3f})")
    ax.set_yscale("log"); ax.set_ylim(2e-3,8); ax.set_xlim(lo,hi)
    ax.set_title(f"ecm{ECM}: WHIZARD beam √s by stage"); ax.set_xlabel("√s of e+e- pair [GeV]"); ax.legend(fontsize=9.5,loc="upper left")
AX[0].set_ylabel("normalized")
fig.suptitle("WHIZARD beam-chain truth (FCC-ee winter2023 wzp6_ee_munumuqq): nominal → Gaussian BES → ISR",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96]); png=f"{EOSW}/whizard_beam_truth.png"; fig.savefig(png,dpi=130); print(f"[plot] {png}")
