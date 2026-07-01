#!/usr/bin/env python3
# ── DAY7: the gen-pole "offset" is the BW WIDTH CONVENTION + wrong reference mass, NOT 4f modeling ────
# gen_pe = the gen self-fit (per-event fold using gen-truth √s', the "physical pole" target).  Measured
# (conv_mw_closure_anatomy.py, FINE_J0=1 NMW=121, full N) for three configurations:
#   (1) DAY3-6 convention:  fixed-width BW, ΓW=2.085, compared to PDG mW=80.379
#   (2) correct reference:  fixed-width BW, ΓW=2.049, compared to GENERATOR mW=80.419 (WHIZARD SM.mdl)
#   (3) the fix:            RUNNING-width BW, ΓW=2.049, compared to GENERATOR mW=80.419
# WHIZARD winter2023 wzp6_ee_munumuqq generated with mW=80.419, ΓW=2.049 (SM.mdl default scheme).
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_bw_width_closure.py
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
ECMS=np.array([157,160,163]); PDG=80.379; GEN=80.419

# measured gen_pe [GeV] (conv_mw_closure_anatomy.py, FINE_J0=1 NMW=121, full N=92218)
genpe_fix2085 = np.array([80.2156, 80.2051, 80.3953])   # fixed BW, ΓW=2.085 (DAY3-6)
genpe_fix2049 = np.array([80.2169, 80.2101, 80.3948])   # fixed BW, ΓW=2.049
genpe_run2049 = np.array([80.4180, 80.4164, 80.4572])   # running BW, ΓW=2.049  (the fix)

s1 = 1000*(genpe_fix2085 - PDG)   # DAY3-6:  fixed, vs PDG
s2 = 1000*(genpe_fix2049 - GEN)   # fixed, vs generator mW
s3 = 1000*(genpe_run2049 - GEN)   # running, vs generator mW

fig,ax=plt.subplots(figsize=(9.2,6.0)); plt.rcParams.update({"font.size":12})
ax.axhspan(-10,10,color="green",alpha=0.10,label="±10 MeV (closure)")
ax.axhline(0,color="0.4",lw=1)
ax.plot(ECMS,s1,"o-",color="C3",lw=2.2,ms=9,label="DAY3–6: FIXED-width BW vs PDG 80.379")
ax.plot(ECMS,s2,"s--",color="C1",lw=2.0,ms=8,label="FIXED-width BW vs GENERATOR mW 80.419 (Γ=2.049)")
ax.plot(ECMS,s3,"D-",color="C2",lw=2.6,ms=10,label="RUNNING-width BW vs GENERATOR mW 80.419 (Γ=2.049) — THE FIX")
for x,y in zip(ECMS,s1): ax.annotate(f"{y:+.0f}",(x,y),textcoords="offset points",xytext=(6,6),color="C3",fontsize=10)
for x,y in zip(ECMS,s3): ax.annotate(f"{y:+.0f}",(x,y),textcoords="offset points",xytext=(6,8),color="C2",fontsize=10,fontweight="bold")
ax.axvline(2*GEN,color="0.6",ls=":",lw=1.2); ax.text(2*GEN,ax.get_ylim()[0]*0.92," 2mW=160.84 (threshold)",color="0.4",fontsize=9,rotation=90,va="bottom")
ax.set_xticks(ECMS); ax.set_xlabel("ECM [GeV]"); ax.set_ylabel("gen self-fit − reference mW  [MeV]")
ax.set_title("gen-pole 'offset' = BW width convention + wrong reference, NOT 4f modeling\n"
             "running-width BW + generator mW/ΓW ⇒ 157/160 close to ~0 (−1,−3); 163 residual +38",fontsize=11.5)
ax.legend(fontsize=9.8,loc="center right"); ax.grid(alpha=0.25)
fig.tight_layout(); png=f"{EOSW}/bw_width_closure_day7.png"; fig.savefig(png,dpi=130); print(f"[plot] {png}")
print("scatter across ECM:  DAY3-6 =",f"{s1.max()-s1.min():.0f} MeV   running-fix =",f"{s3.max()-s3.min():.0f} MeV")
