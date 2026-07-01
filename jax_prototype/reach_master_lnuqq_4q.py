#!/usr/bin/env python3
# NEXT-1 / STEP-0 C5 fix: corrected channel×√s mW statistical-reach comparison.
# 4q (fully hadronic) vs ℓνqq (semileptonic, μ+e COMBINED) — BOTH from the inclusive
# Pythia8 p8_ee_WW sample (same generator/pole ⇒ apples-to-apples), forward-fold, NO kinfit.
# Inputs are the slope-1-corrected per-event σ (σ_1ev = σ_full/slope · √Ng) from the
# STEP-0 battery (4q) and the lnuqq_fold_battery (this session). σ_1ev is the ROBUST,
# assumption-free resolution; the absolute reach carries the stated σ_WW/L/ε (×1.5).
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
import os, math, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
EOSW="/eos/user/m/mdefranc/www/mW/conv_mw_4q"; os.makedirs(EOSW, exist_ok=True)

# ── per-event slope-1 σ [MeV·√evt] = σ_full/slope · √Ng, at the chosen working point ──
def s1ev(sig_full, slope, Ng): return (sig_full/slope)*math.sqrt(Ng)

# 4q (fully hadronic), ALL no-d45 (NEXT-2 done DAY11): @160/240/365 from p8 inclusive, raw bin1.0 (slope≈1).
FOURQ = {
 160: dict(s1=s1ev(6.79,1.0710,963916), floor=s1ev(1.27,1.0050,1003608), cut="no-d45"),
 240: dict(s1=s1ev(6.98,0.9788,373720), floor=s1ev(1.58,1.0001,436609), cut="no-d45"),
 365: dict(s1=s1ev(6.16,0.9862,380317), floor=s1ev(1.53,1.0003,466299), cut="no-d45"),
}
# ℓνqq (semileptonic, μ+e combined, p8 inclusive) — this session. Working point: bin0.5 @160
# (slope≈1.01), bin1.0 @240/365 (slope≈0.99, small closure).
LNUQQ = {
 160: dict(s1=s1ev(3.56,1.0086,351472), floor=s1ev(2.14,1.0055,352595), eps=0.937),
 240: dict(s1=s1ev(4.25,0.9886,349059), floor=s1ev(1.72,0.9994,368838), eps=0.915),
 365: dict(s1=s1ev(5.35,0.9798,286672), floor=s1ev(1.77,0.9995,345971), eps=0.858),
}

# ── N → integrated luminosity assumptions (stated, ~×1.5 uncertain; STAT-ONLY) ──
SIGWW = {160:4.0, 240:16.0, 365:10.0}   # pb (e+e-→WW), representative
LUMI  = {160:10.0, 240:5.0, 365:1.5}    # ab⁻¹ (FCC-ee baseline)
BR4Q, BRLNU = 0.457, 0.288              # WW→qqqq ; WW→(e+μ)νqq direct
EPS_COMMON = 0.90                        # common FF selection eff for the RATIO (cancels)

def reach(s1, sigWW, L_ab, BR, eps): return s1/math.sqrt(L_ab*1e6*sigWW*BR*eps)

print(f"{'ECM':>4} {'chan':>6} {'σ1ev':>7} {'floor':>7} | {'σ_mW(@L)':>9}  (σWW,L,BR,ε)")
rows={}
for ecm in (160,240,365):
    for name,d,BR,eps in [("4q",FOURQ[ecm],BR4Q,EPS_COMMON),("lnuqq",LNUQQ[ecm],BRLNU,EPS_COMMON)]:
        r=reach(d['s1'],SIGWW[ecm],LUMI[ecm],BR,eps)
        rf=reach(d['floor'],SIGWW[ecm],LUMI[ecm],BR,eps)
        rows[(ecm,name)]=dict(s1=d['s1'],floor=d['floor'],reach=r,floor_reach=rf)
        print(f"{ecm:>4} {name:>6} {d['s1']:7.0f} {d['floor']:7.0f} | {r:9.3f}  (floor {rf:.3f})")
print("\nratio σ_4q / σ_ℓνqq  (equal ε ⇒ ε cancels; = s1-ratio × √(BRℓν/BR4q)):")
for ecm in (160,240,365):
    ratio=rows[(ecm,'4q')]['reach']/rows[(ecm,'lnuqq')]['reach']
    s1r=FOURQ[ecm]['s1']/LNUQQ[ecm]['s1']
    print(f"  {ecm}: ℓνqq {ratio:.2f}× better   (per-event 4q/ℓνqq = {s1r:.2f}×)")

# ── master plot: reach vs √s, both channels (+gen floors) ──
ecms=[160,240,365]
fig,ax=plt.subplots(figsize=(8.2,5.8))
c4q,clnu="C3","C0"
ax.plot(ecms,[rows[(e,'4q')]['reach'] for e in ecms],"o-",color=c4q,ms=9,lw=2,label="4q (hadronic), forward-fold")
ax.plot(ecms,[rows[(e,'lnuqq')]['reach'] for e in ecms],"s-",color=clnu,ms=9,lw=2,label="ℓνqq (μ+e), forward-fold")
ax.plot(ecms,[rows[(e,'4q')]['floor_reach'] for e in ecms],"o:",color=c4q,ms=6,alpha=.5,label="4q gen floor")
ax.plot(ecms,[rows[(e,'lnuqq')]['floor_reach'] for e in ecms],"s:",color=clnu,ms=6,alpha=.5,label="ℓνqq gen floor")
for e in ecms:
    ax.annotate(f"{rows[(e,'4q')]['reach']:.2f}",(e,rows[(e,'4q')]['reach']),textcoords="offset points",xytext=(6,6),color=c4q,fontsize=9)
    ax.annotate(f"{rows[(e,'lnuqq')]['reach']:.2f}",(e,rows[(e,'lnuqq')]['reach']),textcoords="offset points",xytext=(6,-12),color=clnu,fontsize=9)
ax.set_xlabel("√s [GeV]"); ax.set_ylabel(r"$\sigma_{m_W}$ (statistical) [MeV]")
ax.set_title("WW→mW forward-fold reach: 4q vs ℓνqq(μ+e), same p8 generator\n"
             "σ_WW=4/16/10 pb, L=10/5/1.5 ab⁻¹, ε=0.90 (stated, ×1.5; STAT-ONLY)")
ax.set_xticks(ecms); ax.grid(alpha=.3); ax.legend(fontsize=9); ax.set_ylim(0,None)
plt.tight_layout(); p=f"{EOSW}/reach_master_lnuqq_4q.png"; plt.savefig(p,dpi=120); print(f"\n[plot] {p}")
json.dump({f"{e}_{n}":rows[(e,n)] for e in ecms for n in ("4q","lnuqq")},
          open("jax_prototype/anatomy_results/reach_master_lnuqq_4q.json","w"),indent=2)
