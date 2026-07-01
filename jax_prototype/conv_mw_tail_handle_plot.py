#!/usr/bin/env python3
# ── DAY6 headline: which DATA-APPLICABLE cut removes the ecm163 deep-ISR estimator floor (−27 MeV)? ──
# Reads anatomy_results/anatomy_ecm163_d6_*.json (full N, exact scheme, hist0.25 unless tagged kde).
# Three quantities per cut (gen_pe_full = 80.3955 = the universal physical pole):
#   resid_vs_pole = exa_rec(cut) − gen_pe(full)   [HONEST absolute residual vs the true pole]
#   floor_self    = exa_id(cut)  − gen_pe(cut)    [estimator self-bias on the kept set]
#   target_shift  = gen_pe(cut)  − gen_pe(full)   [estimand drift = COST of cutting]
# Finding: reco_WW_m cut is INERT (estimand drift ≈ 0 ⇒ removes a representative slice; floor stays −25).
#          kinfit_chi2 cut WORKS (resid −27 → ~+8), tracking the gen-√s'-cut ceiling — but via a blunt
#          fit-quality cut (mostly invalid kinfits), and like the gen cut it drifts the estimand up.
import os, json, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RDIR = os.path.join(os.path.dirname(__file__), "anatomy_results")
EOSW = "/eos/user/m/mdefranc/www/mW/conv_mw"; ECM = 163; FULL = None

def load(tag):
    p = os.path.join(RDIR, f"anatomy_ecm{ECM}{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None

base = load("_d6_base")
FULL = base["gen_pe"]
def row(d):
    erc=d["decomp"]["exact"]["reco"]; gp=d["gen_pe"]; eid=d["decomp"]["exact"]["identity"]
    return dict(frac=100*d.get("proxy_frac",0.0), resid=1000*(erc-FULL), floor=1000*(eid-gp),
                tgt=1000*(gp-FULL), est=d.get("est","hist"))
b = row(base); b["frac"]=0.0
print(f"baseline resid_vs_pole={b['resid']:+.1f} MeV  (gen_pe_full={FULL:.4f})")

def series(prefix, labels):
    out=[b.copy()]
    for lb in labels:
        d=load(f"{prefix}{lb}")
        if d: out.append(row(d))
    out=sorted(out, key=lambda r:r["frac"]); return out

reco = series("_d6_reco_", ["f0144","f03","f05","f08","f12","f20"])
chi2 = series("_d6_chi2_", ["f0144","f02","f03","f04","f05","f08","f12","f20"])
gen  = series("_d6_gen_",  ["f005","f007","f010","f0144","f03","f05","f08","f12","f20"])

print(f"\n{'cut':12s} {'frac%':>5s} {'resid/pole':>10s} {'floor':>7s} {'tgt_shift':>9s}")
for nm,S in [("reco_WW_m",reco),("kinfit_chi2",chi2),("gen √s'",gen)]:
    for r in S: print(f"{nm:12s} {r['frac']:5.1f} {r['resid']:+10.1f} {r['floor']:+7.1f} {r['tgt']:+9.1f}")
    print()

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
sty = {"reco_WW_m":("C3","o","reco_WW_m cut (√s' proxy → INERT)"),
       "kinfit_chi2":("C2","s","kinfit_chi2 cut (fit-quality → WORKS)"),
       "gen √s'":("C0","^","gen √s' cut (ideal ceiling)")}
for nm,S in [("reco_WW_m",reco),("kinfit_chi2",chi2),("gen √s'",gen)]:
    c,m,lab=sty[nm]; F=[r["frac"] for r in S]
    ax[0].plot(F,[r["resid"] for r in S],m+"-",color=c,label=lab)
    ax[2].plot(F,[r["floor"] for r in S],m+"-",color=c,label=lab)
    ax[1].plot([r["tgt"] for r in S],[r["resid"] for r in S],m+"-",color=c,label=lab)
ax[0].set_xlabel("cut fraction removed [%]"); ax[0].set_ylabel("resid vs pole [MeV]")
ax[0].set_title("absolute residual: exa_rec(cut)−80.3955")
ax[1].set_xlabel("estimand drift gen_pe(cut)−gen_pe(full) [MeV]"); ax[1].set_ylabel("resid vs pole [MeV]")
ax[1].set_title("MECHANISM: chi2 & gen on one curve;\nreco stuck at drift≈0 (representative slice)")
ax[2].set_xlabel("cut fraction removed [%]"); ax[2].set_ylabel("floor_self [MeV]")
ax[2].set_title("estimator self-bias (floor) collapse")
for a in ax: a.axhline(0,color="k",lw=0.6,ls=":"); a.legend(fontsize=7.5); a.grid(alpha=0.25)
fig.suptitle(f"DAY6: data-applicable handle for the ecm{ECM} deep-ISR floor — exact, hist0.25, full N", fontsize=12)
fig.tight_layout()
png=f"{EOSW}/tail_handle_day6_ecm{ECM}.png"; fig.savefig(png,dpi=120); print(f"[plot] {png}")
