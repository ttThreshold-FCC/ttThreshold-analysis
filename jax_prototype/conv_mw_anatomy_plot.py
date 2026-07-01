#!/usr/bin/env python3
# Multi-panel closure-anatomy figure from anatomy_results/anatomy_ecm{157,160,163}.json
#   (A) closure decomposition (floor + smearing) per ECM, both weight schemes
#   (B) radiator-width ladder: gen_marg(f) − gen_pe vs f, 3 ECMs (gap closes as radiator narrows)
#   (C) UNIVERSAL gap(δ): gen_marg − gen_pe vs off-shellness δ=√s'−(m_qq+m_lν), 3 ECMs overlaid
#   (D) across-ECM: gen targets vs ECM + threshold; mean-δ bar; total gap split (width vs per-event)
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
import os, json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

TAG = sys.argv[1] if len(sys.argv) > 1 else ""     # e.g. "_n40000" for subsample json
RDIR = os.path.join(os.path.dirname(__file__), "anatomy_results")
OUT  = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(OUT, exist_ok=True)
ECMS = [157, 160, 163]; COL = {157:"C0", 160:"C1", 163:"C2"}
R = {}
for e in ECMS:
    p = os.path.join(RDIR, f"anatomy_ecm{e}{TAG}.json")
    if os.path.exists(p): R[e] = json.load(open(p))
if not R: raise SystemExit(f"no anatomy_ecm*{TAG}.json in {RDIR}")

fig, ax = plt.subplots(2, 2, figsize=(15, 11))
MW = 80.379

# ── (A) closure decomposition ─────────────────────────────────────────────────────────────────────
axA = ax[0,0]; x = np.arange(len(R)); w = 0.36
for gi, sch in enumerate(("grid","exact")):
    floor = [1000*R[e]["decomp"][sch]["floor"] for e in R]
    smear = [1000*R[e]["decomp"][sch]["smear"] for e in R]
    clos  = [1000*R[e]["decomp"][sch]["closure"] for e in R]
    xs = x + (gi-0.5)*w
    axA.bar(xs, floor, w*0.9, color="C0" if gi==0 else "C3", alpha=.55, label=f"{sch}: floor (binning)")
    axA.bar(xs, smear, w*0.9, bottom=floor, color="C0" if gi==0 else "C3", alpha=.95, hatch="//",
            label=f"{sch}: smearing")
    for xi, c in zip(xs, clos): axA.plot(xi, c, "kD", ms=7, zorder=5)
axA.axhline(0, color="k", lw=.8)
axA.plot([], [], "kD", ms=7, label="total closure")
axA.set_xticks(x); axA.set_xticklabels([f"ecm{e}" for e in R])
axA.set_ylabel("matched closure [MeV]"); axA.set_title("(A) closure = estimator-floor + smearing  (per scheme)")
axA.legend(fontsize=8, ncol=2, loc="lower left"); axA.grid(axis="y", alpha=.3)

# ── (B) radiator-width ladder ───────────────────────────────────────────────────────────────────
axB = ax[0,1]
for e in R:
    f = np.array(R[e]["ladder_f"]); m = np.array(R[e]["ladder_mW"]); gp = R[e]["gen_pe"]
    axB.plot(f, 1000*(m-gp), "o-", color=COL[e], label=f"ecm{e}  (gap={1000*(m[-1]-gp):+.0f} MeV)")
axB.axhline(0, color="k", ls="--", lw=.8, label="gen_pe (per-event √s')")
axB.set_xlabel("radiator width scale  f   (0 = δ-fn at ⟨√s'⟩, 1 = full)")
axB.set_ylabel("gen_marg(f) − gen_pe  [MeV]")
axB.set_title("(B) width ladder: spread-bias slope FLIPS SIGN across threshold\n"
              "(157/160 below: rise with f ; 163 above: fall)")
axB.annotate("above threshold:\nspread-bias < 0", (0.5, 1000*(R[163]['ladder_mW'][3]-R[163]['gen_pe'])) if 163 in R else (0.5,20),
             fontsize=8, color="C2", ha="center")
axB.legend(fontsize=9); axB.grid(alpha=.3)

# ── (C) universal gap(δ) ────────────────────────────────────────────────────────────────────────
axC = ax[1,0]
for e in R:
    d = np.array(R[e]["margin_delta"]); g = 1000*np.array(R[e]["margin_gap"])
    axC.plot(d, g, "o-", color=COL[e], label=f"ecm{e}  ⟨δ⟩={R[e]['mean_margin']:.1f}")
    axC.axvline(R[e]["mean_margin"], color=COL[e], ls=":", lw=1, alpha=.7)
axC.set_xlabel(r"off-shellness  $\delta=\sqrt{s'}-(m_{qq}+m_{\ell\nu})$  [GeV]")
axC.set_ylabel(r"gap = gen_marg − gen_pe  [MeV]")
axC.set_title("(C) gap grows with off-shellness δ within an ECM;\n"
              "LEVEL set by √s'−2mW regime (157 below→high, 163 above→low)")
axC.legend(fontsize=9); axC.grid(alpha=.3)

# ── (D) across-ECM summary ────────────────────────────────────────────────────────────────────────
axD = ax[1,1]
es = sorted(R); xe = np.array(es)
gpe  = np.array([R[e]["gen_pe"]   for e in es])
gmg  = np.array([R[e]["gen_marg"] for e in es])
axD.plot(xe, 1000*(gpe-MW), "s-", color="C2", label="gen_pe − PDG (per-event √s')")
axD.plot(xe, 1000*(gmg-MW), "o-", color="C1", label="gen_marg − PDG (marginalized)")
axD.axhline(0, color="k", ls="--", lw=.8, label="PDG 80.379")
for e in es:
    axD.annotate(f"gap {1000*(R[e]['gen_marg']-R[e]['gen_pe']):+.0f}", (e, 1000*(gmg[es.index(e)]-MW)),
                 textcoords="offset points", xytext=(0,8), ha="center", fontsize=8, color="C1")
ax2 = axD.twinx()
ax2.bar(xe, [R[e]["mean_margin"] for e in es], width=1.1, color="gray", alpha=.18)
ax2.set_ylabel("⟨δ⟩ off-shellness [GeV] (bars)", color="gray")
axD.axvline(2*MW, color="purple", ls=":", lw=1.2); axD.text(2*MW, axD.get_ylim()[1], " 2mW", color="purple", fontsize=8, va="top")
axD.set_xlabel("ECM [GeV]"); axD.set_ylabel("gen target − PDG  [MeV]")
axD.set_title("(D) targets vs ECM: gap shrinks above threshold as ⟨δ⟩ falls")
axD.legend(fontsize=8, loc="upper center"); axD.grid(alpha=.3)

plt.tight_layout()
png = f"{OUT}/closure_anatomy{TAG}.png"; plt.savefig(png, dpi=115)
print(f"[plot] {png}")

# ── compact text table ────────────────────────────────────────────────────────────────────────────
print(f"\n{'ECM':>4} {'gen_pe':>9} {'gen_marg':>9} {'gap':>6} | {'grid clos/floor/smear':>24} | {'exact clos/floor/smear':>24}")
for e in es:
    d = R[e]["decomp"]
    print(f"{e:>4} {R[e]['gen_pe']:>9.4f} {R[e]['gen_marg']:>9.4f} {1000*(R[e]['gen_marg']-R[e]['gen_pe']):>+5.0f} | "
          f"{1000*d['grid']['closure']:>+7.0f} {1000*d['grid']['floor']:>+7.0f} {1000*d['grid']['smear']:>+7.0f}   | "
          f"{1000*d['exact']['closure']:>+7.0f} {1000*d['exact']['floor']:>+7.0f} {1000*d['exact']['smear']:>+7.0f}")
