#!/usr/bin/env python3
# ── DECOMPOSE the gen √s' spectrum:  P_gen(√s') = L(√s') · σε(√s')  (exact, pointwise) ───────────────
# The reduced collision energy √s' (= gen_WW_m = m_lνuqq, true ν) of a SELECTED WW→lνqq event is, by
# 4-momentum conservation, the post-radiation CM energy.  Its distribution factorizes EXACTLY as
#     P_gen(√s') = L(√s') · σε(√s') / N
# where  L(√s') = dProb/d√s' that the collision happens at reduced energy √s' = the LUMINOSITY spectrum
# (ISR ⊗ beam-energy-spread; theory, well known — sub-GeV core), and σε(√s') = σ_WW→lνqq(√s')·ε(√s')
# = the production×selection cross-section (the 4-fermion / off-shell shape, the unknown for DATA).
# This is a PRODUCT, not a convolution: L and σε are independent given √s'.
#
# The current analytic model (conv_mw_gen_validation.py) uses σε ≈ Z_CC03 = ∫dm_h dm_l BW·BW·PS (the
# doubly-resonant CC03 production weight).  Question this script settles by MEASUREMENT:
#   Is "model √s' too narrow / too high" the radiator's fault (L) or the production weight's (σε)?
# We EXTRACT σε_eff(√s') = P_gen(√s')/L(√s') from the data and overlay it on Z_CC03(√s').  If σε_eff has
# a much fatter below-threshold tail than Z_CC03, the CC03 weight (not the ISR radiator) is the culprit,
# and the fix is a 4f-shaped σ_WW(√s') — the SAME physics as the known #1 absolute-mW lever (4f).
#
# Research-validated luminosity kernel (FCC-ee WW threshold):
#   ISR structure function, KKMC η-convention, soft-exponentiated (YFS) + O(α) and O(α²) hard terms
#     η = (2α/π)(ln(s/m_e²) − 1) = 0.1129 @ 160 GeV   (β_SHERPA = η/2);  leading power (1−x)^(η/2−1)
#   ⊗ Gaussian BES σ(√s') (sample-measured 0.116/0.119/0.121 GeV; research CDR ~0.150);
#   beamstrahlung NEGLIGIBLE at FCC-ee (mean ~1.4 MeV, Υ≪1).
#   NOTE: conv_mw_gen_validation.py's lumi_density has a convention slip — it uses (1−x)^(β−1) with
#   β:=η (≈0.113) instead of the correct (1−x)^(η/2−1); that makes its core slightly TOO BROAD, so the
#   true radiator core is even narrower ⇒ reinforces "the width is σε, not the radiator". Fixed here.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_sqrts_decomp.py
import os, numpy as np
from math import gamma
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import uproot

GW      = float(os.environ.get("GW", "2.085"))
MW_REF  = float(os.environ.get("MW_REF", "80.379"))
SIG_BES_ECM = {157: 0.116, 160: 0.119, 163: 0.121}     # sample-measured BES on √s' [GeV]
EOSW    = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW, exist_ok=True)
ALPHA   = 1.0/137.035999; ME = 0.000510999; GAMMA_E = 0.5772156649

# ── BW + log_Z (= ∫BW·BW·PS production weight, CC03) — verbatim machinery from the analysis ──────────
LOGZ_N = 256; _GX,_GWl = np.polynomial.legendre.leggauss(LOGZ_N); _W2 = _GWl[:,None]*_GWl[None,:]
def log_Z(m_WW, mW, gW=GW):
    m_WW = np.atleast_1d(np.asarray(m_WW,float)); mwgw = mW*gW; mW2 = mW*mW
    s = m_WW**2; t_min = np.arctan(-mW2/mwgw); t_max = np.arctan((s-mW2)/mwgw)
    hd = 0.5*(t_max-t_min); hs = 0.5*(t_max+t_min); t = hd[:,None]*_GX[None,:]+hs[:,None]
    m = np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv = 1/m
    mh = m[:,:,None]; ml = m[:,None,:]; ih = inv[:,:,None]; il = inv[:,None,:]; sE = s[:,None,None]
    lam = (sE-(mh+ml)**2)*(sE-(mh-ml)**2)
    integ = np.where(lam>0, np.sqrt(np.maximum(lam,0))*ih*il/(4*sE), 0.0)
    Z = np.sum(_W2[None]*integ,axis=(1,2))*hd*hd; return np.log(np.maximum(Z,1e-300))

# ── ISR luminosity kernel: correct η-convention KKMC radiator (YFS-exp + O(α),O(α²) hard) ⊗ BES ──────
def isr_radiator(x, ecm):
    """KKMC η-convention structure function D(x,s), x = E'/E retained per beam (single-radiator form)."""
    s = ecm*ecm; eta = (2*ALPHA/np.pi)*(np.log(s/ME**2) - 1.0)
    x = np.clip(x, 1e-12, 1-1e-12); om = 1.0 - x
    pref = np.exp((eta/2)*(0.75 - GAMMA_E))/gamma(1+eta/2)                       # YFS resummation ~1.04
    soft = pref*(eta/2)*om**(eta/2-1.0)*(1.0 + (3.0/8.0)*eta + eta*eta*(9.0/128.0 - np.pi**2/48.0))
    hard1 = -(eta/4.0)*(1.0+x)
    hard2 = (eta*eta/32.0)*(-4.0*(1.0+x)*np.log(om) - ((1.0+3.0*x*x)/om)*np.log(x) + 5.0 + x)
    return np.maximum(soft + hard1 + hard2, 0.0)

def lumi_density(grid, ecm, sig_bes):
    """L(√s') on `grid`: ISR radiator (in √s') ⊗ Gaussian BES, normalized to ∫=1 over the fine support.
    The radiator is ZERO for √s' >= ECM (ISR can only LOSE energy); BES is the only thing that smears
    a little above ECM.  (The earlier conv_mw_gen_validation.py clipped x to the endpoint singularity for
    √s'>ECM, creating a spurious huge plateau above ECM that pulled the kernel mean ABOVE ECM and crushed
    the real low-√s' ISR tail — the root cause of 'model √s' too narrow / peaked too high'.)"""
    s = ecm*ecm; fine = np.linspace(ecm-60.0, ecm+3.0, 40000); x = (fine/ecm)**2
    p = np.where(fine < ecm, isr_radiator(np.minimum(x, 1-1e-15), ecm) * (2*fine/s), 0.0)  # |dx/d√s'|=2√s'/s
    if sig_bes > 0:
        dxg = fine[1]-fine[0]; n = int(6*sig_bes/dxg); kx = np.arange(-n, n+1)*dxg
        ker = np.exp(-0.5*(kx/sig_bes)**2); ker /= ker.sum(); p = np.convolve(p, ker, mode="same")
    p = np.maximum(p, 0.0); p /= np.trapz(p, fine)
    return np.interp(grid, fine, p, left=0.0, right=0.0)

def wmean_rms(grid, dens):
    dens = np.maximum(dens, 0.0); A = np.trapz(dens, grid)
    if A <= 0: return float("nan"), float("nan")
    dens = dens/A; mu = np.trapz(grid*dens, grid); return float(mu), float(np.sqrt(max(np.trapz((grid-mu)**2*dens, grid),0)))

ECMS = [157, 160, 163]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, AX = plt.subplots(len(ECMS), 2, figsize=(15, 4.6*len(ECMS)))
hdr = f"{'ECM':>4s} | {'gen mean':>9s} {'gen rms':>8s} | {'L·Z_CC03 mean':>13s} {'rms':>6s} | {'L·σε_eff mean':>13s} {'rms':>6s}"
print(hdr); print("-"*len(hdr))
for r, ECM in enumerate(ECMS):
    ROOT = f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
    a = uproot.open(ROOT)["events"].arrays(["gen_Whad_m","gen_Wlep_m","gen_WW_m"], library="np")
    ok = (np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&
          (a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    sqs = a["gen_WW_m"][ok]; sig = SIG_BES_ECM[ECM]

    lo, hi = ECM-16.0, ECM+2.0; BW_ = 0.20
    edges = np.arange(lo, hi+BW_, BW_); cen = 0.5*(edges[:-1]+edges[1:])
    Pg, _ = np.histogram(sqs, bins=edges, density=True)                          # P_gen(√s') density

    L   = lumi_density(cen, ECM, sig)                                            # luminosity kernel
    Zcc = np.exp(log_Z(cen, MW_REF))                                             # CC03 ∫BW·BW·PS
    # extract σε_eff = P_gen / L  (only where L is meaningfully supported)
    Lthr = L.max()*1e-4; good = L > Lthr
    sige = np.full_like(cen, np.nan); sige[good] = Pg[good]/L[good]

    # forward models (both normalized to unit area over the window for shape comparison)
    mZ = L*Zcc;                  mZ = mZ/np.trapz(np.where(np.isfinite(mZ),mZ,0), cen)
    mE = np.where(good, L*sige, 0.0); mE = mE/np.trapz(mE, cen)
    Pgn = Pg/np.trapz(Pg, cen)

    gmu,grms = wmean_rms(cen, Pgn); zmu,zrms = wmean_rms(cen, mZ); emu,erms = wmean_rms(cen, mE)
    print(f"{ECM:>4d} | {gmu:9.3f} {grms:8.3f} | {zmu:13.3f} {zrms:6.3f} | {emu:13.3f} {erms:6.3f}")

    # ── panel A: √s' densities ──
    ax = AX[r,0]
    ax.step(cen, Pgn, where="mid", color="k", lw=1.7, label="gen √s'")
    ax.plot(cen, mZ, color="C2", lw=2.4, label=f"model L·Z_CC03 (rms {zrms:.2f})")
    ax.plot(cen, mE, color="C3", lw=2.0, ls="--", label=f"L·σε_eff = gen by constr. (rms {erms:.2f})")
    ax.axvline(2*MW_REF, color="0.5", ls=":", lw=1.2); ax.text(2*MW_REF, ax.get_ylim()[1]*0.5, " 2m_W", color="0.4", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 5); ax.set_xlim(lo, hi)
    ax.set_title(f"ecm{ECM}: √s' spectrum  (gen rms {grms:.2f})"); ax.set_xlabel("√s' [GeV]"); ax.legend(fontsize=9)
    # ── panel B: extracted σε_eff(√s') vs CC03 Z ──
    ax = AX[r,1]
    sn = lambda v: v/np.nanmax(v[good])
    ax.step(cen[good], sn(sige)[good], where="mid", color="k", lw=1.8, label="σε_eff = P_gen / L (extracted)")
    ax.plot(cen, Zcc/np.nanmax(Zcc[good]), color="C2", lw=2.4, label="Z_CC03 = ∫BW·BW·PS (current model)")
    ax.axvline(2*MW_REF, color="0.5", ls=":", lw=1.2); ax.text(2*MW_REF, 1.0, " 2m_W", color="0.4", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 2.0); ax.set_xlim(lo, hi)
    ax.set_title(f"ecm{ECM}: production×selection σε(√s')  (norm. to peak)"); ax.set_xlabel("√s' [GeV]"); ax.legend(fontsize=9)

fig.suptitle("√s' = L(√s')·σε(√s') decomposition — is the model too narrow because of the radiator L or the CC03 weight σε?", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.985]); png=f"{EOSW}/sqrts_decomp_day7.png"; fig.savefig(png,dpi=130); print(f"\n[plot] {png}")
