#!/usr/bin/env python3
# ── Validate the ISR-convolved double-BW model against GEN-level quantities ──────────────────────────
# Our lineshape (build_Ptrue, copied verbatim from conv_mw_closure_anatomy.py):
#   P(m_h,m_l) = ∫d√s' R(√s')·BW(m_h;mW,Γ)·BW(m_l;mW,Γ)·PS(√s',m_h,m_l)/Z(√s')
# where R(√s') = ISR/radiator √s' spectrum and PS = √λ(s',m_h²,m_l²)/s' (2-body phase space).
# We overlay the model's predicted marginals on the GEN distributions for:
#   m_qq   = gen_Whad_m         (hadronic-W mass; BW×PS prediction)
#   m_lnu  = gen_Wlep_m         (leptonic-W mass; BW×PS prediction)
#   m_lnuqq= gen_WW_m = √s'     (full system / collision energy after ISR)
#   ISR    = |gen_isr| ≈ ECM−√s' (energy radiated)
# For m_qq/m_lnu the radiator R is taken data-driven from gen √s' (= what the analysis uses), so those test
# the BW×PS CORE.  For √s'/ISR we also overlay an ANALYTIC leading-log ISR structure function (Kuraev–Fadin,
# single effective radiator ⊗ Gaussian BES) to check the ISR physics independently.  "Match somehow", not exact:
# residuals expose 4f/off-shell/running-width effects the CC03 double-BW omits.
#
# source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
# python3 jax_prototype/conv_mw_gen_validation.py
import os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import uproot
from scipy.interpolate import RegularGridInterpolator  # noqa (kept for parity w/ anatomy)

BW_RUN  = int(os.environ.get("BW_RUN", "1"))           # 1=running-width BW Γ(m)=gW·m²/mW (DAY7 fix, default); 0=fixed
GW      = float(os.environ.get("GW", "2.049"))         # WHIZARD SM.mdl generator width wW=2.049 (was PDG 2.085)
MW_REF  = float(os.environ.get("MW_REF", "80.419"))    # WHIZARD generator mW=80.419 (was PDG 80.379); the gen peak tests it
SIG_BES_ECM = {157: 0.116, 160: 0.119, 163: 0.121}     # measured BES on √s' [GeV] (memory: depth-1 e± pair)
EOSW    = "/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW, exist_ok=True)
ALPHA   = 1.0/137.035999; ME = 0.000510999

# ── BW + log_Z + build_Ptrue (verbatim machinery from conv_mw_closure_anatomy.py) ───────────────────
LOGZ_N = 256; _GX,_GWl = np.polynomial.legendre.leggauss(LOGZ_N); _W2 = _GWl[:,None]*_GWl[None,:]
def log_Z(m_WW, mW, gW=GW):
    m_WW = np.atleast_1d(np.asarray(m_WW,float)); mwgw = mW*gW; mW2 = mW*mW
    s = m_WW**2; t_min = np.arctan(-mW2/mwgw); t_max = np.arctan((s-mW2)/mwgw)
    hd = 0.5*(t_max-t_min); hs = 0.5*(t_max+t_min); t = hd[:,None]*_GX[None,:]+hs[:,None]
    m = np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv = 1/m
    if BW_RUN:   # importance-reweight the fixed-width arctan sampling onto running-width Γ(m)=gW·m²/mW
        dd = m*m-mW2; wm = gW*m*m/mW; rf = (m*m/mW2)*(dd*dd+mwgw*mwgw)/(dd*dd+wm*wm)
    else: rf = np.ones_like(m)
    mh = m[:,:,None]; ml = m[:,None,:]; ih = inv[:,:,None]; il = inv[:,None,:]; sE = s[:,None,None]
    rh = rf[:,:,None]; rl = rf[:,None,:]
    lam = (sE-(mh+ml)**2)*(sE-(mh-ml)**2)
    integ = np.where(lam>0, np.sqrt(np.maximum(lam,0))*ih*il*rh*rl/(4*sE), 0.0)
    Z = np.sum(_W2[None]*integ,axis=(1,2))*hd*hd; return np.log(np.maximum(Z,1e-300))
def bw_fixed(m, mW, gW=GW): mwgw = mW*gW; d = m*m-mW*mW; return mwgw/(d*d+mwgw*mwgw)
def bw_run(m, mW, gW=GW):   d = m*m-mW*mW; wm = gW*m*m/mW; return wm/(d*d+wm*wm)
def bw(m, mW, gW=GW): return bw_run(m,mW,gW) if BW_RUN else bw_fixed(m,mW,gW)

TLO, THI, NT = 28.0, 96.0, 137
tg = np.linspace(TLO, THI, NT); dt = tg[1]-tg[0]
TH, TL = np.meshgrid(tg, tg, indexing="ij")
def radiator_obs(mWW_g, ecm):
    cnt, edg = np.histogram(mWW_g, bins=60, range=(max(TLO, ecm-45), ecm+1.5))
    sn = 0.5*(edg[:-1]+edg[1:]); w = cnt.astype(float); m = w>0
    return sn[m], w[m]
def build_Ptrue(mW, rad):
    sn, sw = rad; bwh = bw(tg, mW); BWout = bwh[:,None]*bwh[None,:]
    u = np.zeros((NT,NT)); logZ = log_Z(sn, mW)
    for k,(sp,wk) in enumerate(zip(sn,sw)):
        s = sp*sp; lam = (s-(TH+TL)**2)*(s-(TH-TL)**2)
        PS = np.where(lam>0, np.sqrt(np.maximum(lam,0))/s, 0.0)
        u += (wk/np.exp(logZ[k])) * BWout * PS
    Z = u.sum()*dt*dt; return u/np.maximum(Z,1e-300)

# ── ISR luminosity spectrum: leading-log structure function (Kuraev–Fadin, soft-exponentiated) ⊗ BES ─
def lumi_density(sqrts, ecm, sig_bes):
    """dL/d√s' = SF(x=s'/s) |dx/d√s'| ⊗ Gaussian(σ_BES). This is the e+e- collision-energy spectrum
    BEFORE the WW production weighting (NO threshold)."""
    s = ecm*ecm; beta = (2*ALPHA/np.pi)*(np.log(s/ME**2)-1.0)
    fine = np.linspace(ecm-60.0, ecm+3.0, 8000); x = np.clip((fine/ecm)**2, 1e-12, 1-1e-12)
    D = beta*(1-x)**(beta-1)*(1+0.75*beta) - 0.5*beta*(1+x)        # KF radiator in x=s'/s (soft-exp factor)
    # radiator is ZERO for √s'>=ECM (ISR can only LOSE energy); previously x was clipped to the endpoint
    # singularity above ECM, creating a spurious huge plateau that pulled the kernel mean ABOVE ECM and
    # crushed the real low-√s' ISR tail — that bug, NOT the radiator, made the model √s' look "too narrow".
    p = np.where(fine < ecm, np.maximum(D,0.0)*(2*fine/s), 0.0)    # |dx/d√s'| = 2√s'/s
    if sig_bes > 0:                                                 # convolve with Gaussian BES on √s'
        dxg = fine[1]-fine[0]; n = int(6*sig_bes/dxg); kx = np.arange(-n, n+1)*dxg
        ker = np.exp(-0.5*(kx/sig_bes)**2); ker /= ker.sum(); p = np.convolve(p, ker, mode="same")
    p /= np.trapz(p, fine); return fine, p

def model_sqrts_density(sqrts, ecm, mW, sig_bes):
    """Predicted √s' spectrum for SELECTED WW = luminosity(SF⊗BES) × WW production rate Z(√s').
    Z(√s')=∫dm_h dm_l BW·BW·PS rises from threshold (off-shell tail below 2mW), so it suppresses the
    bare-SF low-√s' tail — this is why the luminosity alone looked nothing like gen √s'."""
    fine, lum = lumi_density(None, ecm, sig_bes)
    Z = np.exp(log_Z(fine, mW)); p = lum*Z; p /= np.trapz(p, fine)
    return np.interp(sqrts, fine, p, left=0, right=0)

def norm_hist(x, bins, rng):
    h, e = np.histogram(x, bins=bins, range=rng, density=True); c = 0.5*(e[:-1]+e[1:]); return c, h

def stats(x): return float(np.mean(x)), float(np.std(x))
def model_stats(grid, dens):  # mean/RMS of a density on a grid
    dens = dens/np.trapz(dens, grid); mu = np.trapz(grid*dens, grid)
    return float(mu), float(np.sqrt(np.trapz((grid-mu)**2*dens, grid)))

ECMS = [157, 160, 163]
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13})
fig, AX = plt.subplots(len(ECMS), 4, figsize=(24, 5.6*len(ECMS)))
print(f"{'ECM':>4s} {'quantity':9s} {'gen mean':>9s} {'gen rms':>8s} {'model mean':>10s} {'model rms':>9s}")
for r, ECM in enumerate(ECMS):
    ROOT = f"outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full/wzp6_ee_munumuqq_noCut_ecm{ECM}.root"
    a = uproot.open(ROOT)["events"].arrays(
        ["gen_Whad_m","gen_Wlep_m","gen_WW_m","reco_jet1_p","reco_jet2_p","reco_lep_p",
         "gen_isr_px","gen_isr_py","gen_isr_pz"], library="np")
    ok = (np.isfinite(a["gen_Whad_m"])&np.isfinite(a["gen_Wlep_m"])&np.isfinite(a["gen_WW_m"])&
          (a["reco_jet1_p"]>0)&(a["reco_jet2_p"]>0)&(a["reco_lep_p"]>0)&
          (a["gen_Whad_m"]+a["gen_Wlep_m"]<a["gen_WW_m"]))
    mqq=a["gen_Whad_m"][ok]; mlv=a["gen_Wlep_m"][ok]; sqs=a["gen_WW_m"][ok]
    isrE=np.sqrt(a["gen_isr_px"][ok]**2+a["gen_isr_py"][ok]**2+a["gen_isr_pz"][ok]**2)

    P = build_Ptrue(MW_REF, radiator_obs(sqs, ECM))        # the analysis model at PDG mW
    Pmh = P.sum(1)*dt; Pml = P.sum(0)*dt                   # marginals (already normalized: ∫=1)

    # m_qq
    ax=AX[r,0]; c,h=norm_hist(mqq,80,(60,95)); ax.step(c,h,where="mid",color="k",label="gen m_qq")
    ax.plot(tg,Pmh,color="C3",lw=2.6,label="model BW×PS")
    ax.set_xlim(60,95); ax.set_title(f"ecm{ECM}: m_qq (hadronic W)"); ax.legend(fontsize=11); ax.set_xlabel("GeV")
    print(f"{ECM:>4d} {'m_qq':9s} {stats(mqq)[0]:9.3f} {stats(mqq)[1]:8.3f} {model_stats(tg,Pmh)[0]:10.3f} {model_stats(tg,Pmh)[1]:9.3f}")
    # m_lnu
    ax=AX[r,1]; c,h=norm_hist(mlv,80,(60,95)); ax.step(c,h,where="mid",color="k",label="gen m_lνu")
    ax.plot(tg,Pml,color="C0",lw=2.6,label="model BW×PS")
    ax.set_xlim(60,95); ax.set_title(f"ecm{ECM}: m_lνu (leptonic W)"); ax.legend(fontsize=11); ax.set_xlabel("GeV")
    print(f"{ECM:>4d} {'m_lnu':9s} {stats(mlv)[0]:9.3f} {stats(mlv)[1]:8.3f} {model_stats(tg,Pml)[0]:10.3f} {model_stats(tg,Pml)[1]:9.3f}")
    sig = SIG_BES_ECM[ECM]
    # m_lnuqq = √s'  —  gen vs luminosity(SF⊗BES) vs full model (luminosity × Z production weighting)
    ax=AX[r,2]; lo=ECM-12
    c,h=norm_hist(sqs,90,(lo,ECM+1.5)); ax.step(c,h,where="mid",color="k",lw=1.6,label="gen √s'")
    sg=np.linspace(lo,ECM+1.5,500)
    fl,pl=lumi_density(None,ECM,sig); ax.plot(sg,np.interp(sg,fl,pl/np.trapz(pl[fl>=lo],fl[fl>=lo])),color="C1",ls="--",lw=1.8,label="SF⊗BES (luminosity)")
    pm=model_sqrts_density(sg,ECM,MW_REF,sig); ax.plot(sg,pm,color="C2",lw=2.6,label="model = SF⊗BES × Z(√s')")
    ax.set_xlim(lo,ECM+1.5); ax.set_title(f"ecm{ECM}: m_lνuqq = √s'  (σ_BES={sig})"); ax.legend(fontsize=10); ax.set_xlabel("GeV"); ax.set_yscale("log"); ax.set_ylim(1e-3,5)
    # ⚠ measure the model RMS on the FULL √s' support, NOT the plot window [lo,ECM+1.5]: the gen RMS (stats(sqs))
    # is over the full sample, so windowing only the model chops its low-√s' tail and FALSELY reports it "too
    # narrow" (DAY7 artifact: model 1.54 vs gen 2.70 was model-on-window vs gen-on-full; matched, both ≈2.1).
    sg_full=np.linspace(ECM-60.0,ECM+3.0,4000); pm_full=model_sqrts_density(sg_full,ECM,MW_REF,sig)
    mm=model_stats(sg_full,pm_full); print(f"{ECM:>4d} {'sqrts':9s} {stats(sqs)[0]:9.3f} {stats(sqs)[1]:8.3f} {mm[0]:10.3f} {mm[1]:9.3f}")
    # ISR energy = ECM − √s'
    ax=AX[r,3]; maxi=ECM-lo
    c,h=norm_hist(isrE,80,(0,maxi)); ax.step(c,h,where="mid",color="k",lw=1.6,label="gen |p_ISR|")
    c2,h2=norm_hist(ECM-sqs,80,(0,maxi)); ax.step(c2,h2,where="mid",color="C7",ls="--",label="gen ECM−√s'")
    eg=np.linspace(0.02,maxi,500); pis=model_sqrts_density(ECM-eg,ECM,MW_REF,sig); ax.plot(eg,pis/np.trapz(pis,eg),color="C2",lw=2.4,label="model (ECM−√s')")
    ax.set_xlim(0,maxi); ax.set_yscale("log"); ax.set_ylim(1e-3,5); ax.set_title(f"ecm{ECM}: ISR energy"); ax.legend(fontsize=10); ax.set_xlabel("GeV")
    print(f"{ECM:>4d} {'ISR':9s} {stats(isrE)[0]:9.3f} {stats(isrE)[1]:8.3f} {stats(ECM-sqs)[0]:10.3f} {stats(ECM-sqs)[1]:9.3f}")

_bwlab = "running-width" if BW_RUN else "fixed-width"
fig.suptitle(f"ISR-convolved double-BW [{_bwlab}] vs GEN  (WHIZARD generator mW={MW_REF}, Γ_W={GW}); √s' model = SF⊗BES × Z(√s')", fontsize=14)
fig.tight_layout(); png=f"{EOSW}/gen_validation.png"; fig.savefig(png,dpi=130); print(f"\n[plot] {png}")
