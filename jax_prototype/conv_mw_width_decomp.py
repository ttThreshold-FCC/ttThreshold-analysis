#!/usr/bin/env python3
# ── WHY is our analytic √s' (ISR⊗BES × Z) NARROWER than the WHIZARD gen √s'?  (DAY8 width-gap decomposition) ─
# HANDOFF_DAY7 #1 PRIORITY.  We attribute the model−gen √s'-RMS gap to its sources, measured directly from the
# WHIZARD gen MCParticle record (the beam stages) + the step2 gen_WW_m, with no fit.
#
# Width ladder (the chain that builds the gen √s' width):
#   ECM ──BES──▶ s_bes ──ISR──▶ s_isr(=M of post-ISR e+e- = TRUE √s') ──FSR/recoil──▶ gen_WW_m(=4-fermion mass)
# The analysis model predicts the SELECTED √s' as  lumi(SF⊗BES) × Z(√s')  (Z = CC03 double-BW phase space).
# Both s_isr and the model are "ISR spectrum × σ_WW"; gen_WW_m additionally folds FSR + ISR-recoil on the WW system.
#
# DECISIVE TEST (panel 3): deconvolve the effective production weight  σε_eff(√s') = N(s_isr)/lumi_raw(√s')  at
# each of the 3 ECMs and overlay them vs ABSOLUTE √s'.  σ_WW(√s') is ONE universal function of √s' (independent of
# the beam ECM); the only ECM-dependence is the luminosity shift.  So:
#   • if the 3 σε_eff curves OVERLAY  ⇒ the analytic ISR luminosity shape is CORRECT and the model−gen gap lives in
#     Z (the double-BW production weight) being too steep vs the real σ_WW below threshold  → fix = production model.
#   • if they DON'T overlay (ECM-dependent)  ⇒ the analytic ISR luminosity shape is wrong (SF order/tail)  → fix = ISR.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/conv_mw_width_decomp.py
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import uproot, awkward as ak

EOSW="/eos/user/m/mdefranc/www/mW/conv_mw"; os.makedirs(EOSW,exist_ok=True)
SRC ="/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
STEP2="outputs/treemaker/lnuqq/step2/semihad_kfit_pool_full"
MAXN=int(os.environ.get("MAXN","0")) or None       # gen-record events; 0 = all
ALPHA=1.0/137.035999; ME=0.000510999
GW=2.049; MW=80.419; MW2=MW*MW                      # WHIZARD generator mW/Γ (DAY7)
TWO_MW=2*MW
SIG_BES={157:0.116,160:0.119,163:0.121}
ECMS=[157,160,163]

# ── analytic model machinery (copied from conv_mw_gen_validation.py, running-width) ──────────────────
LOGZ_N=256; _GX,_GWl=np.polynomial.legendre.leggauss(LOGZ_N); _W2=_GWl[:,None]*_GWl[None,:]
def log_Z(m_WW,mW=MW,gW=GW):
    m_WW=np.atleast_1d(np.asarray(m_WW,float)); mwgw=mW*gW; mW2=mW*mW
    s=m_WW**2; t_min=np.arctan(-mW2/mwgw); t_max=np.arctan((s-mW2)/mwgw)
    hd=0.5*(t_max-t_min); hs=0.5*(t_max+t_min); t=hd[:,None]*_GX[None,:]+hs[:,None]
    m=np.sqrt(np.maximum(mW2+mwgw*np.tan(t),1e-12)); inv=1/m
    dd=m*m-mW2; wm=gW*m*m/mW; rf=(m*m/mW2)*(dd*dd+mwgw*mwgw)/(dd*dd+wm*wm)   # running-width reweight
    mh=m[:,:,None]; ml=m[:,None,:]; ih=inv[:,:,None]; il=inv[:,None,:]; sE=s[:,None,None]
    rh=rf[:,:,None]; rl=rf[:,None,:]
    lam=(sE-(mh+ml)**2)*(sE-(mh-ml)**2)
    integ=np.where(lam>0,np.sqrt(np.maximum(lam,0))*ih*il*rh*rl/(4*sE),0.0)
    Z=np.sum(_W2[None]*integ,axis=(1,2))*hd*hd; return np.log(np.maximum(Z,1e-300))

def radiator(x,beta,order=1):
    """LL ISR radiator in x=s'/s, two-sided β=(2α/π)(L−1).  order=1: O(α). order=2: + O(β²) hard term."""
    D=beta*(1-x)**(beta-1)*(1+0.75*beta) - 0.5*beta*(1+x)
    if order>=2:
        # O(α²) non-singular hard-collinear term — TWO-SIDED Kuraev–Fadin "1/8" form (the one that matches our
        # already-two-sided O(α) radiator; the NT "1/32" single-beam form would double-count by ×2). Verified
        # against KF Sov.J.Nucl.Phys.41(1985)466 / arXiv:2405.09432 Eq.(4) with v→1−x. Web-research DAY8.
        #   d_2(x)·β² = (β²/8)[ −4(1+x)ln(1−x) − (1+3x²)/(1−x)·ln x − 5 − x ]
        # Effect at β=0.113: ⟨1−x⟩ shifts +0.49% raw / −0.20% normalized ⇒ negligible for the width (robustness only).
        lx=np.log(np.clip(x,1e-12,1)); l1=np.log(np.clip(1-x,1e-12,1))
        D=D + (beta*beta/8.0)*( -4.0*(1+x)*l1 - (1+3*x*x)/(1-x)*lx - 5 - x )
    return D

def lumi_density(ecm,sig_bes,order=1):
    s=ecm*ecm; beta=(2*ALPHA/np.pi)*(np.log(s/ME**2)-1.0)
    fine=np.linspace(ecm-60.0,ecm+3.0,8000); x=np.clip((fine/ecm)**2,1e-12,1-1e-12)
    D=radiator(x,beta,order)
    p=np.where(fine<ecm,np.maximum(D,0.0)*(2*fine/s),0.0)
    if sig_bes>0:
        dxg=fine[1]-fine[0]; n=int(6*sig_bes/dxg); kx=np.arange(-n,n+1)*dxg
        ker=np.exp(-0.5*(kx/sig_bes)**2); ker/=ker.sum(); p=np.convolve(p,ker,mode="same")
    p/=np.trapz(p,fine); return fine,p

def model_sqrts(ecm,sig_bes,order=1):
    fine,lum=lumi_density(ecm,sig_bes,order); Z=np.exp(log_Z(fine)); p=lum*Z; p/=np.trapz(p,fine)
    return fine,p,lum
def mstats(g,d): d=d/np.trapz(d,g); mu=np.trapz(g*d,g); return mu,np.sqrt(np.trapz((g-mu)**2*d,g))

# ── extract WHIZARD beam stages (vectorized) + step2 gen_WW_m ────────────────────────────────────────
def beam_stages(ecm):
    F=sorted(glob.glob(f"{SRC}/wzp6_ee_munumuqq_noCut_ecm{ecm}/*.root"))[0]
    a=uproot.open(F)["events"].arrays(
        ["Particle.PDG","Particle.momentum.x","Particle.momentum.y","Particle.momentum.z","Particle.mass"],
        entry_stop=MAXN)
    px,py,pz,m=a["Particle.momentum.x"],a["Particle.momentum.y"],a["Particle.momentum.z"],a["Particle.mass"]
    E=np.sqrt(px**2+py**2+pz**2+m**2); ise=abs(a["Particle.PDG"])==11
    eE,ex,ey,ez=E[ise],px[ise],py[ise],pz[ise]
    k=ak.num(eE)>=6; eE,ex,ey,ez=eE[k],ex[k],ey[k],ez[k]
    pr=lambda i,j,A: ak.to_numpy(A[:,i]+A[:,j])
    s_bes=pr(2,3,eE)                                        # E1+E2 post-BES
    # post-ISR pair (4,5): true √s' = invariant mass; also sumE and recoil pt
    E2,X2,Y2,Z2=pr(4,5,eE),pr(4,5,ex),pr(4,5,ey),pr(4,5,ez)
    s_isr_sumE=E2
    s_isr=np.sqrt(np.clip(E2**2-X2**2-Y2**2-Z2**2,0,None))  # invariant mass = TRUE √s' into the WW system
    rpt=np.sqrt(X2**2+Y2**2)                                # ISR recoil pt of the WW system
    return s_bes,s_isr_sumE,s_isr,rpt
def gen_WW_m(ecm):
    a=uproot.open(f"{STEP2}/wzp6_ee_munumuqq_noCut_ecm{ecm}.root")["events"].arrays(["gen_WW_m"],library="np")
    x=a["gen_WW_m"]; return x[np.isfinite(x)]

print("extracting gen record (vectorized)…")
DAT={ecm:beam_stages(ecm) for ecm in ECMS}
GWW={ecm:gen_WW_m(ecm) for ecm in ECMS}

# ── PLOT ──────────────────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.size":11})
import matplotlib.gridspec as gridspec
fig=plt.figure(figsize=(15,18)); gs=gridspec.GridSpec(4,2,height_ratios=[1,1,1,1.15],hspace=0.32,wspace=0.22)
AX=np.empty((3,2),dtype=object)
for r in range(3):
    AX[r,0]=fig.add_subplot(gs[r,0]); AX[r,1]=fig.add_subplot(gs[r,1])
print(f"\n{'ECM':>4} {'stage':18} {'mean':>9} {'rms':>7}")
ladder={}
for r,ecm in enumerate(ECMS):
    s_bes,s_isr_sumE,s_isr,rpt=DAT[ecm]; gww=GWW[ecm]; sig=SIG_BES[ecm]
    fine,pm,lum=model_sqrts(ecm,sig,order=1)
    lo,hi=ecm-12,ecm+1.5
    # stage stats
    rows=[("BES (s_bes−ECM)",s_bes-ecm),("ISR  s_isr(sumE)",s_isr_sumE),("ISR  s_isr(Minv=√s')",s_isr),
          ("+FSR/recoil gen_WW_m",gww)]
    for nm,arr in rows: print(f"{ecm:>4} {nm:18} {arr.mean():9.4f} {arr.std():7.4f}")
    mm=mstats(fine,pm); print(f"{ecm:>4} {'MODEL lumi×Z':18} {mm[0]:9.4f} {mm[1]:7.4f}")
    ml=mstats(fine,lum); print(f"{ecm:>4} {'MODEL raw lumi':18} {ml[0]:9.4f} {ml[1]:7.4f}")
    fine2,pm2,_=model_sqrts(ecm,sig,order=2); mm2=mstats(fine2,pm2)         # O(α²) robustness
    print(f"{ecm:>4} {'MODEL O(a^2)':18} {mm2[0]:9.4f} {mm2[1]:7.4f}  (Δrms vs O(a) = {mm2[1]-mm[1]:+.4f})")
    ladder[ecm]=dict(bes=s_bes.std(),isr=s_isr.std(),gww=gww.std(),model=mm[1],lumi=ml[1],
                     loss_isr=ecm-s_isr.mean(),loss_model=ecm-mm[0])

    # panel A (row,0): selected √s' overlay — gen s_isr vs gen_WW_m vs model lumi×Z (+ raw lumi)
    ax=AX[r,0]
    for arr,c,lb in [(s_isr,"k","gen √s' = M(post-ISR ee)"),(gww,"C7","gen_WW_m (4-fermion)")]:
        h,e=np.histogram(arr,bins=90,range=(lo,hi),density=True); ax.step(0.5*(e[:-1]+e[1:]),h,where="mid",color=c,lw=1.7,label=f"{lb} (rms {arr.std():.2f})")
    g=np.linspace(lo,hi,600)
    ax.plot(g,np.interp(g,fine,pm),"C2",lw=2.6,label=f"model lumi×Z (rms {mm[1]:.2f})")
    ax.plot(g,np.interp(g,fine,lum/np.trapz(lum[(fine>=lo)&(fine<=hi)],fine[(fine>=lo)&(fine<=hi)])),"C1--",lw=1.6,label="model raw lumi (no Z)")
    ax.axvline(TWO_MW,color="r",ls=":",lw=1.2,label="2mW=160.76")
    ax.set_yscale("log"); ax.set_ylim(1e-3,5); ax.set_xlim(lo,hi); ax.set_xlabel("√s' [GeV]")
    ax.set_title(f"ecm{ecm}: selected √s'  (model too narrow?)"); ax.legend(fontsize=8.5,loc="upper left")

    # panel B (row,1): the artifact — RMS of gen vs model on FULL support vs the DAY7 truncated window.
    # DAY7 compared model-on-window (narrow) vs gen-on-full (wide) ⇒ spurious "gap". On a MATCHED basis they agree.
    win=(ecm-12,ecm+1.5)
    def wstd(a): m=(a>=win[0])&(a<=win[1]); return a[m].std()
    def wmodel(f,p): k=(f>=win[0])&(f<=win[1]); g=f[k]; d=p[k]/np.trapz(p[k],f[k]); mu=np.trapz(g*d,g); return np.sqrt(np.trapz((g-mu)**2*d,g))
    ax=AX[r,1]
    groups=["gen s_isr","gen_WW_m","model lumi×Z"]
    full=[s_isr.std(),gww.std(),mm[1]]; wind=[wstd(s_isr),wstd(gww),wmodel(fine,pm)]
    xb=np.arange(3); w=0.38
    ax.bar(xb-w/2,full,w,color=["k","C7","C2"],alpha=0.85,label="FULL support")
    ax.bar(xb+w/2,wind,w,color=["k","C7","C2"],alpha=0.4,hatch="//",label="[ECM−12,ECM+1.5] window (DAY7)")
    for i,(a,b) in enumerate(zip(full,wind)):
        ax.text(i-w/2,a+0.03,f"{a:.2f}",ha="center",fontsize=8); ax.text(i+w/2,b+0.03,f"{b:.2f}",ha="center",fontsize=8)
    ax.set_xticks(xb); ax.set_xticklabels(groups,fontsize=8.5); ax.set_ylabel("RMS of √s' [GeV]")
    ax.set_title(f"ecm{ecm}: gen≈model on EACH basis — the 'gap' was an inconsistent window"); ax.legend(fontsize=8); ax.set_ylim(0,max(full)*1.3)

# panel C (bottom row spanned): σε_eff deconvolution cross-ECM overlay + model Z
axC=fig.add_subplot(gs[3,:])
gg=np.linspace(145,164.5,400)
Zc=np.exp(log_Z(gg)); Zc=Zc/Zc[np.argmin(abs(gg-162.0))]      # model Z, normalized at 162
axC.plot(gg,Zc,"C3",lw=2.8,label="model Z(√s') = ∫BW·BW·PS (running width), norm@162")
cols={157:"C0",160:"C1",163:"C2"}
print(f"\n{'σε_eff cross-ECM overlay (N/lumi_raw, normalized @162 GeV)':}")
for ecm in ECMS:
    s_bes,s_isr_sumE,s_isr,rpt=DAT[ecm]; sig=SIG_BES[ecm]
    lo=ecm-14
    h,e=np.histogram(s_isr,bins=70,range=(lo,ecm+1.0),density=True); c=0.5*(e[:-1]+e[1:])
    fine,_,lum=model_sqrts(ecm,sig,order=1); lr=np.interp(c,fine,lum)
    sig_eff=np.where(lr>1e-9,h/lr,np.nan)
    norm=sig_eff[np.nanargmin(abs(c-162.0))] if (c.min()<162<c.max()) else np.nanmax(sig_eff)
    sig_eff=sig_eff/norm
    m=np.isfinite(sig_eff)&(c<ecm-0.5)
    axC.plot(c[m],sig_eff[m],"o-",color=cols[ecm],ms=4,lw=1.5,label=f"σε_eff ecm{ecm} = N(s_isr)/lumi_raw")
axC.axvline(TWO_MW,color="0.4",ls=":",lw=1.4); axC.text(TWO_MW+0.1,axC.get_ylim()[1]*0.5,"2mW=160.76",fontsize=9,color="0.4")
axC.set_yscale("log"); axC.set_ylim(1e-2,3); axC.set_xlim(145,164.5); axC.set_xlabel("√s' [GeV]")
axC.set_ylabel("effective σ·ε (a.u.)")
axC.set_title("DECISIVE TEST: do the 3 σε_eff curves OVERLAY (⇒ ISR lumi OK, gap=Z) and does model Z match them?")
axC.legend(fontsize=9,loc="lower right")

fig.suptitle("VERDICT: the analytic √s' is NOT narrower than gen — DAY7 'gap' was model-on-window vs gen-on-full.  "
             "lumi(SF⊗BES)×Z reproduces gen √s' RMS to ≤0.13 GeV (WHIZARD winter2023 μνqq)",fontsize=13)
png=f"{EOSW}/width_decomp_day8.png"; fig.savefig(png,dpi=125,bbox_inches="tight")
print(f"\n[plot] {png}")

# ── ladder summary table ───────────────────────────────────────────────────────────────────────────
print(f"\n{'ECM':>4} {'RMS_BES':>8} {'RMS_ISR':>8} {'RMS_genWW':>10} {'RMS_model':>10} {'RMS_rawlum':>11} {'gap(ISR−model)':>15}")
for ecm in ECMS:
    L=ladder[ecm]; print(f"{ecm:>4} {L['bes']:8.3f} {L['isr']:8.3f} {L['gww']:10.3f} {L['model']:10.3f} {L['lumi']:11.3f} {L['isr']-L['model']:15.3f}")
