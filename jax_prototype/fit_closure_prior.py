#!/usr/bin/env python3
# Fit the kfit ENERGY-CLOSURE prior r = E_ISR - |p_ISR| (ISR-system invariant mass)
# with the SAME spike_dcb2g model used for gen_isr_px/pz, reusing fit_resolutions'
# fit + norm + emit helpers so the header constant matches conventions exactly.
# Emits  SDCBG_GEN_ISR_SYS_MASS_<ecm>  for kinfit_inputs/dcb_params.h.
#
# r is one-sided at truth (mass>=0: 96% spike at 0 + positive tail); we SYMMETRIZE
# (like gen_WW_m_minus_m_ee) so the fit gives a clean spike + symmetric heavy tail
# usable for both signs of r at reco, with the MEASURED width/tail replacing the
# tuned sigma_R Gaussian. The heavy DCB tail is what tolerates the jet-scale
# residual (the reason the sigma_R scan wanted ~1.5) WITHOUT a quadratic blow-up.
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/fit_closure_prior.py
import sys, os, numpy as np, uproot
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import quad as _quad
from scipy.stats import median_abs_deviation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (fit_resolutions.py)
import fit_resolutions as F

ECMS=[157,160,163]
ROOT_TMPL="outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm{ecm}.root"
DELTA=0.001; SIGRES=0.001; FWMAX=0.45; NBINS=200; CLIP=(0.5,99.5)
OUT="/eos/user/m/mdefranc/www/mW/jax_lnuqq"; os.makedirs(OUT,exist_ok=True)

def r_gen_of(a, ecm):
    bes=a["gen_ee_m_minus_ecm"]; m_ee=ecm+bes; E_ee=np.sqrt(m_ee**2+a["gen_ee_pz"]**2)
    E_WW=np.sqrt(a["gen_WW_m"]**2+a["gen_WW_px"]**2+a["gen_WW_py"]**2+a["gen_WW_pz"]**2)
    isr_p=np.sqrt(a["gen_isr_px"]**2+a["gen_isr_py"]**2+a["gen_isr_pz"]**2)
    return (E_ee-E_WW)-isr_p

lines=[]
fig,axes=plt.subplots(1,3,figsize=(17,5))
for j,ecm in enumerate(ECMS):
    t=uproot.open(ROOT_TMPL.format(ecm=ecm))["events"]
    br=["gen_ee_m_minus_ecm","gen_ee_pz","gen_WW_m","gen_WW_px","gen_WW_py","gen_WW_pz",
        "gen_isr_px","gen_isr_py","gen_isr_pz"]
    a=t.arrays(br,library="np")
    vals=r_gen_of(a,ecm); vals=vals[np.isfinite(vals)]
    std_raw=np.std(vals)
    vals=np.concatenate([vals,-vals])                       # symmetrize (mass is one-sided)
    f_delta=float((np.abs(vals)<DELTA).mean())              # spike fraction
    body=vals[np.abs(vals)>=DELTA]
    lo,hi=np.percentile(body,[CLIP[0],CLIP[1]]); body_c=body[(body>=lo)&(body<=hi)]
    counts,edges=np.histogram(body_c,bins=NBINS); centers=0.5*(edges[:-1]+edges[1:]); m=counts>0
    mu0=float(centers[np.argmax(counts)]); sig0=float(median_abs_deviation(body_c,scale="normal"))
    popt,_,ok,chi2=F.fit_dcb2g(centers[m],counts[m],mu0,sig0,f_wide_max=FWMAX,mu_fix=0.0)
    ndof=max(int(m.sum())-10,1)
    if popt is None or chi2/ndof>2.5:
        p2,_,ok2,c2=F.fit_dcb2g_iminuit(centers[m],counts[m],mu0,sig0,f_wide_max=FWMAX,mu_fix=0.0)
        if p2 is not None and (popt is None or c2<chi2): popt,ok,chi2=p2,ok2,c2
    N_f,mu_c,sc,aL,nL,aR,nR,fw,muw,sw=popt
    mu_c=float(mu_c); sc=abs(float(sc)); aL=abs(float(aL)); nL=abs(float(nL))
    aR=abs(float(aR)); nR=abs(float(nR)); fw=abs(float(fw)); muw=float(muw); sw=abs(float(sw)); N_f=abs(float(N_f))
    chi2n=chi2/ndof
    p=dict(model="spike_dcb2g",f_delta=f_delta,sig_res=SIGRES,mu=mu_c,sigma=sc,aL=aL,nL=nL,aR=aR,nR=nR,
           f_wide=fw,mu_wide=muw,sigma_wide=sw)
    yfn=lambda x: F.dcb_gauss(x,N_f,mu_c,sc,aL,nL,aR,nR,fw,muw,sw)
    integ,_=_quad(lambda x: yfn(x)/N_f, edges[0], edges[-1], limit=300)
    p["norm"]=float(1.0/max(integ,1e-300))
    line=(f"constexpr SpikeDcbGaussParams SDCBG_GEN_ISR_SYS_MASS_{ecm} = "
          f"{{ {F._cpp_struct_initializer(p)} }};  // chi2/ndf={chi2n:.2f}  {'OK' if ok else 'WARN'}")
    lines.append(line)
    print(f"[ecm{ecm}] r std(raw,one-sided)={std_raw:.3f}  f_delta(sym)={f_delta:.3f}  "
          f"body: sigma={sc:.3f} aL/nL={aL:.2f}/{nL:.1f} aR/nR={aR:.2f}/{nR:.1f} fw={fw:.2f} sw={sw:.3f}  chi2/ndf={chi2n:.2f}")
    # plot
    ax=axes[j]
    allv=np.concatenate([body_c]); h,e2=np.histogram(allv,bins=200,range=(-3,3),density=False)
    cc=0.5*(e2[:-1]+e2[1:]); bw=e2[1]-e2[0]
    ax.step(cc,h,where="mid",color="C0",lw=1.2,label=f"r_gen (sym, no-spike)")
    xs=np.linspace(-3,3,600)
    Ntot=len(body_c)
    ax.plot(xs, yfn(xs)*p["norm"]*Ntot*bw, "C3", lw=1.8, label=f"spike_dcb2g body")
    ax.set_yscale("log"); ax.set_xlabel("r = E_ISR-|p_ISR| [GeV]"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"ecm{ecm}  f_Δ={f_delta*100:.1f}% σ={sc:.2f} χ²/ndf={chi2n:.2f}")
fig.suptitle("kfit energy-closure prior  SDCBG_GEN_ISR_SYS_MASS  (symmetrized spike+DCB tail; replaces tuned σ_R)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{OUT}/closure_prior_fit.png",dpi=110)
print(f"\nwrote {OUT}/closure_prior_fit.png")
print("\n===== paste into kinfit_inputs/dcb_params.h =====")
for l in lines: print(l)
