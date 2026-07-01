#!/usr/bin/env python3
# ── Fully-differentiable WW->lnuqq (semileptonic) kinematic fit (JAX) ─────────
# Faithful port of FCCAnalyses::WWFunctions::kinFit (WWKinReco.h) to JAX + batched
# Levenberg-Marquardt. 16 params, mW FREE-FLOATING (rescale 2 GeV, no prior) —
# exactly as the C++ lnuqq fit. NO jet->W pairing ambiguity (2 jets = hadronic W,
# lepton+MET = leptonic W) -> one fit/event. lnuqq is well-conditioned (Minuit
# already ~83% valid at ecm160), so this validates the differentiable approach on
# the easier channel before returning to 4q.
#
#   source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
#   python3 jax_prototype/kinfit_lnuqq_jax.py <lnuqq_step2.root> [ecm]
import sys, time, os
# Multi-core: expose NDEV fake CPU devices so pmap shards the per-event LM across
# cores (must be set BEFORE jax is imported). Default 32 of the 64 cores.
NDEV=int(os.environ.get("NDEV","32"))
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={NDEV}")
os.environ.setdefault("OMP_NUM_THREADS","1")   # 1 BLAS thread/device (16x16 eigh is tiny); avoid oversubscription
import numpy as np, jax, jax.numpy as jnp, uproot
sys.path.insert(0, os.path.dirname(__file__))
import jaxfit_common as J

KF_MW_INIT=80.419; KF_GW_FIXED=2.049; KF_MW_PHYS_SIGMA=2.0
KF_GW_PRIOR_SIGMA=0.01*KF_GW_FIXED
KF_GW_PRIOR_LOG_NORM=np.log(2.0*np.pi*KF_GW_PRIOR_SIGMA**2)
BARRIER_SIGMA=float(os.environ.get("BARRIER_SIGMA","0.05"))  # soft m_loss>0 wall (C++ hard=1e-4)
TRUST=float(os.environ.get("TRUST","1.0"))
# ISR energy-conservation model:
#   "mloss"  = OLD: penalize m_WW - (ECM+besm) by spike_dcb + barrier (punishes ISR events;
#              double-counts the ISR pz prior since ECM-m_WW ~ |p_WW| for collinear ISR).
#   "photon" = NEW (ISR_FIT_PLAN variant B): treat the WW recoil as a collinear massless ISR
#              photon. p_isr=|beam-WW momentum|, r=(ECM+besm)-E_WW-|p_isr| = ISR-system mass
#              residual (~0, gen std 0.45). Penalize r by a tight Gaussian; the ISR 3-momentum
#              spectrum priors (ix,iy,iz) absorb the recoil softly -> ISR events not punished.
#   "kfit"   = NEWEST (ISR_FIT_PLAN "explicit k"): ISR photon momentum (px_γ,py_γ,k) are
#              EXPLICIT fit params (reusing the 3 neutrino slots y[5],y[8],y[11]); the
#              neutrino 3-momentum is DERIVED from 4-momentum conservation (p_nu = initial
#              − visible − p_γ), E_nu=|p_nu| (massless, exact → no quadratic branch). MET
#              measurement DROPPED (≡ −visible, redundant). Energy conservation is the only
#              soft residual r_E=(ECM+besm)−E_vis−E_nu−E_γ (≈ ISR-system mass, σ_R). This
#              removes the redundant free-nu_pz ↔ k longitudinal freedom → cures the free-mW
#              runaway tail seen in "photon" mode.
KF_ISR_MODE=os.environ.get("KF_ISR_MODE","mloss")
SIGMA_R=float(os.environ.get("KF_SIGMA_R","0.45"))   # ISR-system mass-residual prior width [GeV]
SIGMA_R_LOG_NORM=np.log(2.0*np.pi*SIGMA_R**2)
# kfit energy-closure prior:
#   "gauss"  = (r/SIGMA_R)^2 with SIGMA_R = the RECO-level closure width (~1.5). DEFAULT.
#   "sdcbg"  = MEASURED truth spike+DCB shape SDCBG_GEN_ISR_SYS_MASS — REJECTED 2026-06-15:
#             the truth closure is a near-delta at 0 (single-photon ISR-system mass ~0.42),
#             but the fit's r is dominated by RECO jet-energy residual (~1.5). Imposing the
#             truth shape (86% delta at 0) over-constrains -> WORSE (runaway 8.8→14.8%, Whad
#             bias -0.57→-1.29). So the closure penalty must be sized to the RECO residual,
#             not the truth ISR-system mass. (Truth measurement still useful: confirmed BES
#             is fully in besm, corr(r,BES)=0.002.)  See measure_closure_prior.py.
KF_CLOSURE=os.environ.get("KF_CLOSURE","gauss")
# y-rescale (conditioning only) for the explicit ISR params in "kfit" mode: px_γ,py_γ ~0.4,
# k longitudinal ~few GeV (steeply falling, p99~9). Centered at 0 (ISR spectrum peaks at 0).
ISR_SX=float(os.environ.get("KF_ISR_SX","0.4")); ISR_SY=float(os.environ.get("KF_ISR_SY","0.4"))
ISR_SZ=float(os.environ.get("KF_ISR_SZ","2.0"))
# CONDITIONING TEST (kfit only): KF_ISR_SMOOTH=1 replaces the spike-DCB ISR prior
# (C0 near-delta spike at 0 → stiff, 56% of chi2 variance) with a SMOOTH unit-Gaussian-
# in-y prior (y5²+y8²+y11², i.e. σ=ISR_S{X,Y,Z} on the photon momenta). KF_ISR_SMOOTH_W
# widens it (prior σ ×W) to trade ISR constraint for smoothness/conditioning.
ISR_SMOOTH=int(os.environ.get("KF_ISR_SMOOTH","0"))
ISR_SMOOTH_W=float(os.environ.get("KF_ISR_SMOOTH_W","1.0"))

HEADER=os.path.join(os.path.dirname(__file__),"..","kinfit_inputs","dcb_params.h")
P=J.load_params(HEADER)

# PR pack layout (per event, BINNED priors passed as data): 9 tuples concatenated.
# j1_p(10) j1_t(10) j1_q(10) j2_p(10) j2_t(10) j2_q(10) lep_p(13) lep_t(10) lep_q(10) = 93
_OFF={"j1p":(0,10),"j1t":(10,20),"j1q":(20,30),"j2p":(30,40),"j2t":(40,50),"j2q":(50,60),
      "lp":(60,73),"lt":(73,83),"lq":(83,93)}
PR_LEN=93
def _sl(PR,k): a,b=_OFF[k]; return PR[a:b]

def build_chi2(ecm, isr_mode=None):
    mode=isr_mode or KF_ISR_MODE
    s=lambda n: f"{n}_{ecm}"
    pn=P[s("AG3G_MET_P_RESP")]; tn=P[s("AG3G_MET_THETA_RESOL")]; qn=P[s("AG3G_MET_PHI_RESOL")]
    bm =P[s("GAUSS_GEN_EE_M_MINUS_ECM")]; bz=P[s("GAUSS_GEN_EE_PZ")]
    ix =P[s("SDCBG_GEN_ISR_PX")]; iy=P[s("SDCBG_GEN_ISR_PY")]; iz=P[s("SDCBG_GEN_ISR_PZ")]
    ml =P[s("SDCBG_GEN_WW_M_MINUS_M_EE")]; rsys=P[s("SDCBG_GEN_ISR_SYS_MASS")]
    ECM=float(ecm)
    def vec(p,th,ph):
        st=jnp.sin(th)
        return jnp.array([p, p*st*jnp.cos(ph), p*st*jnp.sin(ph), p*jnp.cos(th)])  # (E,px,py,pz) massless
    def M(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))
    def chi2(y, D, PR):
        j1p,j1t,j1f, j2p,j2t,j2f, lp,lt,lf, mp,mt,mf = [D[i] for i in range(12)]
        pj1=_sl(PR,"j1p"); tj1=_sl(PR,"j1t"); qj1=_sl(PR,"j1q")
        pj2=_sl(PR,"j2p"); tj2=_sl(PR,"j2t"); qj2=_sl(PR,"j2q")
        pl =_sl(PR,"lp");  tl =_sl(PR,"lt");  ql =_sl(PR,"lq")
        mW=KF_MW_INIT+KF_MW_PHYS_SIGMA*y[0]; gW=KF_GW_FIXED+KF_GW_PRIOR_SIGMA*y[1]
        s1=J.y2x(y[2],pj1); s2=J.y2x(y[3],pj2); sl=J.y2x(y[4],pl)
        t1=J.y2x(y[6],tj1); t2=J.y2x(y[7],tj2)
        p1=J.y2x(y[9],qj1); p2=J.y2x(y[10],qj2)
        tlep=J.y2x(y[12],tl); plep=J.y2x(y[13],ql)
        besm=J.y2x(y[14],bm); besz=J.y2x(y[15],bz)
        J1=vec(j1p/s1, j1t-t1, j1f-p1); J2=vec(j2p/s2, j2t-t2, j2f-p2)
        L =vec(lp/sl,  lt-tlep, lf-plep)
        bes=J.gauss_n2ll(besm,bm)+J.gauss_n2ll(besz,bz)
        scale_jl=J.dcb_gauss_n2ll(s1,pj1)+J.dcb_gauss_n2ll(s2,pj2)+J.dcb_er3g_n2ll(sl,pl)
        ang_jl=(J.dcb_gauss_n2ll(t1,tj1)+J.dcb_gauss_n2ll(t2,tj2)+J.dcb_gauss_n2ll(tlep,tl)
               +J.dcb_gauss_n2ll(p1,qj1)+J.dcb_gauss_n2ll(p2,qj2)+J.dcb_gauss_n2ll(plep,ql))
        if mode=="kfit":
            # Explicit ISR photon (px_γ,py_γ,k) in the old neutrino slots; neutrino DERIVED.
            pgx=ISR_SX*y[5]; pgy=ISR_SY*y[8]; pgz=ISR_SZ*y[11]
            Eg=jnp.sqrt(pgx*pgx+pgy*pgy+pgz*pgz+1e-12)
            Vis=J1+J2+L
            nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=besz-Vis[3]-pgz
            Enu=jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12)         # massless neutrino (exact)
            Nu=jnp.array([Enu,nux,nuy,nuz])
            if ISR_SMOOTH:
                isr=(y[5]*y[5]+y[8]*y[8]+y[11]*y[11])/(ISR_SMOOTH_W*ISR_SMOOTH_W)  # smooth Gaussian, no spike
            else:
                isr=J.spike_dcb_n2ll(pgx,ix)+J.spike_dcb_n2ll(pgy,iy)+J.spike_dcb_n2ll(pgz,iz)
            r=(ECM+besm)-Vis[0]-Enu-Eg                          # energy non-closure (≈ ISR-sys mass)
            mlt=J.spike_dcb_n2ll(r,rsys) if KF_CLOSURE=="sdcbg" else (r/SIGMA_R)**2+SIGMA_R_LOG_NORM
            scale=scale_jl; ang=ang_jl                          # NO neutrino prior terms
        else:
            sn=J.y2x(y[5],pn); tnu=J.y2x(y[8],tn); pnu=J.y2x(y[11],qn)
            Nu=vec(mp/sn, mt-tnu, mf-pnu)
            Wh0=J1+J2; Wl0=L+Nu; WW0=Wh0+Wl0
            isr=J.spike_dcb_n2ll(-WW0[1],ix)+J.spike_dcb_n2ll(-WW0[2],iy)+J.spike_dcb_n2ll(besz-WW0[3],iz)
            if mode=="photon":
                # WW recoils against a collinear massless ISR photon: enforce energy-momentum
                # consistency (r~0) instead of m_WW~ECM. p_isr = |beam - WW| 3-momentum.
                p_isr=jnp.sqrt(WW0[1]**2+WW0[2]**2+(besz-WW0[3])**2)
                r=(ECM+besm)-WW0[0]-p_isr
                mlt=(r/SIGMA_R)**2+SIGMA_R_LOG_NORM
            else:
                m_WW0=jnp.sqrt(jnp.maximum(WW0[0]**2-(WW0[1]**2+WW0[2]**2+WW0[3]**2),1e-12))
                m_loss=m_WW0-(ECM+besm); m_abs=jnp.sqrt(m_loss*m_loss+1e-8)
                mlt=J.spike_dcb_n2ll(m_abs,ml)+jnp.where(m_loss>0.0,(m_loss/BARRIER_SIGMA)**2,0.0)
            scale=scale_jl+J.asymg3g_n2ll(sn,pn)
            ang=ang_jl+J.asymg3g_n2ll(tnu,tn)+J.asymg3g_n2ll(pnu,qn)
        Wh=J1+J2; Wl=L+Nu; WW=Wh+Wl
        mh=M(Wh); mlep=M(Wl)
        mwgw=mW*gW; dh=mh*mh-mW*mW; dl=mlep*mlep-mW*mW
        bw_h=mwgw/(dh*dh+mwgw*mwgw); bw_l=mwgw/(dl*dl+mwgw*mwgw)
        s_ww=WW[0]**2-(WW[1]**2+WW[2]**2+WW[3]**2)
        lam=(s_ww-(mh+mlep)**2)*(s_ww-(mh-mlep)**2); lam=jnp.sqrt(lam*lam+1e-24)
        m_WW=jnp.sqrt(jnp.maximum(s_ww,1e-12))
        bw=-2.0*(jnp.log(bw_h)+jnp.log(bw_l))+4.0*jnp.log(jnp.pi)-jnp.log(lam)+2.0*jnp.log(s_ww)+2.0*J.log_Z_ontf(m_WW,mW,gW)
        gw=y[1]*y[1]+KF_GW_PRIOR_LOG_NORM
        return bw+bes+isr+mlt+scale+ang+gw          # NO mW prior (free)
    return chi2

def build_prior_packs(ecm, D):
    """Host-side: per-event BINNED prior packs (natural + jet1<->jet2 swapped).
    D = (N,12) reco array. Returns PR_nat, PR_swap each (N,93)."""
    g=lambda b: J.get_bins(P,b,ecm)
    J1P,J1Pe=g("DCBG_JET1_P_RESP");   J1T,J1Te=g("DCBG_JET1_THETA_RESOL"); J1Q,J1Qe=g("DCBG_JET1_PHI_RESOL")
    J2P,J2Pe=g("DCBG_JET2_P_RESP");   J2T,J2Te=g("DCBG_JET2_THETA_RESOL"); J2Q,J2Qe=g("DCBG_JET2_PHI_RESOL")
    LP,LPe  =g("DCBER3G_LEP_P_RESP"); LT,LTe =g("DCBG_LEP_THETA_RESOL");   LQ,LQe =g("DCBG_LEP_PHI_RESOL")
    j1p,j1t=D[:,0],D[:,1]; j2p,j2t=D[:,3],D[:,4]; lp=D[:,6]
    ac1=np.abs(np.cos(j1t)); ac2=np.abs(np.cos(j2t))
    pb=J.pick_bin_idx
    # lepton pack (same in both)
    lep=np.concatenate([LP[pb(lp,LPe)], LT[pb(lp,LTe)], LQ[pb(lp,LQe)]],axis=1)
    # natural: jet1<-JET1 family, jet2<-JET2 family
    nat=np.concatenate([J1P[pb(j1p,J1Pe)],J1T[pb(j1p,J1Te)],J1Q[pb(ac1,J1Qe)],
                        J2P[pb(j2p,J2Pe)],J2T[pb(j2p,J2Te)],J2Q[pb(ac2,J2Qe)], lep],axis=1)
    # swapped: jet1<-JET2 family (at jet1 reco), jet2<-JET1 family (at jet2 reco)
    swp=np.concatenate([J2P[pb(j1p,J2Pe)],J2T[pb(j1p,J2Te)],J2Q[pb(ac1,J2Qe)],
                        J1P[pb(j2p,J1Pe)],J1T[pb(j2p,J1Te)],J1Q[pb(ac2,J1Qe)], lep],axis=1)
    return nat.astype(np.float64), swp.astype(np.float64)

def build_prior_pack_pool(ecm, D):
    """POOLED jet prior: both jets use the SAME jet1+jet2-pooled family (DCBG_JET_*),
    binned by each jet's own p (|cosθ| for φ). Single family -> jet1/jet2 symmetric
    -> NO swap needed. Returns PR_pool (N,93). Compare vs build_prior_packs swap."""
    g=lambda b: J.get_bins(P,b,ecm)
    JP,JPe=g("DCBG_JET_P_RESP");   JT,JTe=g("DCBG_JET_THETA_RESOL"); JQ,JQe=g("DCBG_JET_PHI_RESOL")
    LP,LPe=g("DCBER3G_LEP_P_RESP");LT,LTe=g("DCBG_LEP_THETA_RESOL"); LQ,LQe=g("DCBG_LEP_PHI_RESOL")
    j1p,j1t=D[:,0],D[:,1]; j2p,j2t=D[:,3],D[:,4]; lp=D[:,6]
    ac1=np.abs(np.cos(j1t)); ac2=np.abs(np.cos(j2t)); pb=J.pick_bin_idx
    lep=np.concatenate([LP[pb(lp,LPe)], LT[pb(lp,LTe)], LQ[pb(lp,LQe)]],axis=1)
    pool=np.concatenate([JP[pb(j1p,JPe)],JT[pb(j1p,JTe)],JQ[pb(ac1,JQe)],
                         JP[pb(j2p,JPe)],JT[pb(j2p,JTe)],JQ[pb(ac2,JQe)], lep],axis=1)
    return pool.astype(np.float64)

def build_gof(ecm, isr_mode=None):
    mode=isr_mode or KF_ISR_MODE
    # Goodness-of-fit: per-event mode-referenced residual sum gof=Σ(term−term_at_mode),
    # which is >=0 and ~chi2-distributed (unlike raw -2logL, which carries large prior
    # log-norm offsets). Priors referenced to their mode (param at prior mu / y=0);
    # ISR/m_loss to 0; BW to both-W-on-pole (isolates the dijet-mass-compat residual).
    s=lambda n: f"{n}_{ecm}"
    pn=P[s("AG3G_MET_P_RESP")]; tn=P[s("AG3G_MET_THETA_RESOL")]; qn=P[s("AG3G_MET_PHI_RESOL")]
    bm =P[s("GAUSS_GEN_EE_M_MINUS_ECM")]; bz=P[s("GAUSS_GEN_EE_PZ")]
    ix =P[s("SDCBG_GEN_ISR_PX")]; iy=P[s("SDCBG_GEN_ISR_PY")]; iz=P[s("SDCBG_GEN_ISR_PZ")]
    ml =P[s("SDCBG_GEN_WW_M_MINUS_M_EE")]; rsys=P[s("SDCBG_GEN_ISR_SYS_MASS")]; ECM=float(ecm)
    def vec(p,th,ph):
        st=jnp.sin(th); return jnp.array([p,p*st*jnp.cos(ph),p*st*jnp.sin(ph),p*jnp.cos(th)])
    def M(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))
    def dg(x,p): return J.dcb_gauss_n2ll(x,p)-J.dcb_gauss_n2ll(p[0],p)   # ref at mu
    def a3(x,p): return J.asymg3g_n2ll(x,p)-J.asymg3g_n2ll(p[0],p)
    def er(x,p): return J.dcb_er3g_n2ll(x,p)-J.dcb_er3g_n2ll(p[0],p)
    def sp(x,p): return J.spike_dcb_n2ll(x,p)-J.spike_dcb_n2ll(0.0,p)    # ref at 0
    def gof(y,D,PR):
        j1p,j1t,j1f,j2p,j2t,j2f,lp,lt,lf,mp,mt,mf=[D[i] for i in range(12)]
        pj1=_sl(PR,"j1p");tj1=_sl(PR,"j1t");qj1=_sl(PR,"j1q")
        pj2=_sl(PR,"j2p");tj2=_sl(PR,"j2t");qj2=_sl(PR,"j2q")
        pl=_sl(PR,"lp");tl=_sl(PR,"lt");ql=_sl(PR,"lq")
        mW=KF_MW_INIT+KF_MW_PHYS_SIGMA*y[0]; gW=KF_GW_FIXED+KF_GW_PRIOR_SIGMA*y[1]
        s1=J.y2x(y[2],pj1);s2=J.y2x(y[3],pj2);sl=J.y2x(y[4],pl)
        t1=J.y2x(y[6],tj1);t2=J.y2x(y[7],tj2)
        p1=J.y2x(y[9],qj1);p2=J.y2x(y[10],qj2)
        tlep=J.y2x(y[12],tl);plep=J.y2x(y[13],ql); besm=J.y2x(y[14],bm);besz=J.y2x(y[15],bz)
        J1=vec(j1p/s1,j1t-t1,j1f-p1);J2=vec(j2p/s2,j2t-t2,j2f-p2);L=vec(lp/sl,lt-tlep,lf-plep)
        r_scale=dg(s1,pj1)+dg(s2,pj2)+er(sl,pl); r_ang=dg(t1,tj1)+dg(t2,tj2)+dg(tlep,tl)+dg(p1,qj1)+dg(p2,qj2)+dg(plep,ql)
        if mode=="kfit":
            pgx=ISR_SX*y[5]; pgy=ISR_SY*y[8]; pgz=ISR_SZ*y[11]; Eg=jnp.sqrt(pgx*pgx+pgy*pgy+pgz*pgz+1e-12)
            Vis=J1+J2+L; nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=besz-Vis[3]-pgz
            Enu=jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12); Nu=jnp.array([Enu,nux,nuy,nuz])
            r_isr=sp(pgx,ix)+sp(pgy,iy)+sp(pgz,iz); r_cl=(ECM+besm)-Vis[0]-Enu-Eg
            r_mloss=sp(r_cl,rsys) if KF_CLOSURE=="sdcbg" else r_cl**2/SIGMA_R**2
        else:
            sn=J.y2x(y[5],pn);tnu=J.y2x(y[8],tn);pnu=J.y2x(y[11],qn)
            Nu=vec(mp/sn,mt-tnu,mf-pnu); WW0=J1+J2+L+Nu
            r_isr=sp(-WW0[1],ix)+sp(-WW0[2],iy)+sp(besz-WW0[3],iz)+a3(sn,pn)+a3(tnu,tn)+a3(pnu,qn)
            if mode=="photon":
                p_isr=jnp.sqrt(WW0[1]**2+WW0[2]**2+(besz-WW0[3])**2); r_mloss=((ECM+besm)-WW0[0]-p_isr)**2/SIGMA_R**2
            else:
                m_loss=M(WW0)-(ECM+besm); m_abs=jnp.sqrt(m_loss*m_loss+1e-8)
                r_mloss=(J.spike_dcb_n2ll(m_abs,ml)+jnp.where(m_loss>0.0,(m_loss/BARRIER_SIGMA)**2,0.0))-J.spike_dcb_n2ll(0.0,ml)
        Wh=J1+J2;Wl=L+Nu; mh=M(Wh);mlep=M(Wl)
        mwgw=mW*gW; dh=mh*mh-mW*mW; dl=mlep*mlep-mW*mW
        bw_h=mwgw/(dh*dh+mwgw*mwgw); bw_l=mwgw/(dl*dl+mwgw*mwgw)
        r_bw=-2.0*(jnp.log(bw_h)+jnp.log(bw_l))-4.0*jnp.log(mwgw)            # >=0, dijet-mass compat
        r_bes=((besm-bm[0])/bm[1])**2+((besz-bz[0])/bz[1])**2; r_gw=y[1]*y[1]
        return r_bw+r_isr+r_mloss+r_bes+r_gw+r_scale+r_ang
    return gof

GOF_TERMS=["bw","isr","mloss","bes","gw","scale","ang"]
def build_gof_terms(ecm, isr_mode=None):
    mode=isr_mode or KF_ISR_MODE
    # Same mode-referenced residuals as build_gof, but returned PER TERM so we can
    # see which contributions drive the high-GoF (low p-value) tail. Order = GOF_TERMS.
    s=lambda n: f"{n}_{ecm}"
    pn=P[s("AG3G_MET_P_RESP")]; tn=P[s("AG3G_MET_THETA_RESOL")]; qn=P[s("AG3G_MET_PHI_RESOL")]
    bm =P[s("GAUSS_GEN_EE_M_MINUS_ECM")]; bz=P[s("GAUSS_GEN_EE_PZ")]
    ix =P[s("SDCBG_GEN_ISR_PX")]; iy=P[s("SDCBG_GEN_ISR_PY")]; iz=P[s("SDCBG_GEN_ISR_PZ")]
    ml =P[s("SDCBG_GEN_WW_M_MINUS_M_EE")]; rsys=P[s("SDCBG_GEN_ISR_SYS_MASS")]; ECM=float(ecm)
    def vec(p,th,ph):
        st=jnp.sin(th); return jnp.array([p,p*st*jnp.cos(ph),p*st*jnp.sin(ph),p*jnp.cos(th)])
    def M(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))
    def dg(x,p): return J.dcb_gauss_n2ll(x,p)-J.dcb_gauss_n2ll(p[0],p)
    def a3(x,p): return J.asymg3g_n2ll(x,p)-J.asymg3g_n2ll(p[0],p)
    def er(x,p): return J.dcb_er3g_n2ll(x,p)-J.dcb_er3g_n2ll(p[0],p)
    def sp(x,p): return J.spike_dcb_n2ll(x,p)-J.spike_dcb_n2ll(0.0,p)
    def gof_terms(y,D,PR):
        j1p,j1t,j1f,j2p,j2t,j2f,lp,lt,lf,mp,mt,mf=[D[i] for i in range(12)]
        pj1=_sl(PR,"j1p");tj1=_sl(PR,"j1t");qj1=_sl(PR,"j1q")
        pj2=_sl(PR,"j2p");tj2=_sl(PR,"j2t");qj2=_sl(PR,"j2q")
        pl=_sl(PR,"lp");tl=_sl(PR,"lt");ql=_sl(PR,"lq")
        mW=KF_MW_INIT+KF_MW_PHYS_SIGMA*y[0]; gW=KF_GW_FIXED+KF_GW_PRIOR_SIGMA*y[1]
        s1=J.y2x(y[2],pj1);s2=J.y2x(y[3],pj2);sl=J.y2x(y[4],pl)
        t1=J.y2x(y[6],tj1);t2=J.y2x(y[7],tj2)
        p1=J.y2x(y[9],qj1);p2=J.y2x(y[10],qj2)
        tlep=J.y2x(y[12],tl);plep=J.y2x(y[13],ql); besm=J.y2x(y[14],bm);besz=J.y2x(y[15],bz)
        J1=vec(j1p/s1,j1t-t1,j1f-p1);J2=vec(j2p/s2,j2t-t2,j2f-p2);L=vec(lp/sl,lt-tlep,lf-plep)
        r_scale=dg(s1,pj1)+dg(s2,pj2)+er(sl,pl); r_ang=dg(t1,tj1)+dg(t2,tj2)+dg(tlep,tl)+dg(p1,qj1)+dg(p2,qj2)+dg(plep,ql)
        if mode=="kfit":
            pgx=ISR_SX*y[5]; pgy=ISR_SY*y[8]; pgz=ISR_SZ*y[11]; Eg=jnp.sqrt(pgx*pgx+pgy*pgy+pgz*pgz+1e-12)
            Vis=J1+J2+L; nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=besz-Vis[3]-pgz
            Enu=jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12); Nu=jnp.array([Enu,nux,nuy,nuz])
            r_isr=sp(pgx,ix)+sp(pgy,iy)+sp(pgz,iz); r_cl=(ECM+besm)-Vis[0]-Enu-Eg
            r_mloss=sp(r_cl,rsys) if KF_CLOSURE=="sdcbg" else r_cl**2/SIGMA_R**2
        else:
            sn=J.y2x(y[5],pn);tnu=J.y2x(y[8],tn);pnu=J.y2x(y[11],qn)
            Nu=vec(mp/sn,mt-tnu,mf-pnu); WW0=J1+J2+L+Nu
            r_isr=sp(-WW0[1],ix)+sp(-WW0[2],iy)+sp(besz-WW0[3],iz)
            r_scale=r_scale+a3(sn,pn); r_ang=r_ang+a3(tnu,tn)+a3(pnu,qn)
            if mode=="photon":
                p_isr=jnp.sqrt(WW0[1]**2+WW0[2]**2+(besz-WW0[3])**2); r_mloss=((ECM+besm)-WW0[0]-p_isr)**2/SIGMA_R**2
            else:
                m_loss=M(WW0)-(ECM+besm); m_abs=jnp.sqrt(m_loss*m_loss+1e-8)
                r_mloss=(J.spike_dcb_n2ll(m_abs,ml)+jnp.where(m_loss>0.0,(m_loss/BARRIER_SIGMA)**2,0.0))-J.spike_dcb_n2ll(0.0,ml)
        Wh=J1+J2;Wl=L+Nu; mh=M(Wh);mlep=M(Wl)
        mwgw=mW*gW; dh=mh*mh-mW*mW; dl=mlep*mlep-mW*mW
        bw_h=mwgw/(dh*dh+mwgw*mwgw); bw_l=mwgw/(dl*dl+mwgw*mwgw)
        r_bw=-2.0*(jnp.log(bw_h)+jnp.log(bw_l))-4.0*jnp.log(mwgw)
        r_bes=((besm-bm[0])/bm[1])**2+((besz-bz[0])/bz[1])**2; r_gw=y[1]*y[1]
        return jnp.array([r_bw,r_isr,r_mloss,r_bes,r_gw,r_scale,r_ang])
    return gof_terms

def build_postfit(ecm, isr_mode=None):
    # Post-fit W/WW masses from the fitted y (same kinematics as build_chi2).
    mode=isr_mode or KF_ISR_MODE
    s=lambda n: f"{n}_{ecm}"
    pn=P[s("AG3G_MET_P_RESP")]; tn=P[s("AG3G_MET_THETA_RESOL")]; qn=P[s("AG3G_MET_PHI_RESOL")]
    bm=P[s("GAUSS_GEN_EE_M_MINUS_ECM")]; bz=P[s("GAUSS_GEN_EE_PZ")]
    def vec(p,th,ph):
        st=jnp.sin(th); return jnp.array([p,p*st*jnp.cos(ph),p*st*jnp.sin(ph),p*jnp.cos(th)])
    def M(v): return jnp.sqrt(jnp.maximum(v[0]**2-(v[1]**2+v[2]**2+v[3]**2),1e-12))
    def pf(y,D,PR):
        j1p,j1t,j1f,j2p,j2t,j2f,lp,lt,lf,mp,mt,mf=[D[i] for i in range(12)]
        J1=vec(j1p/J.y2x(y[2],_sl(PR,"j1p")), j1t-J.y2x(y[6],_sl(PR,"j1t")), j1f-J.y2x(y[9],_sl(PR,"j1q")))
        J2=vec(j2p/J.y2x(y[3],_sl(PR,"j2p")), j2t-J.y2x(y[7],_sl(PR,"j2t")), j2f-J.y2x(y[10],_sl(PR,"j2q")))
        L =vec(lp/J.y2x(y[4],_sl(PR,"lp")),  lt-J.y2x(y[12],_sl(PR,"lt")), lf-J.y2x(y[13],_sl(PR,"lq")))
        if mode=="kfit":
            besz=J.y2x(y[15],bz); pgx=ISR_SX*y[5]; pgy=ISR_SY*y[8]; pgz=ISR_SZ*y[11]
            Vis=J1+J2+L; nux=-Vis[1]-pgx; nuy=-Vis[2]-pgy; nuz=besz-Vis[3]-pgz
            Nu=jnp.array([jnp.sqrt(nux*nux+nuy*nuy+nuz*nuz+1e-12),nux,nuy,nuz])
        else:
            Nu=vec(mp/J.y2x(y[5],pn),  mt-J.y2x(y[8],tn),  mf-J.y2x(y[11],qn))
        Wh=J1+J2; Wl=L+Nu; WW=Wh+Wl
        return jnp.array([M(Wh), M(Wl), M(WW)])
    return pf

# NOTE: the standalone main()/BFGS driver was removed (2026-07-01 review): it wired the
# 3-arg chi2(y,D,PR) into make_bfgs_solver (2-arg) and built no prior pack, so it crashed.
# The working lnuqq driver is jax_prototype/recovery_lnuqq_mp.py (calls chi2(y,Di,PRi)).
