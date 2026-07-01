#!/usr/bin/env python3
# Diagnose lnuqq JAX: is it a SOLVER problem (my LM) or a chi2 PORT bug (wrong
# surface)? Run an independent optimizer (scipy BFGS + L-BFGS-B) on the SAME JAX
# chi2+grad for a few events. If BFGS reaches |grad|~0 but my LM doesn't -> LM bug.
# If BFGS also lands at mW~83 / fails -> chi2 surface (port) bug.
import sys, os
os.environ.setdefault("OMP_NUM_THREADS","4")
import numpy as np, jax, jax.numpy as jnp, uproot
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(__file__))
import kinfit_lnuqq_jax as K   # reuse build_chi2 + constants

ecm=160
chi2=K.build_chi2(ecm)
g=jax.jit(jax.grad(chi2,argnums=0)); f=jax.jit(chi2)

t=uproot.open(sys.argv[1] if len(sys.argv)>1 else
  "outputs/treemaker/lnuqq/step2/semihad/wzp6_ee_munumuqq_noCut_ecm160.root")["events"]
cols=["reco_jet1_p","reco_jet1_theta","reco_jet1_phi","reco_jet2_p","reco_jet2_theta","reco_jet2_phi",
      "reco_lep_p","reco_lep_theta","reco_lep_phi","reco_met_p","reco_met_theta","reco_met_phi"]
a=t.arrays(cols+["kinfit_mW","kinfit_valid"],library="np")
D=np.stack([a[c] for c in cols],axis=1).astype(np.float64)
ok=np.all(np.isfinite(D),axis=1)&(D[:,0]>0)&(D[:,3]>0)&(D[:,6]>0)&(D[:,9]>0)
idx=np.where(ok)[0][:25]
# My LM with trust-cap effectively OFF (1e3) and more iters, single-event.
import jaxfit_common as J
myfit=jax.jit(J.make_solver_mn(chi2, 16, niter=120, nls=40))
print(f"{'ev':>4} {'mW_bfgs':>8} {'mWlm':>8} {'mW_cpp':>8} {'lm_|g|':>9} {'lm_EDM':>10} {'cppV':>4}")
for i in idx:
    Di=jnp.array(D[i])
    fun=lambda y: float(f(jnp.array(y),Di))
    jac=lambda y: np.asarray(g(jnp.array(y),Di),dtype=np.float64)
    r=minimize(fun, np.zeros(16), jac=jac, method="BFGS", options={"maxiter":2000,"gtol":1e-6})
    mWb=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*r.x[0]
    y,c,gn,edm=myfit(Di); y=np.asarray(y)
    mWl=K.KF_MW_INIT+K.KF_MW_PHYS_SIGMA*float(y[0])
    print(f"{i:>4} {mWb:>8.3f} {mWl:>8.3f} {a['kinfit_mW'][i]:>8.3f} {float(gn):>9.2e} {float(edm):>10.2e} {int(a['kinfit_valid'][i]):>4}")
