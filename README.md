# ttThreshold-analysis — WW → m_W reconstruction (`wwTh_scan`)

FCC-ee W-mass reconstruction from WW events, in two decay channels and with two
complementary m_W extraction methods:

- **Channels** — `ℓνqq` (semileptonic) and `4q` (fully hadronic).
- **Methods** — a per-event **kinematic fit** (ROOT Minuit2, gen-level BES + ISR
  priors, BW joint-pdf normalization) and a **convolution / forward-fold**
  estimator (folded Breit-Wigner with m_W as the only free parameter).

Samples are the FCC-ee `winter2023/IDEA` fast-sim productions: the per-ECM
`wzp6_ee_munumuqq` (muon-only ℓνqq, ECM 157/160/163) and the inclusive
`p8_ee_WW_ecm{160,240,365}`. Mass convention: **gen quarks are massive, reco
jets are massless**.

## Setup

Install the FCCAnalyses framework:

```
git clone --branch pre-edm4hep1 https://github.com/HEP-FCC/FCCAnalyses.git
cd FCCAnalyses
source ./setup.sh
fccanalyses build -j 6
cd ..
```

Clone this repository (the `wwTh_scan` branch — where this pipeline lives) next
to `FCCAnalyses/`:

```
git clone -b wwTh_scan git@github.com:ttThreshold-FCC/ttThreshold-analysis.git
cd ttThreshold-analysis
```

Set up the environment **every time** before running (`setup.sh` just sources
`../FCCAnalyses/setup.sh` — edit the path if your layout differs):

```
source setup.sh
```

## Pipeline

Each channel runs a **two-step treemaker**: step 1 produces the response/
resolution branches; `fit_resolutions.py` fits the detector-resolution priors
into a C++ header; step 2 re-runs and applies the per-event kinematic fit that
consumes those priors.

```
step1 treemaker ──▶ fit_resolutions.py ──▶ [build log-Z table] ──▶ step2 treemaker + kinfit ──▶ diagnostics + plots
   (response)          (dcb_params.h)         (logz_table.bin)          (kinfit_mW per event)
```

### ℓνqq (semileptonic) — end to end

```
./run_pipeline.sh
```

which is equivalent to:

```
fccanalysis run treemaker_lnuqq_step1.py --ncores 12     # response branches
python3 fit_resolutions.py                               # → kinfit_inputs/dcb_params.h
tools/ensure_logz_table.sh                               # → kinfit_inputs/logz_table.bin (BW normalization)
for ecm in 157 160 163; do                               # 3 ECMs in parallel
    WW_ECM=$ecm fccanalysis run treemaker_lnuqq_step2.py --ncores 4 &
done; wait
python3 kinfit_diagnostics.py                            # convergence / closure report
python3 plot_kinfit_results.py                           # mW overlays + nuisance posteriors/pulls
```

Individual stages are also wrapped by `run_step1.sh`, `run_fit_resolutions.sh`,
`run_step2.sh` (launches the 3 ECMs in parallel), and `run_plot_kinfit.sh`.

### 4q (fully hadronic)

Same shape, with the fully-hadronic treemakers and the best-of-3 jet→W pairing
kinematic fit (`WWFunctions/WWKinReco4q.h`):

```
fccanalysis run treemaker_4q_step1.py                        # cluster 4 jets, match to gen quarks
WW_CHANNEL=4q python3 fit_resolutions.py                     # → kinfit_inputs_4q/dcb_params_4q.h
fccanalysis run treemaker_4q_step2.py                        # kinfit over all 3 pairings, keep lowest χ²
```

Standalone (no-fit) Breit-Wigner jet→W pairing for pairing studies and as a ZZ→4q
χ² discriminant: `treemaker_4q_bwpairing.py` (`WWFunctions/BWPairing.h`).

### Forward-fold m_W estimator (convolution method)

A second, independent extraction that folds the true W lineshape through an
empirical resolution kernel and fits m_W as the single free parameter. It runs
directly on the inclusive `p8_ee_WW` sample, so it also works at ECM 240/365
where the per-ECM kinfit priors do not exist. Inputs come from the `_ff`
treemakers (kinfit stripped out):

```
# ℓνqq forward-fold inputs (μ+e), then fit
WW_SAMPLE=p8_ee_WW_ecm240 WW_TAG=ff_lnuqq240 fccanalysis run treemaker_lnuqq_step2_ff.py
python3 jax_prototype/conv_mw_fit.py 240        # MODE=gen_pe|gen_marg|reco

# 4q forward-fold inputs, then fit (with jet→W pairing discriminant)
WW_SAMPLE=p8_ee_WW_ecm240 WW_TAG=ff_4q240 fccanalysis run treemaker_4q_step2_ff.py
PAIRING=all python3 jax_prototype/conv_mw_4q.py 240
```

`jax_prototype/` also holds the closure/lineshape/pairing diagnostic and plot
scripts used to develop and validate the method.

## Common environment knobs

Most treemakers honour a small set of env vars so tests run in parallel without
clobbering each other:

| var | effect |
|-----|--------|
| `WW_ECM` | ℓνqq step2: restrict to one ECM (157/160/163) for parallel launches |
| `WW_SAMPLE` | override the input dataset (e.g. `p8_ee_WW_ecm365`, `p8_ee_ZZ_ecm160`) |
| `WW_TAG` | suffix the output dir so side-by-side runs don't collide |
| `WW_FRACTION` | fraction of the (large inclusive) sample to process |
| `WW_LEPTON_PDG` | ℓνqq `_ff`: `both` (e+μ) / `13` (μ) / `11` (e) |
| `WW_CHANNEL=4q` | switch `fit_resolutions.py` to the fully-hadronic priors |
| `PAIRING`, `MODE`, `MAXN` | `conv_mw_*` forward-fold: jet→W pairing, validation stage, event cap |

## Repository layout

| path | contents |
|------|----------|
| `treemaker_lnuqq_*.py`, `treemaker_4q_*.py` | per-channel step1 / step2 / `_ff` treemakers |
| `treemaker_common.py` | shared FCCAnalyses `Define` helpers (beam/ISR kinematics, kinfit driver) |
| `WWFunctions/` | C++ headers: `WWFunctions.h` (selectors), `WWKinReco.h` (ℓνqq fit), `WWKinReco4q.h` (4q fit), `BWPairing.h` |
| `fit_resolutions.py` | detector-resolution prior fits → `kinfit_inputs*/dcb_params*.h` |
| `tools/build_logz_table.cxx`, `ensure_logz_table.sh` | BW joint-pdf log-Z normalization table |
| `jax_prototype/conv_mw*.py` | convolution / forward-fold m_W estimator + diagnostics |
| `kinfit_diagnostics.py`, `plot_kinfit_*.py`, `plot_bw_pairing.py` | convergence/closure reports and plots |
| `run_*.sh`, `setup.sh` | pipeline launchers |
| `report/` | LaTeX write-up of the kinfit pipeline |
| `combine/`, `impacts/`, `condor/`, `jobs/` | ttbar-threshold datacard/impacts/batch tooling (separate workflow) |

For the FCCAnalyses framework itself, see the
[FCCAnalyses tutorial](https://hep-fcc.github.io/fcc-tutorials/main/fast-sim-and-analysis/fccanalyses/doc/starterkit/FccFastSimAnalysis/Readme.html).
