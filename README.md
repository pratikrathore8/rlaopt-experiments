# rlaopt experiments

This repository contains reproducible benchmark suites comparing rlaopt with other optimization and numerical linear algebra methods. It is intended to grow beyond least squares and ridge regression: each problem family should define its own mathematical model, competitors, accuracy contract, configuration, runner, and analysis while sharing the repository's environment, result-tracking, and orchestration conventions.

The first implemented suite is controlled synthetic ridge regression. The synthetic empirical-risk-minimization (ERM) development suite now implements the problem generators, solver adapters, execution grid, solver-independent accuracy checks, and CPU/CUDA calibration needed to prepare the later real-data benchmarks.

## Benchmark suite organization

Repository-level dependencies and reproducibility policy live in `pyproject.toml`, `uv.lock`, and `containers/`. The current ridge suite uses `configs/synthetic.toml` and the `problem.py`, `solvers.py`, `diagnostics.py`, `runner.py`, and `plotting.py` modules. As additional benchmark families are introduced, their problem-specific configuration and implementation should be placed in named subpackages rather than added as conditionals to the ridge model. Shared concerns—stable run identifiers, atomic records, W&B recovery, hardware metadata, and Slurm submission—should remain reusable across suites.

Every suite must document:

- its mathematical problem and data-generating process;
- the normalization and parameter scales used across problem sizes;
- solver-specific mappings and a method-independent success criterion;
- what setup, transfers, and synchronization are included in runtime;
- resource limits, failure handling, and the exact hardware scope; and
- suite-specific limitations and confirmatory versus exploratory analyses.

## Real-data corpus and preparation

The real-data ERM study uses the following frozen training datasets. When a source
publishes separate training and test files, only its training file is used. Sources
without a separate test file contribute all their rows because this is an optimization
benchmark rather than a predictive-accuracy study. Fashion-MNIST is the exception:
OpenML stores its original 60,000 training and 10,000 test examples together, so only
the first 60,000 official training examples are selected.

| application | dataset | source | base training shape | solver shape | feature map |
|---|---|---|---:|---:|---|
| elastic net | ACSIncome | OpenML 43141 | $1{,}664{,}500\times11$ | $1{,}664{,}500\times1{,}000$ | Gaussian, nominal bandwidth 1 |
| elastic net | E2006-tfidf | LIBSVM training file | $16{,}087\times150{,}360$ | unchanged | none |
| elastic net | Real-sim | LIBSVM | $72{,}309\times20{,}958$ | unchanged | none |
| elastic net | YearPredictionMSD | LIBSVM training file | $463{,}715\times90$ | $463{,}715\times4{,}367$ | ReLU |
| elastic net | Yolanda | OpenML 42705 | $400{,}000\times100$ | $400{,}000\times1{,}000$ | Gaussian, nominal bandwidth 1 |
| multinomial | CIFAR-10 | LIBSVM training file | $50{,}000\times3{,}072$ | unchanged | none |
| multinomial | RCV1 multiclass | LIBSVM training file | $15{,}564\times47{,}236$ | unchanged | none |
| multinomial | SVHN | LIBSVM training file | $73{,}257\times3{,}072$ | unchanged | none |
| multinomial | News20 | LIBSVM classic training file | $15{,}935\times62{,}061$ | unchanged | none |
| multinomial | Fashion-MNIST | OpenML 40996 official training rows | $60{,}000\times784$ | unchanged | none |

The News20 entry is the classic 15,935-row multiclass training file, not LIBSVM's
newer 18,846-observation collection. RCV1 likewise uses `rcv1_train.multiclass`, not
the binary RCV1 formulation. Its training file contains 51 observed labels even though
the combined training/test collection is described as having 53 topics; the training-only
optimization problem therefore uses 51 contiguous classes. Sparse LIBSVM matrices remain
CSR and every nonzero row is scaled to unit Euclidean norm; zero rows remain zero. Dense
OpenML features are standardized columnwise to population mean zero and standard deviation
one using only the selected training rows. Continuous OpenML regression targets are
standardized in the same way. LIBSVM regression targets retain their source values.
Multinomial labels are deterministically mapped from sorted source labels to contiguous
integers starting at zero. Missing or nonfinite values and unexpected shapes are hard errors.

The nonlinear maps reproduce the PROMISE implementation with NumPy seed 2468. For
$G\in\mathbb{R}^{m\times p}$ with independent standard Gaussian entries and
$\beta_j$ independent and uniform on $[0,2\pi]$, Gaussian features use

$$
W=\frac{G}{b\sqrt{m}},
\qquad
Z=\sqrt{\frac{2}{m}}\cos(XW^{\mathsf T}+\beta),
$$

with $b=1$ and $m=1000$. This is explicitly called the PROMISE convention because
its frequency scaling differs from textbook fixed-bandwidth random Fourier features.
ReLU features use

$$
W=\frac{G}{\sqrt{m}},
\qquad
Z=\max\{XW^{\mathsf T},0\},
$$

with $m=4367$. `materialize_random_features` regenerates the complete float64 feature
matrix deterministically in RAM or VRAM and applies its nonlinearity in place; expanded
matrices are never stored. Benchmark workers will call this function before solver timing,
and feature generation and transfer will be recorded separately.

Production workers verify the processed matrix and target digests before loading them. Sparse
base matrices remain compact on disk but are converted to the same explicit dense float64
matrix supplied to every solver; random-feature outputs are likewise fully materialized. For
elastic net, the response is used as prepared and the intercept remains unregularized. Letting
$\bar y$ denote its training mean, the data-dependent regularization scale is

$$
\lambda_{\max}=\frac{1}{n}\left\lVert X^{\mathsf T}
\left(y-\bar y\mathbf{1}\right)\right\rVert_\infty.
$$

Both penalties use $\lambda_1=\lambda_2=\gamma\lambda_{\max}$ for each configured fraction
$\gamma$. The vanilla and box-constrained variants therefore share exactly the same matrix,
response, intercept convention, and penalty values. Cache validation, feature materialization,
device transfer, and computation of $\lambda_{\max}$ occur during worker startup and are
excluded from solver time.

The frozen, time-constrained production grid is `configs/real_erm.toml`. It uses one fixed
run seed, 300, and one measured solve per configuration. Runtime values are therefore single
observations rather than estimates of timing variability, which is a limitation of this
campaign. The bounded elastic-net experiment uses $\gamma=0.1$. Each multinomial JAXopt method is run twice: once with its default JIT-compiled optimization loop and once with `jit=False`; both executions
include the complete `run` call in solver time and use the same native tolerance.
Vanilla elastic net is omitted from this time-constrained campaign; its implementation and
calibration remain available for follow-up work. Each backend manifest contains 40 jobs: 25
multinomial and 15 bounded elastic-net jobs. CPU and CUDA together contain 80 jobs. Each solve
has a one-hour hard timeout.

Prepare the compact source and base-matrix cache with

```bash
scripts/prepare_real_data.sh --data-root /scr/pratikr/rlaopt-real-data --all
```

Use repeated `--dataset NAME` arguments to select datasets, `--redownload` to replace
the raw downloads atomically and rebuild their processed artifacts, or `--reprocess`
to rebuild from the current verified raw files. Each processed directory contains the
float64 matrix, target, and JSON provenance with source and artifact SHA-256 digests.
The data root is explicit because `/scr` is node-local on this cluster; production
jobs must point to a prepared cache visible on their execution node.

Stage and verify the cache once on each production node after rebuilding the SIF:

```bash
for node in soal-8 soal-9 soal-12; do
  sbatch --nodelist="$node" --export=ALL,DATA_ROOT=/scr/pratikr/rlaopt-real-data \
    slurm/stage_real_data.sh
done
```

The staging job is idempotent: it verifies reusable artifacts by metadata, SHA-256 digest,
shape, and dtype; an invalid processed artifact is rebuilt from the verified raw download.
`REDOWNLOAD=1` forces a new source download and `REPROCESS=1` forces preprocessing. The
production launcher explicitly binds the configured node-local data root into the container
and refuses to start a `real_erm` worker if the processed cache directory is absent.

Generate the two production manifests with

```bash
uv run rlaopt-bench manifest --config configs/real_erm.toml \
  --backend cpu --output artifacts/real-erm/manifests/cpu.jsonl
uv run rlaopt-bench manifest --config configs/real_erm.toml \
  --backend cuda --output artifacts/real-erm/manifests/cuda.jsonl
```

`run-manifest-job` and `slurm/run_array.sh` dispatch `real_erm` jobs through the same
isolated worker lifecycle used by the synthetic ERM suite. Before loading data, the executor
checks the suite, problem type, dataset, data root, bounds or regularization fraction, problem
ID, backend, solver, master seed, and derived solver seed against the TOML configuration.
Every persisted run ID includes the run seed, preventing the three timing observations for
a fixed real-data problem from overwriting one another. Startup failures and per-solve
timeouts are retained as atomic result records.

Create a fresh, immutable campaign plan with ten jobs per batch:

```bash
uv run --frozen python scripts/plan_real_production.py \
  --config configs/real_erm.toml \
  --output artifacts/real-erm-production-YYYYMMDD
```

The planner freezes and checksums a copy of the TOML file and every JSONL batch, splits CPU
jobs between `soal-8` and `soal-9` without confounding a solver with one node, and writes at
most seven jobs per 12-hour batch.
Worker startup is capped at 30 minutes and each solve at 60 minutes, so seven
worst-case runs take at most 10.5 hours and leave 1.5 hours of batch margin. The 80-run grid fits in one QOS-safe submission wave containing 12 array elements.
Each node's array is throttled to one task, so only one of our benchmark processes runs on
a node at a time.

After the previous wave has finished, submit the next one with

```bash
scripts/submit_real_production_wave.sh \
  artifacts/real-erm-production-YYYYMMDD WAVE_INDEX
```

The submission helper refuses a dirty repository, another active real-data production wave,
more than six tasks per node, more than 18 tasks in a wave, or any submission that would
exceed the user-wide 20-job QOS limit. Individual launcher failures are reported while the
remaining jobs in that batch continue, and all successful or solver-level failure outcomes
remain atomic records under the campaign directory.

## SCS bounded elastic-net backend supplement

The SCS adapters share the same conic formulation, timing boundaries, and external
accuracy checks in both ERM suites. Solver IDs explicitly select these backends:

| Solver ID | Device | Linear solver |
|---|---|---|
| `scs` | CPU | Sparse direct QDLDL (original baseline) |
| `scs_cpu_indirect` | CPU | Indirect CG |
| `scs_cuda` | CUDA | GPU indirect CG (original baseline) |
| `scs_cuda_direct` | CUDA | Sparse direct cuDSS |

`configs/real_erm_scs_backends.toml` adds only CPU indirect and GPU direct for the
same five bounded elastic-net problems used in the original real-data campaign.
It retains seed 300, regularization fraction 0.1, one cold measured solve, 100,000
iterations, and a one-hour solve timeout. Both added backends reuse
`eps_abs = eps_rel = 1e-7`; their native tolerances are explicitly **not recalibrated**.
Records preserve the transfer provenance and the existing external KKT/feasibility checks.
Internal SCS setup and factorization remain inside solver timing; benchmark-side conic
construction is recorded separately. Each solver ID has separate result records.

Rebuild the existing container with `sbatch slurm/build_cuda.sh`. The recipe builds
all four SCS 3.2.11 backends in float64 with 32-bit indices, reusing cuDSS 0.7.1 from
the pinned Julia artifact. A missing CUDA backend raises an error, with no fallback.
After the build, `slurm/check_cpu.sh` and `slurm/check_cuda.sh` run small correctness
checks of both SCS backends on their respective device. These checks use the frozen
production tolerance; they do not perform a calibration sweep.

Prepare one solve per task with:

```bash
uv run --frozen python scripts/plan_real_production.py \
  --config configs/real_erm_scs_backends.toml \
  --output artifacts/real-erm-scs-backends-20260906-64threads \
  --max-jobs-per-batch 1 --gpu-concurrency 2 --gpu-cpu-threads 64
```

The ten tasks fit in one wave. CPU tasks are split 3/2 between `soal-8` and `soal-9`,
with one 64-core task at a time on each node. GPU tasks run on `soal-12`, with up to
two concurrent tasks, each reserving one H200, 64 host cores, and 128 GB host memory.
This matches the original campaign's host-thread allocation. Report shared-node
execution when comparing timings. After the build and checks,
submit wave 0 with `scripts/submit_real_production_wave.sh
artifacts/real-erm-scs-backends-20260906-64threads 0` from a clean checkout.

### YearPredictionMSD-rf tolerance sensitivity follow-up

The 64-thread SCS GPU-direct run at `eps_abs = eps_rel = 1e-7` reported native
convergence after 350 iterations and 365.56 seconds, but missed the common
stationarity and feasibility targets: `0.0117797 > 1e-4` and `5.00225e-6 > 1e-6`,
respectively. This result remains in
`artifacts/real-erm-scs-backends-20260906-64threads`; it is not replaced.

A separate, post-production sensitivity study tests the two native tolerances
`1e-8` and `1e-9`, both specified before running the follow-up. The self-contained
configurations are `configs/real_erm_scs_yearpredictionmsd_1e-8.toml` and
`configs/real_erm_scs_yearpredictionmsd_1e-9.toml`. Only YearPredictionMSD-rf and
`scs_cuda_direct` are selected. This study was motivated by the observed accuracy
miss and must not be described as held-out calibration. Native-tolerance calibration
remains false and the motivation is recorded in `native_tolerance_source`.

Each attempt is a fresh, cold solve of the same seeded problem with the same
regularization, SCS 3.2.11 container, 100,000-iteration ceiling, one-hour solve limit,
one H200, 64 host CPU threads, and 128 GiB host-memory allocation. At most two GPU
attempts run concurrently. Dataset preparation and conic construction remain outside
the solve timer; native setup and factorization remain inside it. The external
stationarity (`1e-4`) and feasibility (`1e-6`) targets are unchanged.

The two outputs are kept separately under
`artifacts/scs-yearpredictionmsd-tolerance-20260906/1e-8` and `1e-9`, each with its
own frozen config, manifest, logs, and result records. Always identify these records
by campaign and native tolerance when aggregating: the solver/problem/seed identity
is intentionally the same. Report every attempt's runtime, native status, and
external metrics, including misses or resource failures; report tuning cost
separately from any selected solve time. These results supplement the original
frozen-protocol comparison.

Both follow-ups finished with a 3,600-second solve timeout and returned no solution
for external evaluation. Neither tolerance supplies an accuracy-qualified runtime.
Together with the original `1e-7` solve, the measured solver cost is 7,565.56 seconds.
The refined plot labels the final outcome Timeout at the one-hour limit; its audit
retains the original accuracy miss and both timed-out attempts.

## Synthetic ERM development suite

This suite develops the three problem classes intended for the later real-data study on deterministic synthetic data first. All generated arrays and all solver computations use float64. The calibration, smoke, and scaling-pilot grids generate features on CPU from seeded Gaussian streams, then center and scale each column to unit population root-mean-square; the separate conditioning diagnostic uses the SORF construction documented below. Problem generation and host-to-device transfer are measured separately from solver time and are not included in the primary solve-time comparison. Every competitor receives the same materialized data and mathematical formulation.

### Problem definitions and scaling

Here, a *teacher* is the hidden ground-truth parameter used only to generate a synthetic response: it generates class probabilities in multinomial regression and the noiseless signal in elastic-net regression. No solver receives the teacher. Because labels and noise are sampled and the fitted problems are regularized, the teacher is generally not the minimizer of the realized empirical objective and is not used as an accuracy oracle.

The box-constrained multinomial logistic-regression problem has no intercept and solves

$$
\min_{B\in[\ell,u]^{p\times K}}
\frac{1}{n}\sum_{i=1}^{n}
\left[
\log\left(\sum_{k=1}^{K}\exp(x_i^{\mathsf T}B_{:k})\right)
-x_i^{\mathsf T}B_{:y_i}
\right].
$$

Labels are sampled from a softmax teacher. The seeded Gaussian teacher is centered across classes to remove the softmax shift ambiguity, scaled by $1/\sqrt{p}$, and, if necessary, rescaled to remain within 80% of the explicitly configured coefficient box. This matches the no-intercept bounded formulation used in Chapter 4 of the thesis.

The vanilla and bounded elastic-net variants share exactly the same generated $X$, response $y$, teacher, and unregularized fitted intercept $b$. They solve

$$
\min_{w,b}
\frac{1}{2n}\lVert Xw+b\mathbf{1}-y\rVert_2^2
+\lambda_1\lVert w\rVert_1
+\frac{\lambda_2}{2}\lVert w\rVert_2^2,
$$

with either $w\in\mathbb{R}^p$ for vanilla elastic net or $w\in[0,1]^p$ for bounded elastic net. A sparse nonnegative teacher generates the signal. Independent centered Gaussian noise is scaled to an explicitly configured population-RMS noise-to-signal ratio. The completed response is divided by its centered population RMS but is not centered: its variance is one, while its nonzero mean preserves the configured teacher intercept and makes fitted-intercept stationarity nontrivial. Regularization follows the thesis convention

$$
\lambda_{\max}
=\frac{1}{n}\left\lVert X_c^{\mathsf T}y_c\right\rVert_\infty,
\qquad
\lambda_1=\lambda_2=\gamma\lambda_{\max},
$$

where $X_c$ and $y_c$ are centered to account for the fitted intercept and the fraction $\gamma>0$ is explicit in the experiment configuration.

### Solver-independent accuracy contract

Native solver status and native stopping tolerances are not treated as cross-method accuracy evidence. After timing, the testbed recomputes every primary metric from the returned primal solution in float64 using the canonical problem above. The problem objects implement the formulation-dependent objective, gradients, KKT residuals, constraint violations, and dual certificate; the suite diagnostics layer records those values and applies frozen pass thresholds. Solver-specific tolerances are calibrated against these common metrics on a separate calibration grid and frozen before production. A production record distinguishes `native_success`, meaning that the solver reached its calibrated native stopping rule; `external_success`, meaning that the returned solution passed the common accuracy and feasibility thresholds; and `runtime_eligible`, which normally equals `native_success` and determines inclusion in runtime summaries. A native success that misses the external target remains in the runtime comparison but is visible in the separately reported external pass rate and residual distribution. Constraint satisfaction is not tested against exact zero: a constrained solution passes when its measured maximum violation is at most the frozen feasibility tolerance. Objective value, native residuals, and native dual variables are retained as secondary diagnostics. Iterate histories are retained when a supported interface exposes primal iterates; black-box interfaces retain their final native diagnostics instead.

For box-constrained multinomial logistic regression, let

$$
G=\frac{1}{n}X^{\mathsf T}(P-Y),
\qquad
P=\mathrm{softmax}(XB).
$$

The primary metric is a primal KKT stationarity residual: $G_{jk}=0$ in the interior, $G_{jk}\ge 0$ at the lower bound, and $G_{jk}\le 0$ at the upper bound. Coordinates within the frozen feasibility tolerance of a bound are evaluated using that bound's one-sided condition; all others are evaluated as interior coordinates. The maximum violation of these coordinate conditions is reported together with, and must pass alongside, the separate maximum box violation

$$
\max_{j,k}\left\{(\ell-B_{jk})_+,(B_{jk}-u)_+\right\}.
$$

No dual variables or solver-independent step size are required.

For vanilla elastic net, the primary metric is a relative primal-dual gap. Given a returned $(w,b)$ with residual $r=Xw+b\mathbf{1}-y$, the testbed constructs its own dual-feasible point

$$
\nu=\frac{r-\overline r\mathbf{1}}{n},
\qquad \mathbf{1}^{\mathsf T}\nu=0.
$$

For the positive $\lambda_2$ used here, define

$$
P(w,b)=\frac{1}{2n}\lVert r\rVert_2^2
+\lambda_1\lVert w\rVert_1
+\frac{\lambda_2}{2}\lVert w\rVert_2^2
$$

and

$$
D(\nu)=-y^{\mathsf T}\nu-\frac{n}{2}\lVert\nu\rVert_2^2
-\frac{1}{2\lambda_2}
\left\lVert S_{\lambda_1}(-X^{\mathsf T}\nu)\right\rVert_2^2,
$$

where $S$ is elementwise soft thresholding. The reported relative gap is

$$
\frac{P-D}{\max\{1,|P|,|D|\}}.
$$

Thus competitors do not need to expose dual variables; the benchmark constructs the certificate uniformly. A primal KKT residual is retained as a secondary diagnostic.

For bounded elastic net, on the feasible region the smooth-plus-linear coordinate gradient is

$$
g=\frac{1}{n}X^{\mathsf T}(Xw+b\mathbf{1}-y)
+\lambda_2w+\lambda_1\mathbf{1},
$$

and intercept stationarity requires $\mathbf{1}^{\mathsf T}(Xw+b\mathbf{1}-y)/n=0$. The primary metric combines the corresponding one-sided box KKT conditions—$g_j=0$ in the interior, $g_j\ge0$ at zero, and $g_j\le0$ at one—with intercept stationarity. As above, coordinates within the frozen feasibility tolerance of a bound use its one-sided condition. Stationarity must pass alongside the separately reported maximum constraint violation

$$
\max_j\left\{(-w_j)_+,(w_j-1)_+\right\}.
$$

No dual variables are required. Native dual information from rlaopt ADMM, SCS, or Clarabel is saved only as a secondary diagnostic because their internal conic and splitting formulations need not use comparable dual coordinates.

| problem | primary accuracy metric | separate feasibility check | dual information required from solver |
|---|---|---|---|
| box-constrained multinomial logistic regression | external primal KKT stationarity | coefficient-box violation at frozen tolerance | no |
| vanilla elastic net | external relative primal-dual gap | none | no; testbed constructs $\nu$ |
| bounded elastic net | external primal KKT stationarity, including intercept | $[0,1]$ violation at frozen tolerance | no |

Real-data ERM production uses $10^{-4}$ for the applicable optimization metric—primal KKT stationarity for the two box-constrained formulations and relative primal-dual gap for vanilla elastic net—and retains $10^{-6}$ for explicit constraint feasibility. The looser optimization target reflects practical statistical-estimation accuracy, while the tighter feasibility threshold prevents a solver from gaining apparent speed by returning a materially infeasible point. These are external qualification thresholds, not solver-native tolerances. Native tolerances are calibrated separately below, and every returned solution is re-evaluated against the applicable external metric in float64.

### Timing and competitor-interface policy

SAPPHIRE will be compared with projected gradient and JAXopt L-BFGS-B for bounded multinomial logistic regression. SAPPHIRE will be compared with scikit-learn, cuML, and JAXopt proximal gradient for vanilla elastic net. The bounded elastic-net comparison will use rlaopt ADMM, SCS, and Clarabel. CPU and GPU results will be reported separately, and unsupported backend/problem combinations will be labeled rather than emulated.

rlaopt 0.1.0's least-squares atom evaluates mean squared error without the factor of $1/2$ used by the canonical elastic-net objective. Its adapter therefore passes $2\lambda_1$ and $2\lambda_2$ to rlaopt, making the complete native objective exactly twice the canonical objective and leaving its minimizer unchanged. The record identifies this convention with `native_objective_scale = 2.0`. All externally reported objectives, gaps, KKT residuals, and pass decisions are recomputed from the returned primal solution using the canonical formulation; they are never compared on rlaopt's doubled scale.

For bounded elastic net, the direct conic adapters use $z=(r,w,b)$ with $r=Xw+b\mathbf{1}-y$. The quadratic matrix is diagonal with blocks $I/n$, $\lambda_2 I$, and zero; the linear term on $w$ is $\lambda_1\mathbf{1}$ because $w\geq0$ makes $\lVert w\rVert_1=\mathbf{1}^{\mathsf T}w$. Equality rows enforce the residual definition, and nonnegative-cone rows encode $0\leq w\leq1$. An executable equivalence check verifies that this conic objective equals the canonical objective and that its constraints have the intended signs. rlaopt ADMM accepts a half-scaled least-squares expression, so its adapter represents the canonical objective directly and otherwise retains the default ADMM configuration. Only the SAPPHIRE decomposition requires the doubled-objective workaround described above. SCS 3.2.11 is called through its direct Python interface. The CPU baseline selects its sparse-direct QDLDL backend. The CUDA baseline selects SCS's sparse-indirect GPU backend explicitly with `gpu = true` and `use_indirect = true`; absence of the compiled `_scs_gpu` module is a hard error and never falls back to CPU.

Every competitor, including SCS and Clarabel, is called through its direct solver interface. Benchmark-side construction of the common problem arrays and any format conversion needed to pass them to an interface occurs before the timed region and is recorded separately. Primary solver time begins immediately before the native setup/solve call and therefore includes all symbolic analysis, scaling, factorization, preconditioner construction, tracing, JIT compilation, and other numerical setup performed by that solver. Repeated timings reuse the same immutable problem arrays but create a fresh solver instance unless a separately labeled warm-start experiment explicitly permits state reuse. Small-instance equivalence checks compare every direct interface against the canonical objective and KKT diagnostics implemented by this repository.

SCS's Python CUDA interface accepts host SciPy arrays and performs its own device transfer during solver construction. Consequently, the SCS CUDA timed region includes that unavoidable native transfer and all GPU linear-system setup. Copying the common PyTorch problem to the host and constructing the conic arrays occur before this region and are recorded as preparation; copying the returned primal solution to the benchmark device is recorded as extraction. The image rebuilds the locked SCS 3.2.11 source distribution in float64 with 32-bit indices, the GPU-indirect backend, and SCS's standard transposed GPU matrix layout. The source archive is verified against the SHA-256 already recorded in `uv.lock`. Because the 3.2.11 release archive's Meson GPU target references source variables before defining them and omits explicit cuBLAS/cuSPARSE linkage, the image applies the Meson-only portions of the official post-release fixes `335311fe87c33751817bd1f7d30f5ca5a0ee0618` and `3564b3b051ebbf516ff4d21bb6482db65abc4665`; the vendored patch records both commits verbatim for auditability.

Clarabel is invoked without CVXPY, JuMP, or another modeling layer. Its CPU baseline uses the `:qdldl` direct linear solver and converts SciPy CSC arrays directly to Julia `SparseMatrixCSC` arrays without materializing dense matrices. Its GPU baseline uses the CuClarabel branch through JuliaCall with the full-float64 `:cudss` linear solver; the mixed-precision `:cudssmixed` mode is excluded from the float64 study. Julia is initialized before PyTorch in Clarabel worker processes to avoid their observed shared-library loading conflict. Each measured solve constructs a fresh Clarabel solver, and its timed region includes both Clarabel's native setup and solve phases; one-shot configurations use no untimed solver warmup, so any first-call Julia compilation is also included. Host construction of $P,q,A,b$, sparse-format conversion, host-to-device transfer, and solution extraction remain separately recorded preparation costs.

The CUDA image uses a small dynamically linked Python 3.12 launcher because CUDA.jl GPU compilation is not reliable when JuliaCall is embedded in Ubuntu's statically linked system-Python executable; the image test verifies the resulting interpreter links `libpython3.12`. It pins Julia 1.10.12 by the official archive SHA-256 and resolves the CuClarabel branch at commit `ffa325c89fa90b7e86b745fa61b1dca64daf3a06`. `julia/cuclarabel/Manifest.toml` locks the complete Julia dependency graph, including CUDA.jl 5.11.3, CUDSS.jl 0.6.5, CUDSS_jll 0.7.1+0, and PythonCall.jl 0.9.31; the Python environment pins the matching JuliaCall 0.9.31 and CuPy CUDA 13 package. The Julia project pins CUDA.jl to its CUDA 13.0 artifact runtime before precompilation, which is required when the image build has no visible NVIDIA driver and allows CUDSS_jll to select its matching CUDA 13 artifact; the base toolkit alone does not contain the separately distributed cuDSS library. Julia package images are precompiled with `JULIA_CPU_TARGET=generic` so that the immutable image remains portable across the cluster's heterogeneous CPU families; solver execution still uses the native BLAS, sparse-solver, and GPU libraries packaged for each backend. The generated Apptainer SIF and its checksum remain the executable-environment artifact for production runs.

CuClarabel's current CuPy bridge wraps device pointers without taking ownership and converts zero-based CSR indices to Julia's one-based indices in place. This can leave Julia and cuDSS referring to allocations returned to CuPy's memory pool, producing the [stochastic use-after-free reported upstream](https://github.com/oxfordcontrol/Clarabel.jl/issues/233). The adapter therefore creates dedicated CuPy inputs and immediately copies every aliased vector and CSR component into Julia-owned device memory; the index shift mutates only the Julia-owned copies. These device-to-device copies are separately recorded as preparation rather than native solver time. This is an interface-safety requirement, not a benchmark optimization. The pre-production probe solves the same deterministic bounded elastic-net QP with CPU QDLDL and GPU cuDSS, recomputes objective, KKT stationarity, and box violation from both returned primal solutions, and checks CPU/GPU agreement. It then performs ten sequential GPU solves of the pilot's $(2^{14},2^{11})$ configuration using one persistent Julia runtime and CuPy memory pool to expose cross-solve lifetime failures before production.

The real-dataset identities, sizes, and preprocessing are frozen above; production resource limits will be frozen after the prepared matrices and solver workspaces are measured. Solver versions are already pinned by the Python and Julia lockfiles and the immutable CUDA image. The study will include instances demonstrably too large for the considered interior-point baselines; such claims will be supported by explicit memory estimates or observed structured resource failures rather than assumed from dimensions alone.

The initial development grid is intentionally small and is defined in `configs/synthetic_erm_smoke.toml`. It uses $(n,p)\in\{(1024,64),(4096,256)\}$, five classes for multinomial regression, three data seeds, one measured repetition per seed, and regularization fractions $\gamma\in\{0.1,0.01\}$. The vanilla and bounded elastic-net variants share one configuration block, which prevents their shapes or data-generating parameters from drifting apart. This original smoke grid retains its strict $10^{-6}$ thresholds for stationarity, feasibility, and relative duality gap so that the completed synthetic campaign remains reproducible; those historical thresholds do not define real-data production qualification. Native-tolerance calibration state is tracked separately by backend so freezing CPU values cannot accidentally qualify provisional CUDA values.

The held-out scaling pilot is defined in `configs/synthetic_erm_pilot.toml`. Guided by the real-data dimensions in [Figure 4.5 of the thesis](https://web.stanford.edu/~udell/doc/rathore26_thesis.pdf), it scales samples through $(2^{14},2^{11})$, $(2^{16},2^{11})$, and $(2^{18},2^{11})$, adds the feature-scaling point $(2^{16},2^{13})$, and adds $(2^{13},2^{14})$ for the two elastic-net variants as an underdetermined high-feature stress case. The thesis datasets can be sparse, whereas this testbed materializes dense float64 matrices, so the pilot matches their computational regimes without copying their largest dimensions literally. Its largest design matrix occupies 4 GiB before solver working memory. One held-out seed, one cold measured repetition, no solver warmup, and only the less-regularized $\gamma=0.01$ endpoint keep the pilot diagnostic rather than production-sized. It contains 12 multinomial, 15 vanilla-elastic-net, and 15 bounded-elastic-net jobs per backend: 42 per backend and 84 total.

The separate conditioning diagnostic in `configs/synthetic_erm_conditioning.toml` changes spectral shape without changing total feature scale. For $r=\min(n,p)$ and covariance-decay exponent $\alpha$, it uses independent SORF factors from [Yu et al. (2016)](https://arxiv.org/abs/1610.09072) and singular values

$$
s_k = c_\alpha k^{-\alpha/2},
\qquad
c_\alpha = \sqrt{\frac{np}{\sum_{j=1}^{r}j^{-\alpha}}},
\qquad k=1,\ldots,r.
$$

Consequently, the nonzero eigenvalues of $X^{\mathsf T}X$ decay as $k^{-\alpha}$ and

$$
\lVert X\rVert_F^2 = \sum_{k=1}^{r}s_k^2 = np.
$$

The latter exactly matches the total squared magnitude of the column-standardized Gaussian generator, whose columns each have population RMS one. Holding this quantity fixed prevents a change in convergence from being attributed merely to rescaling the loss or regularization. The diagnostic crosses $\alpha\in\{0,1/2,1,2\}$ with one held-out $(2^{12},2^{10})$ problem per family, one seed, one regularization fraction, and every applicable solver. It therefore contains 36 jobs per backend. It is an exploratory sensitivity check; the production study will use real datasets rather than these prescribed spectra.

Generate the pilot manifests through the pinned image from a clean repository root:

```bash
source containers/cuda.env
for backend in cpu cuda; do
  apptainer exec --bind "$PWD:$PWD" --pwd "$PWD" "$RLAOPT_CUDA_IMAGE" \
    /opt/rlaopt-experiments/.venv/bin/rlaopt-bench manifest \
    --config configs/synthetic_erm_pilot.toml \
    --backend "$backend" \
    --output "artifacts/synthetic-erm-pilot/manifests/$backend.jsonl"
done
```

Submit the 84 benchmark configurations as six Slurm tasks, below the soal limits of 20 submitted and 12 running jobs. The two CPU chunks are pinned to different nodes and the four CUDA chunks occupy at most the four H200 NVLs:

```bash
BACKEND=cpu MANIFEST=artifacts/synthetic-erm-pilot/manifests/cpu.jsonl \
CONFIG=configs/synthetic_erm_pilot.toml OUTPUT_DIR=artifacts/synthetic-erm-pilot \
CHUNK_COUNT=2 \
  sbatch --array=0 --nodelist=soal-8 --job-name=erm-pilot-cpu-8 \
  --output=erm-pilot-cpu-8-%A_%a.out slurm/run_chunk_array.sh

BACKEND=cpu MANIFEST=artifacts/synthetic-erm-pilot/manifests/cpu.jsonl \
CONFIG=configs/synthetic_erm_pilot.toml OUTPUT_DIR=artifacts/synthetic-erm-pilot \
CHUNK_COUNT=2 \
  sbatch --array=1 --nodelist=soal-9 --job-name=erm-pilot-cpu-9 \
  --output=erm-pilot-cpu-9-%A_%a.out slurm/run_chunk_array.sh

BACKEND=cuda MANIFEST=artifacts/synthetic-erm-pilot/manifests/cuda.jsonl \
CONFIG=configs/synthetic_erm_pilot.toml OUTPUT_DIR=artifacts/synthetic-erm-pilot \
CHUNK_COUNT=4 BENCHMARK_CPU_THREADS=16 \
  sbatch --array=0-3%4 --nodelist=soal-12 --gres=gpu:h200nvl:1 \
  --cpus-per-task=16 --job-name=erm-pilot-cuda \
  --output=erm-pilot-cuda-%A_%a.out slurm/run_chunk_array.sh
```

Each chunk processes its round-robin subset sequentially and continues after an individual launcher failure, while each benchmark retains the configured 15-minute startup and solve limits. Use a fresh output directory for every campaign.

The `manifest` command expands every synthetic-ERM grid into one self-contained JSON record per problem, solver, and backend. Feature, target, and solver randomness use independently derived deterministic streams from the recorded master seed; the complete problem specification and resulting problem ID are stored in every job. Across regularization fractions and the two elastic-net variants, jobs reuse the same feature and target seeds, so only the regularization and boundedness change. Per backend, the smoke manifest contains 18 multinomial jobs, 36 vanilla elastic-net jobs, and 36 bounded elastic-net jobs, for 90 jobs total. The corresponding `run-manifest-job` command reads one zero-based manifest index without loading the full file and obtains all execution controls from the same strict suite configuration.

Independent CPU and CUDA recalibration against the real-data production targets selected the same native tolerances for corresponding methods: $10^{-5}$ for multinomial SAPPHIRE; $10^{-4}$ for projected gradient and JAXopt L-BFGS-B; $10^{-4}$ for vanilla-elastic-net SAPPHIRE, JAXopt proximal gradient, and coordinate descent in scikit-learn or cuML; $10^{-6}$ for bounded-elastic-net ADMM; $10^{-7}$ for SCS; and $10^{-9}$ for Clarabel with QDLDL or cuDSS. Every selected candidate achieved both native and external success on every configured calibration case. The CPU and CUDA selections agree exactly. Complete recalibration records are stored beneath `artifacts/synthetic-erm-calibration-1e-4`; the earlier strict-$10^{-6}$ campaign remains separate. The configuration uses a batch size of 256 for the rlaopt methods. Calibration and smoke runs retain their 10,000-iteration ceiling, while the scaling pilot and future production runs permit up to 100,000 iterations subject to the 15-minute hard timeout. The larger ceiling is only a safety bound: every solver still stops immediately upon satisfying its frozen native criterion. Each problem family and elastic-net variant has its own strict execution-control block whose native-tolerance keys must exactly match its CPU and CUDA solver lists.

Native-tolerance calibration uses `configs/synthetic_erm_calibration.toml`, not the smoke, pilot, or production grids. That file is the single source of truth for the predetermined candidate grid $10^{-4},10^{-5},\ldots,10^{-10}$, two calibration-only master seeds, three easy-to-hard shapes, common external thresholds, iteration and time limits, data-generation parameters, and solver lists. Its production-facing targets are $10^{-4}$ for stationarity and relative duality gap and $10^{-6}$ for feasibility. The calibration file deliberately contains no per-solver native-tolerance maps: calibration supplies each candidate directly, while frozen native-tolerance maps belong only in executable smoke, pilot, and production configurations. CLI candidate overrides are rejected for this suite. Elastic-net calibration crosses those cases with both configured regularization fractions. For each solver, backend, and problem family independently, the `calibrate` command selects the loosest candidate for which every calibration record has both native success and external success. Multinomial logistic regression and bounded elastic net require stationarity at most $10^{-4}$ and feasibility at most $10^{-6}$; vanilla elastic net requires relative duality gap at most $10^{-4}$. Every candidate/case pair runs in a fresh worker process, preventing solver state, warm starts, or JIT state from leaking between candidates. Deterministic seeds reproduce the same mathematical case in each process, and problem generation remains outside solver timing.

For example, calibrate the CPU projected-gradient adapter with

```bash
uv run rlaopt-bench calibrate \
  --config configs/synthetic_erm_calibration.toml \
  --backend cpu \
  --problem-type multinomial \
  --solver projected_gradient \
  --output artifacts/synthetic-erm-calibration-1e-4
```

The command retains an atomic trial record for every candidate and case, then writes `calibration.json` containing the complete pass matrix and selected tolerance. A missing passing candidate is a calibration failure rather than permission to relax the common external target. CPU and CUDA implementations are calibrated separately; both recalibration campaigns completed independently and selected the same values. Those values will be frozen in the real-data configuration before any held-out production run, and calibration seeds remain disjoint from all reported instances.

Submit all nine CPU problem/solver calibrations from the repository root with

```bash
OUTPUT_DIR=artifacts/synthetic-erm-calibration-1e-4 \
  sbatch slurm/calibrate_synthetic_erm_cpu.sh
```

The array is throttled to two concurrent tasks on `soal-8` and `soal-9`. Every task requests 64 physical CPU threads and 128 GB, runs one problem/solver combination, and writes to its own subtree beneath `artifacts/synthetic-erm-calibration-1e-4`. Array-task logs use `cpu-erm-calibrate-<array-job-id>_<task-id>.out` and are ignored by Git. Because the calibration determines accuracy qualification rather than benchmark runtime, concurrent tasks may occupy disjoint cores on the same node; their timings are not used in performance figures.

After the CUDA environment and both direct solver probes pass, submit the corresponding nine GPU calibrations with

```bash
OUTPUT_DIR=artifacts/synthetic-erm-calibration-1e-4 \
  sbatch slurm/calibrate_synthetic_erm_cuda.sh
```

The array is throttled to the four H200 NVLs on `soal-12`. Every task requests one GPU, 16 CPU threads, and 64 GB, and writes to the same backend-separated calibration tree as the CPU array. Its 24-hour allocation is a ceiling for the complete candidate grid; the 15-minute per-solve timeout in the calibration configuration remains in force. Calibration runtimes are not used in performance figures, so concurrently occupying distinct GPUs does not contaminate reported benchmark timings.

All three synthetic-ERM executors validate the recorded suite, problem type, problem ID, configured seed, derived solver seed, solver, backend, shape, and data-generating parameters before starting an isolated worker. Problem generation and worker startup occur outside solver timing. A bounded Clarabel worker initializes Julia before importing PyTorch and retains that runtime while constructing fresh native solver state for every warmup and measured solve; other workers do not initialize Julia. Configured warmups use fresh solver calls with at most ten iterations and are not recorded when successful; measured repetitions also use fresh solver calls. A timeout or exception is persisted as a result rather than aborting the manifest, and the local atomic JSON record remains authoritative if optional W&B logging fails.

The multinomial adapters use each pinned package's default algorithmic configuration: rlaopt SAPPHIRE defaults to SAGA with its rank-10 Nyström preconditioner, JAXopt projected gradient defaults to acceleration with backtracking, and JAXopt L-BFGS-B defaults to the zoom line search. The adapters override only the experiment's iteration and native-tolerance limits; the batch size and random seed required by SAPPHIRE; and settings required to represent and time the common problem. JAX remains JIT-enabled in its normal execution mode, but the primary one-shot runtime includes first-call tracing and compilation because every job fits an independently generated dataset. Any future warm, amortized JAX timing will be labeled and reported separately. Feature and response arrays are explicit dynamic arguments rather than captured compilation constants, preventing dataset values from being embedded in the executable. JAX and PyTorch share device-resident arrays through DLPack, so format conversion is outside the timed region without introducing a hidden host transfer. rlaopt 0.1.0's Nyström spectral estimator creates its random power-iteration vector in PyTorch's default dtype rather than the requested problem dtype. The isolated SAPPHIRE adapter therefore temporarily sets the default dtype to float64 for the entire solve and restores it afterward; it does not replace or disable the Nyström preconditioner. SAPPHIRE computes gradients functionally, so its optimization variables do not need leaf-level autograd tracking. The adapters construct the multinomial coefficients, elastic-net weights, and fitted intercept with `requires_grad=False`; this prevents the SAGA table and averaged-gradient state from retaining a graph across iterations without changing SAPPHIRE's mathematical updates or default configuration.

The remaining sections describe the implemented synthetic ridge suite.

## Synthetic ridge suite: mathematical model

For every shape $(n,p)$, let $r=\min(n,p)$. Large Haar factors are prohibitively expensive to generate by Gaussian QR, so the suite uses independent Structured Orthogonal Random Feature (SORF) factors. Let $H_d$ denote the normalized $d\times d$ Walsh--Hadamard matrix and let each $D_i$ be an independently seeded diagonal matrix of Rademacher signs. Following [Yu et al. (2016)](https://arxiv.org/abs/1610.09072), define

$$
Q_d = H_d D_1 H_d D_2 H_d D_3.
$$

Every factor in this product is orthogonal, so $Q_d$ is orthogonal. Independent sign streams give $Q_n^{(L)}$ and $Q_p^{(R)}$. For decay exponent $\alpha$, define

$$
s_k = k^{-\alpha/2}, \qquad k=1,\ldots,r,
$$

and

$$
X = Q_n^{(L)}[:,1:r] \,\mathrm{diag}(s)\, Q_p^{(R)}[:,1:r]^{\mathsf T}.
$$

The implementation starts from the rectangular diagonal matrix containing $s$ and applies the SORF transforms directly along its row and column axes with the standard fast Walsh--Hadamard butterfly. It never materializes either dense singular-vector factor. The final $X$ is nevertheless an ordinary dense float64 tensor; the compact right-factor signs are retained only for the analytic oracle and are never used by a solver adapter. Thus the nonzero eigenvalues of $X^{\mathsf T}X$ are exactly $k^{-\alpha}$ and $\lVert X\rVert_2=1$. The three profiles are $\alpha \in \{1/2,1,2\}$. This normalization prevents sample count from changing the regularization scale.

The response is generated from an independent Gaussian vector $z$, normalized as $\widehat z=z/\lVert z\rVert_2$, with

$$
y = Q_n^{(L)}[:,1:r]\widehat z, \qquad \lVert y\rVert_2=1.
$$

Every method solves, in float64,

$$
\min_w \frac{1}{2}\lVert Xw-y\rVert_2^2 + \frac{\lambda}{2}\lVert w\rVert_2^2,
$$

for $\lambda \in \{10^{-2},10^{-4},10^{-6}\}$. The exact reference solution is derived from the known SVD:

$$
w_\star
= Q_p^{(R)}[:,1:r] \,\mathrm{diag}\left(\frac{s_k}{s_k^2+\lambda}\right)\widehat z.
$$

We report the effective dimension

$$
d_{\mathrm{eff}}(\lambda)
= \sum_{k=1}^{r}\frac{s_k^2}{s_k^2+\lambda},
$$

the standard condition number

$$
\kappa(X^{\mathsf T}X+\lambda I)
= \frac{1+\lambda}{p^{-\alpha}+\lambda},
$$

and the ratios $r/p$ and $r/d_{\mathrm{eff}}$. Every configured shape has $n\ge p$, so $X$ has full column rank.


## Synthetic ridge suite: fixed experiment grid

There are eight unique shapes:

| family | $(n,p)$ |
|---|---|
| square scaling | $(2^{10},2^{10})$, $(2^{12},2^{12})$, $(2^{14},2^{14})$, $(2^{16},2^{16})$ |
| sample scaling ($p=2^{14}$) | $(2^{14},2^{14})$, $(2^{15},2^{14})$, $(2^{16},2^{14})$ |
| feature scaling ($n=2^{16}$) | $(2^{16},2^{10})$, $(2^{16},2^{12})$, $(2^{16},2^{14})$, $(2^{16},2^{16})$ |


Each shape uses three decay profiles, three master seeds, and three ridge values. Factor, response, and Nyström randomness use separately derived deterministic streams. A job is $(\text{hardware},\text{solver},\text{shape},\alpha,\text{seed})$ and reuses the same data matrix $X$ for all three ridge values: 288 jobs per backend, 576 total, and 1,728 ridge trials before timing repetitions.

We use the following hardware and solver combinations:

| backend | hardware | solvers |
|---|---|---|
| CPU | 64 physical cores on soal-8/soal-9 | rlaopt Nyström-PCG, rlaopt identity-PCG (CG), SciPy LSQR, PyTorch augmented QR |
| GPU | NVIDIA H200 NVL on soal-12 | both rlaopt variants, cuML Ridge/LSMR, PyTorch augmented QR |

rlaopt always receives $X^{\mathsf T}X$ as a linear operator over the fully materialized dense $X$ (i.e., it never forms the Gram matrix) and $B=X^{\mathsf T}y$, with `reg=lambda`. Nyström PCG fixes $\mathtt{rank\_init}=\mathtt{rank\_max}=\min(128,p)$, `base_damping=lambda`, and adaptive damping. Ridge is not folded into the operator, so it is never counted twice. It does not receive the SORF factors or a fast-transform operator; its products use the same dense $X$ supplied to every competitor. SciPy uses a `LinearOperator` wrapper around the same dense $X$ and $\mathtt{damp}=\sqrt{\lambda}$. PyTorch uses the augmented system

$$
\begin{bmatrix}X\\ \sqrt{\lambda}I\end{bmatrix}w
= \begin{bmatrix}y\\0\end{bmatrix},
$$

since `torch.linalg.lstsq` has no ridge argument. The adapter is named `torch_lstsq_qr` in code and selects the unpivoted QR-based `gels` driver on both CPU and CUDA; the augmented matrix has full column rank for every positive ridge value; the stable manifest identifier remains `torch_qr`. cuML uses `Ridge(alpha=lambda, fit_intercept=False, solver="lsmr")`.

## Synthetic ridge suite: accuracy, stopping, and timing

The cross-method success criterion is the externally recomputed relative KKT residual

$$
\frac{\left\lVert (X^{\mathsf T}X+\lambda I)w-X^{\mathsf T}y\right\rVert_2}
{\left\lVert X^{\mathsf T}y\right\rVert_2}
\le 10^{-6}.
$$

Native status does not establish external mathematical accuracy. Runtime summaries use `runtime_eligible` under the frozen calibrated native rule, while the external KKT pass rate and relative error to $w_\star$ are reported separately.

Native tolerances live in `configs/tolerances.toml`. Calibrate one tolerance per solver/backend on representative easy, middle, and hard cases, choose the loosest value that passes every external KKT check, then freeze the file before the production sweep. rlaopt stores a residual point per iteration in both the atomic JSON record and W&B; SciPy records its final LSQR diagnostics; cuML records `n_iter_` when exposed. Direct QR has no iteration history. Runtime figures include every run that completes under its frozen native stopping rule; the external KKT pass rate and marginal misses are reported separately, so native-completed cuML or SciPy runs are not dropped solely for narrowly missing the common diagnostic threshold.

Run calibration with, for example, `uv run rlaopt-bench calibrate --backend cpu --solver scipy_lsqr --candidates 1e-4 1e-5 1e-6 1e-7 1e-8`. The command writes all underlying records plus `calibration.json`; copy the selected value into `configs/tolerances.toml` only after inspecting every case.

Use `configs/calibration.toml` for the fixed easy, middle, and hard calibration regimes. Each problem is generated once and retained while all candidate tolerances run with independently initialized solvers. The current tolerances remain frozen after the switch to SORF because they are relative stopping criteria; they are not selected again on production endpoints. The cases in `configs/pilot.toml` instead provide held-out transfer checks. Any KKT failure is reported as evidence that a tolerance did not transfer and is not silently tuned away. CUDA calibration runs through `slurm/calibrate_cuda.sh` inside the pinned image, while 64-core CPU calibration runs through `slurm/calibrate_cpu.sh` on soal-8 or soal-9.

For pilot, production, and maximum-size runs, iterative solvers stop at native convergence, $2p$, or 15 minutes, whichever comes first; QR has only the 15-minute solve timeout. Calibration uses a 5-minute solve limit and smoke tests use 1 minute. Problem generation has a separate 15-minute startup limit and remains outside solver timing. A persistent spawned worker retains the generated problem but places every timed native call behind a parent-enforced process boundary; if a solver exceeds the configured solve limit, the parent terminates that worker and regenerates the same deterministic problem before continuing with the next ridge value. rlaopt additionally checks elapsed time cooperatively on every iteration. A timeout or exception is never runtime-eligible. A native success that misses the external KKT target is retained in runtime summaries and recorded as an external miss rather than silently dropped.

SORF generation, dense materialization, and analytic oracle work are excluded from runtime. Each backend generates the same seed-defined problem natively on its own device; generation is numerically cross-checked but is not part of solver timing. Timed regions include preconditioner construction, factorization, and all internal solver setup. GPU timings synchronize immediately before and after the solve and therefore describe GPU-resident inputs. Where a configuration requests one warm-up, it is untimed; iterative warm-ups are capped at 10 iterations, while direct QR performs a full warm-up call. The pilot gives every successful configuration three timed repetitions to measure same-instance variability. Production uses one timed run for each of three problem seeds and summarizes runtime across seeds by its median/min/max. A first-run timeout or accuracy failure is recorded once during the primary sweep and flagged for manual audit rather than automatically consuming two more production attempts. Records include peak process RSS on CPU and peak PyTorch allocator use on CUDA; scheduler and `nvidia-smi` accounting remain necessary because the CUDA allocator value does not include every cuML allocation.

## Reproducible environment

Install the exact uv release first, then sync the lockfile:

```bash
curl -LsSf https://astral.sh/uv/0.12.7/install.sh | sh
uv sync --frozen
```

The project pins Python 3.12, uv 0.12.7, rlaopt 0.1.0 from PyPI, NumPy 2.4.2, SciPy 1.18.1, PyTorch 2.13.0, matplotlib 3.11.1, and W&B 0.29.0. NumPy 2.4.2 is the newest release compatible with cuML 26.8.0's pinned Numba CUDA implementation; NumPy 2.5 removes `numpy.row_stack`, which that implementation still imports. `uv.lock` pins the transitive CPU environment and a separate `gpu` dependency group pins the standalone `cuml-cu13==26.8.0` wheel and its dependencies. GPU jobs use the official NVIDIA CUDA 13.0.2 devel image configured in `containers/cuda.env`; the devel flavor supplies NVRTC for Numba. Its `linux/amd64` base manifest is fixed by digest, and `containers/cuda.def.in` derives a project image containing uv, the frozen GPU environment, and the benchmark package. The H200 supports CUDA 13, and the soal cluster's driver is newer than CUDA 13's minimum requirement.

W&B defaults to offline mode. Atomic JSON files under `artifacts/records/` are the source of truth and remain recoverable if W&B fails; sync them later with `wandb sync` if desired.

## Synthetic ridge suite: workflow

Create one manifest per backend:

```bash
uv run rlaopt-bench manifest --backend cpu --output artifacts/cpu.jsonl
uv run rlaopt-bench manifest --backend cuda --output artifacts/cuda.jsonl
```

Use `configs/smoke.toml` for one tiny case per solver, `configs/pilot.toml` for the reduced pre-production sweep, and `configs/max_size.toml` for the largest-shape memory probe. Pass the desired file through `--config` when creating a manifest and running its jobs.

Run a small calibration/smoke case before freezing tolerances:

```bash
WANDB_MODE=offline uv run rlaopt-bench run-job \
  --n 256 --p 256 --alpha 1 --seed 0 --solver scipy_lsqr --backend cpu
```

For a direct smoke grid, submit an array sized from the manifest and route it to a campaign-specific output directory:

```bash
BACKEND=cpu MANIFEST=artifacts/smoke-cpu.jsonl \
  CONFIG=configs/smoke.toml OUTPUT_DIR=artifacts/smoke-campaign \
  sbatch --array="0-$(($(wc -l < artifacts/smoke-cpu.jsonl)-1))" slurm/run_array.sh
```

`CONFIG` must be the same file used to generate `MANIFEST`; it defaults to `configs/synthetic.toml`. For example, submit the CPU smoke grid with

```bash
uv run rlaopt-bench manifest \
  --config configs/smoke.toml \
  --backend cpu \
  --output artifacts/smoke-cpu.jsonl

BACKEND=cpu \
CONFIG=configs/smoke.toml \
MANIFEST=artifacts/smoke-cpu.jsonl \
OUTPUT_DIR=artifacts/smoke-campaign \
  sbatch --array="0-$(($(wc -l < artifacts/smoke-cpu.jsonl)-1))" slurm/run_array.sh
```

For production CPU timing, split the manifest with `scripts/shard_cpu_manifest.py`. The soal QOS counts every pending array element toward its 20-job submission limit, so do not submit the 288 manifest records as individual array elements. Instead, submit six chunks for each CPU shard and six chunks for CUDA. Each chunk executes its round-robin subset sequentially and continues after a launcher failure. Use `--array=0-5%1` for all three arrays, pin the CPU arrays to soal-8 and soal-9, and pin the CUDA array to soal-12. This creates 18 submitted tasks and runs at most one benchmark on each node. `CHUNK_COUNT` must equal the array size, and every array must share the same campaign-specific `OUTPUT_DIR`.

Build the derived GPU image once from a login node by submitting:

```bash
sbatch slurm/build_cuda.sh
```

The build uses the immutable `linux/amd64` CUDA digest in `containers/cuda.env`, installs uv 0.12.7, and runs `uv sync --frozen --no-dev --group gpu` during the image build. The resulting `containers/rlaopt-cuda-13.0.2.sif` is intentionally ignored by Git. The build writes its SHA-256 digest once to the tracked `.sif.sha256` sidecar; array tasks read that small file instead of repeatedly hashing the multi-gigabyte image over NFS. Preserve the SIF and checksum with the experiment artifacts; rebuilding from the same inputs is auditable, but the checksum proves which exact image a run used. If the base tag changes, run `scripts/resolve_cuda_digest.sh` and review the new digest before editing `containers/cuda.env`.

After the build succeeds, submit `sbatch slurm/check_cuda.sh` before the GPU smoke grid. It executes the baked environment, performs a float64 cuML LSMR fit alongside PyTorch and rlaopt, and solves a bounded elastic-net smoke problem through the compiled SCS GPU-indirect backend on one H200 NVL. Inspect `cuda-check-<job-id>.out`; success includes `PROJECT_ENVIRONMENT_INTEROPERABLE=true` and ends with `SCS_CUDA_INTERFACE_OK=true`. Run `sbatch slurm/check_cuclarabel.sh` separately because its Julia-first import order requires a fresh process. Both CPU and CUDA jobs submitted through `slurm/run_array.sh` use this SIF; CPU jobs omit `--nv`, so no GPU is exposed. Using one immutable image keeps package versions identical and avoids the severe metadata latency observed when importing PyTorch through thousands of files in a shared NFS virtual environment. SIF startup and checksum lookup occur outside the timed solver region and should be reported as orchestration overhead rather than solver runtime. The array launcher exports the host repository commit and dirty-tree state because Git is not installed in the immutable image. Use a fresh `artifacts/` directory for each production campaign so plotting cannot aggregate records from older generator or image versions.

Before any production rerun, verify the solver mappings on one shared problem with `uv run python scripts/verify_solver_equivalence.py --backend cpu` and `sbatch slurm/verify_solver_equivalence_cuda.sh`. The probe checks every production ridge value against the analytic SVD oracle and checks every pair of backend-local solutions. This explicitly detects inconsistent regularization conventions, including an accidental `lambda` versus `sqrt(lambda)` mapping or sample-count normalization.

For the maximum-size CPU probe, use `/usr/bin/time -v` around one `run-job` command and compare its maximum resident set size with the `peak_memory_bytes` record. On CUDA, compare the recorded peak PyTorch allocation with `nvidia-smi` and scheduler accounting; allocator memory does not include every cuML/CUDA allocation.

Generate figures only after auditing failures:

```bash
uv run rlaopt-bench plot --input artifacts/records --output artifacts/figures
```

The figure command creates log-log median-runtime scatterplots with native-success seed min--max ranges for fixed-$p$, fixed-$n$, and square families, faceted by $\alpha$ and $\lambda$, plus a machine-readable failure summary. Shared endpoints appear in each applicable scaling panel. Paper analysis should additionally report iteration/matvec throughput, setup time, memory, convergence traces, effective dimension, condition numbers, and GPU-resident CPU/GPU speedups. Points from different spectral profiles or ridge values are never placed in the same panel.

## Post-production accuracy refinement

Calibration supplies an initial native tolerance expected to satisfy the common
accuracy criterion; it does not guarantee accuracy on every production instance.
The refinement trigger is **native success (or normal completion for interfaces
without a convergence flag) together with failure of the common external metric**.
This rule is applied to every solver. Native nonconvergence, timeouts, worker
failures, and missing results remain separate outcomes; they do not trigger this
accuracy-refinement ladder. Original production records are retained unchanged.

The production audit found six CPU SciPy LSQR and seven GPU cuML LSMR ridge
measurements with native success/completion but relative KKT above `1e-6`, all at
native tolerance `1e-9`. Their residuals ranged from approximately `1.05e-6` to
`3.09e-6`. These are 13 individual seed/shape/spectrum/ridge measurements, not 13
entire three-lambda jobs. No ridge Nyström PCG, unpreconditioned CG, or QR native
success missed the external target. The real-data audit identified the SCS
GPU-direct YearPredictionMSD-rf case discussed above; multinomial had no such miss.

The ridge follow-up tries `1e-10` first and `1e-11` only if the first attempt again
has native success but fails the common metric. Each attempt uses a fresh worker,
the same problem seed, shape, spectrum and ridge coefficient, the original node,
64 physical CPU cores, and one H200 for GPU trials. It preserves one untimed warmup
(capped at 10 iterations), one measured solve, the `2*p` measured iteration ceiling,
and the 900-second startup and per-solve limits. The external KKT target stays at
`1e-6`. Trial outputs are separated by original trial ID and native tolerance.
Both SCS diagnostic tolerances were submitted as a predeclared two-point sweep;
the ridge ladder conditionally stops when an attempt passes or fails natively.

The original ridge image digest was
`b54e5e682a428d9dadd7cbd906f64a8e826daf414aa81997eaa600cb84fe891e`.
The prepared follow-up uses the current pinned image
`3f1bfeab2a437cd18cdb21449708fd1eca60d1841e571d0556d7b9b06fde6d2d`, which adds ERM
support. The targeted ridge solver implementations are unchanged from the original
production commit. The extracted SORF generator was checked against the original
implementation on nine small square/tall/wide CPU cases with bit-identical `X` and
`y`. Both image digests and the source-record checksum are retained; this is not
claimed to use an identical container image.

Prepare the audited candidates and review submission commands without launching:

```bash
.venv/bin/python scripts/plan_ridge_refinement.py \
  --output artifacts/ridge-tolerance-refinement-20260906
.venv/bin/python scripts/submit_ridge_refinement.py \
  artifacts/ridge-tolerance-refinement-20260906 0 --dry-run
```

The prepared plan contains three CPU tasks on each of `soal-8` and `soal-9` and
seven GPU tasks on `soal-12`. Wave 0 has the six CPU tasks and six GPU tasks; wave 1
has the remaining GPU task. CPU concurrency is one per node and GPU concurrency
two, with 64 threads and 128 GiB host memory per task. The launcher verifies frozen
manifests, source records and the actual image hash, requires a clean checkout,
refuses overlap with active benchmark jobs on the selected nodes, and enforces the
20-job QOS limit. Use `--node soal-9` (or another listed node) to submit that node
independently; submission receipts are kept separately per wave and node.
To submit a reviewed wave, remove `--dry-run`; wait for wave 0 to finish before
submitting wave 1. Submission status and job IDs are recorded in `submission-<wave>-<node>.json`
under the plan directory.

For reporting, retain the frozen-protocol results and present refinement results
as a disclosed post-production study. Report all attempted tolerances and their
native status, external residuals and measured runtime. A qualifying runtime is
from the first attempted setting that meets both native and external requirements;
if none qualifies, display the final completed attempt's outcome (for example,
timeout), while preserving the original accuracy miss and every attempted setting. Report
cumulative measured solver time across the original and refinement attempts
separately from the qualifying time. Also report refinement wall time, which
includes new worker startup, generation and warmup; original per-trial preparation
cost is not reconstructed from shared production jobs. Each trial's `summary.json`
retains these quantities. Do not silently overwrite the original measurements or
present the follow-up tolerance choices as held-out calibration.

Suggested paper wording:

> Native tolerances were initially selected on a separate calibration set. We
> evaluated returned production solutions against fixed solver-independent accuracy
> criteria. Following an audit of the production results, we performed a disclosed
> accuracy-refinement study on every trial that reported native success or normal
> completion but failed the common criterion. Refinement used a predefined finite
> sequence of stricter native tolerances while retaining the original problem and
> accuracy thresholds. We preserved all attempts and report qualifying solve time
> separately from cumulative measured solver time, including unsuccessful accuracy
> attempts. The refinement study is post-production and is not held-out calibration.

## Paper figures across the production suites

JAXopt APG denotes accelerated projected gradient. Both production JIT variants retain JAXopt's default `acceleration=True`; turning off JIT does not turn off acceleration. Figures explicitly identify JIT on/off and label the seven real-data runs that exhausted 100,000 iterations as “Iteration limit,” distinct from one-hour timeouts. Dataset labels include samples-by-features dimensions after feature generation. Ridge preconditioning figures cover all tested shapes with direct speedup ratios, and runtime figures annotate unsuccessful methods with readable failure names and seed counts.

Run `.venv/bin/python scripts/make_paper_figures.py` from the repository root to generate the ridge, bounded multinomial, bounded elastic-net, and CPU/GPU runtime-ratio figures. Outputs are PDF and PNG files under `artifacts/paper-figures`, with CSV audit tables covering every expected trial. See [FIGURES.md](FIGURES.md) for reproduction details and suggested captions, and [RESULTS_NARRATIVE.md](RESULTS_NARRATIVE.md) for the presentation rationale.

Plotting uses LaTeX math rendering with Computer Modern fonts. Dataset names are lowercase, with `-rf` suffixes for random features. The original plotting mode uses frozen calibrated native success. The current ridge and bounded elastic-net figures use the accuracy-refinement rule documented below: first native-and-external passing attempt, or the final completed failure outcome if none passes. The main multinomial comparison explicitly labels JIT-disabled JAXopt; the JIT-enabled appendix and main-text discussion disclose its faster performance. Timeouts are censored, resource failures are separate outcomes, and CPU/GPU ratios with CPU timeouts are lower bounds. Ridge whiskers are min–max ranges over three seeds; the one-seed real-data runs have no uncertainty bars.

## Synthetic ridge suite: production outcome and limitations

The ridge production campaign completed under `artifacts/production-20260831`. Its manifests contain the expected 288 jobs per backend and 1,728 solver/ridge trials. The campaign retained 1,701 atomic records: 27 CPU QR trials produced no record, while the corresponding 27 CUDA QR trials recorded solver errors. The retained records also contain six CPU identity-PCG timeouts, six SciPy LSQR external KKT misses, and seven cuML LSMR external KKT misses; these outcomes remain visible in `figures/failure_summary.json` rather than being silently discarded. Fifteen-minute per-solve limits remained fixed, and generation time and memory were measured separately.

Limitations: SORF factors are structured random orthogonal matrices rather than Haar draws; the response lies in $\mathrm{range}(X)$ and has no observation noise; rank 128 is a fixed resource budget, not tuned per instance; CPU and GPU plots represent only the named machines; GPU-resident timing excludes transfer; and direct methods may exceed memory. Follow-up sensitivity studies can vary Nyström rank, add a controlled orthogonal/noisy response component, and measure end-to-end transfer costs.

The accuracy-refinement plotting command and captions are documented in
[FIGURES.md](FIGURES.md#figures-incorporating-accuracy-refinement). This mode updates
only ridge scaling and bounded elastic net; attempt costs stay in audit files.

### Differentiable optimization demonstration

```sh
.venv/bin/python scripts/run_differentiable_optimization.py
.venv/bin/python scripts/make_paper_figures.py --differentiable-only
```

The script generates seeded synthetic data, differentiates through the lasso solver, and saves the validation objective in `artifacts/differentiable-optimization/result.json`. Its data and solver settings are specified directly in the script. The plot uses Iterations on the linear x-axis and Objective Value on the logarithmic y-axis. Full figure regeneration includes this saved result when available.
