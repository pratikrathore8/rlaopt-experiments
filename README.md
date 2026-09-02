# rlaopt experiments

This repository contains reproducible benchmark suites comparing rlaopt with other optimization and numerical linear algebra methods. It is intended to grow beyond least squares and ridge regression: each problem family should define its own mathematical model, competitors, accuracy contract, configuration, runner, and analysis while sharing the repository's environment, result-tracking, and orchestration conventions.

The first implemented suite is controlled synthetic ridge regression. A synthetic empirical-risk-minimization (ERM) suite for developing the later real-data benchmarks is now under development; its problem-generation and accuracy contracts are documented below before its solver adapters and experiment grid are implemented.

## Benchmark suite organization

Repository-level dependencies and reproducibility policy live in `pyproject.toml`, `uv.lock`, and `containers/`. The current ridge suite uses `configs/synthetic.toml` and the `problem.py`, `solvers.py`, `diagnostics.py`, `runner.py`, and `plotting.py` modules. As additional benchmark families are introduced, their problem-specific configuration and implementation should be placed in named subpackages rather than added as conditionals to the ridge model. Shared concerns—stable run identifiers, atomic records, W&B recovery, hardware metadata, and Slurm submission—should remain reusable across suites.

Every suite must document:

- its mathematical problem and data-generating process;
- the normalization and parameter scales used across problem sizes;
- solver-specific mappings and a method-independent success criterion;
- what setup, transfers, and synchronization are included in runtime;
- resource limits, failure handling, and the exact hardware scope; and
- suite-specific limitations and confirmatory versus exploratory analyses.

## Synthetic ERM suite (under development)

This suite develops the three problem classes intended for the later real-data study on deterministic synthetic data first. All generated arrays and all solver computations use float64. Features are generated on CPU from seeded Gaussian streams, then each column is centered and scaled to unit population root-mean-square. Problem generation and host-to-device transfer are measured separately from solver time and are not included in the primary solve-time comparison. Every competitor receives the same materialized data and mathematical formulation.

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

Native solver status and native stopping tolerances are not treated as cross-method accuracy evidence. After timing, the testbed recomputes every primary metric from the returned primal solution in float64 using the canonical problem above. The problem objects implement the formulation-dependent objective, gradients, KKT residuals, constraint violations, and dual certificate; the suite diagnostics layer records those values and applies frozen pass thresholds. Solver-specific tolerances will be calibrated against these common metrics on a separate calibration grid and frozen before production. A production record distinguishes `native_success`, meaning that the solver reached its calibrated native stopping rule; `external_success`, meaning that the returned solution passed the common accuracy and feasibility thresholds; and `runtime_eligible`, which normally equals `native_success` and determines inclusion in runtime summaries. A native success that misses the external target remains in the runtime comparison but is visible in the separately reported external pass rate and residual distribution. Constraint satisfaction is not tested against exact zero: a constrained solution passes when its measured maximum violation is at most the frozen feasibility tolerance. Objective value, native residuals, and native dual variables are retained as secondary diagnostics. Iterate histories are retained when a supported interface exposes primal iterates; black-box interfaces retain their final native diagnostics instead.

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

### Timing and competitor-interface policy

SAPPHIRE will be compared with projected gradient and JAXopt L-BFGS-B for bounded multinomial logistic regression. SAPPHIRE will be compared with scikit-learn, cuML, and JAXopt proximal gradient for vanilla elastic net. The bounded elastic-net comparison will use rlaopt ADMM, SCS, and Clarabel. CPU and GPU results will be reported separately, and unsupported backend/problem combinations will be labeled rather than emulated.

rlaopt 0.1.0's least-squares atom evaluates mean squared error without the factor of $1/2$ used by the canonical elastic-net objective. Its adapter therefore passes $2\lambda_1$ and $2\lambda_2$ to rlaopt, making the complete native objective exactly twice the canonical objective and leaving its minimizer unchanged. The record identifies this convention with `native_objective_scale = 2.0`. All externally reported objectives, gaps, KKT residuals, and pass decisions are recomputed from the returned primal solution using the canonical formulation; they are never compared on rlaopt's doubled scale.

Every competitor, including SCS and Clarabel, is called through its direct solver interface. Benchmark-side construction of the common problem arrays and any format conversion needed to pass them to an interface occurs before the timed region and is recorded separately. Primary solver time begins immediately before the native setup/solve call and therefore includes all symbolic analysis, scaling, factorization, preconditioner construction, and other numerical setup performed by that solver. Repeated timings reuse the same immutable problem arrays but create a fresh solver instance unless a separately labeled warm-start experiment explicitly permits state reuse. Small-instance equivalence checks compare every direct interface against the canonical objective and KKT diagnostics implemented by this repository.

Clarabel is invoked without CVXPY, JuMP, or another modeling layer. Its CPU baseline uses the `:qdldl` direct linear solver. Its GPU baseline uses the CuClarabel branch through JuliaCall with the full-float64 `:cudss` linear solver; the mixed-precision `:cudssmixed` mode is excluded from the float64 study. Julia JIT compilation is triggered by an untimed miniature solve before measurement. Each measured solve still constructs a fresh Clarabel solver, and its timed region includes both Clarabel's native setup and solve phases. Host construction of $P,q,A,b$, sparse-format conversion, and host-to-device transfer remain separately recorded preparation costs.

The CUDA image pins Julia 1.10.12 by the official archive SHA-256 and resolves the CuClarabel branch at commit `ffa325c89fa90b7e86b745fa61b1dca64daf3a06`. `julia/cuclarabel/Manifest.toml` locks the complete Julia dependency graph, including CUDA.jl 5.11.3, CUDSS.jl 0.6.5, CUDSS_jll 0.7.1+0, and PythonCall.jl 0.9.31; the Python environment pins the matching JuliaCall 0.9.31 and CuPy CUDA 13 package. The generated Apptainer SIF and its checksum remain the executable-environment artifact for production runs.

CuClarabel's current CuPy bridge wraps device pointers without taking ownership and converts zero-based CSR indices to Julia's one-based indices in place. The adapter therefore creates dedicated CuPy CSR buffers for each solve, retains their Python owners until `solve!` and solution extraction finish, and never reuses the mutated matrices. This is an interface-safety requirement, not a benchmark optimization. The pre-production smoke probe solves the same deterministic bounded elastic-net QP with CPU QDLDL and GPU cuDSS, recomputes objective, KKT stationarity, and box violation from both returned primal solutions, and checks CPU/GPU agreement.

Final dataset sizes, the synthetic grid, calibrated thresholds, resource limits, and exact solver versions will be added and frozen before production. The study will include instances demonstrably too large for the considered interior-point baselines; such claims will be supported by explicit memory estimates or observed structured resource failures rather than assumed from dimensions alone.

The initial development grid is intentionally small and is defined in `configs/synthetic_erm_smoke.toml`. It uses $(n,p)\in\{(1024,64),(4096,256)\}$, five classes for multinomial regression, three data seeds, one measured repetition per seed, and regularization fractions $\gamma\in\{0.1,0.01\}$. The vanilla and bounded elastic-net variants share one configuration block, which prevents their shapes or data-generating parameters from drifting apart. The configured accuracy thresholds are explicitly marked `calibrated = false`; production orchestration must not treat them as frozen thresholds until solver-specific tolerance calibration is complete.

The `manifest` command expands every currently executable synthetic-ERM grid into one self-contained JSON record per problem, solver, and backend. Feature, target, and solver randomness use independently derived deterministic streams from the recorded master seed; the complete problem specification and resulting problem ID are stored in every job. Across regularization fractions, vanilla elastic-net jobs reuse the same feature and target seeds, so only the regularization changes. Per backend, the smoke manifest contains 18 multinomial jobs and 36 vanilla elastic-net jobs, for 54 jobs total. Bounded elastic-net jobs will be added only when their solver adapters are executable. The corresponding `run-manifest-job` command reads one zero-based manifest index without loading the full file and obtains all execution controls from the same strict suite configuration.

The development configuration gives every multinomial and vanilla elastic-net solver a provisional native tolerance of $10^{-6}$, uses a SAPPHIRE batch size of 256, and permits at most 10,000 solver iterations subject to the 15-minute hard timeout. Each problem family has its own strict execution-control block whose native-tolerance keys must exactly match its CPU and CUDA solver lists. Both native tolerances and the common external accuracy thresholds are marked uncalibrated and recorded independently; these values enable smoke execution but are not production settings.

The multinomial and vanilla elastic-net executors validate the recorded suite, problem type, problem ID, configured seed, derived solver seed, solver, backend, shape, and all data-generating parameters before starting an isolated worker. Problem generation and worker startup occur outside solver timing. Configured warmups use fresh solver calls with at most ten iterations and are not recorded when successful; measured repetitions also use fresh solver calls. A timeout or exception is persisted as a result rather than aborting the manifest, and the local atomic JSON record remains authoritative if optional W&B logging fails.

The multinomial adapters use each pinned package's default algorithmic configuration: rlaopt SAPPHIRE defaults to SAGA with its rank-10 Nyström preconditioner, JAXopt projected gradient defaults to acceleration with backtracking, and JAXopt L-BFGS-B defaults to the zoom line search. The adapters override only the experiment's iteration and native-tolerance limits; the batch size and random seed required by SAPPHIRE; and settings required to represent and time the common problem. JAX is configured for float64 and performs one untimed compilation solve before a fresh measured solve. JAX and PyTorch share device-resident arrays through DLPack, so format conversion is outside the timed region without introducing a hidden host transfer. rlaopt 0.1.0's Nyström spectral estimator creates its random power-iteration vector in PyTorch's default dtype rather than the requested problem dtype. The isolated SAPPHIRE adapter therefore temporarily sets the default dtype to float64 for the entire solve and restores it afterward; it does not replace or disable the Nyström preconditioner.

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

The build uses the immutable `linux/amd64` CUDA digest in `containers/cuda.env`, installs uv 0.12.7, and runs `uv sync --frozen --no-dev --group gpu` during the image build. The resulting `containers/rlaopt-cuda-13.0.2.sif` is intentionally ignored by Git. The build writes its SHA-256 digest once to the tracked `.sif.sha256` sidecar; array tasks read that small file instead of repeatedly hashing the 6.6 GB image over NFS. Preserve the SIF and checksum with the experiment artifacts; rebuilding from the same inputs is auditable, but the checksum proves which exact image a run used. If the base tag changes, run `scripts/resolve_cuda_digest.sh` and review the new digest before editing `containers/cuda.env`.

After the build succeeds, submit `sbatch slurm/check_cuda.sh` before the GPU smoke grid. It executes the baked environment and performs a float64 cuML LSMR fit alongside PyTorch and rlaopt on one H200 NVL. Inspect `cuda-check-<job-id>.out`; success ends with `PROJECT_ENVIRONMENT_INTEROPERABLE=true`. Both CPU and CUDA jobs submitted through `slurm/run_array.sh` use this SIF; CPU jobs omit `--nv`, so no GPU is exposed. Using one immutable image keeps package versions identical and avoids the severe metadata latency observed when importing PyTorch through thousands of files in a shared NFS virtual environment. SIF startup and checksum lookup occur outside the timed solver region and should be reported as orchestration overhead rather than solver runtime. The array launcher exports the host repository commit and dirty-tree state because Git is not installed in the immutable image. Use a fresh `artifacts/` directory for each production campaign so plotting cannot aggregate records from older generator or image versions.

Before production, verify the solver mappings on one shared problem with `uv run python scripts/verify_solver_equivalence.py --backend cpu` and `sbatch slurm/verify_solver_equivalence_cuda.sh`. The probe checks every production ridge value against the analytic SVD oracle and checks every pair of backend-local solutions. This explicitly detects inconsistent regularization conventions, including an accidental `lambda` versus `sqrt(lambda)` mapping or sample-count normalization.

For the maximum-size CPU probe, use `/usr/bin/time -v` around one `run-job` command and compare its maximum resident set size with the `peak_memory_bytes` record. On CUDA, compare the recorded peak PyTorch allocation with `nvidia-smi` and scheduler accounting; allocator memory does not include every cuML/CUDA allocation.

Generate figures only after auditing failures:

```bash
uv run rlaopt-bench plot --input artifacts/records --output artifacts/figures
```

The figure command creates log-log median-runtime scatterplots with native-success seed min--max ranges for fixed-$p$, fixed-$n$, and square families, faceted by $\alpha$ and $\lambda$, plus a machine-readable failure summary. Shared endpoints appear in each applicable scaling panel. Paper analysis should additionally report iteration/matvec throughput, setup time, memory, convergence traces, effective dimension, condition numbers, and GPU-resident CPU/GPU speedups. Points from different spectral profiles or ridge values are never placed in the same panel.

## Synthetic ridge suite: expected cost and limitations

The revised $2^{16}$ endpoints require a new pilot before assigning a production wall-clock estimate. Fifteen-minute per-solve limits remain fixed; generation time and memory are measured separately during the maximum-size probe.

Limitations: SORF factors are structured random orthogonal matrices rather than Haar draws; the response lies in $\mathrm{range}(X)$ and has no observation noise; rank 128 is a fixed resource budget, not tuned per instance; CPU and GPU plots represent only the named machines; GPU-resident timing excludes transfer; and direct methods may exceed memory. Follow-up sensitivity studies can vary Nyström rank, add a controlled orthogonal/noisy response component, and measure end-to-end transfer costs.
