# rlaopt experiments

This repository contains reproducible benchmark suites comparing rlaopt with other optimization and numerical linear-algebra methods. It is intended to grow beyond least squares and ridge regression: each problem family should define its own mathematical model, competitors, accuracy contract, configuration, runner, and analysis while sharing the repository's environment, result-tracking, and orchestration conventions.

The first implemented suite is controlled synthetic ridge regression. Its fixed-budget design is documented below; it is not a claim that rank 128 is optimal for every spectrum or that its solver set applies to future problem families.

## Benchmark-suite organization

Repository-level dependencies and reproducibility policy live in `pyproject.toml`, `uv.lock`, and `containers/`. The current ridge suite uses `configs/synthetic.toml` and the `problem.py`, `solvers.py`, `diagnostics.py`, `runner.py`, and `plotting.py` modules. As additional benchmark families are introduced, their problem-specific configuration and implementation should be placed in named subpackages rather than added as conditionals to the ridge model. Shared concerns—stable run identifiers, atomic records, W&B recovery, hardware metadata, and Slurm submission—should remain reusable across suites.

Every suite must document:

- its mathematical problem and data-generating process;
- the normalization and parameter scales used across problem sizes;
- solver-specific mappings and a method-independent success criterion;
- what setup, transfers, and synchronization are included in runtime;
- resource limits, failure handling, and the exact hardware scope; and
- suite-specific limitations and confirmatory versus exploratory analyses.

The sections that follow describe only the synthetic ridge suite.

## Synthetic ridge suite: mathematical model

For every shape `(n,p)`, let `r=min(n,p)`. Draw independent Gaussian matrices, compute thin QR factorizations, and canonicalize each QR sign using the diagonal of `R`. This gives deterministic orthonormal factors `U in R^(n x r)` and `V in R^(p x r)`. For decay exponent `alpha`, define

```text
s_k = k^(-alpha/2),  k=1,...,r
X = U diag(s) V^T.
```

Thus the nonzero eigenvalues of `X^T X` are exactly `k^(-alpha)` and `||X||_2=1`. The three profiles are `alpha in {1/2,1,2}`. This normalization is equivalent to starting from the statistical convention `G/sqrt(n)` and then prescribing the population spectrum; it prevents sample count from changing the regularization scale.

The response is generated from an independent Gaussian vector `z`, normalized as `zhat=z/||z||`, with

```text
y = U zhat,                 ||y||_2 = 1.
```

Every method solves, in float64,

```text
min_w  1/2 ||Xw-y||_2^2 + lambda/2 ||w||_2^2,
```

for `lambda in {1e-2,1e-4,1e-6}`. The exact reference solution is derived from the known SVD—not from a separate numerical solve:

```text
w_star = V diag(s_k/(s_k^2+lambda)) zhat.
```

This is the closed-form ridge solution for this construction, not a special “Tropp formula.” The design follows the normalized prescribed-spectrum models used in randomized numerical linear algebra. We deliberately do not add a second entrywise Gaussian noise matrix: doing so would destroy the exact spectrum and oracle. Here `lambda` supplies the isotropic floor in the normal equations.

We report effective dimension `sum_k s_k^2/(s_k^2+lambda)`, full condition number `(1+lambda)/lambda`, active condition number `(1+lambda)/(s_r^2+lambda)`, `r/p`, and `r/d_eff`.

## Synthetic ridge suite: fixed experiment grid

There are eight unique shapes:

| family | `(n,p)` |
|---|---|
| square anchors | `(2^8,2^8)`, `(2^10,2^10)`, `(2^12,2^12)` |
| sample scaling (`p=2^12`) | `(2^14,2^12)`, `(2^16,2^12)`, `(2^18,2^12)` |
| feature scaling (`n=2^18`) | `(2^18,2^8)`, `(2^18,2^10)` |

Each shape uses three decay profiles, three master seeds, and three ridge values. Factor, response, and Nyström randomness use separately derived deterministic streams. A job is `(hardware,solver,shape,alpha,seed)` and reuses one resident `X` for all three ridge values: 288 jobs per backend, 576 total, and 1,728 ridge trials before timing repetitions.

Only the intended paper hardware is in scope for now:

| backend | hardware | solvers |
|---|---|---|
| CPU | 64 physical cores on soal-8/soal-9 | rlaopt Nyström-PCG, rlaopt identity-PCG (CG), SciPy LSQR, PyTorch augmented QR |
| GPU | NVIDIA H200 NVL on soal-12 | both rlaopt variants, cuML Ridge/LSMR, PyTorch augmented QR |

rlaopt always receives the matrix-free `X^T X` operator and `B=X^T y`, with `reg=lambda`. Nyström-PCG fixes `rank_init=rank_max=min(128,p)`, `base_damping=lambda`, and adaptive damping. Ridge is not folded into the operator, so it is never counted twice. SciPy uses a `LinearOperator` for `X` and `damp=sqrt(lambda)`. PyTorch uses the augmented system `[X; sqrt(lambda)I]w=[y;0]`, since `torch.linalg.lstsq` has no ridge argument. cuML uses `Ridge(alpha=lambda, fit_intercept=False, solver="lsmr")`.

## Synthetic ridge suite: accuracy, stopping, and timing

The cross-method success criterion is the externally recomputed relative KKT residual

```text
||(X^T X + lambda I)w - X^T y||_2 / ||X^T y||_2 <= 1e-6.
```

Native solver status alone never counts as success. Relative error to `w_star` is a secondary diagnostic. Normwise backward error and objective gap are omitted because they add little beyond KKT residual plus exact solution error for this controlled quadratic.

Native tolerances live in `configs/tolerances.toml`. Calibrate one tolerance per solver/backend on representative easy, middle, and hard cases, choose the loosest value that passes every external KKT check, then freeze the file before the production sweep. rlaopt records a residual point per iteration; SciPy records its LSQR diagnostics; cuML records `n_iter_` when exposed. Direct QR has no iteration history.

Iterative solvers stop at native convergence, `2p` iterations, or five minutes, whichever comes first. QR has only the five-minute process timeout. The Slurm wrapper provides a hard process boundary; rlaopt additionally checks elapsed time cooperatively on every iteration. A timeout or a native “success” that misses KKT is displayed as a failure, never silently dropped.

Problem generation, analytic oracle work, CPU pinning, and host-to-device transfer are excluded from runtime. Timed regions include preconditioner construction, factorization, and all internal solver setup. GPU timings synchronize immediately before and after the solve and therefore describe GPU-resident inputs. One warm-up is untimed. A first successful run under 30 seconds receives three timed repetitions and is summarized by its median/min/max; slower configurations run once. Peak CUDA allocator memory is recorded (CPU peak RSS should be supplied by the scheduler).

## Reproducible environment

Install the exact uv release first, then sync the lockfile:

```bash
curl -LsSf https://astral.sh/uv/0.12.7/install.sh | sh
uv sync --frozen
```

The project pins Python 3.12, uv 0.12.7, rlaopt 0.1.0 from PyPI, NumPy 2.5.2, SciPy 1.18.1, PyTorch 2.13.0, matplotlib 3.11.1, and W&B 0.29.0. `uv.lock` pins the transitive CPU environment. cuML/RAPIDS 26.08 is supplied only on the H200 through the image in `containers/rapids.env`; replace its placeholder with the immutable registry digest before production. A mutable tag is not sufficient evidence of the GPU environment.

We do not store multi-gigabyte matrices or their checksums. A stable problem ID plus the pinned code, package lock, dimensions, exponents, and explicit seeds reproduce each matrix. Record the git commit, Slurm job ID, node, driver, CUDA runtime, GPU model, thread variables, and final container digest alongside paper artifacts.

W&B defaults to offline mode. Atomic JSON files under `artifacts/records/` are the source of truth and remain recoverable if W&B fails; sync them later with `wandb sync` if desired.

## Synthetic ridge suite: workflow

Create one manifest per backend:

```bash
uv run rlaopt-bench manifest --backend cpu --output artifacts/cpu.jsonl
uv run rlaopt-bench manifest --backend cuda --output artifacts/cuda.jsonl
```

Run a small calibration/smoke case before freezing tolerances:

```bash
WANDB_MODE=offline uv run rlaopt-bench run-job \
  --n 256 --p 256 --alpha 1 --seed 0 --solver scipy_lsqr --backend cpu
```

Submit a manifest (the CPU manifest has 288 lines):

```bash
BACKEND=cpu MANIFEST=artifacts/cpu.jsonl sbatch --array=0-287 slurm/run_array.sh
```

For the GPU allocation, add the site-specific GPU constraint and execute the same script inside the pinned RAPIDS digest. Generate figures only after auditing failures:

```bash
uv run rlaopt-bench plot --input artifacts/records --output artifacts/figures
```

The figure command creates log-log runtime scatterplots for fixed-`p`, fixed-`n`, and square families and a machine-readable failure summary. Paper analysis should additionally report iteration/matvec throughput, setup time, memory, convergence traces, effective dimension, condition numbers, and GPU-resident CPU/GPU speedups. Never connect points across different `alpha` or `lambda` without facet/legend separation.

## Synthetic ridge suite: expected cost and limitations

With all target nodes available concurrently, the sweep should take roughly 4–10 wall-clock hours; queueing, retries, and slow tail jobs make one to two days a realistic end-to-end allowance. The reduced `n<=2^18` grid and five-minute cap keep the study tractable.

Reviewer-visible limitations are intentional: the response lies in `range(X)` and has no observation noise; rank 128 is a fixed resource budget, not tuned per instance; CPU and GPU plots represent only the named machines; GPU-resident timing excludes transfer; and direct methods may exceed memory. Follow-up sensitivity studies can vary Nyström rank, add a controlled orthogonal/noisy response component, and measure end-to-end transfer costs, but they must be labeled separately from this confirmatory grid.

## Implementation history

The first suite's implementation history is split into reproducible environment, generator, solver adapters, calibration/diagnostics, tracking/orchestration, aggregation/figures, and documentation. Future suites should follow the same reviewable separation without treating ridge-specific mathematics or solver interfaces as repository-wide abstractions.
