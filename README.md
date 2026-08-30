# rlaopt experiments

This repository contains reproducible benchmark suites comparing rlaopt with other optimization and numerical linear algebra methods. It is intended to grow beyond least squares and ridge regression: each problem family should define its own mathematical model, competitors, accuracy contract, configuration, runner, and analysis while sharing the repository's environment, result-tracking, and orchestration conventions.

The first implemented suite is controlled synthetic ridge regression. Its fixed-budget design is documented below.

## Benchmark suite organization

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

For every shape $(n,p)$, let $r=\min(n,p)$. Draw independent Gaussian matrices, compute thin QR factorizations, and canonicalize each QR sign using the diagonal of $R$. This gives deterministic orthonormal factors $U \in \mathbb{R}^{n \times r}$ and $V \in \mathbb{R}^{p \times r}$. For decay exponent $\alpha$, define

$$
s_k = k^{-\alpha/2}, \qquad k=1,\ldots,r,
$$

and

$$
X = U \mathrm{diag}(s) V^{\mathsf T}.
$$

Thus the nonzero eigenvalues of $X^{\mathsf T}X$ are exactly $k^{-\alpha}$ and $\lVert X\rVert_2=1$. The three profiles are $\alpha \in \{1/2,1,2\}$. This normalization is equivalent to starting from the statistical convention $G/\sqrt{n}$ and then prescribing the population spectrum; it prevents sample count from changing the regularization scale.

The response is generated from an independent Gaussian vector $z$, normalized as $\widehat z=z/\lVert z\rVert_2$, with

$$
y = U\widehat z, \qquad \lVert y\rVert_2=1.
$$

Every method solves, in float64,

$$
\min_w \frac{1}{2}\lVert Xw-y\rVert_2^2 + \frac{\lambda}{2}\lVert w\rVert_2^2,
$$

for $\lambda \in \{10^{-2},10^{-4},10^{-6}\}$. The exact reference solution is derived from the known SVD:

$$
w_\star
= V\,\mathrm{diag}\left(\frac{s_k}{s_k^2+\lambda}\right)\widehat z.
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

and the ratios $r/p$ and $r/d_{\mathrm{eff}}$. This formula uses the fact that every configured shape has $n\ge p$ and therefore $X$ has full column rank.

## Synthetic ridge suite: fixed experiment grid

There are eight unique shapes:

| family | $(n,p)$ |
|---|---|
| square anchors | $(2^8,2^8)$, $(2^{10},2^{10})$, $(2^{12},2^{12})$ |
| sample scaling ($p=2^{12}$) | $(2^{14},2^{12})$, $(2^{16},2^{12})$, $(2^{18},2^{12})$ |
| feature scaling ($n=2^{18}$) | $(2^{18},2^8)$, $(2^{18},2^{10})$ |

Each shape uses three decay profiles, three master seeds, and three ridge values. Factor, response, and Nyström randomness use separately derived deterministic streams. A job is $(\text{hardware},\text{solver},\text{shape},\alpha,\text{seed})$ and reuses the same data matrix $X$ for all three ridge values: 288 jobs per backend, 576 total, and 1,728 ridge trials before timing repetitions.

We use the following hardware and solver combinations:

| backend | hardware | solvers |
|---|---|---|
| CPU | 64 physical cores on soal-8/soal-9 | rlaopt Nyström-PCG, rlaopt identity-PCG (CG), SciPy LSQR, PyTorch augmented QR |
| GPU | NVIDIA H200 NVL on soal-12 | both rlaopt variants, cuML Ridge/LSMR, PyTorch augmented QR |

rlaopt always receives $X^{\mathsf T}X$ as a linear operator (i.e., it never forms the Gram matrix) and $B=X^{\mathsf T}y$, with `reg=lambda`. Nyström PCG fixes $\mathtt{rank\_init}=\mathtt{rank\_max}=\min(128,p)$, `base_damping=lambda`, and adaptive damping. Ridge is not folded into the operator, so it is never counted twice. SciPy uses a `LinearOperator` for $X$ and $\mathtt{damp}=\sqrt{\lambda}$. PyTorch uses the augmented system

$$
\begin{bmatrix}X\\ \sqrt{\lambda}I\end{bmatrix}w
= \begin{bmatrix}y\\0\end{bmatrix},
$$

since `torch.linalg.lstsq` has no ridge argument. The adapter is named `torch_lstsq_qr` in code and selects the QR-based `gelsy` driver on CPU and `gels` on CUDA; the stable manifest identifier remains `torch_qr`. cuML uses `Ridge(alpha=lambda, fit_intercept=False, solver="lsmr")`.

## Synthetic ridge suite: accuracy, stopping, and timing

The cross-method success criterion is the externally recomputed relative KKT residual

$$
\frac{\left\lVert (X^{\mathsf T}X+\lambda I)w-X^{\mathsf T}y\right\rVert_2}
{\left\lVert X^{\mathsf T}y\right\rVert_2}
\le 10^{-6}.
$$

Native solver status alone never counts as success. Relative error to $w_\star$ is a secondary diagnostic.

Native tolerances live in `configs/tolerances.toml`. Calibrate one tolerance per solver/backend on representative easy, middle, and hard cases, choose the loosest value that passes every external KKT check, then freeze the file before the production sweep. rlaopt stores a residual point per iteration in both the atomic JSON record and W&B; SciPy records its final LSQR diagnostics; cuML records `n_iter_` when exposed. Direct QR has no iteration history.

Run calibration with, for example, `uv run rlaopt-bench calibrate --backend cpu --solver scipy_lsqr --candidates 1e-4 1e-5 1e-6 1e-7 1e-8`. The command writes all underlying records plus `calibration.json`; copy the selected value into `configs/tolerances.toml` only after inspecting every case.

Use `configs/calibration.toml` for the fixed easy, middle, and hard calibration regimes. Each problem is generated once and retained while all candidate tolerances run with independently initialized solvers. These cases select a candidate; the largest cases in `configs/pilot.toml` are held out to check that the selected relative tolerance transfers to larger dimensions before it is frozen. CUDA calibration runs through `slurm/calibrate_cuda.sh` inside the pinned image, while 64-core CPU calibration runs through `slurm/calibrate_cpu.sh` on soal-8 or soal-9.

Iterative solvers stop at native convergence, $2p$ iterations, or five minutes, whichever comes first. QR has only the five-minute timeout. A persistent spawned worker retains the generated problem but places every timed native call behind a parent-enforced process boundary; if a solver exceeds five minutes, the parent terminates that worker and regenerates the same deterministic problem before continuing with the next ridge value. rlaopt additionally checks elapsed time cooperatively on every iteration. A timeout, exception, or native “success” that misses KKT is written as a structured failure, never silently dropped.

Problem generation, analytic oracle work, CPU pinning, and host-to-device transfer are excluded from runtime. Timed regions include preconditioner construction, factorization, and all internal solver setup. GPU timings synchronize immediately before and after the solve and therefore describe GPU-resident inputs. One warm-up is untimed. Every configuration whose first timed run passes the external KKT criterion receives three timed repetitions, summarized by its median/min/max. A first-run timeout or accuracy failure is recorded once during the primary sweep and flagged for manual audit rather than automatically consuming two more production attempts. Records include peak process RSS on CPU and peak PyTorch allocator use on CUDA; scheduler and `nvidia-smi` accounting remain necessary because the CUDA allocator value does not include every cuML allocation.

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

Submit a manifest using an array sized from the file (the full CPU manifest has 288 lines):

```bash
BACKEND=cpu MANIFEST=artifacts/cpu.jsonl \
  CONFIG=configs/synthetic.toml \
  sbatch --array="0-$(($(wc -l < artifacts/cpu.jsonl)-1))" slurm/run_array.sh
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
  sbatch --array="0-$(($(wc -l < artifacts/smoke-cpu.jsonl)-1))" slurm/run_array.sh
```

Build the derived GPU image once from a login node by submitting:

```bash
sbatch slurm/build_cuda.sh
```

The build uses the immutable `linux/amd64` CUDA digest in `containers/cuda.env`, installs uv 0.12.7, and runs `uv sync --frozen --no-dev --group gpu` during the image build. The resulting `containers/rlaopt-cuda-13.0.2.sif` is intentionally ignored by Git. The build writes its SHA-256 digest once to the tracked `.sif.sha256` sidecar; array tasks read that small file instead of repeatedly hashing the 6.6 GB image over NFS. Preserve the SIF and checksum with the experiment artifacts; rebuilding from the same inputs is auditable, but the checksum proves which exact image a run used. If the base tag changes, run `scripts/resolve_cuda_digest.sh` and review the new digest before editing `containers/cuda.env`.

After the build succeeds, submit `sbatch slurm/check_cuda.sh` before the GPU smoke grid. It executes the baked environment and performs a float64 cuML LSMR fit alongside PyTorch and rlaopt on one H200 NVL. Inspect `cuda-check-<job-id>.out`; success ends with `PROJECT_ENVIRONMENT_INTEROPERABLE=true`. Both CPU and CUDA jobs submitted through `slurm/run_array.sh` use this SIF; CPU jobs omit `--nv`, so no GPU is exposed. Using one immutable image keeps package versions identical and avoids the severe metadata latency observed when importing PyTorch through thousands of files in a shared NFS virtual environment. SIF startup and checksum lookup occur outside the timed solver region and should be reported as orchestration overhead rather than solver runtime.

For the maximum-size CPU probe, use `/usr/bin/time -v` around one `run-job` command and compare its maximum resident set size with the `peak_memory_bytes` record. On CUDA, compare the recorded peak PyTorch allocation with `nvidia-smi` and scheduler accounting; allocator memory does not include every cuML/CUDA allocation.

Generate figures only after auditing failures:

```bash
uv run rlaopt-bench plot --input artifacts/records --output artifacts/figures
```

The figure command creates log-log runtime scatterplots for fixed-$p$, fixed-$n$, and square families, faceted by $\alpha$ and $\lambda$, plus a machine-readable failure summary. Paper analysis should additionally report iteration/matvec throughput, setup time, memory, convergence traces, effective dimension, condition numbers, and GPU-resident CPU/GPU speedups. Points from different spectral profiles or ridge values are never placed in the same panel.

## Synthetic ridge suite: expected cost and limitations

With all target nodes available concurrently, the sweep should take roughly 4–10 wall-clock hours; queueing, retries, and slow tail jobs make one to two days a realistic end-to-end allowance.

Limitations: the response lies in $\mathrm{range}(X)$ and has no observation noise; rank 128 is a fixed resource budget, not tuned per instance; CPU and GPU plots represent only the named machines; GPU-resident timing excludes transfer; and direct methods may exceed memory. Follow-up sensitivity studies can vary Nyström rank, add a controlled orthogonal/noisy response component, and measure end-to-end transfer costs.
