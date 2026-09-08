# rlaopt experiments

Code and results for the experiments in the rlaopt paper. The repository compares CPU and GPU solvers on three problems and includes a small differentiable optimization example.

| Experiment | Methods compared | Configuration |
|---|---|---|
| Synthetic ridge regression | Nyström PCG, CG, QR, CPU LSQR, GPU LSMR | [synthetic.toml](configs/synthetic.toml) |
| Bounded multinomial regression | SAPPHIRE and JAXopt APG/L-BFGS-B, with and without JIT | [real_erm.toml](configs/real_erm.toml) |
| Bounded elastic net | NysADMM, SCS indirect/direct, Clarabel/cuClarabel | [real_erm.toml](configs/real_erm.toml), [SCS supplement](configs/real_erm_scs_backends.toml) |
| Differentiable optimization | Tune a lasso penalty by differentiating through proximal gradient | [Python example](scripts/run_differentiable_optimization.py) |

Unbounded elastic-net benchmarking is no longer supported because it is not part of the paper. The lasso differentiation example is a separate demonstration and remains included.

## Setup

Use Python 3.12 and uv 0.12.7. From the repository root:

```sh
uv sync --frozen
```

This creates `.venv` from the pinned dependencies in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). For development tools, use `uv sync --frozen --group dev`.

The paper's cluster runs use an Apptainer image with CUDA, all four SCS backends, and Julia/cuClarabel. Installing the local environment alone does not build those native GPU interfaces. See [cluster setup and execution](docs/cluster.md).

## Reproduce the paper figures

The saved paper results are included in [results/paper](results/paper/README.md). After setup, regenerate all figures with:

```sh
.venv/bin/python scripts/make_paper_figures.py
```

This uses the original trials, additional SCS backends, completed stricter-tolerance reruns, and saved differentiable optimization trace. No solver runs or dataset downloads are needed. PDFs, PNGs, plotted values, and an input audit are written to **`artifacts/paper-figures`**. [FIGURES.md](FIGURES.md) explains the figures and outcome labels.

Use `--paper-results PATH` to relocate the saved bundle, or explicit `--ridge`, `--real`, and refinement arguments to plot another campaign. `--output PATH` changes the figure destination. The manuscript lives in the separate `rlaopt-paper` repository; this plotter does not update it automatically.

Datasets, container images, and unrelated experiment outputs remain outside Git.

### Differentiable optimization

This small example needs no downloaded dataset or GPU:

```sh
.venv/bin/python scripts/run_differentiable_optimization.py
.venv/bin/python scripts/make_paper_figures.py --differentiable-only \
  --differentiable artifacts/differentiable-optimization/result.json
```

The script uses fixed synthetic data and saves its objective trace. The plot has a linear iteration axis and a logarithmic objective axis. Pass the same `--differentiable` option to include a newly generated trace in full figure regeneration. The default uses the saved paper trace.

## Run experiments

Start with a small CPU ridge run to check the environment:

```sh
WANDB_MODE=offline .venv/bin/rlaopt-bench run-job \
  --config configs/smoke.toml \
  --n 256 --p 256 --alpha 1 --seed 0 \
  --solver scipy_lsqr --backend cpu \
  --output artifacts/smoke-local
```

For the bounded problems, generate a manifest and execute one entry:

```sh
.venv/bin/rlaopt-bench manifest \
  --config configs/synthetic_erm_smoke.toml --backend cpu \
  --output artifacts/erm-smoke/cpu.jsonl

.venv/bin/rlaopt-bench run-manifest-job \
  --config configs/synthetic_erm_smoke.toml \
  --manifest artifacts/erm-smoke/cpu.jsonl --index 0 \
  --output artifacts/erm-smoke/results
```

A manifest lists the exact problem, seed, solver, and backend for every job. Use the same configuration when generating and executing it. Synthetic bounded problems are used for smoke tests and tolerance calibration; the paper's bounded-problem results use real datasets.

For full runs, prepare the [datasets](docs/datasets.md) and follow the [cluster workflow](docs/cluster.md). Write each new campaign to a new output directory so it cannot overwrite completed results. Use `rlaopt-bench --help` for all commands.

## How to interpret the results

A run counts as successful only if the solver finishes successfully **and** its returned solution passes the common accuracy check:

| Problem | Common check |
|---|---|
| Ridge | Relative residual ≤ `1e-6` |
| Bounded multinomial / bounded elastic net | Stationarity ≤ `1e-4` and feasibility violation ≤ `1e-6` |

The ridge results summarize three seeds using the median and min-max range. Each real-data result is one cold solve. Timeouts, iteration limits, memory failures, index overflow, and worker errors are reported separately.

Native tolerances are calibrated before production. Runs that finish successfully but fail the common check are rerun at stricter native tolerances. The figures use the first qualifying attempt in tolerance order; the attempt ledger also reports the total cost of the original run and reruns. All 13 ridge refinements passed. The two stricter GPU-direct SCS attempts on YearPredictionMSD-rf timed out.

See [benchmark methods](docs/benchmark_methods.md) for the mathematical problems, timing boundaries, and tolerance procedure.

## Repository guide

| Location | Contents |
|---|---|
| `configs/` | Paper configurations, synthetic smoke/calibration grids, and rerun tolerances |
| `src/rlaopt_experiments/` | Data preparation, problem definitions, solver adapters, runners, and records |
| `scripts/` | Figure generation, production planning, refinements, and environment diagnostics |
| `slurm/`, `containers/`, `julia/` | Cluster jobs and native solver environment |
| `tests/` | Checks for models, adapters, manifests, execution, and plotting |
| `results/paper/` | Saved paper inputs and checksums, included in Git |
| `artifacts/` | New local results and generated figures; ignored by Git |

Atomic JSON trial records are the source of truth. W&B is optional and defaults to offline mode. Old campaign configurations remain readable for the retained problem families; their retired unbounded settings do not create jobs.

## Development checks

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

GPU- and Julia-specific behavior also requires the environment checks described in the cluster guide. CPU unit tests alone do not validate those native backends.
