# Paper figures

Use the full regeneration command in [README.md](README.md#reproduce-the-paper-figures). It incorporates the additional SCS backends and the completed stricter-tolerance reruns, and writes PDFs, PNGs, and audit tables to `artifacts/paper-figures`.

The plotter reads saved results; it does not launch benchmarks. Use `--output` to write elsewhere. Original trial records are never modified.

## Figures

| File stem | Contents |
|---|---|
| `ridge_scaling` | Fixed-sample feature sweep at $n=2^{16}$ and $\lambda=10^{-6}$, on CPU and GPU |
| `ridge_preconditioning` | Matched-seed CG/PCG runtime ratios across the eight matrix shapes |
| `bounded_multinomial` | Real-data comparison with JAXopt JIT disabled |
| `bounded_elastic_net` | Real-data comparison with SCS indirect before direct on both backends |
| `cpu_gpu_speedup` | Matched CPU/GPU runtime ratios for the rlaopt methods |
| `appendix_bounded_multinomial_jit` | Multinomial comparison with JAXopt JIT enabled |
| `appendix_ridge_*` | Remaining ridge parameter and shape sweeps |
| `diff_solver_plot` | Lasso tuning objective versus iterations, with a logarithmic objective axis |

Full regeneration includes the saved paper differentiable example. To recreate that trace and update only its figure, follow the two commands in the README. These commands do not copy figures into the separate manuscript repository.

## Reading the plots

Runtime points require native solver success and passing the common accuracy checks. Ridge points show the median across three seeds with min-max whiskers; incomplete seed counts are annotated. Real-data points represent one cold solve. Timing boundaries and accuracy formulas are in [benchmark methods](docs/benchmark_methods.md).

Ridge refinement diamonds identify groups containing a stricter-tolerance result. Selection follows tolerance order, not minimum runtime. A point shows the qualifying solve's time; the audit also retains the cumulative cost of finding that tolerance. All 13 ridge refinements passed. The stricter GPU-direct SCS attempts on YearPredictionMSD-rf both timed out, so its final outcome is Timeout.

Timeout arrows sit at the solve limit: 900 seconds for ridge and 3600 seconds for real data. Iteration limits, memory exhaustion, index overflow, worker errors, and missing records have separate labels. A missing record alone is not evidence of out-of-memory failure. CPU/GPU ratios compare complete runs on different systems, not just GPU kernel execution. Hatched preconditioning cells and speedup arrows indicate timeout-based lower bounds.

The CPU SCS worker errors on ACSIncome-rf and YearPredictionMSD-rf mean that the worker connection closed before returning a solution. Diagnostic logs show `EOFError`, followed by `BrokenPipeError` during cleanup. Slurm reported failure, not an out-of-memory event; the underlying cause remains unknown. Those entries should be described as worker errors.

The largest CPU QR cases have attributable warmup OOM evidence. Nine failed warmup configurations prevented measurements for three ridge values each, accounting for 27 absent records. These are not 27 independently observed timed QR failures. Source log paths are retained in the audit.

## Audit files

| File | Contents |
|---|---|
| `ridge_outcomes.csv` | All 1,728 expected ridge outcomes, including missing records and residuals |
| `real_outcomes.csv` | All 90 expected outcomes, including the SCS supplement, native status, common checks, and failure evidence |
| `refinement_attempts.csv`, `.json` | Original and stricter-tolerance attempts, selection, tolerances, metrics, and cumulative measured solver cost |
| `speedup_values.csv` | Matched CPU/GPU ratios and lower-bound flags |
| `preconditioning_values.csv` | CG/PCG ratios, matched-seed counts, and lower-bound flags |
| `dataset_regimes.csv` | Processed dimensions, sparsity, and feature-generation metadata |
| `audit.json` | Eligibility policy, outcome counts, and hashes of consumed records, manifests, logs, configurations, and plotting code |

The original nonqualifying SCS solve and both follow-up timeouts remain separate audit entries. Their cumulative measured solver cost is 7565.56 seconds. A displayed timeout is never selected as a successful solve. Startup and warmup times are excluded from these measured solve totals.
