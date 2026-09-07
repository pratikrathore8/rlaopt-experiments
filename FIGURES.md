# Production figures

To reproduce the original frozen-protocol figures from the repository root
(use the [accuracy-refinement command below](#figures-incorporating-accuracy-refinement)
for the current ridge and bounded elastic-net figures):

```bash
MPLCONFIGDIR=/tmp/pratikr/mpl-paper .venv/bin/python scripts/make_paper_figures.py
```

The script takes `--ridge`, `--real`, and `--output` arguments. Defaults are the frozen campaigns `artifacts/production-20260831`, `artifacts/real-erm-production-20260905`, and output directory `artifacts/paper-figures`. The ridge manifest is expanded using `configs/synthetic.toml`; the script rejects duplicate and unexpected trials. No solver runs or dataset downloads are required. Matplotlib's built-in LaTeX math renderer and Computer Modern fonts avoid a system TeX dependency. Every figure is exported as a vector PDF and a 220-dpi PNG.

## Main figures and suggested captions

The following paragraphs are paper captions, not text embedded in the figures. Explanations of dimensions, OOM, timeout arrows, and hatching belong here. Figure legends use “NysADMM.” Speedup values are centered visually over their arrows on the logarithmic axis.

### Ridge scaling

[PDF](artifacts/paper-figures/ridge_scaling.pdf) · [PNG](artifacts/paper-figures/ridge_scaling.png)

Solver runtime for ridge regression with $n=2^{16}$ and $\lambda=10^{-6}$ as the feature dimension increases. Columns vary spectral decay $\alpha$; rows show 64-core CPU and H200 execution. Points are median runtimes over three seeds that reached their calibrated native stopping criteria; whiskers span their minimum and maximum, not confidence intervals. Any incomplete success count is annotated. The dotted line and labeled y-axis tick denote the 900-second (15-minute) solve limit. Named failure annotations identify the solver, cause, and affected seed count; they have no runtime coordinate. For example, “PyTorch QR: GPU OOM (3/3 seeds)” explains why its runtime point is absent. “No record” means no retained result, not confirmed OOM. Full ridge-parameter and shape sweeps appear in the appendix. Timing includes solver setup and excludes problem generation and data transfer.

`ridge_scaling` is the main-paper name for the fixed-sample feature sweep at $\lambda=10^{-6}$, not an additional experiment. `appendix_ridge_fixed_n_*` shows the same shapes at the other ridge values; `appendix_ridge_fixed_p_*` varies samples with $p=2^{14}$; `appendix_ridge_square_*` varies $n=p$. All show both backends and all three spectral profiles.

The largest CPU QR cases are now annotated “Warmup OOM.” The nine affected manifest configurations all failed while the parent was waiting for a full QR warmup, and the six corresponding Slurm chunk logs report nine OOM kills in total, with matching failure counts in each chunk. Their failures prevented measured runs for all three ridge values, explaining the 27 absent records. These are not 27 independently observed timed QR failures. The audit exports `WARMUP_OOM` and the source log path; the raw records remain absent. Generic “No record” is reserved for missing outcomes without this attributable evidence.

### Ridge preconditioning

[PDF](artifacts/paper-figures/ridge_preconditioning.pdf) · [PNG](artifacts/paper-figures/ridge_preconditioning.png)

Median matched-seed runtime ratio $T_{\mathrm{CG}}/T_{\mathrm{Nystr\ddot{o}m\ PCG}}$ across all eight tested matrix shapes, both backends, and all tested $\alpha$ and $\lambda$. Values greater than one favor Nyström PCG. Annotations and the shared linear color scale show the ratio directly, without a logarithmic transformation. Exact cells require matched native successes on all seeds. CG timeouts yield lower bounds based on 900 seconds; these cells are hatched and excluded from the quantitative color scale. Lower-bound labels are rounded downward. Every shape also has a separate two-panel PDF/PNG for flexible paper layout.

### Bounded multinomial logistic regression

[PDF](artifacts/paper-figures/bounded_multinomial.pdf) · [PNG](artifacts/paper-figures/bounded_multinomial.png)

Solver runtimes on five real datasets, with one seed and one production parameter setting. APG means accelerated projected gradient: JAXopt retains its default acceleration in both JIT configurations. The main comparison uses explicitly labeled JIT-disabled JAXopt. Filled circular markers indicate calibrated native success, with a stable color per solver throughout the figures. Dimensions below dataset names give samples $\times$ features. Upward arrows at the labeled one-hour boundary indicate censored solve times. Outcome matrices use full outcome names. All seven production cases previously labeled native nonconvergence reached the 100,000-iteration limit before the one-hour timeout: SAPPHIRE on fashion-mnist and svhn on both backends, and APG on fashion-mnist on CPU with JIT and GPU with/without JIT. They are labeled “Iteration limit,” even if an external diagnostic passed. The JIT-enabled appendix is substantially faster; this figure alone does not establish superiority over JAXopt generally. Improving `rlaopt` computational performance remains future work.

### Bounded elastic net

[PDF](artifacts/paper-figures/bounded_elastic_net.pdf) · [PNG](artifacts/paper-figures/bounded_elastic_net.png)

Solver runtimes for bounded elastic net, grouped by dense and sparse data. Dataset names ending in `-rf` use random features. Each configuration uses one seed and one regularization setting. Successful times are plotted numerically; timeouts are arrows at the one-hour boundary. Outcome matrices distinguish native success, timeout, native nonconvergence, host/device memory exhaustion, index overflow, solver/worker error, and missing records. Nyström ADMM succeeds on large dense GPU problems where the tested conic implementations encounter resource or runtime limits, while sparse conic factorization can be much faster on sparse data. The current `rlaopt` path materializes sparse input densely, whereas the conic paths retain exact sparsity. The updated figure includes SCS indirect first and SCS direct second in both
CPU and GPU panels. Runtime eligibility and completed refinement outcomes follow
the accuracy-refinement caption below. The “Worker error” entries for CPU SCS on acsincome-rf and yearpredictionmsd-rf mean that the isolated worker connection closed before a solution or measured runtime was returned; the cause of termination is undetermined. These entries do not establish native nonconvergence or out-of-memory failure.

### CPU/GPU runtime ratios

[PDF](artifacts/paper-figures/cpu_gpu_speedup.pdf) · [PNG](artifacts/paper-figures/cpu_gpu_speedup.png)

Matched CPU/GPU solver runtime ratios for the three `rlaopt` methods. Ridge fixes $n=2^{16}$ and $\lambda=10^{-6}$; each point is the median of three seed-specific ratios, with min–max whiskers. Real-data points are individual matched runs. Right-pointing arrows give lower bounds $3600/T_{\mathrm{GPU}}$ when the CPU solve timed out and the GPU solve succeeded. Configurations without an eligible GPU result or with non-timeout CPU failures are excluded. Hardware is a 64-core CPU system versus an H200 NVL system; these are recorded solver invocation ratios, not kernel-only measurements. Ridge timing is device-resident. Real-data timing includes internal solver setup, required transfers, and JIT compilation when enabled, but excludes dataset loading, random features, and benchmark-side conic construction and format conversion. The GPU system has device memory in addition to host memory.

## Appendix and audit data

- `appendix_bounded_multinomial_jit`: same runtime/status layout with JIT enabled. Solver-side compilation is included in the recorded invocation time.
- `appendix_ridge_*`: every remaining combination of shape family and ridge parameter, each showing all three spectral decays and both backends. The hardest fixed-sample panel is the main `ridge_scaling` figure. Shared endpoints appear in each applicable family.
- `real_outcomes.csv`: all 80 expected trials, with native status, external success, runtime, plotted classification, and failure-evidence path. Failed-run elapsed values remain available here but are never plotted as successful runtimes.
- `ridge_outcomes.csv`: all 1,728 expected trials, including missing records and external KKT diagnostics.
- `speedup_values.csv`: individual matched ratios and explicit lower-bound flags.
- `preconditioning_values.csv`: ratios for all eight matrix shapes, both backends, all spectral profiles and ridge parameters; matched-seed counts and lower-bound flags are explicit.
- `dataset_regimes.csv`: dimensions after random features, dense/sparse classification, geometry, and feature-generation metadata. This supplies the paper's dataset table.
- `audit.json`: outcome counts, eligibility policy, and SHA-256 hashes of source records, manifests, diagnostic logs, configuration, and plotting code.

CSV status codes are `OK` for calibrated native success, `TO` for timeout, `ITER` for iteration limit, `NC` for other native nonconvergence, `HOST` for host OOM, `GPU` for device OOM, `IDX` for index overflow, `ERR` for solver/worker error, and `MISS` for a missing record without attributable failure evidence. Figures spell out these outcomes. Native successes are not filtered by external diagnostic thresholds. A missing record alone is not evidence of OOM: CPU host OOM labels require an attributable scheduler event, and the yearpredictionmsd-rf GPU OOM label uses the diagnostic rerun's explicit device-allocation failure. The two CPU SCS worker failures remain worker errors rather than inferred OOMs.


SCS worker-error evidence: the diagnostic reruns
`artifacts/real-erm-production-20260905/logs/rerun-scs-acsincome-17282392_26.out`
and `artifacts/real-erm-production-20260905/logs/rerun-scs-yearpredictionmsd-17282396_35.out`
both show `EOFError` while receiving the solver response, followed by a secondary
`BrokenPipeError` during worker cleanup. Slurm reports `FAILED`, exit code `1:0`,
for both reruns, not `OUT_OF_MEMORY`. The recorded batch MaxRSS values are
119,109,168 KiB and 65,115,053 KiB respectively, against a 128 GiB request; these
sampled memory values do not prove or exclude a transient memory failure. No
worker exit signal was retained, so the caption leaves the underlying cause open.

Additional checks on 2026-09-06 found `/var/crash` empty on both CPU nodes.
Kernel journal and `dmesg` access require privileges unavailable to this session;
no historical worker signal could be recovered. Evidence is retained in
`artifacts/scs-failure-logs-17300671.out`, `artifacts/scs-failure-logs-17300673.out`,
`artifacts/scs-crash-reports-17300717.out`, and `artifacts/scs-crash-reports-17300718.out`.
A controlled rerun with explicit child exit-code/signal capture and native SCS
logging would be needed to investigate a reproducible cause without administrator logs.

## Figures incorporating accuracy refinement

To update only the main ridge scaling and bounded elastic-net figures:

```bash
MPLCONFIGDIR=/tmp/pratikr/mpl-paper .venv/bin/python scripts/make_paper_figures.py \
  --accuracy-refinement \
  --ridge-refinement artifacts/ridge-tolerance-refinement-20260906 \
  --real-supplement artifacts/real-erm-scs-backends-20260906-64threads \
  --real-refinement artifacts/scs-yearpredictionmsd-tolerance-20260906/1e-8 \
  --real-refinement artifacts/scs-yearpredictionmsd-tolerance-20260906/1e-9
```

These two figures update the existing files in `artifacts/paper-figures`.
Original experiment records remain unchanged. By default this mode updates only these two figures. Add `--all-figures`
to regenerate every existing main and appendix figure with the same refined
results and eligibility rule; this includes preconditioning, multinomial, and
CPU/GPU speedup figures. No separate cost figure is generated. Rerunning the command incorporates newly completed trials; an attempt is
included only once its trial summary is saved. Until all planned trials finish,
this is a provisional snapshot. A trial with no completed refinement retains its
original outcome. If no attempt passes, the displayed outcome comes from the last
completed attempt; pending runs are never inferred to have timed out.

**Refined ridge caption:** Points summarize the first attempted native tolerance
that meets both native success/completion and the fixed common relative KKT target
of `1e-6`, for each seed. Diamonds mark groups containing at least one refined
measurement; whiskers and incomplete-seed counts retain their existing meanings.
The original calibrated setting is tried first, followed by the documented finite
refinement ladder only for native-success accuracy misses. Selection follows
attempt order, not minimum runtime. All attempts, original measurements, selected
tolerances and cumulative measured solver cost are retained in the accompanying
audit files. The displayed runtime is the qualifying solve's runtime, not the
cumulative cost of finding that setting. This is a disclosed post-production
refinement study, not held-out calibration.

**Refined bounded elastic-net caption addition:** The CPU and GPU panels include
both SCS direct and indirect backends. Runtime points require both native success
and the unchanged common stationarity and feasibility checks. “Accuracy miss”
means native success without satisfying those checks and has no qualifying runtime
coordinate. SCS GPU-direct on YearPredictionMSD-rf returned native success in
365.56 seconds at `1e-7` but missed both common targets; follow-ups at `1e-8` and
`1e-9` each reached the 3,600-second solve timeout without returning a solution.
The figure therefore labels the refined outcome **Timeout** and places an arrow
at the one-hour limit. This states that the tested stricter setting did not return
a qualifying solution within the budget; it does not assert that every possible
tolerance would time out. The original accuracy miss and both timeouts remain
separate entries in the audit.
Their aggregate measured solver cost, including the original attempt, is 7,565.56
seconds. Worker-error causes and attributable OOM labels retain the evidence rules
above. CPU-indirect results still in progress remain “No record.”

`refinement_attempts.csv` and `.json` retain each original/refinement attempt, native
and external status, residuals, tolerance, runtime, selection flag, and cumulative
measured cost. The outcome CSVs identify refined measurements, selected tolerance,
attempt count, last refinement status, displayed tolerance, and original outcome.
The attempt audit distinguishes the displayed attempt from an accuracy-qualified
selected attempt; a timeout is displayed but never selected as a successful solve. `audit.json` records the eligibility rule
and hashes all consumed records, completed summaries, manifests and plotting code.
Startup and warmup wall times remain available in the ridge trial summaries; they
are excluded from the cumulative measured solver times in the plotting audit.

## Differentiable optimization

[PDF](artifacts/paper-figures/diff_solver_plot.pdf) · [PNG](artifacts/paper-figures/diff_solver_plot.png)

The lasso tuning objective versus iterations, with a logarithmic objective axis. Run `scripts/run_differentiable_optimization.py`, then `scripts/make_paper_figures.py --differentiable-only`. The latter updates the existing figure and trace CSV; full figure regeneration also includes the saved result.
