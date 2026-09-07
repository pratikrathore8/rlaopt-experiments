# Results Presentation Plan

## Central message

The experiments should present `rlaopt` as a unified randomized-linear-algebra framework rather than claim that it is the fastest solver in every setting. The main empirical conclusion is:

> `rlaopt` supports smooth, composite, and constrained optimization within one framework. Its methods are most useful for ill-conditioned dense problems and large dense GPU workloads, while specialized sparse, direct, or JIT-compiled solvers remain preferable when the problem matches their structure.

The paper should identify both the regimes in which `rlaopt` is useful and those in which a specialized method is a better choice. This is more informative and defensible than presenting the benchmark as a universal solver competition.

## Evidence by problem class

| Problem and regime | Main observation | Role in the paper |
|---|---|---|
| Ill-conditioned dense least squares | Nyström PCG provides the clearest performance advantage, particularly as conditioning and scale make unpreconditioned or direct methods expensive. | Primary algorithmic performance result |
| Tall, dense bounded elastic net | Nyström ADMM is robust on GPU and is the only successful tested GPU method on ACSIncome and YearPredictionMSD. It also outperforms SCS on RealSim and Yolanda. | Primary constrained-optimization and GPU result |
| Sparse bounded elastic net | cuClarabel can be dramatically faster when sparse KKT factorization is favorable; E2006 is the clearest example. | Defines the sparse/direct-solver regime where `rlaopt` is currently less useful |
| Box-constrained multinomial logistic regression | SAPPHIRE obtains meaningful CPU-to-GPU acceleration but is slower than the specialized JIT-enabled JAXopt methods. | Demonstrates modeling breadth and honestly delineates a current performance limitation |

### Least squares

The least-squares experiments should lead the results section because they provide the strongest direct evidence for randomized preconditioning. Emphasize:

- runtime scaling with samples and features;
- sensitivity to spectral decay and conditioning;
- the CPU-to-GPU transition; and
- successful accuracy at the calibrated stopping criteria.

The claim should be that Nyström PCG is especially effective for large, dense, ill-conditioned systems. It should not be generalized to sparse systems that were not tested through a native sparse `rlaopt` path.

### Bounded elastic net

The bounded elastic-net results show a clear structural split:

- On dense, tall ACSIncome and YearPredictionMSD, GPU ADMM succeeds while the tested conic competitors time out or encounter implementation limits.
- On dense, tall Yolanda, ADMM outperforms SCS, although cuClarabel is faster.
- On sparse RealSim, cuClarabel is fastest, but ADMM still outperforms SCS.
- On sparse, extremely wide E2006, SCS and cuClarabel succeed while ADMM times out.

This should be described as a robustness and regime result, not an across-the-board runtime victory. A suitable conclusion is:

> Nyström ADMM is particularly effective for large, tall, dense learning problems. Sparse direct methods can be substantially faster when they preserve favorable sparsity, but their conic formulations can encounter memory, index-capacity, or runtime failures on large dense instances.

The sparse results require an explicit implementation caveat: the current `rlaopt` execution path materializes the input as a dense PyTorch tensor, whereas SCS and Clarabel receive CSC conic matrices and can recover exact sparsity. The observed sparse-data disadvantage should therefore be presented as a limitation of the current data path, not necessarily an inherent limitation of Nyström preconditioning.

### Box-constrained multinomial logistic regression

The projected-gradient baseline is **APG (accelerated projected gradient)**. Both JIT-enabled and JIT-disabled JAXopt use its default `acceleration=True` (FISTA-style acceleration); disabling JIT does not disable acceleration. All 19 production APG records with returned solver metadata confirm acceleration; one timed-out worker has no returned metadata. Figure legends use “JAXopt APG.”

SAPPHIRE is not the raw-runtime winner against idiomatic JIT-enabled JAXopt. The main figure shows explicitly labeled JIT-disabled JAXopt, with the JIT-enabled comparison in an appendix figure. The main text must state that JIT-enabled JAXopt is substantially faster. Improving the computational performance of `rlaopt` is an important direction for future work. The non-JIT comparison alone does not support a claim of superiority over JAXopt generally.

The defensible conclusion is:

> SAPPHIRE obtains substantial CPU-to-GPU acceleration and is competitive with some non-JIT baselines, but specialized JIT-compiled JAXopt methods are substantially faster on the tested problems.

Do not claim that the current experiments demonstrate an advantage when full gradients are infeasible. SAPPHIRE's stochastic structure makes that a plausible target regime, but the present datasets allow the full-gradient JAXopt methods to run successfully. This point should be described as motivation or future work unless an explicit large-scale experiment is added.

## Cross-cutting claims

The results support the following claims:

1. Randomized preconditioning can provide a substantial advantage for large, dense, ill-conditioned least-squares problems.
2. The same design philosophy extends beyond linear systems to nonsmooth constrained models through Nyström ADMM.
3. GPU execution materially changes which large dense problems `rlaopt` can solve within the resource and time limits.
4. `rlaopt` provides a common modeling and execution framework across problem classes that otherwise require different specialized baselines.
5. Performance depends strongly on matrix geometry, sparsity, conditioning, and the availability of specialized compiler or factorization paths.

Avoid the absolute claim that no competing method can apply to all three problem classes. General conic and differentiable-optimization frameworks can represent broad classes of problems. The narrower supported claim is:

> No single baseline was both a natural formulation and a consistently competitive implementation across all three problem classes.

## Figures and tables

The main paper should favor a small number of figures with a clear purpose:

1. **Least-squares scaling:** runtime against problem scale, separated by spectral decay and backend. Mark timeouts and memory failures rather than assigning them artificial runtimes.
2. **Bounded elastic-net comparison:** grouped runtime by dataset and backend, with distinct markers for success, timeout, memory exhaustion, index-capacity failure, and solver/runtime failure.
3. **CPU-to-GPU speedup:** compare only matched, accuracy-eligible runs. This answers a different question from comparison against other GPU solvers: competitor comparisons measure relative GPU performance, whereas this panel measures whether GPU execution changes `rlaopt`'s useful operating range. Define speedup as

   $$
   \text{speedup}=\frac{T_{\mathrm{CPU}}}{T_{\mathrm{GPU}}}.
   $$

   Use exact ratios only when both matched runs reach their frozen calibrated native stopping criteria. When the CPU run reaches the one-hour limit and the GPU run succeeds, show a right-pointing arrow and report only the lower bound $3600/T_{\mathrm{GPU}}$. Do not assign the CPU run an observed runtime of 3,600 seconds. The panel groups Nyström PCG, Nyström ADMM, and SAPPHIRE. Ridge ratios are formed per matched seed before taking medians and min–max ranges.
4. **Multinomial comparison:** show explicitly labeled JIT-disabled JAXopt baselines in the main figure and the JIT-enabled results in an appendix. Discuss the substantial JIT advantage in the main text.

Use LaTeX math rendering and Computer Modern fonts throughout. Dataset labels are lowercase, with `-rf` appended when random features were applied. The structural headings are simply “dense” and “sparse.” Filled points represent calibrated native successes; external diagnostic disagreements do not change their appearance. Real-data outcome matrices below the runtime panels display every unsuccessful configuration. See [FIGURES.md](FIGURES.md) for generated artifacts, captions, and the reproduction command.

A compact regime table should accompany the figures and identify each dataset as dense or sparse, tall or wide, and transformed or untransformed. This connects the observed performance directly to the computational explanation.

Figure revisions use circular success markers and a fixed color mapping for solvers, “NysADMM” and “H200” as display labels, and samples-by-features dimensions below each real dataset name. Time-limit lines have explicit y-axis ticks. Ridge failures are named by solver and cause with seed counts, replacing ambiguous codes. The largest CPU QR configurations are labeled “Warmup OOM,” supported by matching worker failures and scheduler OOM counts; their measured records were never produced. Preconditioning heatmaps cover every tested shape and show direct speedup ratios on a linear scale. The seven real-data nonconverged records stopped at 100,000 iterations and are labeled “Iteration limit,” separately from timeouts. Explanatory prose belongs in the paper captions rather than inside the figures, including the meanings of dimensions, OOM, arrows, and hatching. Speedup labels are centered over arrows in logarithmic display coordinates.

### Current CPU-to-GPU examples

The bounded elastic-net production results currently give censored lower bounds for GPU ADMM because every matched CPU ADMM run reached the one-hour limit:

| Dataset | GPU runtime | CPU-to-GPU speedup |
|---|---:|---:|
| ACSIncome | 3,093 s | greater than $1.16\times$ |
| RealSim | 579 s | greater than $6.2\times$ |
| YearPredictionMSD | 811 s | greater than $4.4\times$ |
| Yolanda | 965 s | greater than $3.7\times$ |

E2006 should not appear as an ADMM speedup because neither backend completed successfully. For SAPPHIRE, the matched accuracy-eligible examples include approximately $3.2\times$ on CIFAR-10 and $4.6\times$ on News20. RCV1 changes from a CPU timeout to a successful 490-second GPU solve, giving a lower bound of approximately $7.4\times$. Runs such as SVHN for which native and external success classifications disagree should not enter the primary speedup summary; their statuses can be reported separately.

The least-squares suite provides the largest set of exact matched CPU/GPU ratios and anchors this panel. Captions must identify the configured 64-core CPU system and H200 GPU system. These measure the recorded solver invocation: ridge includes setup but excludes generation and data transfer; real ERM includes internal setup, required solver transfers, and JIT compilation when enabled, but excludes data preparation and benchmark-side conic construction. These are not isolated kernel speedups, and the GPU system has device memory in addition to host memory.

## Success and failure reporting

Runtime comparisons include runs that satisfy the predetermined calibrated native solver criterion. External accuracy checks are retained for diagnostics but do not filter these runtime points. Reaching an external threshold without native convergence does not qualify a run. Native and external statuses remain separately available in the exported data.

Unsuccessful outcomes should retain their actual classification:

- timeout;
- host out-of-memory;
- device or sparse-index capacity failure;
- solver/runtime exception; or
- completed without satisfying the required accuracy.

Do not replace these outcomes with the timeout value in runtime plots. Report the fixed resource envelope explicitly, including CPU count, host memory, GPU model and device memory, precision, solver timeout, and whether compilation or setup is included in each timed region.

## Limitations to state explicitly

- The real-data production grid uses one solver seed and one regularization setting, so it characterizes selected computational regimes rather than estimating broad statistical variability.
- `rlaopt` currently materializes sparse real datasets densely.
- The multinomial experiments do not reach a scale at which full-gradient methods become infeasible.
- JAXopt performance is sensitive to JIT compilation; the main non-JIT comparison must be accompanied by the JIT-enabled appendix and a clear main-text acknowledgment of its faster performance.
- Direct conic methods can be exceptionally fast when factorization structure is favorable, but their memory and index requirements can prevent them from handling large dense conic formulations.

## Suggested results-section structure

1. Begin with least squares as the clearest algorithmic success.
2. Use bounded elastic net to demonstrate extension to constrained composite optimization and the importance of GPU execution.
3. Present multinomial logistic regression as a breadth experiment and a transparent comparison against highly optimized specialized methods.
4. Conclude with the regime map: dense versus sparse, tall versus wide, and general framework versus specialized implementation.

The final discussion should emphasize that the experiments are intended to locate useful operating regimes, not manufacture a clean sweep. One clear algorithmic win, one strong constrained-optimization robustness result, one breadth result, and well-explained limitations form a coherent empirical contribution.

## Accuracy-refinement procedure in the paper

Use the methods wording in README.md under “Post-production accuracy refinement.”
Calibration chooses the initial tolerance; it does not certify unseen production
instances. The follow-up trigger is native success/normal completion plus a miss
of the fixed common accuracy criterion, applied symmetrically to all solvers.
The audit identifies 13 ridge measurements and the SCS GPU-direct YearPredictionMSD-rf
measurement. Preserve the original results, disclose the post-production timing of
the decision, and report all attempts. Distinguish the first qualifying solve's
runtime from cumulative measured solver time and refinement wall time. Never
silently treat a native success as an external accuracy pass, or describe the
refinement as held-out calibration. State that ridge refinement preserves its
original warmup policy, and disclose the changed container digest.
