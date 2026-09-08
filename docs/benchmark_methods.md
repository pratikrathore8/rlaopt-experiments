# Benchmark methods

This document describes the experiments reported in the paper. The exact settings are in `configs/`; saved campaign configurations and trial records identify what was actually run.

## Ridge regression

We solve

$$
\min_w \frac12\|Xw-y\|_2^2+\frac\lambda2\|w\|_2^2.
$$

The dense matrix is constructed as $X=U\Sigma_\alpha V^T$, with singular values $\sigma_i=i^{-\alpha/2}$. The columns of $U$ and $V$ come from independent SORF orthogonal transforms. With $g$ standard Gaussian, the response is $y=Ug/\|g\|_2$. SORF is used to construct singular vectors here, not kernel features.

The grid has eight distinct shapes:

- Square: $n=p\in\{2^{10},2^{12},2^{14},2^{16}\}$.
- Fixed features: $p=2^{14}$ and $n\in\{2^{14},2^{15},2^{16}\}$.
- Fixed samples: $n=2^{16}$ and $p\in\{2^{10},2^{12},2^{14},2^{16}\}$.

Each shape uses $\alpha\in\{0.5,1,2\}$, $\lambda\in\{10^{-2},10^{-4},10^{-6}\}$, and seeds 0, 1, 2. Overlapping shapes are deduplicated. Nyström PCG uses rank 128. All solvers receive the materialized matrix, not the SORF factors.

CG and PCG apply the normal matrix through products with $X$ and $X^T$. LSQR and LSMR also avoid explicit Gram formation. LSQR uses damping $\sqrt\lambda$; cuML Ridge uses `alpha=lambda`, no intercept, and its LSMR solver. QR solves the augmented system with matrix $[X;\sqrt\lambda I]$ and target $[y;0]$.

The common relative residual is

$$
\frac{\|(X^TX+\lambda I)w-X^Ty\|_2}{\|X^Ty\|_2}\leq10^{-6}.
$$

Some legacy records call this field `relative_kkt`; it is the same metric. Data generation and initial device placement are outside the timer. One warmup is excluded: ten iterations for iterative methods, a full solve for QR. Measured time includes solver setup, preconditioning, and QR augmentation/factorization. Each seed has one measured solve, a 900-second solve limit, and a separate 900-second startup limit. Iterative methods allow at most $2p$ iterations.

## Bounded multinomial regression

We minimize mean multiclass negative log likelihood over coefficients $B\in[-1,1]^{p\times K}$, without an intercept. The datasets are CIFAR-10, SVHN, Fashion-MNIST, News20, and RCV1. SAPPHIRE is compared with JAXopt accelerated projected gradient and L-BFGS-B. JIT-enabled and JIT-disabled runs are labeled separately; JIT compilation is included in cold solve time.

## Bounded elastic net

We solve

$$
\min_{0\leq w\leq1,\ b\in\mathbb R}
\frac{1}{2n}\|Xw+b\mathbf1-y\|_2^2
+\lambda_1\|w\|_1+\frac{\lambda_2}{2}\|w\|_2^2.
$$

The intercept is unregularized. Both penalties are $0.1\lambda_{\max}$, where $\lambda_{\max}=\|X^T(y-\bar y\mathbf1)\|_\infty/n$. The datasets are ACSIncome-rf, YearPredictionMSD-rf, Yolanda-rf, E2006-tfidf, and Real-sim.

SCS and Clarabel use a residual-variable QP: introduce $r=Xw+b\mathbf1-y$, minimize $\|r\|^2/(2n)+\lambda_2\|w\|^2/2+\lambda_1\mathbf1^Tw$, and retain the box constraints. This avoids explicit Gram formation. Results describe this formulation, not an optimization over alternative formulations.

## Common bounded-problem checks

Every returned solution is checked in float64. Let $\delta=10^{-6}$ and $(a)_+=\max(a,0)$. For a coordinate $z$ with gradient $g$ and bounds $[\ell,u]$, its stationarity violation is $(-g)_+$ when $z\leq\ell+\delta$, $(g)_+$ when $z\geq u-\delta$, and $|g|$ otherwise. Feasibility is the maximum amount by which any coordinate exceeds its bounds. These are absolute maxima, without dimension-dependent scaling.

For multinomial regression, use $G=X^T(P-Y)/n$, where $P$ contains softmax probabilities and $Y$ contains one-hot labels. For bounded elastic net, use $g=X^Tr/n+\lambda_2w+\lambda_1\mathbf1$, and include $|\mathbf1^Tr/n|$ to check that the derivative with respect to the intercept is close to zero.

The stationarity threshold is $10^{-4}$ and the feasibility threshold is $10^{-6}$. No external duality gap is used. Native conic-solver gaps may still appear as diagnostic fields in records.

## Timing and hardware

Real-data experiments use seed 300, one cold solve, no warmup, at most 100,000 iterations, a 3600-second solve limit, and a separate 1800-second startup limit. Data loading, preprocessing, random features, and input-format conversion are outside the solve timer. Native setup, factorization, required internal transfers, and compilation inside the solver call are included. GPU timings synchronize device work.

Paper benchmark tasks receive 64 physical CPU cores and 128 GiB host memory. CPU nodes soal-8/9 use AMD EPYC 7763 processors. GPU node soal-12 uses AMD EPYC 9565 processors and H200 NVL GPUs, with one GPU per task. CPU-to-GPU ratios compare complete solver runs on those systems. If the CPU times out and the GPU qualifies, the ratio is reported as a lower bound.

## Tolerances and reruns

Initial native tolerances are `1e-6` for CG/PCG, `1e-9` for LSQR/LSMR, `1e-7` for SAPPHIRE, `1e-6` for JAXopt, `1e-7` for NysADMM/SCS, and `1e-10` for Clarabel/cuClarabel. Native stopping tests differ, so these numbers are not interchangeable.

Synthetic calibration searches for a tolerance that passes every common check on its calibration instances. The ridge calibration was performed before the production generator changed to SORF; its tolerances remained frozen. Bounded calibration uses separate synthetic data, not the real production datasets.

Production results are checked again. Native-successful results that miss the common threshold receive stricter-tolerance reruns with the same problem and resource limits. This is post-production refinement, not held-out calibration. The plotted time is the first qualifying attempt in tolerance order; it is not the total cost of finding that tolerance. `refinement_attempts.csv` retains all attempts and cumulative measured solver time. If none qualifies, the final completed attempt determines the displayed failure.

Six CPU LSQR and seven GPU LSMR results passed after tightening `1e-9` to `1e-10`. No `1e-11` attempt was needed. GPU direct SCS on YearPredictionMSD-rf returned a nonqualifying solution at `1e-7` after 365.56 seconds; `1e-8` and `1e-9` each timed out after 3600 seconds. Its final label is Timeout, and its cumulative measured cost is 7565.56 seconds. Original records are retained unchanged.
