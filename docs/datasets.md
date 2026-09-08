# Datasets and preprocessing

The paper uses training data only. Fashion-MNIST uses the first 60,000 official training rows of its 70,000-row OpenML file. The source URLs, expected shapes, and checksums are defined in [data.py](../src/rlaopt_experiments/suites/real_erm/data.py).

| application | dataset | source | base training shape | solver shape | feature map |
|---|---|---|---:|---:|---|
| bounded elastic net | ACSIncome | OpenML 43141 | $1{,}664{,}500\times11$ | $1{,}664{,}500\times1{,}000$ | Gaussian, nominal bandwidth 1 |
| bounded elastic net | E2006-tfidf | LIBSVM training file | $16{,}087\times150{,}360$ | unchanged | none |
| bounded elastic net | Real-sim | LIBSVM | $72{,}309\times20{,}958$ | unchanged | none |
| bounded elastic net | YearPredictionMSD | LIBSVM training file | $463{,}715\times90$ | $463{,}715\times4{,}367$ | ReLU |
| bounded elastic net | Yolanda | OpenML 42705 | $400{,}000\times100$ | $400{,}000\times1{,}000$ | Gaussian, nominal bandwidth 1 |
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

## Random features

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
matrices are never stored. Benchmark workers call this function before solver timing,
and feature generation and transfer are recorded separately.


The same feature values are used across solvers. Sparse input is retained by adapters that support it; rlaopt materializes dense arrays. No standardization is applied after the random-feature map. The intercept in bounded elastic net is unregularized.

## Preparing data

```sh
scripts/prepare_real_data.sh --data-root /scr/pratikr/rlaopt-real-data --all
```

Use `--dataset NAME` instead of `--all` to prepare a subset. The command saves the source files, processed arrays, and checksums. Existing verified files are reused. On the cluster, `/scr` is local to each node: stage the cache on every execution node using `slurm/stage_real_data.sh`. See [cluster instructions](cluster.md).
