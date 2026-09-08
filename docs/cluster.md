# Stanford cluster setup and execution

This guide records how we ran the benchmarks on Stanford's SC cluster. The account, partition, node names, and `/scr` storage paths below are specific to that environment. The supplied Slurm scripts are examples for that cluster, not portable submission commands.

To reproduce the figures on another machine, use the saved results and the single plotting command in the [README](../README.md#reproduce-the-paper-figures). No cluster access is required.

To rerun benchmarks elsewhere, start with the local manifest and execution commands in the [README](../README.md#run-experiments). Copy the configuration, set its data paths for your machine, and install the solver dependencies required by the selected backend. For a Slurm deployment, adapt the account, partition, node constraints, GPU requests, thread counts, and container/cache paths before submitting. The production planning scripts also encode soal node assignments and need adaptation. Preserve the problem settings, precision, accuracy checks, and timing boundaries described in [benchmark methods](benchmark_methods.md); record any hardware or resource differences alongside the new timings.

Run the commands below from the repository root on the Stanford SC cluster.

## Native solver environment

[containers/cuda.env](../containers/cuda.env) pins the CUDA base image and native solver sources. The derived Apptainer image includes the locked Python environment, four float64 SCS backends, Julia, Clarabel, and cuClarabel/cuDSS.

```sh
sbatch slurm/build_cuda.sh
```

Wait for the build to complete, then check the environment:

```sh
sbatch slurm/check_cuda.sh
sbatch slurm/check_cuclarabel.sh
```

The image and its `.sha256` sidecar identify the environment used by a run. Preserve them with campaign artifacts. Existing images contain a snapshot of the Python package; rebuild for changed package code, and retain the old image when reproducing old runs. CPU jobs use the same image without exposing a GPU. Julia/Clarabel checks use a separate process because native-library initialization order matters.

The environment pins rlaopt 0.1.0, PyTorch 2.13.0, NumPy 2.4.2, SciPy 1.18.1, cuML 26.8.0, JAXopt 0.8.5, JAX 0.11.1, SCS 3.2.11, Julia 1.10.12, and cuDSS 0.7.1. Clarabel/cuClarabel use revision `ffa325c89fa90b7e86b745fa61b1dca64daf3a06`. Inspect the saved record metadata for the exact image and versions used in a campaign.

## Data

The datasets and preprocessing are described in [datasets.md](datasets.md). `/scr` is node-local. The staging script downloads, prepares, and verifies each node’s cache, reusing existing verified files:

```sh
sbatch --nodelist=soal-8 slurm/stage_real_data.sh
sbatch --nodelist=soal-9 slurm/stage_real_data.sh
sbatch --nodelist=soal-12 slurm/stage_real_data.sh
```

Check the staging logs before submission. The data root in the configuration must match the cache visible on the execution node.

## Small ridge array

```sh
.venv/bin/rlaopt-bench manifest \
  --config configs/smoke.toml --backend cpu \
  --output artifacts/smoke-cpu.jsonl

BACKEND=cpu CONFIG=configs/smoke.toml \
MANIFEST=artifacts/smoke-cpu.jsonl OUTPUT_DIR=artifacts/smoke-campaign \
  sbatch --account=soal --partition=soal --nodelist=soal-8 \
    --array="0-$(($(wc -l < artifacts/smoke-cpu.jsonl)-1))" slurm/run_array.sh
```

For production ridge, generate manifests from `configs/synthetic.toml`. `scripts/shard_cpu_manifest.py` divides the CPU manifest between two nodes; `slurm/run_chunk_array.sh` executes groups of jobs per array element. Inspect each script's arguments before submission and use a fresh campaign directory.

## Real-data production

Create a plan first:

```sh
.venv/bin/python scripts/plan_real_production.py \
  --config configs/real_erm.toml \
  --output artifacts/new-real-campaign \
  --gpu-concurrency 2 --gpu-cpu-threads 64
```

The plan contains 40 jobs per backend: 25 multinomial and 15 bounded elastic-net jobs. It freezes the configuration, batches jobs into waves, and creates checksums. Review `plan.json`, then submit one wave:

```sh
scripts/submit_real_production_wave.sh artifacts/new-real-campaign 0
```

The launcher requires a clean Git checkout and verifies the frozen inputs. Wait for a wave to finish before submitting the next. Follow-up GPU campaigns use at most two jobs concurrently; CPU campaigns run one batch per node. Historical campaigns may have different concurrency, recorded in their plans.

Use `configs/real_erm_scs_backends.toml` for the additional SCS backends. The two `real_erm_scs_yearpredictionmsd_*.toml` files describe the completed stricter-tolerance attempts. Do not mix their outputs into the base campaign directory; the plotter merges them explicitly.

## Calibration and ridge refinement

`slurm/calibrate_synthetic_erm_cpu.sh` and `slurm/calibrate_synthetic_erm_cuda.sh` each launch six calibration tasks covering multinomial and bounded elastic net. Their resource settings are independent of production timing settings. Ridge calibration uses `configs/calibration.toml` and the corresponding CPU/CUDA calibration scripts.

`scripts/plan_ridge_refinement.py` prepares stricter-tolerance jobs from original nonqualifying records. `.venv/bin/python scripts/submit_ridge_refinement.py PLAN WAVE --dry-run` prints the submission commands; omit `--dry-run` to submit. Keep the original records and frozen plan with the results. The completed paper refinements are listed in the root README.
