# Cluster setup and execution

Run commands from the repository root. The Slurm scripts target the Stanford soal cluster; adapt account, partition, nodes, and local storage paths for another system.

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

Prepare the base cache as described in [datasets.md](datasets.md). `/scr` is node-local, so stage it on each worker node before launching real-data jobs:

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
  sbatch --array="0-$(($(wc -l < artifacts/smoke-cpu.jsonl)-1))" slurm/run_array.sh
```

For production ridge, generate manifests from `configs/synthetic.toml`. Use `scripts/shard_cpu_manifest.py` and `slurm/run_chunk_array.sh` to group jobs rather than submit hundreds of array elements. Inspect each script's arguments before submission and use a fresh campaign directory.

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
