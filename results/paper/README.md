# Saved paper results

These files reproduce the paper figures without rerunning solvers or downloading datasets. From the repository root, after installing the Python environment:

```sh
.venv/bin/python scripts/make_paper_figures.py
```

Outputs go to `artifacts/paper-figures`. The bundle contains the original ridge and real-data trials, the additional SCS backends, stricter-tolerance attempts, frozen configurations and manifests, failure-evidence logs, and the differentiable optimization trace. Failed and nonqualifying attempts are retained, not just successful measurements.

Campaign directory names match the original experiments. Trial records, summaries, configurations, and logs are copied byte-for-byte. Historical absolute paths in their metadata describe the original execution environment; plotting resolves refinement records within this bundle. The original figure audit is retained as provenance; each regeneration writes a new audit with the paths it actually used.

To verify the saved files:

```sh
cd results/paper
sha256sum --check SHA256SUMS
```

The checksums cover the saved inputs and original audit. Plotting code is versioned separately in Git. Do not overwrite this snapshot with new benchmark runs. Datasets, container binaries, W&B files, and unrelated development runs are excluded.
