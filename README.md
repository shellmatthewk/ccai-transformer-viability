# CCAI transformer viability

Climate-field reconstruction experiments comparing ordinary kriging, ConvCNP, and a latent-grid
transformer on ECA&D station observations with ERA5 reference targets.

The repository includes experiment code, paper figure scripts, compact result tables, and final
paper figures. Climate datasets, checkpoints, full experiment runs, point-level Parquet exports,
and locally installed third-party packages are intentionally excluded from Git.

## Paper artifacts

- [Paper reproduction guide](PAPER_REPRODUCIBILITY.md): tables, figures, dependencies, and data requirements.
- [Original paper results](paper_results/README.md): temperature and precipitation summaries.
- [Final paper figures](analysis/generated_figures): station sparsity, geographic transfer, and evaluation design.
- [Later development evidence](paper_results/development/README.md): the separate kriging/Engression
  pilot, including its negative transfer result. These results do not replace the original paper tables.

The manuscript itself is not included in this checkout. Versioning these artifacts does not mean
the manuscript has been revised or that the later development experiments are publication-ready.

## Experiment guides

- [Temperature full-context experiments](FULL_CONTEXT_EXPERIMENTS.md)
- [Final temperature evaluation protocol](FINAL_EVALUATION_PROTOCOL.md)
- [Precipitation experiments](PRECIPITATION_EXPERIMENTS.md)
- [CUDA and GPU selection](CUDA.md)
- [Dataset statistics](DATASET_STATS.md)

## Quick start

Create and activate the project virtual environment, install `requirements.txt`, and place the
ECA&D and ERA5 inputs at the paths documented in the experiment guides.

For the prepared precipitation workflow on Windows:

```cmd
scripts\preflight_precipitation.cmd
scripts\run_precipitation_training.cmd
```

Run evaluation and summarization only after all expected training checkpoints have completed.
