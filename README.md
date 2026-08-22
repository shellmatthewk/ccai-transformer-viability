# CCAI transformer viability

Climate-field reconstruction experiments comparing ordinary kriging, ConvCNP, and a latent-grid
transformer on ECA&D station observations with ERA5 reference targets.

The repository contains source code and launch scripts only. Climate datasets, checkpoints, and
generated experiment results are intentionally excluded from Git.

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
