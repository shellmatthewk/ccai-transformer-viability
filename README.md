# CCAI transformer viability

## Paper

**Climate Field Reconstruction Under Station Sparsity and Geographic Shift**  
**Authors:** Matthew Shell and Varun Panuganti  
**Venue:** NeurIPS 2026 Workshop on Tackling Climate Change with Machine Learning

- [Camera-ready paper](Climate_Field_Reconstruction_Under_Station_Sparsity_and_Geographic_Shift_Camera_Ready.pdf)
- Official NeurIPS records: [oral presentation](https://neurips.cc/virtual/2026/loc/sydney/oral/174598) · [poster presentation](https://neurips.cc/virtual/2026/loc/sydney/poster/174597)
- [OpenReview entry](https://openreview.net/forum?id=WHbp01Qvtw)

### Research question

We study whether compact neural models improve climate-field reconstruction from sparse weather stations and remain reliable under geographic shift. We compare a latent-grid transformer and ConvCNP with local ordinary kriging, using ECA&D station observations and ERA5 reference fields.

The experiments assess temperature and precipitation reconstruction, station-network thinning, geographic transfer, and uncertainty calibration. The task is same-day reconstruction: station observations from the reconstruction date are provided as inputs.

### Evaluation design

- **Chronological partitions:** 2010–2017 for training, 2018 for validation, 2019 for calibration, and 2020–2021 for testing.
- **Geographic transfer:** the source model is trained in the western region, at or west of 15°E, and evaluated in the eastern target region.
- **Value normalization:** means and standard deviations are fitted using training-period observations from the selected training region.
- **Separate calibration and testing:** source-calibrated transfer and target-region recalibration are reported separately. Target-region recalibration uses labelled target calibration data and is a distinct diagnostic condition.
- **Uncertainty evaluation:** interval coverage and interval width are reported alongside point-prediction error. ERA5 is treated as a reference analysis product.

### Methods and supporting artifacts

- [Final evaluation protocol](FINAL_EVALUATION_PROTOCOL.md)
- [Temporal and geographic partition definitions](experiment_data.py)
- [Station selection and value normalization](full_context_data.py)
- [Neural evaluation implementation](evaluate_full_context_v2.py)
- [Compact paper results](paper_results/README.md)

Climate-field reconstruction experiments comparing ordinary kriging, ConvCNP, and a latent-grid
transformer on ECA&D station observations with ERA5 reference targets.

The repository includes experiment code, paper figure scripts, compact result tables, and final
paper figures. Climate datasets, checkpoints, full experiment runs, point-level Parquet exports,
and locally installed third-party packages are intentionally excluded from Git.

## Quick start

Create and activate the project virtual environment, install `requirements.txt`, and place the
ECA&D and ERA5 inputs at the paths documented in the experiment guides.

For the prepared precipitation workflow on Windows:

```cmd
scripts\preflight_precipitation.cmd
scripts\run_precipitation_training.cmd
```

Run evaluation and summarization only after all expected training checkpoints have completed.
