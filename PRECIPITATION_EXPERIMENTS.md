# Precipitation reconstruction experiments

This pipeline mirrors the temperature study while keeping all precipitation checkpoints and
results separate. It uses the full available ECA&D precipitation context on each day and samples
ERA5 grid cells as reconstruction targets.

## Data and units

- Station input: `data/rr/daily_precipitation_2010_2021.parquet`
- ERA5 hourly source: `data/rr_era5/era5_hourly_rr/era5_tp_YYYYMM.nc`
- ERA5 daily target: `data/era5_daily_tp_mm_2010_2021.nc`
- Date range: 2010-01-01 through 2021-12-31
- ERA5 grid: 153 latitudes by 281 longitudes
- Daily target: sum of the 24 hourly ERA5 `tp` accumulations, converted from metres to
  millimetres per day
- Validation: every retained day must contain exactly 24 hourly fields

The generated target contains 4,383 complete days, no missing values, and no negative values.
A direct comparison for 2020-01-15 gave a maximum absolute difference of 0.0 mm between the
stored daily field and the raw 24-hour sum.

## Statistical model

Precipitation is nonnegative, strongly right-skewed, and has many exact zeros. ConvCNP and the
latent-grid transformer therefore consume and predict standardized `log1p(mm/day)` values.
The Gaussian NLL is optimized in that transformed space. The reported original-scale point
standard deviation includes a smooth 0.1 floor in normalized transformed space. This prevents
near-zero initial variance from destabilizing ConvCNP while preserving gradients and records the
floor in every checkpoint so evaluation applies the identical predictive scale. The
estimate is the shifted-lognormal predictive mean:

`max(0, exp(mu_log + sigma_log^2 / 2) - 1)`.

This is more appropriate than fitting a Gaussian directly to raw precipitation, but it is not an
explicit zero-inflated distribution. Wet-day Brier score and critical success index at the
1 mm/day threshold are reported so that failures around dry/wet occurrence remain visible.

Conformal residuals are calibrated in `log1p` space. Their interval endpoints are transformed
back to nonnegative mm/day, giving asymmetric prediction intervals on the physical scale.

## Protocol

- Training: 2010-2017
- Validation and early stopping: 2018
- Conformal calibration: 2019
- Test: 2020-2021
- Seeds: 2026 through 2033
- Experiments: global reconstruction and western-source to eastern-target transfer
- Context densities: full network and the same nested sparse retentions used for temperature
- Baseline: tuned local-neighbor ordinary kriging in `log1p` space
- Calibration comparisons: global/source transferred calibration and oracle target-region
  calibration, including matched-density and deployed-full-density variants

Heavy-precipitation days are the upper 5 percent of target-region daily spatial-mean
precipitation anomalies. The seasonal reference is a 31-day calendar climatology fitted only on
2010-2017. This is one-sided: unusually dry days are not labeled as heavy precipitation.

Uncertainty is summarized across days and across independent training seeds. Grid target samples
within a day are not treated as independent replicates. The evaluator reports day-bootstrap
confidence intervals plus the day-clustered 90x90 interval criterion.

## Main metrics

- RMSE and MAE in mm/day
- Gaussian NLL in standardized `log1p` space
- Wet-day Brier score and critical success index at 1 mm/day
- Raw and conformal 90 percent pointwise coverage and interval width
- Day-clustered 90x90 coverage and width
- Fraction of test days with at least 90 percent spatial coverage
- Overall and heavy-precipitation-day subsets

## Run order

From the project directory with the virtual environment available:

1. Prepare the ERA5 target if it is not already present:

   `scripts\prepare_precipitation_targets.cmd`

2. Train both models on the two GPUs:

   `scripts\run_precipitation_training.cmd`

   The latent transformer uses physical GPU 0 (RTX 4080 SUPER). ConvCNP uses physical GPU 1
   (RTX 5060 Ti). Each script skips a checkpoint that already has `best.pt`, so interrupted
   runs can be restarted.

3. After all checkpoints exist, evaluate all eight seeds:

   `scripts\run_precipitation_evaluation.cmd`

4. Run the CPU local-neighbor ordinary-kriging baselines:

   `scripts\run_precipitation_kriging.cmd`

5. Once neural and kriging evaluation outputs are complete, create the strict final summary:

   `scripts\summarize_precipitation_results.cmd`

The summarizer refuses to produce a final table unless every expected neural seed and all three
kriging configurations are present. This prevents an incomplete run matrix from being mistaken
for final results.
