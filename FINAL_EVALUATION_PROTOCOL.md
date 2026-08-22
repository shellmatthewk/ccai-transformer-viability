# Final evaluation protocol

This pipeline supersedes the original extreme-event rows and the earlier 256-station kriging baseline. It does not overwrite any existing checkpoint or result.

## What changes

- Neural models receive all available in-domain ECA&D stations, or deterministic nested 75%, 50%, 25%, and 10% retained networks.
- Evaluation and calibration seeds are fixed independently of the training seed, so every architecture and training seed sees the same station subsets and sampled ERA5 targets.
- Extreme days are defined separately for each evaluated target region. Exact ERA5 daily spatial means are converted to anomalies relative to a 31-day calendar-window climatology fitted on 2010-2017. Days outside the training anomaly 5th/95th percentiles are extreme. This uses no 2020-2021 information to set thresholds.
- The corrected definition selects 106 of 731 global test days and 104 of 731 eastern-target test days.
- Every summary includes a 2,000-replicate bootstrap confidence interval obtained by resampling dates. Grid targets within a day are not treated as independent replicates.
- Neural variation is measured over eight training seeds, 2026 through 2033. Existing checkpoints are reused and later seeds are stored separately.

## Conformal prediction

All conformal scores use the 2019 calibration split and normalized absolute residuals:

`s = abs(y - mean) / predicted_std`.

The pointwise 90% interval is `mean +/- q * predicted_std`, where `q` is the finite-sample split-conformal quantile.

Two density regimes are reported:

- `deployed_full_density`: calibrate at 100% station retention and deploy unchanged at every density. This measures calibration loss after station-network thinning.
- `matched_density`: calibrate and test at the same retained station fraction. This measures how much density-specific recalibration can recover.

Three geographic regimes are reported:

- `global`: global model, global calibration, global test.
- `transferred`: western/source model, western/source calibration, eastern target test.
- `oracle_target`: the same source-trained predictions, calibrated in the eastern target region. This is a diagnostic upper bound, not a deployable no-target-data method.

The output also includes day-clustered 90-by-90 conformal intervals. For each calibration day, it computes the within-day 90% score quantile and then conformalizes those daily quantiles across days at 90%. Under exchangeable days, this targets at least 90% spatial coverage on at least 90% of days. It is intentionally more conservative than pointwise marginal conformal.

## Ordinary kriging

`evaluate_local_kriging.py` is the comparable full-network CPU baseline.

- It finds the nearest stations independently for each ERA5 query using a KD-tree.
- It uses ordinary kriging with an exponential covariance and a 5% nugget.
- Neighborhood size is selected from 16 and 32 stations.
- Length scale is selected from 200, 400, and 800 km.
- Hyperparameters minimize mean daily RMSE on 30 dates spread across the 2018 validation split.
- It uses the same retained ECA&D networks, ERA5 targets, dates, extreme labels, and conformal regimes as the neural models.

A realistic 32-neighbor, 1,024-target benchmark processed about 22 days per second on this machine. The three complete kriging evaluations should therefore take roughly 15-25 minutes in total, with file loading and tuning overhead included.

## Run order

### 1. Train all additional seeds

Seed 2026 is the original run. Seeds 2027 and 2028 use:

```cmd
scripts\run_final_multiseed_training.cmd
```

Seeds 2029 through 2033 use:

```cmd
scripts\run_five_more_seeds_training.cmd
```

The latent transformer runs on physical GPU 0 (RTX 4080), while ConvCNP runs on physical GPU 1 (RTX 5060 Ti). Every checkpoint is stored under its own seed, region, and model directory; existing runs are never overwritten.

### 2. Run corrected neural evaluation

Evaluate seeds 2026 through 2028 with:

```cmd
scripts\run_final_neural_evaluation.cmd
```

After seeds 2029 through 2033 finish, evaluate only those new checkpoints with:

```cmd
scripts\run_five_more_seeds_evaluation.cmd
```

Together these commands evaluate all eight seeds for both models and write results under `final_results\neural`.

### 3. Run full-network local kriging

This CPU job can run independently, though running it after neural training avoids unnecessary data-I/O contention:

```cmd
scripts\run_final_kriging.cmd
```

Results are written under `final_results\kriging`.

### 4. Build paper tables

After neural and kriging evaluation finish:

```cmd
scripts\summarize_final_results.cmd
```

The summary directory contains:

- `all_results.csv`: every model, seed, region, density, subset, and calibration regime.
- `neural_seed_summary.csv`: mean, sample standard deviation, and t interval across the eight training seeds.
- `headline_results.csv`: full-network and 10%-network deployed-calibration rows.

Each individual evaluation also writes a `.daily.csv` file containing one row per date. These files are the appropriate inputs for paired day-level comparisons and plots.

## Interpretation limits

- ERA5 remains a reference analysis product, not error-free ground truth.
- Eight training seeds characterize optimizer and initialization variation; report the individual seeds as well as the mean, standard deviation, and paired-seed differences.
- Pointwise conformal coverage is empirical under spatial and temporal dependence. The day-clustered result is the more defensible dependence-aware reliability summary.
- Oracle target calibration uses target-region calibration labels and must not be described as a zero-shot deployment method.

