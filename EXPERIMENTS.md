# Final experiment pipeline

The earlier `checkpoints/*.pt` files are pilot checkpoints trained on all years.
The final pipeline does not overwrite them.

Temporal partitions:

- training: 2010-2017
- validation: 2018
- conformal calibration: 2019
- final test: 2020-2021

The western/source region is longitude at or west of 15 degrees E. The target
region is east of 15 degrees E. Source training filters both station contexts
and ERA5 target cells, preventing target-region leakage.

Run from Command Prompt:

```bat
scripts\run_global_50epochs.cmd
scripts\run_source_50epochs.cmd
scripts\run_sparse_evaluation.cmd
```

Wait for both global jobs before starting the source jobs, and wait for all four
training jobs before starting evaluation. Training stops before 50 epochs when
validation NLL has not improved for eight epochs after epoch ten. `best.pt` is
the checkpoint used for evaluation; `last.pt` is retained for diagnostics.

Sparse evaluation uses deterministic nested daily contexts: 256, 192, 128, 64,
and 26 stations. All models receive the same context identities and target-cell
samples for a given region and split.

The reported extreme subset is presently defined as test days whose sampled
regional ERA5 spatial mean lies outside the 1st or 99th percentile of training
regional daily means. This is a reproducible temperature-extreme definition,
not an externally curated event catalogue.
