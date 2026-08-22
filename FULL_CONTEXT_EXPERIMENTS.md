# Full-context experiments

These scripts supersede the earlier 256-context neural scripts. They filter
ECA&D observations to the ERA5 grid bounds and use every available station in
the selected geographic region on each day.

The latent-grid transformer cross-attends from a 16 by 16 grid of 256 inducing
tokens to all station observations, processes the inducing tokens with three
self-attention layers, and decodes ERA5 target queries by cross-attention. Its
station-scaling term is linear rather than quadratic in the station count.

Run in order from Command Prompt:

```bat
scripts\run_full_context_global_50epochs.cmd
scripts\run_full_context_source_50epochs.cmd
scripts\run_full_context_evaluation.cmd
```

Wait for both jobs at each training stage. Outputs are written beneath
`experiments_full_context`, leaving pilot and 256-context outputs unchanged.

Sparse evaluation retains 100, 75, 50, 25, and 10 percent of each day's full
in-region station network using fixed nested station-ID rankings. Consequently,
the exact context count varies by date and is recorded as minimum, median, and
maximum in every results row.
