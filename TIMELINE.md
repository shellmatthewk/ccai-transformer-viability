# Scope:

We have 28 days, due August 28th.

The entire scope we have:

3 Baselines - kriging, ConvCNP, Transformer x 3 Out of Distribution - spatial, extreme, sparse, x 2 variables (temp, precipitation) x 3 calibrations x classical and conformal UQ. 

# Priorities 

Take a look at decode grid Swin-TNP for pseudo tokens which makes the cost of everything so much better. https://arxiv.org/abs/2410.06731

## Priority 1:

Both variables, Kriging + Transformer, ID, OOD-Spatial, OOD-Extreme splits, Global/Local-transfer/Oracle-local on those OOD splits

## Priority 2:

ConvCNP baseline, OOD SPARSE + calibration, Classical + conformal UQ <- possible that UQ is included in priority 1

┌───────────┬────────────────────────────────────────────────────────────────────────┬───────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
│   Dates   │                              You (precip)                              │          Collaborator (temp)          │                                Shared / gate                                │
├───────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ Jul 30 –  │ Precip ECA&D+ERA5 pipeline, train/calib/val/test split code            │ Temp pipeline (mirrored) + kriging    │ Milestone: both pipelines produce tensors; kriging sanity-checks on ID      │
│ Aug 5     │                                                                        │ implementation                        │                                                                             │
├───────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ Aug 6 –   │ Transformer arch + quantile heads + precip zero-inflation handling     │ Kriging run across                    │ Gate: ConvCNP go/no-go by Aug 12 — if DeepSensor integration isn't clean by │
│ 12        │                                                                        │ ID/OOD-Spatial/OOD-Extreme (temp)     │  then, cut to P1 fallback, no second attempt                                │
├───────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ Aug 13 –  │ Transformer trained + run across P0 splits (precip)                    │ Transformer run across P0 splits      │ P0 results + 3-arm calibration on OOD-Spatial/OOD-Extreme. This week is the │
│ 19        │                                                                        │ (temp)                                │  actual deliverable — if only this gets done, you have a paper.             │
├───────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ Aug 20 –  │ P1 items only if P0 fully closed: OOD-Sparse, classical-vs-conformal   │ same                                  │ Gate: Aug 24 hard freeze on new experiments, no exceptions                  │
│ 24        │ table                                                                  │                                       │                                                                             │
├───────────┼────────────────────────────────────────────────────────────────────────┼───────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
│ Aug 25 –  │ Writing, figures, results tables, citation-checking (your own          │ same                                  │ Buffer absorbs slip; submit                                                 │
│ 29        │ LITERATURE.md flags citations as unreliable — verify before submit)    │                                       │                                                                             │
└───────────┴────────────────────────────────────────────────────────────────────────┴───────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘
(i did not know how to do this, this is roughly)
## Extra things:

Extra baseline - OR REPLACEMENT, use engression (noise based energy scoring system) (Shen & Meinshausen)

# Why is this useful? (because I keep on second guessing it)

Because something CCAI cares about is data-sparse regions and getting accurate predicitons from places like African and oceans is really valuable, as you can't run this sort of experiment with those locations. 

Computing and accessibility means that if a method isnt usable by a under resourced group, then why even bother? (climate justice)

Having transformer and non-neural network method is very useful as baselines as all the other papers don't use it. 