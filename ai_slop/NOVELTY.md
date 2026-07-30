# Novelty Statement

## What's actually new

Not the architecture (transformer for station→grid reconstruction already exists: Swin-TNP, Aardvark, ADAF). Not "local beats global conformal calibration" (LSCP and the Cluster-Aware DeepKriging paper already show that within a domain — see `LITERATURE.md` §9).

What's new is the combination, applied specifically to failure diagnosis:

1. **Systematic OOD evaluation protocol** for station→grid reconstruction — train West Europe → test East Europe (geographic transfer, not random holdout), plus extreme-event splits (heatwave/cold snap) and station-density ablations. Prior neural work tests in-distribution skill or random spatial holdout; none of it does controlled geographic transfer or extreme-event splits together.
2. **A classical baseline (kriging) throughout**, missing from nearly every recent neural paper in this space — so "is the transformer actually worth it OOD?" has an honest answer instead of being neural-vs-neural.
3. **Calibration-*transfer* diagnostic**: does a conformal calibration set (local or global) fit on one region/period retain its coverage guarantee when deployed on a distribution-shifted one (region, extreme period, or station density)? LSCP and Cluster-Aware DeepKriging test local vs. global calibration *within* an exchangeable domain — neither asks whether the calibration itself survives being moved outside that domain. Our three-arm design (Global / Local-transfer / Oracle-local) isolates coverage lost to transfer from coverage lost to plain model error.
4. **Classical vs. conformal UQ head-to-head** (kriging's native variance intervals vs. conformalized kriging vs. transformer CQR) under the same shift axes — no cited conformal-weather paper makes this comparison; they're all neural-only.

The throughline: this is an **evaluation and diagnosis paper**, not a new method paper. Every piece of machinery (transformer, kriging, split conformal, local calibration) is off-the-shelf; the contribution is the controlled experimental design that reveals where and how each piece breaks.

## Why this fits CCAI

- CCAI's call explicitly welcomes **"analysis of existing methods' limitations,"** not just new methods — this paper is that, directly.
- **Data-sparse regions** (the OOD-Spatial and OOD-Sparse splits) are a stand-in for the CCAI-relevant case: Africa, oceans, and other under-observed regions where climate justice applications most need reconstruction to work, and where it's hardest to verify that it does.
- **Extreme events** (the OOD-Extreme split) are exactly the high-stakes, low-frequency regime where operational climate/disaster-response decisions get made — and exactly where ML models are least tested and most likely to silently fail.
- **Actionable UQ, not just point accuracy**: a coverage-gap number ("trust this model's interval, don't trust that one, in this region/season") is more useful to a practitioner deciding whether to deploy than another RMSE table — this matches CCAI's applied, decision-relevant framing over pure benchmark chasing.
- Both outcomes are publishable at a workshop: "local calibration doesn't fix transfer failure" is as useful a finding for practitioners as "it does" — CCAI reviewers have historically accepted well-executed negative/cautionary results.
