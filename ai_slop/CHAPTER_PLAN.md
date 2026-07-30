# Chapter Plan: When Does Learned Station-to-Grid Reconstruction Fail?

**Venue**: CCAI Workshop (NeurIPS 2026?2027?)
**Format**: 4-page paper
**Deadline**: August 29, 2026 AoE
**Team**: 2 NERDS 

---

# DO NOT TRUST CITATIONS THEY DONT WORK HALF THE TIME AND CAN LEAD TO VERY VERY VERY BAD THIGNS 

## Working Title

*When Does Learned Climate Field Reconstruction Fail? A Controlled Study of Spatial Transfer and Extreme Events*

**Alternative**: *Out-of-Distribution Degradation in Neural Station-to-Grid Climate Reconstruction*

---

## Contribution Reframing (Critical)

**Original claim (no longer viable)**:
> "First transformer architecture for direct station→grid climate reconstruction"

This claim is invalidated by existing work: Swin-TNP, ConvCNP/DeepSensor, Aardvark Weather, ADAF, Manshausen et al., FieldFormer, SLAMS. See `LITERATURE.md` for full analysis.

**New claim (defensible)**:
> "Systematic evaluation of when learned station→grid reconstruction degrades: controlled study of spatial transfer, extreme events, and data sparsity with transformer, ConvCNP, and kriging baselines"

**Why this works**:
1. Prior work reports ID skill; we systematically test OOD failure modes
2. Most prior work lacks classical baselines (kriging)—we include it
3. Random spatial holdout ≠ systematic geographic transfer (train West → test East)
4. Extreme event splits are largely absent from prior evaluations
5. CCAI explicitly welcomes "analysis of existing methods' limitations"

---

## Division of Labor

| Person | Variable | Responsibilities |
|--------|----------|------------------|
| You (ML/modeling strength) | Precipitation | Transformer architecture, training pipeline, handling zero-inflation/skewness, joint model integration |
| Collaborator (math strength) | Temperature | Data pipeline, kriging baseline, theoretical framing, smoother-field experiments |

**Shared**: ConvCNP baseline (via DeepSensor), writing, figures, OOD evaluation design

---

## Chapter 1: Introduction (~0.5 pages)

### Key Points
1. Learned methods for station→grid climate reconstruction have proliferated (Swin-TNP, Aardvark, ADAF, ConvCNP)—but how do they fail?
2. Prior work reports in-distribution skill; systematic OOD evaluation is sparse
3. Climate applications demand reliability in data-sparse regions and during extreme events—exactly where models are least tested
4. **Contribution**:
   - Controlled OOD evaluation: spatial transfer (train West EU → test East EU), extreme events (heatwaves, cold snaps), data sparsity ablations
   - Classical baseline (kriging) absent from most recent neural work
   - Analysis of failure modes: where does transformer attention break down?

### INSIGHT-1
Frame as *evaluation contribution*, not architecture contribution. "We use a representative transformer architecture (similar to Swin-TNP/Aardvark encoder) to study failure modes."

### Evidence Needed
- Citation: Swin-TNP, Aardvark, ADAF, ConvCNP/DeepSensor (establish prior work)
- Citation: these papers report ID metrics but lack systematic OOD splits

---

## Chapter 2: Related Work (~0.25 pages)

### Four threads to cover
1. **Classical DA**: Kalnay (2003), Carrassi et al. (2018) review
2. **Neural station→grid methods**:
   - Swin-TNP (Qu et al., 2024): 20% random pixel holdout, no kriging
   - Aardvark Weather (Nature 2025): end-to-end from obs, no kriging, no spatial transfer
   - ADAF (Xiang et al., 2025): tropical cyclone demo, no kriging
   - Manshausen et al. (2025): 40 stations, left-out test, no kriging
   - ConvCNP/DeepSensor: kriging baseline exists in some applications
3. **Evaluation gap**: ID temporal holdout is standard; systematic spatial transfer and extreme event splits are rare
4. **Set/point-cloud transformers**: Lee et al. (2019) Set Transformer—architectural foundation

### INSIGHT-2 (Revised)
Gap statement: prior work handles irregular inputs well but **evaluation is limited to ID settings**. Our contribution is the OOD evaluation protocol, not the architecture.

### Positioning Table

| Paper | Spatial OOD | Extreme Events | Kriging Baseline | Local/Regional Conformal UQ |
|-------|-------------|----------------|------------------|------------------------------|
| Swin-TNP | Random 20% holdout | No | No | No |
| Aardvark | Not reported | Not reported | No | No |
| ADAF | Sparse sensitivity | Tropical cyclones | No | No |
| Manshausen | Left-out stations | No | No | No |
| ConvCNP/DeepSensor | Varies | No | Sometimes | No |
| LSCP (Jiang & Xie, ICML 2026) | No (within-domain locality only) | No | No | Yes, but not under region transfer |
| Cluster-Aware DeepKriging (Kim et al. 2026) | No (spatial clusters, same domain) | No | No | Yes, incl. global fallback for sparse clusters — but not under region transfer |
| **Ours** | Train West → Test East | Heatwave/cold snap splits | **Yes** | Calibration-*transfer* test: does calibration fit on one region/period survive deployment on a shifted one? |

**Note (see `LITERATURE.md` §9)**: LSCP and the cluster-aware DeepKriging paper already show local beats global calibration *within* a domain — that specific claim is no longer novel. Our angle is calibration validity under distribution shift (spatial, extreme, density), which neither tests.

---

## Chapter 3: Method (~0.75 pages)

### 3.1 Problem Formulation
- **Input**: Set of observations {(lat_i, lon_i, t, var_i, value_i)} — variable-length, irregular
- **Output**: Gridded field F(lat, lon, t, var) on regular 0.25° grid matching ERA5

### 3.2 Architecture (Representative, Not Novel)

We use a standard set-transformer encoder + cross-attention decoder, similar to architectures in Swin-TNP and Aardvark's encoder module.

```
┌─────────────────────────────────────────────────────┐
│  Observation Tokens                                 │
│  (lat, lon, t, var_type, value) → MLP → token      │
│  + sinusoidal positional encoding for (lat,lon,t)  │
│  + learned embedding for var_type                  │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Transformer Encoder                                │
│  Self-attention over observation tokens             │
│  Captures spatial + cross-variable correlations     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  Grid Decoder                                       │
│  Query tokens = regular grid positions (lat,lon,t) │
│  Cross-attention: queries attend to encoded obs    │
│  Output MLP → scalar prediction per grid cell      │
└─────────────────────────────────────────────────────┘
```

### 3.3 Conformal Prediction for Uncertainty Quantification

**Goal**: Prediction intervals [L(x), U(x)] with coverage guarantee P(Y ∈ [L, U]) ≥ 1 - α.

**Method**: Split conformal with conformalized quantile regression (CQR)
1. Train model with quantile heads (pinball loss for α/2 and 1-α/2)
2. On calibration set, compute scores: s_i = max(q_low - y, y - q_high)
3. Find threshold Q̂ = (1-α)(1 + 1/n)-quantile of scores
4. Predict: [q_low - Q̂, q_high + Q̂]

### 3.4 Conformal Calibration Under Distribution Shift (KEY ABLATION, reframed)

**Caveat (see `LITERATURE.md` §9)**: "Local calibration beats global" is already shown by LSCP (Jiang & Xie, ICML 2026) and the cluster-aware DeepKriging paper (Kim et al. 2026) — including the same global-fallback-for-sparse-regions idea we'd have used for the East region. That specific claim is not our contribution. What neither paper tests is whether a calibration fit — local or global — **survives being deployed outside the domain it was calibrated on**. That's the question this ablation now answers.

**Design — three arms, not two**:

| Arm | Calibration set | Deployed on | Purpose |
|-----|-----------------|-------------|---------|
| **Global** | All of Europe, 2019 | ID + all OOD splits | Baseline: single Q̂, ignores region and shift |
| **Local-transfer** | West Europe only, 2019 | East Europe (OOD-Spatial), extreme periods (OOD-Extreme) | Realistic deployment: calibration was never possible in the target region/period |
| **Oracle-local** | East Europe, 2019 (or extreme-period analogue) | Same region/period it was calibrated on | Undeployable upper bound — shows how much coverage loss is attributable to *transfer* specifically vs. residual model error |

The gap between Local-transfer and Oracle-local is the number that matters: it isolates "coverage lost to calibration-transfer" from "coverage lost to the model just being wrong OOD."

**Second axis — calibration validity under density shift**: OOD-Sparse (§4.3) drops 50%/75% of stations at test time. Split conformal's exchangeability assumption is violated not just by region/period shift but by observation-density shift too — the residual distribution calibration was fit on (dense 2019 network) differs from the one it's applied to (sparse test network). Re-run the Global/Local-transfer/Oracle-local comparison under OOD-Sparse as a third, orthogonal stress test (spatial × temporal-extreme × density).

**Third comparison — classical vs. conformal UQ**: Kriging produces variance estimates natively from the variogram (no conformal step needed). Compare (a) raw kriging variance intervals, (b) conformalized kriging (kriging point prediction + CQR-style calibration), and (c) the transformer's CQR intervals, across all three shift axes. No cited conformal-weather paper (Camps-Valls et al., the neural-operator/PINN conformal papers) makes this classical-vs-conformal comparison — they're neural-only.

**Why this matters**: Climate fields are spatially heterogeneous (coastal vs. continental, mountains vs. plains), and climate applications need to know not just "is this method locally well-calibrated" but "can I trust calibration done somewhere else when I deploy here."

### 3.5 Training Details
- **Loss**: MSE + pinball loss for quantiles
- **Precipitation handling**: Two-stage (occurrence classifier + amount regressor) or transformed target (log1p)
- **Augmentation**: Random station dropout to simulate sparser networks
- **Calibration set**: 2019 held out for conformal calibration

### INSIGHT-3 (Revised 2026-07)
Architecture is not the contribution—evaluation protocol + uncertainty calibration is. Local vs. global calibration by itself is **not** novel (LSCP, Cluster-Aware DeepKriging already show it — see `LITERATURE.md` §9). What's novel: using local/global calibration as a diagnostic for whether conformal coverage survives spatial transfer, extreme events, and density shift — a transfer question neither prior paper asks — plus a classical-vs-conformal (kriging variance vs. CQR) UQ comparison neither makes.

---

## Chapter 4: Experimental Setup (~0.75 pages)

### 4.1 Data
- **Observations**: ECA&D (European Climate Assessment & Dataset) daily stations
  - Temperature: ~3000 stations over Europe
  - Precipitation: ~5000 stations (more variable coverage)
- **Target**: ERA5 0.25° reanalysis, matched variables/times
- **Region**: Europe (well-observed, diverse climate zones)
- **Period**: 2010–2019 train, 2020 val, 2021 test

### 4.2 Baselines

| Baseline | Description | Rationale |
|----------|-------------|-----------|
| **Ordinary Kriging** | Classical geostatistics, variogram-based | Absent from most prior neural work; honest classical yardstick |
| **ConvCNP** (via DeepSensor) | Neural process baseline | State-of-the-art neural method with available code; stronger than ConvLSTM-on-IDW |
| **Transformer** | Our implementation | Representative of recent approaches (Swin-TNP, Aardvark encoder) |

**Dropped**: ConvLSTM on IDW-interpolated grid (reviewers would view as strawman)

### 4.3 Evaluation Splits (Core Contribution)

| Split | Description | Rationale |
|-------|-------------|-----------|
| **ID** | Random held-out days in 2021 | Standard; all methods should perform well here |
| **OOD-Spatial** | Train Western Europe (lon < 10°E) → Test Eastern Europe (lon > 15°E) | Systematic geographic transfer; gap region ensures no spatial leakage |
| **OOD-Extreme** | Held-out heatwave (June 2021) and cold snap (Feb 2021) periods | Extreme events are where climate applications need reliability most |
| **OOD-Sparse** | Ablation: drop 50%, 75% of training stations | Tests robustness to data-sparse conditions; also a density-shift stress test for conformal calibration validity (§3.4) |

### 4.4 Metrics

**Point prediction**:
- RMSE, MAE, bias
- Spatial correlation with ERA5
- **Degradation ratio**: OOD RMSE / ID RMSE (quantifies failure severity)

**Uncertainty quantification**:
- **Empirical coverage**: % of true values within 90% prediction interval
- **Interval width**: Mean width (sharpness—tighter is better if coverage holds)
- **Coverage gap**: |90% - empirical coverage| (calibration quality)

**Ablation**: Compare global vs. local conformal coverage

### INSIGHT-4
OOD-spatial split speaks to CCAI relevance: data-sparse regions (Africa, oceans) are where reconstruction quality matters most for climate justice applications. If models fail on West→East Europe transfer, they'll fail harder on truly data-sparse regions.

---

## Chapter 5: Results (~1.25 pages)

### 5.1 Main Results Table

| Method | ID RMSE (T/P) | OOD-Spatial RMSE | OOD-Extreme RMSE | Degradation Ratio |
|--------|---------------|------------------|------------------|-------------------|
| Kriging | — | — | — | — |
| ConvCNP | — | — | — | — |
| Transformer | — | — | — | — |

**Hypothesis**: All methods perform similarly ID; differences emerge OOD. Transformer may win ID but degrade more than kriging OOD (or vice versa—both findings are publishable).

### 5.2 Degradation Analysis
- **Figure**: Performance vs. distance from training region boundary
- **Figure**: Performance on extreme percentiles (top 5% hottest days, coldest days) vs. normal days
- **Table**: Degradation ratio by method × split

### 5.3 Conformal Calibration Under Distribution Shift (KEY ABLATION, reframed)

| Arm | ID Coverage | OOD-Spatial Coverage | OOD-Extreme Coverage | OOD-Sparse Coverage |
|-----|-------------|---------------------|---------------------|----------------------|
| Global | 90% (by construction) | —% | —% | —% |
| Local-transfer | 90% (by construction) | —% | —% | —% |
| Oracle-local (upper bound) | 90% (by construction) | —% | —% | —% |

**Hypothesis**: Global and Local-transfer both lose coverage OOD relative to Oracle-local; the size of that gap quantifies coverage loss attributable to calibration-transfer specifically (as opposed to model error).

**Classical vs. conformal UQ**: separate table comparing raw kriging variance intervals, conformalized kriging, and transformer CQR across the same shift axes.

**Figure**: Coverage map of Europe showing where Global calibration fails (coverage < 85%) but Local-transfer/Oracle-local succeed — annotated with the transfer gap.

### 5.4 Attention Analysis (If Space Permits)
- Do attention weights decay with distance? Does this break down OOD?
- Cross-variable attention: do temp queries attend to precip tokens? Does this help or hurt?

### 5.5 Failure Mode Characterization
- Where does each method fail?
  - Kriging: likely fails on non-stationary fields (fronts, gradients)
  - ConvCNP: may struggle with extreme values outside training distribution
  - Transformer: may overfit spatial patterns that don't transfer
- **Actionable insight**: Which method should practitioners choose for which use case?

### INSIGHT-5
Both positive and negative results are publishable:
- "Transformer wins OOD" → neural methods are reliable for climate applications
- "Kriging wins OOD" → caution warranted; classical methods remain competitive
- "All methods degrade similarly" → fundamental limits of current approaches

---

## Chapter 6: Discussion & Conclusion (~0.5 pages)

### Key Findings Summary
- [TBD based on results]

### Limitations
- ERA5 is imperfect ground truth (itself an interpolation)
- Single region (Europe); generalization to other continents untested
- Two variables only; doesn't cover full atmospheric state

### Implications for Climate Applications
- Data-sparse regions: which method degrades most gracefully?
- Extreme events: which method should be trusted during heatwaves?
- Operational guidance for practitioners

### Future Work
- Uncertainty quantification (critical for operational use)
- Multi-region study (Europe, CONUS, data-sparse Africa)
- Physics-informed constraints to improve OOD robustness

---

## Implementation Timeline (5 weeks)

| Week | You (Precipitation) | Collaborator (Temperature) | Shared |
|------|---------------------|---------------------------|--------|
| 1 | Data pipeline: ECA&D precip + ERA5 | Data pipeline: ECA&D temp + ERA5 | Align formats, define OOD splits |
| 2 | Transformer architecture + precip-specific handling | Kriging baseline implementation | ConvCNP via DeepSensor |
| 3 | Training runs on precip | Training runs on temp + kriging eval | All baselines running |
| 4 | OOD evaluation all methods | OOD evaluation, degradation plots | Joint analysis |
| 5 | — | — | Writing, figures, polish |

---

## Key Figures to Produce

1. **Positioning table** (related work): what prior work tested vs. what we test
2. **Architecture diagram** (brief, since not the contribution)
3. **Results table**: methods × splits × (RMSE + coverage)
4. **Degradation plot**: RMSE vs. distance from training region
5. **Coverage map**: where does global calibration fail but local succeed?
6. **Reliability diagram**: predicted vs. empirical coverage (ID vs. OOD)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| All methods perform identically OOD | This is itself a finding ("current methods share failure modes"); emphasize need for new approaches |
| Precipitation model fails to converge | Fall back to single-variable temperature paper |
| ConvCNP baseline doesn't work via DeepSensor | Use simpler neural baseline (MLP on interpolated grid) but acknowledge limitation |
| Running out of time | Prioritize: (1) ID results, (2) OOD-spatial, (3) OOD-extreme, (4) attention analysis |

---

## INSIGHT Collection

- **INSIGHT-1**: Frame as evaluation + UQ contribution, not architecture contribution
- **INSIGHT-2**: Prior work handles irregular inputs; evaluation + uncertainty calibration is the gap
- **INSIGHT-3 (Revised)**: Local vs. global conformal is NOT novel on its own (LSCP, Cluster-Aware DeepKriging pre-empt it); reframed as a calibration-*transfer* diagnostic (Global/Local-transfer/Oracle-local) plus classical-vs-conformal UQ comparison — neither prior paper tests either
- **INSIGHT-8**: OOD-Sparse doubles as a density-shift stress test for conformal validity, not just an RMSE robustness ablation
- **INSIGHT-9**: Point-accuracy degradation (RMSE) and calibration degradation (coverage) should be reported jointly — a model can hold RMSE steady while coverage collapses, or vice versa, under transfer
- **INSIGHT-4**: OOD-spatial connects to climate justice (data-sparse regions)
- **INSIGHT-5**: Both positive and negative results are publishable at workshops
- **INSIGHT-6**: Kriging as baseline differentiates from prior neural-only work
- **INSIGHT-7**: Coverage degradation is more actionable than RMSE degradation—tells practitioners when not to trust the model
