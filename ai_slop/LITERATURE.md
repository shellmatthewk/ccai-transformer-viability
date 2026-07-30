# Literature Review: Learned Station-to-Grid Climate Reconstruction

This document provides an annotated bibliography of existing work on neural methods for reconstructing gridded climate fields from sparse station observations. The focus is on identifying gaps in OOD evaluation that our paper addresses.

---

## Summary: Evaluation Gaps in Prior Work

| Paper | Spatial OOD | Extreme Events | Kriging | Conformal UQ | Key Limitation |
|-------|-------------|----------------|---------|--------------|----------------|
| Swin-TNP | Random 20% holdout | No | No | No | Random ≠ geographic transfer |
| Aardvark Weather | Not reported | Not reported | No | No | Focus on lead time, not spatial transfer |
| ADAF | Sparse sensitivity | Tropical cyclones | No | No | No classical baseline |
| Manshausen et al. | Left-out stations | No | No | No | Only 40 stations; limited OOD |
| ConvCNP/DeepSensor | Varies | No | Sometimes | No | Inconsistent evaluation protocols |
| SLAMS | Global coverage | Not explicit | Unknown | No | Multimodal focus, not OOD |
| FieldFormer | Mesh-free design | Not mentioned | No | No | General framework, not climate-specific |
| Conformal for weather | N/A (gridded forecasts) | Yes | No | **Yes** | Not station→grid reconstruction |

**Our contribution fills**:
1. Systematic geographic transfer (train West → test East)
2. Extreme event splits (heatwaves, cold snaps)
3. Kriging baseline throughout
4. **Conformal prediction for station→grid reconstruction** (first application)

---

## 1. Gridded Transformer Neural Processes (Swin-TNP)

**Citation**: Qu, Y., Bruinsma, W. P., Markou, S., Ferguson, J., Vaughan, A., Hosking, J. S., Turner, R. E., & Requeima, J. (2024). Gridded transformer neural processes for large unstructured spatio-temporal data. *arXiv preprint arXiv:2410.06731*.

**Links**:
- [arXiv](https://arxiv.org/abs/2410.06731)
- [GitHub](https://github.com/cambridge-mlg/gridded-tnp)
- [ICML 2025 Poster](https://icml.cc/virtual/2025/poster/45467)

### Summary
Gridded TNPs use specialized encoders and decoders to handle unstructured spatio-temporal data. The architecture employs gridded pseudo-tokens with efficient attention mechanisms, enabling scalability to large datasets that standard transformer neural processes cannot handle.

### Dataset
- **Reanalysis**: ERA5
- **Stations**: HadISD (~9,957 weather station locations globally)
- **Variables**: 2m temperature (t2m), skin temperature (skt)
- **Grid**: Coarsened to 180×360 global grid
- **Period**: Train 2009-2017, validate 2018, test 2019

### OOD Evaluation
- **Spatial holdout**: 20% of weather station pixels randomly excluded from all model inputs during training/validation
- **Temporal holdout**: Standard train/val/test by year
- **Extreme events**: **Not evaluated**
- **Geographic transfer**: **Not evaluated** (random holdout ≠ systematic region-to-region transfer)

### Baselines
- ConvCNP with U-Net backbone (64×128 grid)
- PT-TNP with 256 pseudo-tokens
- **Kriging**: **Not included**

### Key Quote
> "We evaluate the model on held-out stations to assess spatial generalization."

### Gap for Our Paper
Random spatial holdout tests interpolation, not extrapolation to new geographic regions. No classical baseline makes it hard to assess whether neural complexity is justified.

---

## 2. ConvCNP and DeepSensor

**Primary Reference**: Alan Turing Institute. *DeepSensor: A Python package for modelling with neural processes*. GitHub: alan-turing-institute/deepsensor.

**Links**:
- [GitHub Repository](https://github.com/alan-turing-institute/deepsensor)
- [Documentation](https://alan-turing-institute.github.io/deepsensor/)

**Related Papers**:
- Gordon, J., Bruinsma, W. P., Foong, A. Y. K., Requeima, J., Dubois, Y., & Turner, R. E. (2020). Convolutional conditional neural processes. *ICLR 2020*.
- Vaughan, A., et al. (2022). Convolutional conditional neural processes for local climate downscaling. *Geoscientific Model Development*.

### Summary
DeepSensor provides a framework for training Convolutional Conditional Neural Processes (ConvCNPs) for environmental prediction tasks. ConvCNPs use a CNN-based encoder-decoder architecture that naturally handles irregular spatial observations.

### OOD Evaluation (Varies by Application)
- Some applications include left-out station tests
- **Extreme events**: Generally **not evaluated**
- **Geographic transfer**: **Inconsistent** across papers

### Baselines
- **Regression Kriging (RK)**: Included in some applications as geostatistical baseline
- Various ConvCNP variants, GriddedTNP

### Key Finding
> "Alternative architectures, such as the ConvCNP and GriddedTNP, struggled to generalize effectively when relying solely on static topographic predictors. Performance was benchmarked against Regression Kriging."

### Gap for Our Paper
DeepSensor is the only framework where kriging baselines appear consistently. We should use DeepSensor's ConvCNP as our neural baseline—it's more credible than ConvLSTM on interpolated grids.

---

## 3. Manshausen et al.: Generative Data Assimilation

**Citation**: Manshausen, P., Maletych, V., Cachay, S. R., Kurth, T., Sudharsan, S., & Smith, L. A. (2025). Generative data assimilation of sparse weather station observations at kilometer scales. *Journal of Advances in Modeling Earth Systems*, e2024MS004505.

**Links**:
- [Wiley (Published)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024MS004505)
- [arXiv](https://arxiv.org/abs/2406.16947)
- [NVIDIA PhysicsNeMo](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo/examples/weather/regen/README.html)

### Summary
Uses diffusion models for data assimilation of sparse weather station observations to 3km-resolution surface fields. Trains an unconditional diffusion model on HRRR snapshots, then uses Score-based Data Assimilation (SDA) to condition on sparse observations.

### Dataset
- **Analysis product**: High Resolution Rapid Refresh (HRRR), 3km resolution
- **Stations**: 40 weather stations, central US testbed
- **Variables**: Precipitation, surface winds

### OOD Evaluation
- **Spatial**: Left-out station test (shows 10% lower RMSE than HRRR itself)
- **Extreme events**: Sensitivity tests show learned physics (gust fronts as example)
- **Geographic transfer**: **Not evaluated**

### Baselines
- HRRR system itself (naive baseline)
- **Kriging**: **Not included**

### Acknowledged Limitations
> "Lingering imperfections such as insufficiently disperse ensemble DA estimates."

### Gap for Our Paper
Only 40 stations in a small region. Left-out station test ≠ geographic transfer. No classical baseline.

---

## 4. ADAF: Artificial Intelligence Data Assimilation Framework

**Citation**: Xiang, S., et al. (2025). ADAF: An artificial intelligence data assimilation framework for weather forecasting. *Journal of Advances in Modeling Earth Systems*, 17(1), e2024MS004839.

**Links**:
- [Wiley (Published)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024MS004839)
- [arXiv](https://arxiv.org/abs/2411.16807)
- [GitHub (Microsoft)](https://github.com/microsoft/ADAF)

### Summary
ADAF uses a Swin Transformer architecture to generate km-scale analysis for weather forecasting. The framework assimilates real-world observations from surface stations and satellite imagery within a 3-hour window.

### Architecture
- Encoder-decoder with residual Swin Transformer blocks
- Multi-head Self Attention layers
- Inputs: observations, background field, topography

### Dataset
- **Analysis product**: High Resolution Rapid Refresh (HRRR)
- **Observations**: Real-world surface weather observations + satellite imagery
- **Region**: Contiguous United States (CONUS)
- **Variables**: Four near-surface variables

### OOD Evaluation
- **Extreme events**: **Yes** — demonstrates reconstruction of tropical cyclone wind fields
- **Sparse sensitivity**: Shows robustness to "extremely sparse surface observations" and "low-accuracy backgrounds"
- **Geographic transfer**: **Not evaluated**

### Baselines
- High Resolution Rapid Refresh Data Assimilation System (HRRRDAS)
- **Kriging**: **Not included**

### Performance
16-33% improvement over HRRRDAS in accuracy for near-surface atmospheric conditions.

### Gap for Our Paper
Strong extreme event demonstration (tropical cyclones), but no classical baseline and no systematic spatial transfer evaluation.

---

## 5. SLAMS: Score-based Latent Assimilation in Multimodal Setting

**Citation**: Qu, Y., et al. (2024). Deep generative data assimilation in multimodal setting. *CVPR 2024 EarthVision Workshop*.

**Links**:
- [arXiv](https://arxiv.org/abs/2404.06665)
- [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Qu_Deep_Generative_Data_Assimilation_in_Multimodal_Setting_CVPRW_2024_paper.pdf)

### Summary
SLAMS (Score-based Latent Assimilation in Multimodal Setting) is a diffusion-based framework that assimilates multimodal data (in-situ weather stations + ex-situ satellite imagery) in a unified latent space. First data-driven probabilistic, multimodal DA framework for real-world Earth system modeling.

### Dataset
- **Observations**: In-situ weather station data + satellite imagery
- **Coverage**: Global vertical temperature profile calibration

### OOD Evaluation
- **Data quality robustness**: Claims robustness in "low-resolution, noisy, and sparse data settings"
- **Extreme events**: **Not explicitly evaluated**
- **Geographic transfer**: **Not explicitly evaluated**

### Baselines
- Not detailed in available excerpts

### Key Contribution
Ensemble generation enables uncertainty quantification—addresses a key limitation of deterministic methods.

### Gap for Our Paper
Multimodal focus is orthogonal to our evaluation focus. OOD robustness claimed but not systematically demonstrated.

---

## 6. FieldFormer: Physics-Informed Transformers

**Citation**: (Authors TBD). (2025). FieldFormer: Physics-informed transformers for spatio-temporal field reconstruction from sparse sensors. *arXiv preprint arXiv:2510.03589*.

**Links**:
- [arXiv](https://arxiv.org/abs/2510.03589)

### Summary
FieldFormer is a transformer-based framework for mesh-free spatio-temporal field reconstruction. Uses learnable velocity-scaled distance metrics for anisotropic adaptation. Physics consistency enforced through autograd-based PDE residuals and boundary penalties.

### Architecture
- Local transformer encoder processes neighborhoods
- Coordinate-based: operates directly on (x, y, t, value) observations
- No grid assumption required

### OOD Evaluation
- **Spatial**: Mesh-free design handles irregular domains
- **Extreme events**: **Not mentioned**
- **Data conditions**: Designed for "sparse, irregular, and noisy" observations

### Baselines
- Not detailed in available excerpts

### Gap for Our Paper
General-purpose framework, not climate-specific. No systematic OOD evaluation or classical baselines reported.

---

## 7. Aardvark Weather: End-to-End Forecasting

**Citation**: (Authors TBD). (2025). End-to-end data-driven weather prediction. *Nature*, 641, 1172–1179.

**Links**:
- [Nature (Published)](https://www.nature.com/articles/s41586-025-08897-0)
- [arXiv](https://arxiv.org/abs/2404.00411)
- [Vector Institute](https://vectorinstitute.ai/aardvark-weather/)

### Summary
First end-to-end data-driven weather forecasting system that generates predictions with no input from conventional NWP. Maps raw observations directly to forecasts, bypassing traditional data assimilation entirely.

### Architecture
Three modular components:
1. **Encoder Module**: SetConv layers (from ConvCNP) + Vision Transformer backbone
2. **Processor**: Handles atmospheric dynamics
3. **Decoder**: Produces gridded forecasts

Key: No "first guess" from previous forecasts required—operates on raw observations only.

### Dataset
- **Observations**: Weather stations, ships, radiosondes, satellite instruments (heterogeneous, multimodal)
- **Coverage**: Global
- **Resolution**: 1.41° output grid

### OOD Evaluation
- **Lead time evaluation**: Multiple forecast horizons tested
- **Spatial transfer**: **Not explicitly reported**
- **Extreme events**: **Not explicitly mentioned**

### Baselines
- ECMWF HRES (operational NWP)—outperformed across multiple variables/lead times
- **Kriging**: **Not included**

### Performance
Lower RMSE than operational NWP using "order of magnitude fewer observations" and "orders of magnitude less compute."

### Gap for Our Paper
Impressive operational performance, but evaluation focuses on forecast skill (lead time), not reconstruction robustness under spatial transfer or extreme events.

---

---

## 8. Conformal Prediction for Weather/Climate (NEW ADDITION)

### 8.1 Conformal Prediction for Probabilistic Weather Forecasts

**Citation**: Camps-Valls, G., et al. (2024). Rigorous uncertainty quantification of probabilistic AI weather forecasts with conformal prediction. *arXiv preprint arXiv:2606.19642*.

**Link**: [arXiv](https://arxiv.org/abs/2606.19642)

### Summary
Applies conformal prediction to temperature and precipitation forecasts of leading global weather models (GenCast, NeuralGCM, AIFS-ENS). Provides calibrated uncertainty with mathematical coverage guarantees.

### Key Contributions
- **Distribution-free guarantees**: Finite-sample coverage holds regardless of underlying data distribution
- **Cell-wise calibration**: Marginal coverage guarantees at every spatial and temporal location
- **Extreme weather**: Demonstrated on typhoons, showing spatially-varying uncertainty that correlates with forecast error

### Method
- Conformalized quantile regression (CQR)
- Split conformal prediction with calibration on held-out data
- Local conformal for spatially adaptive bands

### Gap for Our Paper
- Applied to **gridded forecasts** (model output → future state)
- Not applied to **station→grid reconstruction** (sparse obs → dense field)
- We extend conformal prediction to the reconstruction setting

---

### 8.2 Conformal Prediction for Physics-Informed Neural Networks

**Citation**: (Authors TBD). (2024). A conformal prediction framework for uncertainty quantification in physics-informed neural networks. *Journal of Computational Physics* / *arXiv:2509.13717*.

**Link**: [arXiv](https://arxiv.org/abs/2509.13717)

### Summary
Local conformal quantile estimation for PINNs solving PDEs. Produces wider intervals near heat sources, boundaries, and material interfaces—where prediction is harder.

### Key Contributions
- **Spatially adaptive**: Intervals vary with local prediction difficulty
- **Normalized conformal**: Adjusts for heteroscedasticity
- **Interpretable**: Interval width correlates with actual error magnitude

### Relevance
Shows that conformal prediction can handle spatial heterogeneity in physical fields—directly applicable to climate reconstruction where some regions are harder than others.

---

### 8.3 Conformal Prediction for Neural Operators

**Citation**: (Authors TBD). (2024). Conformal prediction for neural operators: Distribution-free uncertainty quantification in physics simulation. *arXiv preprint arXiv:2606.09923*.

**Link**: [arXiv](https://arxiv.org/abs/2606.09923)

### Summary
Extends conformal prediction to neural operators (FNO, DeepONet) for physics simulation, providing prediction bands for entire output fields.

### Gap for Our Paper
- Focuses on simulation (model → model), not reconstruction (obs → field)
- We adapt the field-level conformal framework to station→grid setting

---

## 9. Local/Regional Conformal Calibration — Competing Work (Novelty Check, added 2026-07)

**Why this section exists**: Our Chapter 3.4 ablation (local vs. global conformal calibration) is close to published work. This section documents exactly how, so the paper can be positioned as a diagnostic contribution (calibration behavior under OOD shift) rather than a methodological one (local calibration itself).

### 9.1 LSCP: Localized Spatial Conformal Prediction

**Citation**: Jiang, H., & Xie, Y. (2024). Spatial conformal inference through localized quantile regression. *arXiv preprint arXiv:2412.01098*. (ICML 2026 poster.)

**Links**: [arXiv](https://arxiv.org/abs/2412.01098) · [OpenReview](https://openreview.net/forum?id=CwhODlpFuq) · [ICML 2026](https://icml.cc/virtual/2026/poster/64211)

**Summary**: Model-agnostic framework coupling local quantile regression with conformal calibration for spatial data. Retains finite-sample marginal coverage under exchangeability; under stationarity + spatial mixing, attains asymptotic *conditional* coverage. Framework extends to spatio-temporal settings.

**Overlap with our 3.4 ablation**: Directly demonstrates that local/regional calibration improves conditional coverage over a global baseline — the same qualitative claim as our "local beats global" hypothesis.

**What it does NOT test**: Calibration fit on one spatial region and deployed on a disjoint, systematically different region (train/calibrate West → test East). LSCP's locality is conditioning on neighborhoods *within* an otherwise-exchangeable domain, not transfer across a distribution shift boundary. No extreme-event or station-density-shift stress test either.

### 9.2 Cluster-Aware Conformal Calibration for Spatio-Temporal Prediction

**Citation**: Kim, G., Lim, C. Y., Wang, W.-T., Huang, H.-Y., & Wu, W.-Y. (2026). Cluster-aware conformal calibration for spatio-temporal distributional prediction. *arXiv preprint arXiv:2606.06753*.

**Link**: [arXiv](https://arxiv.org/abs/2606.06753)

**Summary**: DeepKriging-style extension with cluster-adaptive spatial bases and cluster-aware conformal calibration — determines interval widths within spatial clusters, with a **global fallback when calibration samples in a cluster are insufficient**. Demonstrates improved coverage/tail reliability vs. a global conformal baseline on simulation + PM2.5 data.

**Overlap with our 3.4 ablation**: Near-identical high-level design to our global/local split, including the exact "fall back to global when the local calibration set is too small" concern we'd flagged for the East region.

**What it does NOT test**: Clusters are carved out of the training/calibration domain's own spatial pattern (non-uniform sampling), not a held-out region unseen during calibration. No systematic geographic-transfer test, no extreme-event split, no comparison against a classical (kriging-variance) UQ baseline.

**Gap for our paper**: Both 9.1 and 9.2 establish that local calibration helps *within* the calibration domain. Neither asks whether locally- or globally-fit calibration **survives being applied out-of-region** — i.e., whether the conformal guarantee itself transfers under the same spatial/temporal/density shifts the rest of the paper is built around. That transfer question, not "does local beat global," is what our ablation should be framed as answering.

### 9.3 Theoretical Foundations: Split CP Under Non-Exchangeability

**NexCP** — Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. (2023). Conformal prediction beyond exchangeability. *Annals of Statistics*, 51(2), 816–845.
[arXiv](https://arxiv.org/abs/2202.13415) · [Annals of Statistics](https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.full)
Reweights nonconformity scores by a fixed, data-independent weight so that observations believed to share the test distribution count more. Under exchangeability, recovers standard split-CP guarantees; under violation, the coverage gap is governed by the total-variation distance between swapped score vectors. **Relevance**: the standard remedy to cite if our coverage results show meaningful drift under spatial/temporal shift — one sentence noting NexCP as the known fix, without implementing it.

**Gibbs & Candès (2024)** — Gibbs, I., & Candès, E. (2024). Conformal inference for online prediction with arbitrary distribution shifts. *Journal of Machine Learning Research*, 25(162), 1–36.
[JMLR](https://jmlr.org/papers/v25/22-1218.html)
Online conformal variant guaranteeing coverage on average over a stream under arbitrary (possibly discontinuous) distribution shift — weaker than pointwise coverage but requires no stationarity assumption. **Relevance**: alternative remedy to NexCP for the discussion section; note that our setting (fixed train/calibrate/test splits) is closer to Barber et al.'s framing than to this online setting.

**Oliveira, Orenstein, Ramos & Romano (2024)** — Split conformal prediction and non-exchangeable data. *Journal of Machine Learning Research*, 25, article 23-1553. Preprint: *arXiv:2203.15885*.
[JMLR](https://jmlr.org/papers/v25/23-1553.html) · [arXiv](https://arxiv.org/abs/2203.15885)
Concentration-inequality framework showing split CP remains valid for many non-exchangeable processes with a bounded coverage penalty tied to how strongly the process departs from exchangeability (mixing-type dependence measures). Empirically benchmarks split CP against distribution-shift-aware alternatives on **real spatiotemporal climate data**, finding standard split CP competitive on coverage/interval size while being far simpler and faster. **Relevance**: direct literature support for using a contiguous calibration year (2019) rather than a scattered/randomized calibration sample — the disjointness requirement is non-negotiable, but the specific contiguous-year design is now citable rather than just asserted.

---

## Cross-Paper Synthesis

### What Prior Work Does Well
1. **Temporal holdout** is standard (train/val/test splits by year)
2. **Left-out station tests** are common (random spatial holdout)
3. **Neural baselines** are consistently included (ConvCNP, various transformer variants)
4. **Scalability** to large datasets demonstrated (Swin-TNP, Aardvark)

### Critical Gaps Our Paper Fills

| Gap | Affected Papers | Our Approach |
|-----|-----------------|--------------|
| **No classical baselines** | Swin-TNP, Manshausen, ADAF, Aardvark | Include kriging as honest yardstick |
| **Random spatial holdout only** | Swin-TNP, most others | Systematic geographic transfer (train West EU → test East EU) |
| **No extreme event splits** | Most papers except ADAF | Dedicated heatwave/cold-snap holdout periods |
| **OOD degradation not quantified** | All papers | Report degradation ratio (OOD RMSE / ID RMSE) |
| **Data sparsity not systematically tested** | Most papers | Ablation: drop 50%/75% of stations |

### Recommended Positioning

**Frame prior work positively**: "Recent methods (Swin-TNP, Aardvark, ADAF) achieve impressive in-distribution reconstruction skill."

**Identify the gap precisely**: "However, systematic evaluation of out-of-distribution performance—crucial for climate applications in data-sparse regions and during extreme events—remains limited."

**State our contribution clearly**: "We provide a controlled study of when these methods fail, using spatial transfer, extreme event splits, and classical baselines absent from prior work."

**Calibration claim, revised (2026-07)**: Do NOT frame local-vs-global conformal calibration as a novel method — see §9. LSCP (Jiang & Xie, ICML 2026) and the cluster-aware DeepKriging paper (Kim et al. 2026) both already show local calibration beats global within a domain, including the same global-fallback-for-sparse-regions design we'd planned for the East region. Frame instead as: "we use known local-calibration machinery to diagnose whether conformal coverage *itself* survives the spatial/extreme/sparsity distribution shifts central to this paper — a transfer question neither prior work asks." Cite Oliveira et al. (2024) for why a contiguous calibration year is defensible despite temporal dependence, and Barber et al. (2023, NexCP) as the one-sentence remedy if coverage drifts.

---

## References (BibTeX)

```bibtex
@article{qu2024gridded,
  title={Gridded Transformer Neural Processes for Large Unstructured Spatio-Temporal Data},
  author={Qu, Yongquan and Bruinsma, Wessel P and Markou, Stratis and Ferguson, James and Vaughan, Anna and Hosking, J Scott and Turner, Richard E and Requeima, James},
  journal={arXiv preprint arXiv:2410.06731},
  year={2024}
}

@article{manshausen2025generative,
  title={Generative Data Assimilation of Sparse Weather Station Observations at Kilometer Scales},
  author={Manshausen, Peter and Maletych, Veronika and Cachay, Stephan Rasp and Kurth, Thorsten and Sudharsan, Srikaran and Smith, Leonard A},
  journal={Journal of Advances in Modeling Earth Systems},
  year={2025},
  publisher={Wiley}
}

@article{xiang2025adaf,
  title={ADAF: An Artificial Intelligence Data Assimilation Framework for Weather Forecasting},
  author={Xiang, Shuaixin and others},
  journal={Journal of Advances in Modeling Earth Systems},
  volume={17},
  number={1},
  year={2025}
}

@inproceedings{qu2024slams,
  title={Deep Generative Data Assimilation in Multimodal Setting},
  author={Qu, Yongquan and others},
  booktitle={CVPR 2024 EarthVision Workshop},
  year={2024}
}

@article{fieldformer2025,
  title={FieldFormer: Physics-Informed Transformers for Spatio-Temporal Field Reconstruction from Sparse Sensors},
  author={TBD},
  journal={arXiv preprint arXiv:2510.03589},
  year={2025}
}

@article{aardvark2025,
  title={End-to-end Data-Driven Weather Prediction},
  author={TBD},
  journal={Nature},
  volume={641},
  pages={1172--1179},
  year={2025}
}

@misc{deepsensor,
  title={DeepSensor: A Python Package for Modelling with Neural Processes},
  author={Alan Turing Institute},
  howpublished={\url{https://github.com/alan-turing-institute/deepsensor}},
  year={2024}
}

@article{campsvalls2024conformal,
  title={Rigorous Uncertainty Quantification of Probabilistic AI Weather Forecasts with Conformal Prediction},
  author={Camps-Valls, Gustau and others},
  journal={arXiv preprint arXiv:2606.19642},
  year={2024}
}

@article{conformalPINNs2024,
  title={A Conformal Prediction Framework for Uncertainty Quantification in Physics-Informed Neural Networks},
  author={TBD},
  journal={arXiv preprint arXiv:2509.13717},
  year={2024}
}

@article{conformalNeuralOps2024,
  title={Conformal Prediction for Neural Operators: Distribution-Free Uncertainty Quantification in Physics Simulation},
  author={TBD},
  journal={arXiv preprint arXiv:2606.09923},
  year={2024}
}

@article{jiang2024lscp,
  title={Spatial Conformal Inference through Localized Quantile Regression},
  author={Jiang, Hanyang and Xie, Yao},
  journal={arXiv preprint arXiv:2412.01098},
  note={ICML 2026 poster},
  year={2024}
}

@article{kim2026clusteraware,
  title={Cluster-Aware Conformal Calibration for Spatio-Temporal Distributional Prediction},
  author={Kim, Gooyoung and Lim, Chae Young and Wang, Wen-Ting and Huang, Hao-Yun and Wu, Wei-Ying},
  journal={arXiv preprint arXiv:2606.06753},
  year={2026}
}

@article{barber2023nexcp,
  title={Conformal Prediction Beyond Exchangeability},
  author={Barber, Rina Foygel and Cand{\`e}s, Emmanuel J and Ramdas, Aaditya and Tibshirani, Ryan J},
  journal={Annals of Statistics},
  volume={51},
  number={2},
  pages={816--845},
  year={2023},
  note={arXiv:2202.13415}
}

@article{gibbs2024online,
  title={Conformal Inference for Online Prediction with Arbitrary Distribution Shifts},
  author={Gibbs, Isaac and Cand{\`e}s, Emmanuel},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={162},
  pages={1--36},
  year={2024}
}

@article{oliveira2024nonexchangeable,
  title={Split Conformal Prediction and Non-Exchangeable Data},
  author={Oliveira, R. and Orenstein, P. and Ramos, T. and Romano, J. V.},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={23-1553},
  year={2024},
  note={arXiv:2203.15885 — first names not independently verified, confirm before camera-ready}
}
```

---

## Additional Resources

### Code Repositories
- [Swin-TNP (Cambridge MLG)](https://github.com/cambridge-mlg/gridded-tnp)
- [DeepSensor (Turing Institute)](https://github.com/alan-turing-institute/deepsensor)
- [ADAF (Microsoft)](https://github.com/microsoft/ADAF)

### Datasets
- [ECA&D (European Climate Assessment & Dataset)](https://www.ecad.eu/)
- [ERA5 (Copernicus Climate Data Store)](https://cds.climate.copernicus.eu/)
- [HadISD (Met Office)](https://www.metoffice.gov.uk/hadobs/hadisd/)
- [HRRR (NOAA)](https://rapidrefresh.noaa.gov/hrrr/)

### Classical Methods Reference
- Kalnay, E. (2003). *Atmospheric Modeling, Data Assimilation and Predictability*. Cambridge University Press.
- Carrassi, A., Bocquet, M., Bertino, L., & Evensen, G. (2018). Data assimilation in the geosciences: An overview of methods, issues, and perspectives. *WIREs Climate Change*, 9(5), e535.
