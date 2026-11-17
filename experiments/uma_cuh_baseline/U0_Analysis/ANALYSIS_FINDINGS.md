# UMA-M-1.1 U0 / SRP48 DFT validation: residual analysis

**Data analyzed**: user-provided `structures.extxyz` (100 validation configurations), `val_predictions.npz`, and `metrics.json`. This is frozen U0 inference; there is **no LoRA fine-tuning** in these files. The script checks exact `source_index` and flattened force-array alignment, and recomputes both species MAEs and both energy MAEs.

## What is SHA-256?

SHA-256 is a 256-bit content hash used to verify that the DFT file, split manifest, or checkpoint bytes match a pinned artifact. It establishes *byte identity*, not physical correctness, training validity, data provenance by itself, or prediction accuracy. The exact same bytes produce the same hash.

## Raw vs. training-aligned energy

Let `r_i = E_UMA,i - E_DFT,i`. Raw energy MAE is the validation mean of `|r_i|`. For each composition `s`, the original pipeline fits an offset from *training data only*: `c_s = mean_train(E_UMA - E_DFT | s)`. Aligned validation error is `mean_val(|r_i-c_s|)`; forces are unchanged by this constant shift.

| Composition | n | train offset c_s (eV) | val mean raw residual (eV) | raw MAE (eV) | val mean aligned residual (eV) | aligned MAE (eV) |
|---|---:|---:|---:|---:|---:|---:|
| H2Cu54 | 64 | -2.46365 | -1.93588 | 1.93588 | +0.52777 | 0.52777 |
| Cu54 | 36 | -0.14283 | +0.42196 | 0.42196 | +0.56479 | 0.56479 |
| Overall | 100 | by composition | -1.08706 | 1.39087 | +0.54110 | 0.54110 |

**Interpretation**: the training-only offsets do not transfer perfectly to the held-out validation cell family. For Cu54, offset alignment actually increases validation energy MAE. Neither metric should be relabeled as a reaction-barrier error. Fitting offsets on validation data would be leakage; do not do it.

## 1. Energy parity

See `01_energy_parity.png`. Raw total energies show composition-dependent systematic shifts. Training-only calibration moves both sets above the 1:1 line on validation, showing a residual bias.

## 2. H/Cu force parity

See `02_force_parity.png`. Componentwise MAEs are H = **0.032652 eV/Å** (384 components) and Cu = **0.019783 eV/Å** (16,200 components). Typical DFT force RMSs differ: H about 1.684 and Cu about 0.441 eV/Å, so the absolute MAEs alone do not establish that the model is relatively less accurate for H. These are not force vector norm MAEs.

## 3. Per-configuration outliers

See `03_force_outliers.png` and `per_configuration.csv`. Among 64 H-containing validation frames, median per-frame H MAE = **0.02637 eV/Å**, 90th percentile = **0.06614**, and maximum = **0.11495**. The ten worst frames contribute **36.47%** of the summed H absolute error across these frames. Representative high-error original source indices are **1877, 1915, 1811**. Cu errors remain comparatively narrowly distributed (near 0.020 eV/Å per frame).

## 4. H–H and H–Cu geometry dependence

See `04_H_geometry_dependence.png`. Periodic minimum-image distances were computed from saved structures. Mean H MAE by H–H distance: `<0.9 Å` (n=35): **0.03845 eV/Å**; `0.9–1.6 Å` (n=16): **0.02463**; `>=1.6 Å` (n=13): **0.02692**. Pearson r between H–H distance and H MAE = **−0.241**; nearest H–Cu distance versus H MAE = **−0.147**. These are weak exploratory correlations, not causes or formal significance claims. The very largest errors occur at **short, compressed H–H separations** (~0.59–0.71 Å), not preferentially at long H–H separations often associated with dissociation. Very short bonds may have higher true force magnitudes, so compare residuals to DFT force magnitudes before calling them intrinsically harder.

## 5. Same-composition relative energy

For any A and B of the SAME composition and compatible reference cell, `ΔE_DFT = E_DFT,B − E_DFT,A` and `ΔE_UMA = E_UMA,B − E_UMA,A`.

| Composition | n frames | unordered pairs | pairwise ΔE MAE (eV) | pairwise ΔE RMSE (eV) | Pearson r of ΔE |
|---|---:|---:|---:|---:|---:|
| H2Cu54 | 64 | 2,016 | **0.11224** | 0.14385 | 0.99525 |
| Cu54 | 36 | 630 | **0.08788** | 0.10777 | 0.99715 |

See `05_relative_energy.png`. Per-composition constant offsets cancel exactly. **Important**: pairs share many structures, are not independent observations, and are not known minimum-to-saddle reaction pairs. This is a descriptive PES-shape diagnostic, **not** a direct dissociation/adsorption/diffusion barrier validation. Do not compare Cu54 to H2Cu54 energies without the appropriate chemical-potential/gas reference.

## Using the outputs for LoRA

1. Keep `scripts/evaluate.py` as the canonical, unchanged, reproducible UMA inference/standard-MAE pipeline.
2. Place `analyze_u0_residuals.py` into `scripts/` as a separate **post-processing** script. Point it at `data/prepared/val/structures.extxyz` and `reports/U0_val/val_predictions.npz` and `metrics.json`.
3. After an adapted checkpoint has been evaluated with **the same 100 validation configurations**, rerun the same analysis using the adapted model's `val_predictions.npz` and `metrics.json`.
4. Implement LoRA adapters in a separate training/model-adaptation module; not inside the metrics evaluator. Check energy-conserving forces, equivariance, model reload, frozen parameter counts and identical validation samples.
5. Do not use the locked 200-configuration test subset for adapter selection or target-setting.

**Method caveat:** this stand-alone parser assumes the pinned EXTXYZ `Properties=species:S:1:pos:R:3:forces:R:3`. For other exports, use ASE or adapt the reader. No new DFT calculations or model predictions were performed to generate these figures.
