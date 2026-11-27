# UMA-M-1.1：DFT Reference、Frozen U0 與 Full LoRA 三方比較

本報告將 **原始 DFT 資料當作參考標準（reference labels）**，比較 Frozen UMA-M-1.1（U0）與六組 Full LoRA（L-All）在**同一批 100 筆 validation configurations**上的能量與原子力預測。所有結果來自上傳的實際資料，不是示意數值；圖中的 DFT identity line 是預測完全正確時的目標，不代表 DFT 也被當成另一個需要衡量 MAE 的模型。

## 1. 數據來源及配對確認

- **DFT**：`i-x20/uma_cuh_baseline/data/prepared/val/structures.extxyz`，原始 energy 與每顆原子的 Cartesian force components。
- **U0**：新上傳的 `val_predictions(2).npz`，與 `metrics(1).json` 對應。
- **Full LoRA**：`UMA_LAll_results_seed41.zip` 中的 `runs/lall/full_training_seed41/val_predictions.npz`、`validation_metrics.json`、`training.jsonl`。
- **組成**：Cu₅₄H₂ 64 筆、Cu₅₄ 36 筆。總計 5,528 顆原子（其中 H 128 顆、Cu 5,400 顆）。Force component 數量分別為 H 384 與 Cu 16,200。
- **驗證方式**：DFT extxyz 的 SHA-256 與原始 manifest 完全一致；DFT、U0、L-All 的 100 個 `source_index` 均相同，兩組 predictions 的 force-offset arrays 完全相同。所有重新計算的 MAE 均與原始 validation metrics 相符。
- **訓練方式**：六組 LoRA，rank 2，共 343,040 個 trainable parameters；500 筆訓練樣本、2,000 optimizer updates、batch size 1。**未開啟 200 筆 test split**。

## 2. 三方比較：以 DFT 為基準的數值結果

| 目標 | DFT | Frozen U0 | Full LoRA | U0 → LoRA |
|---|---|---:|---:|---:|
| Raw total-energy MAE | 原始 reference energy | 1.390870 eV/config | 1.464874 eV/config | **惡化 5.32%** |
| Aligned energy MAE | 同一 DFT energy | 0.541098 eV/config | 0.249369 eV/config | **改善 53.91%** |
| H-force MAE | 原始 H 力向量 | 0.032652 eV/Å/component | 0.018114 eV/Å/component | **改善 44.52%** |
| Cu-force MAE | 原始 Cu 力向量 | 0.019783 eV/Å/component | 0.006544 eV/Å/component | **改善 66.92%** |

這裡的 force MAE 是 x、y、z 三個 Cartesian components 的**逐分量 MAE**，並非每顆原子的力向量模長 MAE。由於 Cu 原子遠多於 H 原子，分開報告 H 和 Cu 的 force MAE，比將所有 force components 混合平均更能反映研究的物理目標。

![三方比較的 MAE](figures/08_reference_based_mae_summary.png)

**解讀**：Full LoRA 對 H/Cu force 的改善是穩定且明顯的；但不能把 aligned-energy 的改善直接寫成「raw total energy 更準確」，兩種指標測量的系統誤差不同。

## 3. 能量：為什麼 raw MAE 變差，aligned MAE 卻變好？

![原始 total-energy parity](figures/01_energy_raw_parity_by_composition.png)

![固定 offset 的 aligned-energy parity](figures/02_energy_aligned_parity_by_composition.png)

`Aligned energy` 的定義是

\[
\Delta E_{\mathrm{aligned},i}=E_{\mathrm{pred},i}-b_{c(i)}-E_{\mathrm{DFT},i}
\]

其中 `b` 只使用 **U0 training data residuals** 估計，不重新擬合 validation：Cu₅₄ 為 **−0.142832 eV**、Cu₅₄H₂ 為 **−2.463651 eV**。

| Composition | Model | Raw energy MAE (eV) | Signed raw-energy bias (eV) | Aligned energy MAE (eV) |
|---|---|---:|---:|---:|
| Cu₅₄ (36) | U0 | 0.421961 | +0.421961 | 0.564793 |
| Cu₅₄ (36) | L-All | 0.119564 | +0.119564 | 0.262396 |
| Cu₅₄H₂ (64) | U0 | 1.935881 | −1.935881 | 0.527770 |
| Cu₅₄H₂ (64) | L-All | 2.221610 | −2.221610 | 0.242041 |

![Energy residual distributions](figures/03_energy_residuals_raw_vs_aligned.png)

**機制層級的解釋，須保持謹慎**：L-All 在這批資料的 raw predicted energies 平均比 U0 低約 0.292 eV/config；這個變化同時減少 Cu₅₄ 的正偏差，卻加重 Cu₅₄H₂ 的負偏差。因此，跨組成彙總的 raw MAE 稍微增加，而使用原本鎖定的 U0 composition offsets 後，兩組 aligned errors 都下降。這是一個 residual/bias decomposition，**單憑能量數據無法把原因唯一歸結為特定化學鍵或電荷轉移機制**。

不能直接把 Cu₅₄ 和 Cu₅₄H₂ 的未校正 total energies 差當成 H adsorption energy；若要探討 adsorption energetics，仍須引入恰當的 H₂（或其他氫化學勢）參考及一致的計算協定。

### 不需要對齊 offset 的相對能量

為避免 energy correction 完全主導結論，我也固定每個組成中的第一筆來源作 reference，分析

\[
\Delta E_i=E_i-E_{\mathrm{reference}}
\]

| Composition | Reference `source_index` | U0 relative-energy MAE | L-All relative-energy MAE |
|---|---:|---:|---:|
| Cu₅₄ | 3925 | 0.0608 eV | 0.0107 eV |
| Cu₅₄H₂ | 84 | 0.0949 eV | 0.0503 eV |

![Relative energy parity](figures/10_within_composition_relative_energy.png)

這項分析表明：不只固定 energy shift，**同一組成內的能量相對變化也更接近 DFT**。但 reference 結構的相對能量誤差必然為零，這不是一個獨立測試點。

## 4. Hydrogen force：DFT、U0、L-All 的逐分量比較

![Hydrogen force parity](figures/04_H_force_component_parity.png)

H-force MAE 從 **0.03265** 降到 **0.01811 eV/Å**，下降 **44.52%**。384 個 Cartesian components 來自 64 筆 Cu₅₄H₂ 中的 128 顆 H 原子。H 的 DFT force 範圍約為 −7.74 至 +7.74 eV/Å，因此以 full-range parity plot 觀察時，兩個模型都會幾乎沿著對角線；需要再看局部誤差或累積分布才能量化差異。

從 configuration-level MAE 觀察，**64 筆 H-bearing structures 有 61 筆改善**。這表示改善不只是來自少數高權重力分量，但仍有 3 筆在 LoRA 後出現惡化，未來可分析其 H–H / H–Cu 幾何特徵。

## 5. Copper force：大幅改善且分布廣泛

![Copper force parity](figures/05_Cu_force_density_parity.png)

Cu-force MAE 從 **0.01978** 降到 **0.00654 eV/Å**，減少 **66.92%**。Cu 有 5,400 顆原子、16,200 個 components，因此用密度圖而非完整散點圖避免 overplotting。**100/100 筆 validation configurations 的 Cu force MAE 都下降**。

![Force absolute error CDF](figures/06_force_error_cdf.png)

![Force error against DFT magnitude](figures/11_force_error_by_reference_magnitude.png)

誤差 CDF 和 DFT-force magnitude 分組圖都支持同一方向：LoRA 不只是修正幾個 outliers，而是在多種力大小區間降低了預測偏差。這支持「domain-specific fine-tuning improves force accuracy」的結論；但仍不能推論微調模型已能穩定進行分子動力學，因為尚未有 trajectory rollout、energy conservation 或穩定性測試。

## 6. Paired per-configuration improvement 與不確定性

![Same configuration paired MAE](figures/07_per_configuration_paired_errors.png)

| Metric | 改善 configurations | MAE 改善率 | 探索性 95% bootstrap interval |
|---|---:|---:|---:|
| Raw energy | 36/100 | −5.3% | −6.5% 至 −4.3% |
| Aligned energy | 100/100 | +53.9% | +52.8% 至 +55.0% |
| H force | 61/64 | +44.5% | +39.1% 至 +49.8% |
| Cu force | 100/100 | +66.9% | +66.2% 至 +67.6% |

Bootstrap 在 **configuration 層級配對抽樣**，並維持 Cu₅₄／Cu₅₄H₂ 的原本比例；4,000 次重抽樣，seed 41。它只是 validation set 內的**探索性不確定性分析**。由於資料可能來自彼此相關的 trajectory structures，且這批 validation data 已參與 checkpoint selection，這些區間**不能當作獨立 test performance 的正式 confidence interval**。

## 7. Training convergence：訓練過程及收斂解釋

![Training/validation convergence](figures/09_training_and_validation_convergence.png)

首先，L-All 使用 500 筆訓練構型、batch size 1、2,000 optimizer updates，相當於四次隨機排序的資料經過（shuffled passes）。在固定的 100 筆 validation structures 上，normalized loss 從 step 250 的 **0.271973** 下降至 step 2000 的 **0.100064**，下降 **63.2%**。這張圖反映對同一個 validation set 的整體預測表現。

### 7.1 詳細 Training loss（保留原版 Figure）

![Detailed single-sample training loss](figures/12_detailed_training_loss.png)

這是你指定保留的原版 Training convergence 圖。圖中包含每一個 update 的 sample-level training loss、101-update rolling median 及第 90 百分位數，並使用 **logarithmic y-axis**。這個呈現方式特別適合 batch size = 1 的訓練：每一步使用不同 configuration，構型難度差異可能造成數十倍甚至數百倍的 loss 變化，所以不能只根據一個 spike 就宣稱模型發散。

| Training pass | Updates | Median normalized training loss |
|---|---|---:|
| 1 | 1–500 | 0.10498 |
| 2 | 501–1000 | 0.02310 |
| 3 | 1001–1500 | 0.01767 |
| 4 | 1501–2000 | 0.01467 |

從第一輪到第四輪，training loss 的 median 下降約 **86.0%**。同時對每個 `source_index` 配對第一次與第四次遇到模型時的 loss，**496/500 筆（99.2%）呈現下降**。這比只比較單一時間點的 loss 更能確認模型確實學會處理絕大多數訓練構型；但它仍是 training-data 表現，不能代替 validation 或 locked test。

最值得追蹤的 outlier 是 `source_index = 304`：四次遇到它時的 loss 依序為 **115.944、73.724、21.658、4.694**。這說明 step 50 的大幅尖峰並不是持續發散。不過，即使第四次的值已下降，4.694 仍高於多數 training samples，值得後續單獨檢查其 H/Cu 力殘差和原子幾何。

## 8. Six LoRA groups：實際參數預算及選擇研究意義

![Six-group LoRA parameter budget](figures/13_six_group_parameter_budget.png)

這是你指定保留的第二張原版圖，展示六組 LoRA 的實際 trainable parameter counts，而不是 UMA backbone 的總權重數。模型保留 10 個 blocks，依 Early（0–4）與 Late（5–9）分成三種群組：Radial（R）、SO(2)（S）、Atomwise（A）。

| Group | Trainable parameters | Matched module sites |
|---|---:|---:|
| R_early | 39,680 | 15 |
| R_late | 39,680 | 15 |
| S_early | 99,840 | 30 |
| S_late | 99,840 | 30 |
| A_early | 32,000 | 15 |
| A_late | 32,000 | 15 |
| **總計** | **343,040** | **120** |

六組合計僅占 **1,397,012,129 個原始 UMA 參數的約 0.0246%**。其中兩個 SO(2) groups 合計 **199,680（58.2%）**，Radial 佔 23.1%，Atomwise 佔 18.7%。因此，若後續 FA-JOINT / Force-Gradient 選到兩組 S，而 Random 選到兩組 A，雖然都是「選兩組」，可訓練參數量卻很不一樣；這不是公平的 parameter-matched comparison。

**這張圖的主要研究價值**，是讓後續 Selective LoRA 的 budget 約束可以用實際參數量明確定義。六組的參數數量只能說明 capacity/cost，**不能直接推論 S-group 對 accuracy 最重要**；這個問題必須由 probes 與 matched-budget ablation 實驗回答。另外，目前 MoLE 使用 expert-shared LoRA，並不是 32 個 experts 都有獨立 adapters；SO(2) groups 也同時涵蓋 `so2_conv_1` 與 `so2_conv_2`。描述實驗方法時必須保持這個界定。

## 9. 最重要的研究結論與限制

**目前可以支持的論點**：使用極小比例的 trainable parameters，L-All 在相同 DFT validation structures 上顯著改善 Cu 與 H force predictions，也改善固定 U0 calibration 後的 energy residuals，以及同組成內的 relative energy differences。

**目前不能支持的論點**：全面改善 raw absolute-energy predictions、已超越未來選定 adapters 的表現、在獨立 test set 上同樣有效、DFT 與實驗完全一致、或能穩定執行 MD。

**下一階段研究方向**：以同一原始 UMA checkpoint 為起點，固定 500 training frames、optimizer steps、rank、validation/test protocol，進行 selected groups 與 parameter-matched random groups 的比較。Selected/Random 的 budget 需以**實際 trainable parameter count**衡量，而非只看選了幾組。

## 10. 可直接用於英文 Final Report 的結果段落

> The frozen UMA-M-1.1 baseline and the six-group, rank-2 Full LoRA model were compared against the same DFT reference labels across 100 validation configurations (64 Cu54H2 and 36 Cu54). Fine-tuning reduced the hydrogen force-component MAE from 0.03265 to 0.01811 eV/Å and the copper force-component MAE from 0.01978 to 0.00654 eV/Å, corresponding to relative reductions of 44.5% and 66.9%, respectively. Improvements were observed in 61 of 64 hydrogen-bearing configurations and all 100 configurations for copper forces. The composition-aligned energy MAE, evaluated using fixed offsets fitted exclusively on U0 training residuals, decreased from 0.54110 to 0.24937 eV/configuration. However, raw total-energy MAE increased from 1.39087 to 1.46487 eV/configuration, reflecting composition-dependent systematic biases. Within-composition relative-energy comparisons, which do not require fitted energy offsets, also improved for both Cu54 and Cu54H2. During 2,000 optimizer updates, the median training loss declined from 0.10498 in the first 500-update pass to 0.01467 in the fourth, and normalized validation loss decreased from 0.27197 at step 250 to 0.10006 at the final checkpoint. The six LoRA groups comprised 343,040 trainable parameters (0.0246% of the original UMA parameter count), with SO(2) groups accounting for 58.2% of the adapter budget. These results support the effectiveness of low-rank adaptation for improving Cu/H force predictions, while emphasizing that fixed-validation performance should not be interpreted as independent test accuracy or molecular-dynamics reliability.

---

全部逐構型數值、每顆原子的 DFT／U0／L-All force vectors、95% bootstrap 估計、圖表原檔、可重現腳本與來源資料均包含在本資料包中。
