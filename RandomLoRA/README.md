# Three Random LoRA adapter selections

Three random adapter masks were trained from frozen UMA-M-1.1 and evaluated on the same 100-frame validation set. The common budget is an upper bound of 171,520 trainable parameters, equal to 50% of the six-group Full LoRA budget. Seed 42 uses fewer parameters because of the discrete group costs. All three training runs and their validation evaluations completed successfully.

## How the masks were selected

Before the first run, a budget of 171,520 was selected as a preliminary choice because the original protocol left it unspecified. Six group costs were used in this fixed order.

| Group | Parameters |
|---|---:|
| R_early | 39,680 |
| R_late | 39,680 |
| S_early | 99,840 |
| S_late | 99,840 |
| A_early | 32,000 |
| A_late | 32,000 |

The code enumerates combinations in the order above, first by subset size, then in `itertools.combinations` order. A candidate is included if its total cost is at most the budget and adding any omitted group would exceed the budget. These rules give 11 inclusion-maximal candidates. Each selection independently uses Python `random.Random(selection_seed).choice(candidates)`. This sampling is uniform over those 11 candidates, with replacement across seeds. The three resulting masks happen to be distinct. Seeds 42 and 43 were chosen after the first run and before either additional run started. Run `python selection_reproduce.py` to list all candidates, their costs, and the selected zero-based indices.

| Selection seed | Candidate index | Selected groups | Parameters | Budget used |
|---:|---:|---|---:|---:|
| 41 | 6 | R_late, S_late, A_early | 171,520 | 100.0% |
| 42 | 10 | R_early, R_late, A_early, A_late | 143,360 | 83.6% |
| 43 | 0 | R_early, S_early, A_early | 171,520 | 100.0% |

This distribution differs from uniform sampling over all feasible subsets. The unequal realized counts make comparisons subject to different parameter usage even though the budget ceiling is shared. The budget, distribution and validation protocol should be agreed and locked for future selector comparisons.

## Training and validation

Each run uses rank 2, training seed 41, fresh adapters initialized from frozen U0, 500 training configurations, batch size 1, 2,000 updates, AdamW with learning rate 1e-4 and weight decay 1e-3, validation every 250 steps, and the same fixed U0 training-derived energy calibration and normalization. All three selected their best checkpoint at update 2,000. Normalized best validation losses were 0.109830, 0.097066 and 0.117349 for selection seeds 41, 42 and 43 respectively.

| Validation metric | U0 | Full LoRA | Mask 41 | Mask 42 | Mask 43 |
|---|---:|---:|---:|---:|---:|
| Raw energy MAE, eV/config | 1.390870 | 1.464874 | 1.466617 | 1.468706 | 1.463369 |
| Aligned energy MAE, eV/config | 0.541098 | 0.249369 | 0.247749 | 0.221025 | 0.260311 |
| H-force component MAE, eV/Å | 0.032652 | 0.018114 | 0.022613 | 0.022174 | 0.022994 |
| Cu-force component MAE, eV/Å | 0.019783 | 0.006544 | 0.007135 | 0.007618 | 0.006980 |

The three-mask mean and sample standard deviation are 0.243028 ± 0.020064 eV/config for aligned energy, 0.022594 ± 0.000411 eV/Å for H force, and 0.007244 ± 0.000333 eV/Å for Cu force. These three observations describe mask-to-mask variation under one fixed training seed. They do not quantify variation over training seeds, independent datasets, or all possible adapter subsets. They are not confidence intervals.

All three masks improve the listed force MAEs and aligned energy MAE relative to U0, while their raw energy MAEs are higher. Their H and Cu force MAEs are all higher than Full LoRA's. Mask 42 has the best aligned energy MAE among these runs but uses fewer parameters, so its result alone does not identify why it performs differently.

The same 100 validation configurations were used for checkpoint selection and reporting. The original selector configuration proposes separate 50-frame selector-development and checkpoint-validation portions, which these runs did not implement. The 200-frame test set was not used. No force-response or gradient-based selectors have been evaluated here. Thus these results do not establish independent test performance or a complementarity advantage. The pretrained checkpoint copy has not had its full hash reverified.

## Files and reproducibility

- `selection_reproduce.py` reproduces the complete candidate list and all three selections.
- `analysis/` contains individual-run plots, three-mask metrics and convergence plots, analysis code, and machine-readable results. Run `python analysis/analyze_repeats.py` to regenerate the three-mask plots from saved metrics and logs. `analysis/analyze.py` recreates the initial force plots and requires the original validation extxyz.
- `training/` contains model implementation, training and evaluation scripts, configs, logs, predictions, best and final adapters, and optimizer resume states for all runs.
- `reference/` contains saved U0 and Full LoRA predictions and validation results used for comparison.
- The original pretrained model checkpoint and DFT dataset are external dependencies and are not included. Training scripts retain original `/home/ubuntu/uma_cuh_baseline` paths and require the project's UMA/FAIRChem environment.

Source-index ordering, raw energy MAE, aligned energy MAE and H/Cu force MAEs were independently checked against original validation labels for each run. No test labels were inspected.
