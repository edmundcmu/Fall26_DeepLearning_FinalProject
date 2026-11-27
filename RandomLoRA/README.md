# Random LoRA preliminary run

This package records one random adapter selection and a validation comparison with U0 and full candidate-group LoRA for UMA-M-1.1 on Cu/H. The random mask and common parameter budget were not fixed in the original protocol, so this is a preliminary run rather than a completed three-mask study.

## Selection and training

- Budget: 171,520 parameters, or 50% of the full candidate-group LoRA count of 343,040.
- Sampling: enumerate all 11 inclusion-maximal nonempty subsets whose cost is at most the budget, in the order defined by `selection.py`, then select uniformly with Python `random.Random(41).choice`. Inclusion-maximal means no omitted group can be added without exceeding the budget.
- Selected groups: R_late, S_late, A_early. Their respective costs are 39,680, 99,840 and 32,000, totaling 171,520.
- Training seed: 41. Rank: 2. Starting point: frozen UMA-M-1.1 checkpoint, with newly initialized adapters. Full LoRA adapter weights were not loaded.
- Training: 500 structures, batch size 1, 2,000 updates, AdamW, learning rate 0.0001, weight decay 0.001. Validation every 250 updates. Fixed U0 training-derived composition offsets and normalization are used throughout.
- The best validation checkpoint was update 2,000, with normalized loss 0.109830. Training and validation evaluation both exited successfully. The logged average training-loop time was about 0.44 s per update, including validation, excluding model loading.

Run `python selection.py` to reproduce the mask. The maximal-subset distribution is a specific choice and need not match a uniform draw over all feasible subsets. Future methods should share the same budget and predeclared protocol.

## Validation results

Force MAEs use Cartesian components. Energy MAEs are in eV per configuration and force MAEs are in eV/Å. The same 100 configurations were used for checkpoint selection and reporting.

| Metric | U0 | Full LoRA | Random LoRA |
|---|---:|---:|---:|
| Raw energy MAE | 1.390870 | 1.464874 | 1.466617 |
| Aligned energy MAE | 0.541098 | 0.249369 | 0.247749 |
| H-force MAE | 0.032652 | 0.018114 | 0.022613 |
| Cu-force MAE | 0.019783 | 0.006544 | 0.007135 |

Relative to U0, Random LoRA lowers aligned energy MAE by 54.2%, H-force MAE by 30.7%, and Cu-force MAE by 63.9%. Its raw energy MAE is 5.4% higher. Relative to Full LoRA, H and Cu force MAEs are 24.8% and 9.0% higher, respectively. Aligned energy MAE is numerically 0.65% lower, a small difference without an uncertainty estimate.

One mask and one training seed do not establish the distribution of random-selection performance or the value of response-based selection. Results are on the validation set and do not establish test accuracy or diffusion performance. The 200-frame test set was not used.

## Files and reproduction

- `analysis/` contains PNG plots, comparison metrics and `analyze.py`.
- `training/` contains the Random LoRA configuration, scripts, model implementation, adapter checkpoints, resumable optimizer state, logs, and predictions.
- `reference/` contains the validation predictions and metrics used for the U0 and Full LoRA comparison.
- The original pretrained checkpoint and DFT dataset are external dependencies. The validation labels are not included in this package. Training scripts retain their original `/home/ubuntu/uma_cuh_baseline` paths. Their Python environment requires the project UMA/FAIRChem dependencies.

To regenerate the plots in an environment with NumPy, Matplotlib and ASE installed, provide the original validation data:

```bash
python analysis/analyze.py --validation-data /path/to/structures.extxyz
```

The plotting script verifies the source-index ordering of all three prediction files before computing force plots. Aligned-energy and aggregate MAE values come from the saved evaluation metrics, which use fixed U0 training-derived offsets.
