# UMA-M-1.1 U0 Baseline Results

## Experiment
- Model: UMA-M-1.1 (uma-m-1p1)
- Task: oc20
- Method: Frozen pretrained inference (U0)
- Fine-tuning: None
- Reference: SRP48 DFT energies and forces
- Validation configurations: 100

## Validation Results

| Metric | Value |
|---|---:|
| H Force MAE | 0.032652 eV/Angstrom |
| Cu Force MAE | 0.019783 eV/Angstrom |
| Overall Force MAE | 0.020081 eV/Angstrom |
| Raw Energy MAE | 1.390870 eV/configuration |
| Train-offset-aligned Energy MAE | 0.541098 eV/configuration |

## Result Files
- reports/U0_val/metrics.json
- reports/U0_val/val_predictions.npz
- reports/U0_val/train_predictions.npz
- data/prepared/manifest.json

## Limitations
These results represent pretrained UMA inference,
not LoRA fine-tuning or full-backbone fine-tuning.

Energy offsets were fitted using training data only.
The results do not establish hydrogen diffusion
barriers or molecular dynamics accuracy.
