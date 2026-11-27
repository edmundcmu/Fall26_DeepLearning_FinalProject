# Fall26_DeepLearning_FinalProject
This is the final project of Fall 2026 11-785 Introduction to Deep Learning.

## Environment setup

Run the following commands from the project root to create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate uma
```

## Repository layout

```
src/                      FA-JOINT pipeline (run from the repo root: python -m src.<module>)
  data.py  objective.py  probes.py  selection.py  train.py  evaluate.py
  LoRA/                   LoRA adapters: config, sites, layers, model, uma, interface
  utils/                  seeding, device and numeric settings
configs/                  locked experiment settings (adapter_selection.yaml, baseline.yaml)
tests/                    pytest suite (python -m pytest)
experiments/              baseline runs: uma_cuh_baseline (U0), starter_baseline
docs/                     AWS setup guide, future work
aws/                      deployment scripts for the shared EC2 instance
```

List the sites each adapter group captures in a UMA checkpoint:

```bash
python -m src.LoRA --checkpoint <uma-m-1p1.pt>
```
