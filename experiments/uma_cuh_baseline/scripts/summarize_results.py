"""Make an evidence table. Missing model runs stay blank, never become scores."""
import csv
import json
from common import ROOT

rows = []
for path in sorted((ROOT / "reports").glob("*/metrics.json")):
    m = json.loads(path.read_text())
    rows.append({"method": m["label"], "split": m["split"], "status": "measured",
        "H_force_MAE_eV_A": m["forces"]["H"]["mae_eV_per_A"],
        "Cu_force_MAE_eV_A": m["forces"]["Cu"]["mae_eV_per_A"],
        "energy_MAE_eV_configuration": m["raw_energy_mae_eV_per_configuration"],
        "aligned_energy_MAE_eV_configuration": m["train_offset_aligned_energy_mae_eV_per_configuration"],
        "source": str(path.relative_to(ROOT))})
for method in ["U0", "full-transfer-seed41"]:
    if not any(r["method"] == method for r in rows):
        rows.append({"method": method, "split": "not evaluated", "status": "NOT RUN"})
sanity = json.loads((ROOT / "reports/sanity_reference.json").read_text())["val"]
rows.append({"method": "Zero-force / train-composition-mean energy (NOT MLIP)",
    "split": "val", "status": "CPU sanity reference only",
    "H_force_MAE_eV_A": sanity["forces"]["H"]["mae_eV_per_A"],
    "Cu_force_MAE_eV_A": sanity["forces"]["Cu"]["mae_eV_per_A"],
    "energy_MAE_eV_configuration": sanity["raw_energy_mae_eV_per_configuration"],
    "source": "reports/sanity_reference.json"})
columns = ["method", "split", "status", "H_force_MAE_eV_A", "Cu_force_MAE_eV_A",
           "energy_MAE_eV_configuration", "aligned_energy_MAE_eV_configuration", "source"]
with open(ROOT / "reports/baseline_status.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=columns)
    w.writeheader(); w.writerows(rows)
print(ROOT / "reports/baseline_status.csv")
