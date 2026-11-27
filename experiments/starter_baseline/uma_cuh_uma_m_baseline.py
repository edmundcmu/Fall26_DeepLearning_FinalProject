"""Real DFT Cu-H baseline for Meta UMA; NO generated or fabricated scores.

Dataset: https://huggingface.co/datasets/colabfit/reactive_hydrogen_ACS_2023
Sourced from Stark et al. (2023), FHI-aims DFT-SRP, Cu surfaces + H2.

Usage: python uma_cuh_uma_m_baseline.py --inspect-only
       python uma_cuh_uma_m_baseline.py  # defaults to UMA-M-1.1, OC20 task

UMA-M-1.1 is a published energy-conserving checkpoint available through the
official Meta FAIR Chemistry model loader. The dataset contains H2/Cu,
not single-H diffusion; its DFT-SRP reference differs from UMA OC20's RPBE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

DATASET_REPO = "colabfit/reactive_hydrogen_ACS_2023"
DATASET_FILE = "co/co_0.parquet"
DATASET_DOI = "10.60732/d0801836"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dataset_file(work: Path, supplied: str | None) -> Path:
    if supplied:
        p = Path(supplied).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset parquet not found: {p}")
        return p
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset", local_dir=str(work / "dataset")))


def flatten_cell(value):
    """Convert one parquet nested cell to a normal list or number."""
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def source_label(row) -> str:
    names = flatten_cell(row.get("names"))
    if isinstance(names, (tuple, list)) and names:
        name = str(names[0])
    elif isinstance(names, str):
        name = names
    else:
        name = str(row.get("configuration_id", "unknown"))
    # Collapse frames from common parent/source paths. This is a conservative heuristic:
    # verify actual trajectory-family provenance before calling these an independent holdout.
    name = re.split(r"__\d+_results", name, maxsplit=1)[0]
    name = re.split(r"__\d+_forces", name, maxsplit=1)[0]
    return name


def check_config(row, facet: str, n_h: int, n_cu: int):
    try:
        numbers = np.asarray(flatten_cell(row["atomic_numbers"]), dtype=int).reshape(-1)
        positions = np.asarray(flatten_cell(row["positions"]), dtype=float).reshape(-1, 3)
        forces = np.asarray(flatten_cell(row["atomic_forces"]), dtype=float).reshape(-1, 3)
        cell = np.asarray(flatten_cell(row["cell"]), dtype=float).reshape(3, 3)
        energy = float(row["energy"])
        pbc = np.asarray(flatten_cell(row["pbc"]), dtype=bool).reshape(3)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"Invalid ColabFit row / missing DFT label: {exc}") from exc
    if set(numbers) - {1, 29} or np.count_nonzero(numbers == 1) != n_h or np.count_nonzero(numbers == 29) != n_cu:
        return None
    if (numbers.size != positions.shape[0] or numbers.size != forces.shape[0]
            or not np.isfinite(forces).all() or not np.isfinite(positions).all()
            or not np.isfinite(cell).all() or not np.isfinite(energy)):
        raise ValueError("Invalid shape or nonfinite DFT labels")
    src = source_label(row)
    if facet != "all":
        # Strict source tag. A frame without a clear facet source tag is NOT silently assigned.
        names = str(flatten_cell(row.get("names", ""))).lower()
        facet_tag = re.search(r"cu\s*\(?\s*(111|100|110|211)\s*\)?", names)
        if not facet_tag or facet_tag.group(1) != facet.replace("cu", ""):
            return None
    cid = str(row.get("configuration_id", ""))
    return {"id": cid, "group": src, "numbers": numbers, "positions": positions,
            "cell": cell, "pbc": pbc, "dft_forces": forces,
            "dft_energy": energy, "name": str(flatten_cell(row.get("names", "")))}


def prepare_data(parquet: Path, facet: str, n_h: int, n_cu: int):
    import pandas as pd
    df = pd.read_parquet(parquet)
    needed = {"atomic_numbers", "positions", "atomic_forces", "cell", "pbc", "energy"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required ColabFit columns: {sorted(missing)}. Available: {list(df.columns)}")
    items = []
    invalid = 0
    for _, row in df.iterrows():
        try:
            item = check_config(row, facet=facet, n_h=n_h, n_cu=n_cu)
        except ValueError:
            invalid += 1
            continue
        if item is not None:
            items.append(item)
    # Remove exact duplicate geometries (does not eliminate near-duplicates).
    unique = {}
    for item in items:
        key = hashlib.sha256(item["numbers"].tobytes() +
                             np.round(item["positions"], 6).tobytes() +
                             np.round(item["cell"], 6).tobytes()).hexdigest()
        unique.setdefault(key, item)
    result = list(unique.values())
    print(f"Rows in source: {len(df)}; compatible Cu{n_cu}H{n_h}: {len(items)}; "
          f"after exact geometry deduplication: {len(result)}; invalid/skipped rows: {invalid}")
    print("Dataset source groups:", Counter(i["group"] for i in result).most_common(15))
    if not result:
        raise ValueError("No compatible configurations. Try --facet all (strict facet tags may be absent), or inspect parquet.")
    return result, len(df)


def split_data(items, n_cal: int, n_test: int, seed: int):
    """Group-disjoint calibration and test when ≥2 source-name families exist."""
    rng = np.random.default_rng(seed)
    buckets = {}
    for x in items:
        buckets.setdefault(x["group"], []).append(x)
    keys = list(buckets)
    rng.shuffle(keys)
    if len(keys) < 2:
        raise ValueError("Only one source group. Refusing an unmarked non-independent holdout. Choose a broader facet / data source.")
    # Try group boundaries yielding both sufficiently large and disjoint sets.
    best = None
    for cut in range(1, len(keys)):
        a = [x for g in keys[:cut] for x in buckets[g]]
        b = [x for g in keys[cut:] for x in buckets[g]]
        if len(a) >= n_cal and len(b) >= n_test:
            score = abs(len(a) - n_cal) + abs(len(b) - n_test)
            if best is None or score < best[0]:
                best = (score, a, b, keys[:cut], keys[cut:])
    if best is None:
        raise ValueError("Cannot obtain requested source-group-disjoint calibration/test sizes. "
                         f"Found group sizes {[len(buckets[k]) for k in keys]}. "
                         "Reduce --calibration and/or --test or broaden --facet.")
    _, cal, test, akeys, bkeys = best
    rng.shuffle(cal); rng.shuffle(test)
    selected_cal, selected_test = cal[:n_cal], test[:n_test]
    assert not set(x["group"] for x in selected_cal) & set(x["group"] for x in selected_test)
    split_info = {"split_kind": "source-name-family groups (heuristic)",
                  "calibration_groups": akeys, "test_groups": bkeys,
                  "limitation": "Source-name-family grouping may not identify all near-duplicate MD structures."}
    return selected_cal, selected_test, split_info


def load_calculator(model: str, task: str):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Enable a GPU in Colab/runtime.")
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    predictor = pretrained_mlip.get_predict_unit(
        model, device="cuda", inference_settings="batch"
    )
    return FAIRChemCalculator(predictor, task_name=task)


def predict(items, calc):
    from ase import Atoms
    import torch
    rows = []
    for idx, item in enumerate(items):
        atoms = Atoms(numbers=item["numbers"], positions=item["positions"],
                      cell=item["cell"], pbc=item["pbc"])
        atoms.calc = calc
        t0 = time.perf_counter()
        e_pred = float(atoms.get_potential_energy())
        f_pred = np.asarray(atoms.get_forces(), dtype=float)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        if f_pred.shape != item["dft_forces"].shape or not (np.isfinite(f_pred).all() and np.isfinite(e_pred)):
            raise ValueError("UMA predicted nonfinite/incompatible outputs")
        rows.append({**item, "pred_energy": e_pred, "pred_forces": f_pred, "seconds": secs})
        print(f"{idx+1}/{len(items)} | {item['id'][:35]} | {secs:.2f}s", flush=True)
    return rows


def save_report(cal, test, outdir, args, parquet_sha, split_info, all_rows):
    import csv
    import matplotlib.pyplot as plt
    outdir.mkdir(exist_ok=True, parents=True)
    offset = float(np.mean([r["pred_energy"] - r["dft_energy"] for r in cal]))
    force_all_ref = np.concatenate([r["dft_forces"].reshape(-1) for r in test]);
    force_all_pred = np.concatenate([r["pred_forces"].reshape(-1) for r in test])
    h_ref = np.concatenate([r["dft_forces"][r["numbers"] == 1].reshape(-1) for r in test])
    h_pred = np.concatenate([r["pred_forces"][r["numbers"] == 1].reshape(-1) for r in test])
    cu_ref = np.concatenate([r["dft_forces"][r["numbers"] == 29].reshape(-1) for r in test])
    cu_pred = np.concatenate([r["pred_forces"][r["numbers"] == 29].reshape(-1) for r in test])
    en_ref = np.array([r["dft_energy"] for r in test]); en_pred = np.array([r["pred_energy"] for r in test]);
    def mae(a,b): return float(np.mean(np.abs(a-b)))
    result = {"status": "measured_after_successful_real_UMA_DFT_run",
              "model":args.model, "checkpoint_path": None,
              "checkpoint_sha256": None,
              "task":args.task, "dataset": DATASET_REPO, "dataset_file":DATASET_FILE,
              "dataset_sha256":parquet_sha, "dataset_doi":DATASET_DOI,
              "dft_settings": "FHI-aims DFT-SRP48; different from UMA oc20 RPBE reference",
              "composition": f"Cu{args.cu}H{args.h}", "facet_filter":args.facet,
              "calibration_configurations":len(cal), "test_configurations":len(test),
              "split":split_info, "calibration_offset_eV":offset,
              "H_force_MAE_eV_per_A":mae(h_ref,h_pred),
              "Cu_force_MAE_eV_per_A":mae(cu_ref,cu_pred),
              "all_force_MAE_eV_per_A":mae(force_all_ref,force_all_pred),
              "H_force_RMSE_eV_per_A":float(np.sqrt(np.mean((h_ref-h_pred)**2))),
              "energy_raw_MAE_eV_per_configuration":mae(en_ref,en_pred),
              "energy_offset_calibrated_MAE_eV_per_configuration":mae(en_ref,en_pred-offset),
              "energy_range_DFT_eV":float(np.ptp(en_ref)),
              "total_inference_seconds": float(sum(row["seconds"] for row in all_rows)),
              "caveats": ["H2/Cu surfaces, NOT one H atom diffusing on Cu(111)",
                          "DFT-SRP48/FHI-aims reference differs from UMA's OC20/RPBE/VASP reference",
                          "Calibration/test grouping derived heuristically from source names",
                          "Not an adapter or selection-method comparison",
                          "Check DFT constraints and publication-specific energies before final physical claims",
                          "UMA-M predicts forces from its energy; verify numerical consistency for any adapter implementation"]}
    (outdir/"results.json").write_text(json.dumps(result,indent=2))
    rows=[]
    for t in test:
        ef=np.abs(t["dft_forces"] - t["pred_forces"])
        rows.append({"id":t["id"],"group":t["group"],
                     "H_force_MAE_eV_A":float(ef[t["numbers"] == 1].mean()),
                     "Cu_force_MAE_eV_A":float(ef[t["numbers"] == 29].mean()),
                     "DFT_energy_eV":t["dft_energy"],
                     "UMA_energy_eV":t["pred_energy"],
                     "calibrated_UMA_energy_eV":t["pred_energy"]-offset,
                     "prediction_seconds":t["seconds"]})
    with (outdir/"per_configuration.csv").open("w", newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(outdir/"prediction_components.npz", H_ref=h_ref,H_pred=h_pred,
                        Cu_ref=cu_ref,Cu_pred=cu_pred, E_ref=en_ref,E_pred=en_pred,
                        E_pred_cal=en_pred-offset)
    def parity(ref, pred, xlabel,ylabel,title,path):
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(ref,pred,s=10,alpha=0.4,rasterized=True)
        low=min(float(ref.min()),float(pred.min()));high=max(float(ref.max()),float(pred.max()))
        ax.plot([low,high],[low,high],"k--",lw=1,label="Ideal agreement")
        ax.set(xlabel=xlabel,ylabel=ylabel,title=title);ax.legend();fig.tight_layout()
        fig.savefig(outdir/path,dpi=170);plt.close(fig)
    parity(h_ref,h_pred,"DFT H force component (eV/Å)","UMA H force component (eV/Å)",
           "Cu–H baseline: H force parity","H_force_parity.png")
    parity(cu_ref,cu_pred,"DFT Cu force component (eV/Å)","UMA Cu force component (eV/Å)",
           "Cu–H baseline: Cu force parity","Cu_force_parity.png")
    fig,ax=plt.subplots(figsize=(6,4));ax.hist(np.abs(h_ref-h_pred),bins=28,alpha=.7,label="H");
    ax.hist(np.abs(cu_ref-cu_pred),bins=28,alpha=.4,label="Cu")
    ax.set(xlabel="Absolute force component error (eV/Å)",ylabel="Components",
           title="Cu–H DFT comparison: force errors");ax.legend();fig.tight_layout()
    fig.savefig(outdir/"force_error_hist.png",dpi=170);plt.close(fig)
    parity(en_ref,en_pred-offset,"DFT total energy (eV)","UMA energy after calibration (eV)",
           "Cu–H baseline: energy parity","energy_parity.png")
    txt=f"""## 5. Preliminary results — Cu–H zero-shot baseline (executed)

We evaluated unadapted **{args.model}** with the **{args.task}** task on **{len(test)}** held-out
Cu{args.cu}H{args.h} configurations from the published **reactive_hydrogen_ACS_2023** dataset,
using **{len(cal)}** disjoint calibration configurations. Source dataset SHA256:
`{parquet_sha}`. Calibration/test groups were separated using parsed source-name
families; this is a pilot holdout and is not proof of full trajectory independence.

The DFT-reference Cartesian force-component MAE was **{result['H_force_MAE_eV_per_A']:.5f} eV/Å**
for hydrogen and **{result['Cu_force_MAE_eV_per_A']:.5f} eV/Å** for copper
(overall: **{result['all_force_MAE_eV_per_A']:.5f} eV/Å**).
The raw energy MAE was **{result['energy_raw_MAE_eV_per_configuration']:.3f} eV/configuration**;
with a single constant energy offset fitted on calibration-only data, the test error
was **{result['energy_offset_calibrated_MAE_eV_per_configuration']:.5f} eV/configuration**.
Total model evaluation time for all calibration and test systems was
**{result['total_inference_seconds']:.1f} seconds**. See the four accompanying plots.

**Interpretation limits.** These configurations concern molecular hydrogen (H₂) near
copper surfaces, NOT the diffusion of one H atom on Cu(111). Source labels use
FHI-aims DFT-SRP48, whereas UMA's `oc20` task approximates a different
RPBE/VASP reference. Thus these errors include differences between DFT reference
methods, not just UMA model error. No adapters were trained; no force-response
complementarity or greedy selection comparison has been established.
"""
    (outdir/"section_5_cuh_pilot.md").write_text(txt)
    print("\n",txt)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir",default="cuh_uma_m_work")
    parser.add_argument("--parquet",default=None,help="Local copy of co_0.parquet; otherwise download")
    parser.add_argument("--inspect-only",action="store_true",help="Inspect dataset and counts without CUDA or model")
    parser.add_argument("--model",default="uma-m-1p1",help="Published UMA-M-1.1 checkpoint (model ID: uma-m-1p1)")
    parser.add_argument("--task",default="oc20",help="oc20 chosen for Cu/H surface, but DFT settings differ")
    parser.add_argument("--facet",choices=["all","cu111","cu100","cu110","cu211"],default="all")
    parser.add_argument("--cu",type=int,default=54)
    parser.add_argument("--h",type=int,default=2)
    parser.add_argument("--calibration",type=int,default=12)
    parser.add_argument("--test",type=int,default=30)
    parser.add_argument("--seed",type=int,default=17)
    a=parser.parse_args(argv)
    work=Path(a.workdir).resolve();work.mkdir(parents=True,exist_ok=True)
    if a.model != "uma-m-1p1":
        raise ValueError("This notebook/script is locked to released UMA-M-1.1 (uma-m-1p1)")
    src=dataset_file(work,a.parquet)
    print("Dataset parquet:",src,"\nSHA256:",sha256(src))
    candidates,total=prepare_data(src,a.facet,a.h,a.cu)
    if a.inspect_only:
        print("Inspection completed. No model was run; no metrics have been measured.")
        return {"compatible_configurations":len(candidates),"source_rows":total}
    cal,test,split_info=split_data(candidates,a.calibration,a.test,a.seed)
    print("Cal/test split:",len(cal),len(test),split_info)
    calc=load_calculator(a.model,a.task)
    preds=predict(cal+test,calc)
    report=save_report(preds[:len(cal)],preds[len(cal):],work/"results",a,
                       sha256(src),split_info,preds)
    return report


if __name__=="__main__":
    main()
