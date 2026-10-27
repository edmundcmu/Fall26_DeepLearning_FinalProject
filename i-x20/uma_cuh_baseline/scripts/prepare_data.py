"""Audit real DFT labels and build a cell-family-held-out transfer pilot.

The two larger rectangular cell families train; the shortest-a rectangular
family validates; the oblique in-plane family tests. This is deliberately a
cell-family generalization test, not a reconstruction of a published split.
"""
import argparse
from collections import Counter, defaultdict
import json
import numpy as np
from ase.io import read, write
from common import ROOT, sha256, save_json, cell_key, fingerprint, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-size", type=int, default=500)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261023)
    args = p.parse_args()
    lock = json.loads((ROOT / "source_lock.json").read_text())
    raw = ROOT / "data/raw/h2cu.xyz"
    assert sha256(raw) == lock["data_sha256"]
    atoms = read(raw, ":", format="extxyz")
    families, hashes = defaultdict(list), defaultdict(list)
    for i, a in enumerate(atoms):
        assert set(a.numbers).issubset({1, 29})
        assert a.calc is not None and {"energy", "forces"}.issubset(a.calc.results)
        f = a.get_forces(apply_constraint=False)
        assert f.shape == (len(a), 3) and np.isfinite(f).all()
        assert np.isfinite(a.get_potential_energy()) and np.isfinite(a.positions).all()
        assert a.pbc.all() and abs(np.linalg.det(a.cell)) > 1
        assert not a.constraints, "New constraints require an explicit force-mask audit"
        families[cell_key(a)].append(i)
        hashes[fingerprint(a)].append(i)
    # Never silently train on inconsistent duplicate labels.
    duplicates = 0
    removed = set()
    for indices in hashes.values():
        duplicates += len(indices) - 1
        if len(indices) > 1:
            a = atoms[indices[0]]
            for i in indices[1:]:
                b = atoms[i]
                # Only remove verified identically ordered and identically labelled
                # records. Permuted/conflicting records require a manual audit.
                np.testing.assert_array_equal(a.numbers, b.numbers)
                np.testing.assert_allclose(a.positions, b.positions, atol=1e-10, rtol=0)
                np.testing.assert_allclose(a.get_forces(apply_constraint=False), b.get_forces(apply_constraint=False), atol=1e-10, rtol=0)
                assert abs(a.get_potential_energy()-b.get_potential_energy()) < 1e-10
                removed.add(i)
    families = {k: [i for i in ids if i not in removed] for k, ids in families.items()}
    assert len(families) == 4, "Cell-family assignment must be reviewed for a changed source"
    oblique = [k for k in families if abs(np.array(k).reshape(3, 3)[1, 0]) > 1]
    assert len(oblique) == 1
    test_key = oblique[0]
    val_key = min((k for k in families if k != test_key), key=lambda k: k[0])
    pool = {"train": [], "val": families[val_key], "test": families[test_key]}
    for k, ids in families.items():
        if k not in (val_key, test_key):
            pool["train"].extend(ids)
    rng = np.random.default_rng(args.seed)
    selected = {}
    for split, n in [("train", args.train_size), ("val", args.val_size), ("test", args.test_size)]:
        assert 0 < n <= len(pool[split])
        # Stratification uses composition, never label values.
        groups = defaultdict(list)
        for i in pool[split]:
            groups[atoms[i].get_chemical_formula()].append(i)
        counts = {k: int(n * len(v) / len(pool[split])) for k, v in groups.items()}
        for k in sorted(groups, key=lambda k: -(n * len(groups[k]) / len(pool[split]) - counts[k]))[:n-sum(counts.values())]:
            counts[k] += 1
        selected[split] = sorted(int(i) for k in sorted(groups)
                                 for i in rng.choice(groups[k], counts[k], replace=False))
    out = ROOT / "data/prepared"
    if out.exists():
        raise FileExistsError("Prepared split exists; preserve it or move it before generating a new protocol")
    out.mkdir(parents=True)
    inventory = []
    for split, ids in selected.items():
        dest = out / split
        dest.mkdir()
        frames = []
        for i in ids:
            a = atoms[i]
            a.info["source_index"] = i
            a.info["source_sha256"] = lock["data_sha256"]
            frames.append(a)
            inventory.append({"source_index": i, "split": split, "formula": a.get_chemical_formula(),
                              "cell": list(cell_key(a)), "fingerprint": fingerprint(a)})
        write(dest / "structures.extxyz", frames, format="extxyz")
        # Round trip checks preserve raw labels, geometry, species, PBC and ordering.
        reread = read(dest / "structures.extxyz", ":")
        for original, new in zip(frames, reread):
            np.testing.assert_array_equal(original.numbers, new.numbers)
            np.testing.assert_allclose(original.get_forces(apply_constraint=False), new.get_forces(apply_constraint=False), atol=1e-8)
            np.testing.assert_allclose(original.positions, new.positions, atol=1e-8)
            np.testing.assert_allclose(original.cell, new.cell, atol=1e-8)
            assert abs(original.get_potential_energy() - new.get_potential_energy()) < 1e-8
    manifest = {"source_sha256": lock["data_sha256"], "seed": args.seed,
                "split_policy": "cell-family-held-out; composition-stratified subsampling",
                "removed_duplicate_source_indices": sorted(removed),
                "source_trajectory_ids_available": False,
                "pretraining_overlap_audited": False,
                "pool_counts": {k: len(v) for k, v in pool.items()},
                "selected_counts": {k: len(v) for k, v in selected.items()},
                "file_sha256": {k: sha256(out/k/"structures.extxyz") for k in selected},
                "structures": inventory}
    save_json(out / "manifest.json", manifest)
    # A label-only, zero-force/constant-energy sanity reference, NOT UMA or MACE.
    train = [atoms[i] for i in selected["train"]]
    train_e = defaultdict(list)
    for a in train:
        train_e[a.get_chemical_formula()].append(a.get_potential_energy())
    mean_e = {k: float(np.mean(v)) for k, v in train_e.items()}
    dummy = {"model": "training-composition-mean energy; zero force; NOT UMA/MACE",
             "train_energy_means_eV": mean_e}
    for split in ["val", "test"]:
        frames = [atoms[i] for i in selected[split]]
        dummy[split] = metrics(frames, [mean_e[a.get_chemical_formula()] for a in frames],
                               [np.zeros((len(a), 3)) for a in frames])
    save_json(ROOT / "reports/sanity_reference.json", dummy)
    report = {"source_sha256": lock["data_sha256"], "n_configurations": len(atoms),
              "composition_counts": dict(Counter(a.get_chemical_formula() for a in atoms)),
              "single_H_configurations": sum(np.sum(a.numbers == 1) == 1 for a in atoms),
              "exact_duplicates_removed": duplicates, "n_unique_configurations": len(atoms)-len(removed),
              "cell_family_count": len(families),
              "force_labels": "all stored atoms; no constraints/mobility metadata supplied",
              "metadata_keys": sorted(atoms[0].info),
              "cell_families": [{"cell": list(k), "n": len(v),
                  "composition_counts": dict(Counter(atoms[i].get_chemical_formula() for i in v))} for k,v in families.items()],
              "splits": {k: {"n": len(v), "composition_counts": dict(Counter(atoms[i].get_chemical_formula() for i in v))} for k,v in selected.items()},
              "prepared_manifest_sha256": sha256(out / "manifest.json")}
    report["single_H_configurations"] = int(report["single_H_configurations"])
    save_json(ROOT / "reports/data_audit.json", report)
    print(json.dumps({k: report[k] for k in ["n_configurations", "composition_counts", "splits"]}, indent=2))


if __name__ == "__main__":
    main()
