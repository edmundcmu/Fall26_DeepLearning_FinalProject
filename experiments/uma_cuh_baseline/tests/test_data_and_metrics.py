import json
from pathlib import Path
import sys
import subprocess
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import ROOT, metrics, fingerprint, sha256


def test_species_metrics_do_not_hide_h_error():
    a = Atoms("HCu10", positions=np.zeros((11,3)))
    a.calc = SinglePointCalculator(a, energy=0., forces=np.zeros((11,3)))
    f = np.zeros((11,3)); f[0] = [1,2,3]
    m = metrics([a], [0.], [f])
    assert m["forces"]["H"]["mae_eV_per_A"] == 2.
    assert m["forces"]["Cu"]["mae_eV_per_A"] == 0.
    assert np.isclose(m["forces"]["all"]["mae_eV_per_A"], 2/11)


def test_energy_offset_is_supplied_not_fitted_to_test():
    a = Atoms("HCu", positions=np.zeros((2,3)))
    a.calc = SinglePointCalculator(a, energy=5., forces=np.zeros((2,3)))
    m = metrics([a], [10.], [np.zeros((2,3))], {a.get_chemical_formula():3.})
    assert m["raw_energy_mae_eV_per_configuration"] == 5.
    assert m["train_offset_aligned_energy_mae_eV_per_configuration"] == 2.


def test_fingerprint_atom_permutation():
    a = Atoms("HCu", positions=[[1,0,0],[0,0,0]], cell=[5,5,5], pbc=True)
    assert fingerprint(a) == fingerprint(a[[1,0]])


def test_download_and_group_disjointness():
    lock = json.loads((ROOT / "source_lock.json").read_text())
    assert sha256(ROOT / "data/raw/h2cu.xyz") == lock["data_sha256"]
    m = json.loads((ROOT / "data/prepared/manifest.json").read_text())
    groups, ids, fingerprints = {}, {}, {}
    for split in ["train", "val", "test"]:
        rows = [r for r in m["structures"] if r["split"] == split]
        ids[split] = {r["source_index"] for r in rows}
        groups[split] = {tuple(r["cell"]) for r in rows}
        fingerprints[split] = {r["fingerprint"] for r in rows}
        assert len(rows) == m["selected_counts"][split]
        assert sha256(ROOT / f"data/prepared/{split}/structures.extxyz") == m["file_sha256"][split]
    for x,y in [("train","val"),("train","test"),("val","test")]:
        assert not (ids[x] & ids[y])
        assert not (groups[x] & groups[y])
        assert not (fingerprints[x] & fingerprints[y])


def test_test_evaluation_requires_protocol_record(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
        "--checkpoint", "not-a-checkpoint.pt", "--label", "U0", "--split", "test",
        "--output", str(tmp_path / "evaluation")], capture_output=True, text=True)
    assert r.returncode != 0
    assert "Test is sealed" in r.stderr
    assert not (tmp_path / "evaluation").exists()
