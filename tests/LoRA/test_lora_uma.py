"""Tests for src.LoRA on a tiny randomly initialized UMA MoE backbone.

Skipped when fairchem is not installed.
"""
import copy
from pathlib import Path

import pytest
import torch

from src.LoRA import LoRAConfig, LoRAModel

pytest.importorskip("fairchem.core")

ROOT = Path(__file__).resolve().parents[2]


def tiny_uma():
    from fairchem.core.models.uma.escn_md import MLP_EFS_Head
    from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone

    torch.manual_seed(0)
    backbone = eSCNMDMoeBackbone(
        num_experts=4, num_layers=2, sphere_channels=16, hidden_channels=16, edge_channels=16,
        lmax=2, mmax=2, num_distance_basis=16, cutoff=5.0, otf_graph=True, max_neighbors=30,
        ff_type="spectral", direct_forces=False, regress_forces=True,
        dataset_mapping={"oc20": "oc20"}, use_dataset_embedding=True, always_use_pbc=True)
    return backbone, MLP_EFS_Head(backbone)


def tiny_config():
    # The project's group patterns, with block ranges narrowed to the tiny model's two blocks.
    project = LoRAConfig.from_yaml(ROOT / "configs/adapter_selection.yaml")
    groups = {g: [p.replace("[0-4]", "0").replace("[5-9]", "1") for p in pats]
              for g, pats in project.groups.items()}
    return LoRAConfig(groups=groups, rank=2)


def cu_h_slab(rotation=None):
    from ase.build import add_adsorbate, fcc111

    atoms = fcc111("Cu", size=(2, 2, 3), vacuum=6.0)
    add_adsorbate(atoms, "H", 1.5, "fcc")
    add_adsorbate(atoms, "H", 2.5, "hcp")
    atoms.info.update(charge=0, spin=0)
    if rotation is not None:
        atoms.set_cell(atoms.cell.array @ rotation.T, scale_atoms=False)
        atoms.positions = atoms.positions @ rotation.T
    return atoms


def energy_forces(backbone, head, atoms):
    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    data = atomicdata_list_to_batch(
        [AtomicData.from_ase(atoms, task_name="oc20", r_edges=False, r_data_keys=["spin", "charge"])])
    data["pos"].requires_grad_(True)
    out = head(data, backbone(data))
    unwrap = lambda v, k: v[k] if isinstance(v, dict) else v  # noqa: E731
    return unwrap(out["energy"], "energy"), unwrap(out["forces"], "forces")


def test_groups_capture_expected_sites():
    backbone, _ = tiny_uma()
    lora = LoRAModel(backbone, tiny_config())
    kinds = {g: sorted({row["kind"] for row in lora.describe() if row["group"] == g}) for g in lora.group_ids()}
    assert kinds["R_early"] == ["linear"] and len(lora.sites("R_early")) == 3
    assert kinds["S_early"] == ["mole"] and len(lora.sites("S_early")) == 6
    assert kinds["A_early"] == ["linear", "so3"] and len(lora.sites("A_early")) == 3
    assert lora.unassigned == []  # every legal backbone site belongs to a group


def test_starts_at_u0_and_adapters_change_it(randomize_b):
    backbone, head = tiny_uma()
    ref_backbone, ref_head = copy.deepcopy(backbone), copy.deepcopy(head)
    lora = LoRAModel(backbone, tiny_config())
    lora.reset(lora.group_ids(), seed=0)
    lora.activate(lora.group_ids())
    for module in (backbone, head, ref_backbone, ref_head):
        module.eval()
    e, f = energy_forces(backbone, head, cu_h_slab())
    e_ref, f_ref = energy_forces(ref_backbone, ref_head, cu_h_slab())
    # Repeated CPU forwards of the same model already differ by ~1e-7 (float32 scatter
    # order), so equivalence is checked at that noise level rather than bitwise.
    assert torch.allclose(e, e_ref, rtol=0, atol=1e-6) and torch.allclose(f, f_ref, rtol=0, atol=1e-6)
    randomize_b(lora, lora.group_ids(), std=0.05)
    e2, f2 = energy_forces(backbone, head, cu_h_slab())
    assert not torch.allclose(e2, e_ref) and not torch.allclose(f2, f_ref)


def test_adapted_model_stays_rotation_equivariant(randomize_b):
    backbone, head = tiny_uma()
    lora = LoRAModel(backbone, tiny_config())
    lora.reset(lora.group_ids(), seed=1)
    lora.activate(lora.group_ids())
    randomize_b(lora, lora.group_ids(), std=0.05)
    backbone.eval()
    head.eval()
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64))
    rotation = (q * torch.sign(torch.det(q))).numpy()
    e, f = energy_forces(backbone, head, cu_h_slab())
    e_rot, f_rot = energy_forces(backbone, head, cu_h_slab(rotation))
    assert torch.allclose(e_rot, e, rtol=1e-5, atol=1e-5)
    assert torch.allclose(f_rot, f @ torch.as_tensor(rotation, dtype=f.dtype).T, atol=1e-4)


def test_force_loss_backpropagates_to_adapters_only():
    backbone, head = tiny_uma()
    lora = LoRAModel(backbone, tiny_config())
    lora.reset(lora.group_ids(), seed=2)
    lora.activate(lora.group_ids())
    backbone.train()
    head.train()  # fairchem keeps the force graph only in train mode
    _, forces = energy_forces(backbone, head, cu_h_slab())
    forces.pow(2).sum().backward()
    for group in lora.group_ids():
        grads = [p.grad for p in lora.lora_b_parameters(group)]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0, group
    base_grads = [p.grad for n, p in backbone.named_parameters() if ".lora_" not in n]
    assert all(g is None for g in base_grads)


def test_refuses_mole_merge():
    backbone, _ = tiny_uma()
    from fairchem.core.models.uma.nn.mole_utils import replace_MOLE_with_linear

    lora = LoRAModel(backbone, tiny_config())
    with pytest.raises(RuntimeError, match="traineval"):
        replace_MOLE_with_linear(lora.layers["S_early"][0])
