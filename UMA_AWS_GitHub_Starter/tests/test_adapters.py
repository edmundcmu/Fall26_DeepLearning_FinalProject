"""Tests for scripts/adapters.py.

The first part uses small stand-ins with the same forward as fairchem's SO3_Linear
and MOLE. The second part builds a tiny randomly initialized UMA MoE backbone and
is skipped when fairchem is not installed.
"""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from adapters import LoRAConfig, LoRAModel  # noqa: E402


class ToySO3(nn.Module):
    """Same forward as fairchem's SO3_Linear: one channel mix per degree l, bias on l = 0."""

    def __init__(self, cin, cout, lmax):
        super().__init__()
        self.lmax = lmax
        self.weight = nn.Parameter(torch.randn(lmax + 1, cout, cin) / math.sqrt(cin))
        self.bias = nn.Parameter(torch.randn(cout))
        index = torch.cat([torch.full((2 * l + 1,), l) for l in range(lmax + 1)])
        self.register_buffer("expand_index", index, persistent=False)

    def forward(self, x):
        out = torch.einsum("bmi, moi -> bmo", x, self.weight.index_select(0, self.expand_index))
        out[:, 0:1, :] = out.narrow(1, 0, 1) + self.bias
        return out


class Mixing:
    def __init__(self, coefficients):
        self.expert_mixing_coefficients = coefficients


class ToyMOLE(nn.Module):
    """Mixture of linear experts merged by routing coefficients, like fairchem's MOLE."""

    def __init__(self, cin, cout, experts):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(experts, cout, cin) / math.sqrt(cin))
        self.bias = nn.Parameter(torch.randn(cout))
        self.global_mole_tensors = Mixing(torch.softmax(torch.randn(1, experts), dim=1))

    def forward(self, x):
        coeffs = self.global_mole_tensors.expert_mixing_coefficients
        return F.linear(x, torch.einsum("eoi, be -> boi", self.weights, coeffs)[0], self.bias)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.radial = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 6))
        self.so3 = ToySO3(6, 5, lmax=2)
        self.mole = ToyMOLE(5, 7, experts=3)
        self.head = nn.Linear(7, 1)

    def forward(self, s, x):
        # s: (N, 4) invariant scalars; x: (N, 9, 6) features of degree l <= 2
        y = self.so3(x * self.radial(s)[:, None, :])
        return self.head(self.mole(y[:, 0, :])).sum(), y


GROUPS = {"R": [r"^radial\.\d+$"], "S3": [r"^so3$"], "M": [r"^mole$"]}


def toy_config(**overrides):
    return LoRAConfig(groups=GROUPS, **overrides)


def inputs():
    g = torch.Generator().manual_seed(1)
    return torch.randn(3, 4, generator=g), torch.randn(3, 9, 6, generator=g)


def randomize_b(lora, groups, std=0.1):
    with torch.no_grad():
        for p in (p for g in groups for p in lora.lora_b_parameters(g)):
            p.normal_(std=std)


def random_wigner_like(seed=0):
    """Block-diagonal rotation acting on the m index: 1 for l = 0, random SO(2l+1) blocks."""
    g = torch.Generator().manual_seed(seed)
    blocks = [torch.ones(1, 1)]
    for l in (1, 2):
        q, r = torch.linalg.qr(torch.randn(2 * l + 1, 2 * l + 1, generator=g))
        q = q * torch.sign(torch.diagonal(r))
        if torch.det(q) < 0:
            q[:, 0] = -q[:, 0]
        blocks.append(q)
    return torch.block_diag(*blocks)


def test_attach_starts_at_u0_and_freezes_base():
    model, reference = Toy(), Toy()
    lora = LoRAModel(model, toy_config())
    lora.reset(lora.group_ids(), seed=0)
    lora.activate(lora.group_ids())
    s, x = inputs()
    (e, y), (e_ref, y_ref) = model(s, x), reference(s, x)
    assert torch.equal(e, e_ref) and torch.equal(y, y_ref)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in n for n in trainable)
    assert lora.unassigned == ["head"]


def test_linear_update_matches_explicit_weight():
    lora = LoRAModel(Toy(), toy_config())
    lora.reset(["R"], seed=1)
    lora.activate(["R"])
    randomize_b(lora, ["R"])
    layer = lora.layers["R"][0]
    x = torch.randn(5, 4)
    weight = layer.base_layer.weight + layer.delta_weight()[0]
    assert torch.allclose(layer(x), F.linear(x, weight, layer.base_layer.bias), atol=1e-6)
    assert torch.allclose(layer.weight, weight)  # code that reads .weight sees the update


@pytest.mark.parametrize("per_l", [True, False])
def test_so3_update_is_rotation_equivariant(per_l):
    lora = LoRAModel(Toy(), toy_config(so3_per_l=per_l))
    lora.reset(["S3"], seed=2)
    lora.activate(["S3"])
    randomize_b(lora, ["S3"], std=0.5)
    layer = lora.layers["S3"][0]
    assert layer.n_slices == (3 if per_l else 1)
    rot = random_wigner_like()
    x = torch.randn(4, 9, 6)
    rotated_then_layer = layer(torch.einsum("mn, bnc -> bmc", rot, x))
    layer_then_rotated = torch.einsum("mn, bnc -> bmc", rot, layer(x))
    assert torch.allclose(rotated_then_layer, layer_then_rotated, atol=1e-5)


def test_mole_update_does_not_depend_on_routing():
    lora = LoRAModel(Toy(), toy_config())
    lora.reset(["M"], seed=3)
    lora.activate(["M"])
    randomize_b(lora, ["M"])
    layer = lora.layers["M"][0]
    x = torch.randn(4, 5)
    before = layer(x) - layer.base_layer(x)
    layer.base_layer.global_mole_tensors.expert_mixing_coefficients = torch.tensor([[0.8, 0.1, 0.1]])
    assert torch.allclose(layer(x) - layer.base_layer(x), before, atol=1e-6)


@pytest.mark.parametrize("per_l, so3_params", [(True, 3 * 2 * (6 + 5)), (False, 2 * (6 + 5))])
def test_param_count_follows_eq3(per_l, so3_params):
    lora = LoRAModel(Toy(), toy_config(so3_per_l=per_l))
    assert lora.param_count(["R"]) == 2 * (4 + 8) + 2 * (8 + 6)
    assert lora.param_count(["S3"]) == so3_params
    assert lora.param_count(["M"]) == 2 * (5 + 7)
    lora.activate(["R", "M"])
    assert lora.parameter_summary()["trainable"] == lora.param_count(["R", "M"])


def test_group_rank_override():
    lora = LoRAModel(Toy(), toy_config(group_ranks={"M": 4}))
    assert lora.layers["M"][0].rank == 4 and lora.param_count(["M"]) == 4 * (5 + 7)


def test_activate_selects_contributing_and_trainable_groups():
    model, reference = Toy(), Toy()
    lora = LoRAModel(model, toy_config())
    lora.reset(lora.group_ids(), seed=4)
    randomize_b(lora, lora.group_ids())
    lora.activate([])
    s, x = inputs()
    assert torch.equal(model(s, x)[0], reference(s, x)[0])
    lora.activate(["S3"])
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"so3.lora_A", "so3.lora_B"}
    assert lora.active_groups() == ["S3"]
    assert not torch.equal(model(s, x)[0], reference(s, x)[0])


def test_reset_is_seeded_per_group():
    a, b = LoRAModel(Toy(), toy_config()), LoRAModel(Toy(), toy_config())
    a.reset(["R"], seed=7)
    b.reset(["R"], seed=7)
    assert torch.equal(a.layers["R"][0].lora_A, b.layers["R"][0].lora_A)
    b.reset(["R"], seed=8)
    assert not torch.equal(a.layers["R"][0].lora_A, b.layers["R"][0].lora_A)
    assert all(torch.count_nonzero(p) == 0 for p in a.lora_b_parameters("R"))


def test_coefficient_scales_the_update_linearly():
    lora = LoRAModel(Toy(), toy_config())
    lora.reset(["R"], seed=5)
    lora.activate(["R"])
    randomize_b(lora, ["R"])
    layer = lora.layers["R"][0]
    x = torch.randn(3, 4)
    full = layer(x) - layer.base_layer(x)
    lora.set_coefficients({"R": 0.5})
    assert torch.allclose(layer(x) - layer.base_layer(x), 0.5 * full, atol=1e-6)


def test_relative_update_and_rescale():
    lora = LoRAModel(Toy(), toy_config(so3_per_l=False))
    lora.reset(lora.group_ids(), seed=6)
    randomize_b(lora, lora.group_ids())
    for group in lora.group_ids():
        before = lora.relative_update(group)
        lora.rescale_update(group, {site: 0.5 for site in before})
        after = lora.relative_update(group)
        assert all(math.isclose(after[s], 0.5 * before[s], rel_tol=1e-5) for s in before)
    # A shared SO(3) update counts once per degree slice of W.
    layer = lora.layers["S3"][0]
    assert math.isclose(float(layer.update_norm()), float(layer.delta_weight().norm()) * math.sqrt(3), rel_tol=1e-6)


def test_save_and_load_roundtrip(tmp_path):
    source = LoRAModel(Toy(), toy_config())
    source.reset(source.group_ids(), seed=9)
    randomize_b(source, source.group_ids())
    source.set_coefficients({"M": 0.7})
    source.activate(["R", "M"])
    source.save_adapters(tmp_path / "adapters.pt", metadata={"run": "test"})
    target_model = Toy()
    target = LoRAModel(target_model, toy_config())
    assert target.load_adapters(tmp_path / "adapters.pt") == {"run": "test"}
    s, x = inputs()
    assert torch.equal(source.model(s, x)[0], target_model(s, x)[0])
    assert target.active_groups() == ["R", "M"]


def test_dropout_only_at_invariant_sites():
    lora = LoRAModel(Toy(), toy_config(dropout=0.3, invariant_input=[r"^radial\.", r"^so3$"]))
    assert all(isinstance(l.lora_dropout, nn.Dropout) for l in lora.layers["R"])
    assert isinstance(lora.layers["S3"][0].lora_dropout, nn.Identity)  # l > 0 inputs: never
    assert isinstance(lora.layers["M"][0].lora_dropout, nn.Identity)


def test_configuration_errors():
    with pytest.raises(ValueError, match="disjoint"):
        LoRAModel(Toy(), LoRAConfig(groups={"a": [r"^radial\.0$"], "b": [r"^radial\.\d$"]}))
    with pytest.raises(ValueError, match="matched no sites"):
        LoRAModel(Toy(), LoRAConfig(groups={"a": [r"^radial\.0$"], "typo": [r"^radail"]}))
    with pytest.raises(ValueError, match="unknown groups"):
        LoRAConfig(groups=GROUPS, group_ranks={"X": 4})
    model = Toy()
    lora = LoRAModel(model, toy_config())
    with pytest.raises(ValueError, match="already has LoRA"):
        LoRAModel(model, toy_config())
    with pytest.raises(KeyError):
        lora.activate(["X"])


def test_force_loss_reaches_b_but_not_a_at_initialization():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1))
    lora = LoRAModel(model, LoRAConfig(groups={"first": [r"^0$"]}))
    lora.reset(["first"], seed=0)
    lora.activate(["first"])
    pos = torch.randn(5, 3, requires_grad=True)
    forces = -torch.autograd.grad(model(pos).sum(), pos, create_graph=True)[0]
    forces.pow(2).sum().backward()
    layer = lora.layers["first"][0]
    assert torch.isfinite(layer.lora_B.grad).all() and layer.lora_B.grad.abs().sum() > 0
    assert torch.count_nonzero(layer.lora_A.grad) == 0  # B = 0 gives dL/dA = 0


def test_project_config_loads():
    pytest.importorskip("yaml")
    config = LoRAConfig.from_yaml(ROOT / "configs/adapter_selection.yaml")
    assert sorted(config.groups) == ["A_early", "A_late", "R_early", "R_late", "S_early", "S_late"]
    assert config.rank == 2 and config.alpha is None and config.scaling_for("S_early") == 1.0


# --- tiny UMA MoE backbone (requires fairchem) --------------------------------------------

def tiny_uma():
    pytest.importorskip("fairchem.core")
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


def test_tiny_uma_groups_capture_expected_sites():
    backbone, _ = tiny_uma()
    lora = LoRAModel(backbone, tiny_config())
    kinds = {g: sorted({row["kind"] for row in lora.describe() if row["group"] == g}) for g in lora.group_ids()}
    assert kinds["R_early"] == ["linear"] and len(lora.sites("R_early")) == 3
    assert kinds["S_early"] == ["mole"] and len(lora.sites("S_early")) == 6
    assert kinds["A_early"] == ["linear", "so3"] and len(lora.sites("A_early")) == 3
    assert lora.unassigned == []  # every legal backbone site belongs to a group


def test_tiny_uma_starts_at_u0_and_adapters_change_it():
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


def test_tiny_uma_with_adapters_stays_rotation_equivariant():
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


def test_tiny_uma_force_loss_backpropagates_to_adapters_only():
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


def test_tiny_uma_refuses_mole_merge():
    backbone, _ = tiny_uma()
    from fairchem.core.models.uma.nn.mole_utils import replace_MOLE_with_linear

    lora = LoRAModel(backbone, tiny_config())
    with pytest.raises(RuntimeError, match="traineval"):
        replace_MOLE_with_linear(lora.layers["S_early"][0])
