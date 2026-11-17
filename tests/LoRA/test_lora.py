"""Tests for main.LoRA on small stand-ins with the same forward as fairchem's
SO3_Linear and MOLE. Needs only PyTorch; test_lora_uma.py covers a real tiny UMA.
"""
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from main.LoRA import LoRAConfig, LoRALinear, LoRAModel, match_sites

ROOT = Path(__file__).resolve().parents[2]


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


# --- sites -------------------------------------------------------------------------------

def test_match_sites_assigns_groups_and_reports_unassigned():
    sites, unassigned = match_sites(Toy(), toy_config())
    assert [(s.name, s.group, s.kind) for s in sites] == [
        ("radial.0", "R", "linear"), ("radial.2", "R", "linear"), ("so3", "S3", "so3"), ("mole", "M", "mole")]
    assert unassigned == ["head"]


# --- layers ------------------------------------------------------------------------------

def test_layer_works_without_lora_model():
    base = nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2, scaling=1.0, dropout=0.0, site="x", group="g")
    layer.reset(torch.Generator().manual_seed(0))
    layer.enabled = True
    x = torch.randn(2, 4)
    assert torch.equal(layer(x), base(x))  # B = 0
    assert layer.param_count == 2 * (4 + 3)


def test_linear_update_matches_explicit_weight(randomize_b):
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
def test_so3_update_is_rotation_equivariant(per_l, randomize_b):
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


def test_mole_update_does_not_depend_on_routing(randomize_b):
    lora = LoRAModel(Toy(), toy_config())
    lora.reset(["M"], seed=3)
    lora.activate(["M"])
    randomize_b(lora, ["M"])
    layer = lora.layers["M"][0]
    x = torch.randn(4, 5)
    before = layer(x) - layer.base_layer(x)
    layer.base_layer.global_mole_tensors.expert_mixing_coefficients = torch.tensor([[0.8, 0.1, 0.1]])
    assert torch.allclose(layer(x) - layer.base_layer(x), before, atol=1e-6)


def test_coefficient_scales_the_update_linearly(randomize_b):
    lora = LoRAModel(Toy(), toy_config())
    lora.reset(["R"], seed=5)
    lora.activate(["R"])
    randomize_b(lora, ["R"])
    layer = lora.layers["R"][0]
    x = torch.randn(3, 4)
    full = layer(x) - layer.base_layer(x)
    lora.set_coefficients({"R": 0.5})
    assert torch.allclose(layer(x) - layer.base_layer(x), 0.5 * full, atol=1e-6)


def test_relative_update_and_rescale(randomize_b):
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


# --- model -------------------------------------------------------------------------------

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


def test_activate_selects_contributing_and_trainable_groups(randomize_b):
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


def test_save_and_load_roundtrip(tmp_path, randomize_b):
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


# --- config ------------------------------------------------------------------------------

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


def test_project_config_loads():
    pytest.importorskip("yaml")
    config = LoRAConfig.from_yaml(ROOT / "configs/adapter_selection.yaml")
    assert sorted(config.groups) == ["A_early", "A_late", "R_early", "R_late", "S_early", "S_late"]
    assert config.rank == 2 and config.alpha is None and config.scaling_for("S_early") == 1.0
