"""What the selector code needs from adapters.py, plus a CPU stand-in.

PSEUDOCODE: a proposed contract to agree with the adapter lead; adapters.py
implements the real class. FakeAdaptedUMA lets probes, training and selection
be tested on CPU before real UMA adapters exist.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol


class AdaptedUMAProtocol(Protocol):
    def group_ids(self) -> list[str]: ...

    def param_count(self, group_ids: list[str]) -> int: ...
    # c_g = sum over sites of r * (d_in + d_out)   (IDL Eq. 3)

    def reset(self, group_ids: list[str], seed: int) -> None: ...
    # fresh initialization: A random, B = 0, so the output equals U0

    def activate(self, group_ids: list[str]) -> None: ...
    # only these groups contribute and require grad; backbone, routing, heads stay frozen

    def set_coefficients(self, coeffs: dict[str, float]) -> None: ...
    # scale each group's Delta W by a_g (additivity check, IDL §4.3)

    def adapter_parameters(self, group_ids: list[str]) -> Iterator: ...

    def lora_b_parameters(self, group_id: str) -> Iterator: ...
    # B matrices only; used by the Force-Gradient control

    def relative_update(self, group_id: str) -> dict[str, float]: ...
    # per site: ||B A||_F / ||W||_F

    def rescale_update(self, group_id: str, factors: dict[str, float]) -> None: ...
    # per site: B <- factor * B (Delta W = B A scales linearly)

    def predict(self, frames: list, create_graph: bool): ...
    # returns (energy [n_sys], forces [n_atoms, 3]) with forces = -dE/dR

    def save_adapters(self, path) -> None: ...

    def load_adapters(self, path) -> None: ...


class FakeAdaptedUMA:
    """Toy model with the same API for CPU unit tests.

    Each fake group g owns a fixed random linear map J_g from its parameters
    to the stacked (E, F) outputs: prediction = base + sum_g J_g @ theta_g.
    Because J_g is known, probe responses and selection results are predictable.
    """

    def __init__(self, frames, group_sizes: dict[str, int], seed: int):
        # base outputs: random (E, F) per frame; J_g: random (M, size_g) per group
        # theta_g = zeros(size_g) for every group
        raise NotImplementedError

    # reset(g): theta_g = 0;  activate(g): mark trainable
    # predict(frames): base + sum over active g of J_g @ theta_g, returned as torch tensors
    # param_count(g) = size_g;  relative_update(g) = ||theta_g|| / 1.0
