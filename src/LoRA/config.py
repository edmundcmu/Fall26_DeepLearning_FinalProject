"""LoRAConfig: the adapter groups and LoRA hyperparameters (IDL §4.1).

Loaded from the `adapters` section of configs/adapter_selection.yaml.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Heads, conditioning, routing and the initial embedding stay frozen (IDL §4.1).
DEFAULT_EXCLUDE = (
    r"(^|\.)output_heads\.",
    r"(^|\.)energy_block\.",
    r"(^|\.)mix_csd$",
    r"(^|\.)routing_mlp\.",
    r"(^|\.)edge_degree_embedding\.",
)
# Sites whose inputs are invariant scalars, where dropout cannot break equivariance.
DEFAULT_INVARIANT_INPUT = (r"\.rad_func\.net\.\d+$", r"\.scalar_mlp\.\d+$")


@dataclass
class LoRAConfig:
    groups: dict[str, list[str]]  # group id -> regexes matched against full module names
    rank: int = 2  # IDL §4.1
    alpha: float | None = None  # scaling = alpha / rank; None keeps Eq. (3): W + B A
    dropout: float = 0.0
    so3_per_l: bool = True  # one (A, B) pair per degree l instead of one shared pair
    group_ranks: dict[str, int] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    invariant_input: list[str] = field(default_factory=lambda: list(DEFAULT_INVARIANT_INPUT))

    def __post_init__(self):
        if not self.groups:
            raise ValueError("LoRAConfig needs at least one adapter group")
        self.groups = {g: [p] if isinstance(p, str) else list(p) for g, p in self.groups.items()}
        if self.rank < 1 or any(r < 1 for r in self.group_ranks.values()):
            raise ValueError("LoRA ranks must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        unknown = set(self.group_ranks) - set(self.groups)
        if unknown:
            raise ValueError(f"group_ranks names unknown groups: {sorted(unknown)}")
        for pattern in [*sum(self.groups.values(), []), *self.exclude, *self.invariant_input]:
            re.compile(pattern)

    def rank_for(self, group: str) -> int:
        return self.group_ranks.get(group, self.rank)

    def scaling_for(self, group: str) -> float:
        rank = self.rank_for(group)
        return (self.alpha if self.alpha is not None else rank) / rank

    def is_invariant_input(self, site: str) -> bool:
        return any(re.search(p, site) for p in self.invariant_input)

    @classmethod
    def from_dict(cls, values: dict) -> LoRAConfig:
        known = set(cls.__dataclass_fields__)
        extra = set(values) - known
        if extra:
            raise ValueError(f"unknown LoRA config keys: {sorted(extra)}")
        return cls(**{k: v for k, v in values.items() if v is not None or k == "alpha"})

    @classmethod
    def from_yaml(cls, path: str | Path, section: str = "adapters") -> LoRAConfig:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text())[section])
