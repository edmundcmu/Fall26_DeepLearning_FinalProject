"""Reproduce the three independently sampled random adapter masks."""
from itertools import combinations
from random import Random

COSTS = {
    'R_early': 39680, 'R_late': 39680,
    'S_early': 99840, 'S_late': 99840,
    'A_early': 32000, 'A_late': 32000,
}
BUDGET = 171520
names = list(COSTS)
candidates = []
for n in range(1, len(names) + 1):
    for subset in combinations(names, n):
        total = sum(COSTS[name] for name in subset)
        if total <= BUDGET and all(total + COSTS[name] > BUDGET for name in names if name not in subset):
            candidates.append((subset, total))
print(f'Budget: {BUDGET}; inclusion-maximal candidates: {len(candidates)}')
for index, (subset, total) in enumerate(candidates):
    print(f'{index:2d}: {", ".join(subset)} | {total}')
for seed in (41, 42, 43):
    index = Random(seed).randrange(len(candidates))
    subset, total = candidates[index]
    print(f'seed {seed}: candidate {index}, {", ".join(subset)}, {total} parameters')
