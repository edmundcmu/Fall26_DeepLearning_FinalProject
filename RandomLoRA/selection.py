"""Reproduce the preliminary mask selected before training."""
import itertools
import random

costs = {
    'R_early': 39680,
    'R_late': 39680,
    'S_early': 99840,
    'S_late': 99840,
    'A_early': 32000,
    'A_late': 32000,
}
budget = 171520
items = list(costs)
feasible = []
for n in range(1, len(items) + 1):
    for subset in itertools.combinations(items, n):
        total = sum(costs[name] for name in subset)
        if total <= budget and all(total + costs[name] > budget for name in items if name not in subset):
            feasible.append((subset, total))
selection, count = random.Random(41).choice(feasible)
print(f'Candidates: {len(feasible)}')
print(f'Groups: {selection}; parameters: {count}')
