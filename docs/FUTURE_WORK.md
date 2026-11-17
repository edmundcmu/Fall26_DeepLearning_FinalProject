# Future work: multi-fidelity Bayesian optimization (MF-BO) for adapter-mask selection

Status: **not in scope**. The project uses FA-JOINT (IDL §4) as the selection method.
This note records the extension we discussed so it can be added if time and compute permit.

## Idea
Treat the fully trained selector-development loss of a mask S as an expensive black-box
objective and search the feasible masks with a multi-fidelity surrogate:

| Fidelity | Evaluation of mask S | Cost |
|---|---|---|
| z0 (low) | FA-JOINT proxy J(S) from probe responses (Eq. 5) | probes once, then CPU only |
| z1 (mid) | train S for ~200 updates, selector-dev objective | short GPU run |
| z2 (high) | train S for the full 2,000 updates, selector-dev objective | full GPU run |

- Surrogate: GP over (mask, fidelity), e.g. autoregressive co-kriging
  f_z2(S) = rho * f_z0(S) + delta(S), delta ~ GP with a kernel on binary masks.
- Acquisition: cost-aware (EI per unit cost or multi-fidelity knowledge gradient),
  restricted to masks within the parameter budget (e.g. BoTorch).
- Output: the mask with the lowest posterior mean at z2, then the same fresh training as every other method.

## Rules for a fair comparison with FA-JOINT
- Report selection cost separately from final-training cost (IDL §4.3) and compare
  accuracy against selection cost (Pareto), not accuracy alone.
- Score every fidelity on selector-dev only; never checkpoint-validation or test.
- The 4-6 longer-trained masks from the IDL §4.3 proxy check can serve as z2 observations
  if they are scored on selector-dev.
