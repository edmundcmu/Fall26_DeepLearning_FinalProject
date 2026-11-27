"""FA-JOINT budgeted adapter selection for UMA-M on Cu/H (IDL §3-§4).

Pipeline, in order:
    data.py       pinned Cu/H frames and the locked selector split
    objective.py  species-balanced objective (Eq. 2) and energy calibration
    probes.py     adaptation probes -> response signatures q_g
    selection.py  joint residual fit (Eq. 5), greedy selection (Eq. 6), controls
    train.py      fresh adapter training of a selected mask
    evaluate.py   final locked evaluation

Supporting packages: LoRA/ (the adapters) and utils/ (seeding, device).
"""
