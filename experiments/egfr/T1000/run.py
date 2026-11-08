"""EGFR signalling (ODEbase model 1452) over initial EGF concentrations from 5e-3 to 1 uM, 0-1000 s.

    python experiments/egfr/T1000/run.py

The dataset is generated on the first run if data_dir holds none.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

CASE = "egfr"

SETTINGS = dict(
    # Data
    data_dir="datasets/egfr/T1000",
    generate=[],                                # dataset generator options (its defaults), used if data_dir is empty
    build_snapshots=True,                        # build the snapshot files from the dataset
    qoi=["ppERK", "ppMEK", "EGF-pEGFR-2-pShc-Grb2-SOS-Cbl-CIN85-EPn", "RhoGTP", "pROK"],   # QoIs: species the error is measured on
    log_cols=["p"],                             # the EGF concentration is clustered on a log scale
    drop_cols=[],

    # Clustering
    n_clusters=8,                               # clusters to start from
    max_clusters=10000,
    min_cluster_size=150,
    increase_rank=True,                         # grow the basis when a cluster cannot be split
    rng_seed=42,                                # cluster splits (the training/testing trajectories are fixed by the dataset)
    base_seed=0,                                # starting clusters

    # What to sweep
    taus=[5e-2, 5e-3, 1e-3],                    # split tolerances
    etas=[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],        # energy a basis may discard
    ranks=list(range(1, 102, 10)),              # basis ranks of the global-basis ROM: 1, 11, ..., 101
    positivity_modes=(True,),                   # evaluate with the non-negativity correction
    hatrom_positivity=(True,),                  # same for hAT-ROM

    # Evaluation
    n_eval={"test": 199, "train": 200},         # trajectories evaluated per set
)

if __name__ == "__main__":
    from src.run import main
    main(__file__)
