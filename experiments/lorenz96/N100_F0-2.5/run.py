"""Lorenz-96 with N = 100 over forcing values F in (0, 2.5].

    python experiments/lorenz96/N100_F0-2.5/run.py

The dataset is generated on the first run if data_dir holds none.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

CASE = "lorenz96"

SETTINGS = dict(
    # Data
    data_dir="datasets/lorenz96/N100_F0-2.5",
    generate=["--N", "100"],                    # dataset generator options, used if data_dir is empty
    build_snapshots=True,                        # build the snapshot files from the dataset
    n_train=150,                                # training trajectories
    n_test=50,                                  # testing trajectories
    qoi=[0, 25, 50, 75],                        # QoIs: state components the error is measured on
    log_cols=[],
    drop_cols=[],

    # Clustering
    n_clusters=16,                              # clusters to start from
    max_clusters=10000,
    min_cluster_size=50,
    increase_rank=True,                         # grow the basis when a cluster cannot be split
    rng_seed=42,                                # training/testing partition and cluster splits
    base_seed=0,                                # starting clusters

    # What to sweep
    taus=[0.005, 0.01, 0.02, 0.05, 0.25],       # split tolerances, in state units
    etas=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10],   # energy a basis may discard
    ranks=list(range(2, 100, 4)),        
                  # basis sizes of the global-basis ROM
    positivity_modes=(False,),                  # a signed state needs no non-negativity correction
    hybrid_positivity=(),                       # no hybrid sweep

    # Evaluation
    n_eval={"test": 50, "train": 150},          # trajectories evaluated per set
)

if __name__ == "__main__":
    from src.run import main
    main(__file__)
