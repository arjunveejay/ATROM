"""Nelson network along the parameter trajectories of the R0.1_M6.0 gravitational collapse.

    python experiments/nelson/R0.1_M6.0/run.py

Needs the dataset (trajectories.npy and params.npz) in data_dir; see the README.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

CASE = "nelson"

SETTINGS = dict(
    # Data
    data_dir="datasets/grav_collapse/nelson",   # holds trajectories.npy and params.npz
    build_snapshots=True,                        # build the snapshot files from the dataset
    n_train=4096,                               # training trajectories
    n_test=2048,                                # testing trajectories
    qoi=["CO", "e-"],                           # QoIs: species the error is measured on
    log_cols=["nH", "Av", "uv_flux"],           # parameters taken on a log scale in the feature vector
    drop_cols=["Tgrain"],                       # parameters left out of the feature vector

    # Clustering
    n_clusters=8,                               # clusters to start from
    max_clusters=10000,
    min_cluster_size=50,
    increase_rank=True,                         # grow the basis when a cluster cannot be split
    rng_seed=42,                                # training/testing partition and cluster splits
    base_seed=0,                                # starting clusters

    # What to sweep
    taus=[0.01, 0.10, 0.40],                    # split tolerances
    etas=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12],   # energy a basis may discard
    ranks=[1, 3, 5, 7, 9],              # basis sizes of the global-basis ROM
    positivity_modes=(True,),                   # evaluate with the non-negativity correction
    hatrom_positivity=(True,),                  # same for hAT-ROM

    # Evaluation
    n_eval={"test": 2048, "train": 4096},       # trajectories evaluated per set
)

if __name__ == "__main__":
    from src.run import main
    main(__file__)
