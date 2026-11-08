"""Astrochemistry: the Nelson and OSU gas-grain reaction networks, driven by parameter trajectories p(t) from a hydrodynamic simulation of gravitational collapse.

Holds what is fixed for these systems -- the network files, the interval length h, the solver -- and the loading of their datasets. The choices of each experiment live in ``experiments/<case>/<model>/run.py``.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
from scipy.linalg import qr as _qr
from sklearn.preprocessing import StandardScaler

from .. import REPO_ROOT
from .data import (N_LEADING, SUBSTEPS, _snapshot_paths, _idx_path, write_snapshots,
                   open_trajectories, read_params)
from .parser import Network, load_abundances

# Dataset row layout: [idx | t | params | species].
PARAM_NAMES = ["nH", "T", "Tgrain", "Av", "uv_flux"]
N_PARAMS = len(PARAM_NAMES)
SPECIES_COL0 = N_LEADING + N_PARAMS          # first species column
NSEC = 3600 * 24 * 365.25                    # seconds per Julian year

# Must match how the dataset's FOM snapshots were solved.
SELF_SHIELDING = True
DUST_ATTENUATION = False     # uv_flux is already attenuated; would double-count

# Log10-transformed when forming the training/testing partition, independent of the clustering's log_cols.
SPLIT_LOG_COLS = ("nH", "Av", "uv_flux")

# Fixed for both networks: abundances are positive, span decades and are stiff.
_FIXED = dict(
    dt_hydro_yr=250.0,       # interval length h, the time step of the hydrodynamic simulation
    ode_method="BDF",
    scale_method="pareto",   # diagonal scaling: each species scaled by 1/sqrt(std)
    qoi_log=True,            # cluster on log10 of the QoIs
    error_norm="relative",
    positivity=True,         # non-negativity correction on the abundances
    carry_step=True,         # each interval's solve starts from the previous interval's last step size
    atol=1e-6,
    rtol=1e-6,
)
DEFAULTS = {
    "nelson": dict(_FIXED,
                   network=os.path.join(REPO_ROOT, "networks/nelson/gas_reactions.in"),
                   abundances=os.path.join(REPO_ROOT, "networks/nelson/abundances.in")),
    "osu": dict(_FIXED,
                network=os.path.join(REPO_ROOT, "networks/osu_09_2008/gas_reactions.in"),
                abundances=os.path.join(REPO_ROOT, "networks/osu_09_2008/abundances.in")),
}


class ChemistryNetwork(Network):
    """The reaction network together with the map from the parameter p_k to an environment.

    The solver passes each interval's p_k to ``env``. Defining that map on the network allows the same ROM code to be used for a system with a different parameter set.
    """

    @staticmethod
    def env(p):
        return dict(nH=float(p[0]), T=float(p[1]), Tgrain=float(p[2]),
                    Av=float(p[3]), uv_flux=float(p[4]), Tcap_2body=True)


# ---------------------------------------------------------------------------
# Training / testing partition
# ---------------------------------------------------------------------------

def pivoted_qr_split(params, n_tracers, n_times, n_train, n_test, seed):
    """Split trajectories 50/50 at random, then order each half by column-pivoted QR.

    QR on the scaled parameter trajectories picks maximally diverse trajectories; running it per half leaves the training and testing sets equally diverse and identically distributed.
    """
    log_cols = [PARAM_NAMES.index(c) for c in SPLIT_LOG_COLS]
    flat = np.asarray(params, dtype=np.float64).copy()
    flat[:, log_cols] = np.log10(flat[:, log_cols])
    scaler = StandardScaler().fit(flat)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_tracers)
    mid = n_tracers // 2
    if n_train > mid or n_test > n_tracers - mid:
        raise ValueError(f"n_train <= {mid} and n_test <= {n_tracers - mid} "
                         f"are required for a 50-50 split of {n_tracers}.")

    def order(pool):
        X = scaler.transform(
            flat.reshape(n_tracers, n_times, N_PARAMS)[pool].reshape(-1, N_PARAMS)
        ).reshape(len(pool), -1)
        _, _, pivots = _qr(X.T, pivoting=True, mode="economic")
        return pool[pivots]

    return order(shuffled[:mid])[:n_train], order(shuffled[mid:])[:n_test]


def _failed_tracers(mm, starts):
    """Global ids of trajectories whose FOM solve failed.

    A failed solve writes a full block of NaN, so one row per trajectory spots it.
    """
    rows = np.asarray(mm[np.asarray(starts, dtype=np.int64), SPECIES_COL0:])
    return np.where(~np.isfinite(rows).all(axis=1))[0].astype(np.int64)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def build_network(cfg):
    """Construct the reaction network specified by ``cfg``.

    Returns ``(net, qoi_indices)``, where ``qoi_indices`` holds the species index of each entry of ``cfg.qoi``. Raises ``ValueError`` if a QoI is not a species of the network.
    """
    net = ChemistryNetwork(grains=True, self_shielding=SELF_SHIELDING, dust_attenuation=DUST_ATTENUATION)
    net.load_from_disk(cfg.network)
    net.drop_passive_species()
    net.initial_abundances = load_abundances(cfg.abundances)
    missing = [q for q in cfg.qoi if q not in net.species_map]
    if missing:
        raise ValueError(f"QoI {missing} are not in {cfg.network}.")
    return net, np.array([net.species_map[q] for q in cfg.qoi], dtype=int)


def load_context(cfg):
    """One-time setup shared by every fit and evaluation in a run."""
    # --- network ---
    net, qoi_indices = build_network(cfg)
    n_species = len(net.species)

    os.makedirs(cfg.out_dir, exist_ok=True)
    train_snap, test_snap = _snapshot_paths(cfg.out_dir)

    # --- parameters at the interval boundaries t_k ---
    data = read_params(cfg.data_dir)
    grid = data["params"]
    n_tracers, n_times = grid.shape[:2]
    n_steps = n_times - 1
    rpt = SUBSTEPS * n_steps + 1
    dt_hydro = cfg.dt_hydro_yr * NSEC

    # --- FOM snapshots, read only to build the snapshot files ---
    if cfg.build_snapshots:
        mm, starts = open_trajectories(cfg.data_dir, n_tracers, n_steps)
        n_cols = mm.shape[1]
    else:
        mm = None
        n_cols = int(np.load(train_snap, mmap_mode="r").shape[1])
    if n_cols - SPECIES_COL0 != n_species:
        raise ValueError(
            f"Data has {n_cols - SPECIES_COL0} species columns but "
            f"{cfg.network} defines {n_species}. Wrong out_dir, or the feature "
            f"matrices need rebuilding (build_snapshots=True)?")

    # dt_hydro is the interval length h, the span of every per-interval solve; the evaluation grid comes from the stored snapshot times. A mismatch puts t_eval outside the span and scores every block as +inf, so refuse to start.
    dt_seen = float(np.median(np.diff(data["t"])))
    if abs(dt_seen - dt_hydro) > 1e-3 * dt_hydro:
        raise ValueError(
            f"dt_hydro_yr={cfg.dt_hydro_yr:g} does not match the data's interval "
            f"spacing {dt_seen / NSEC:.9g} yr. Set dt_hydro_yr to match.")
    print(f"  {n_tracers} tracers, {n_times} interval boundaries, {n_steps} intervals, "
          f"dt_hydro={cfg.dt_hydro_yr:g} yr")

    # --- training / testing partition ---
    # Reuse a prior split from its sidecars: deterministic given rng_seed, but the pivoted QR is not free to redo every run.
    tr_side, te_side = _idx_path(train_snap), _idx_path(test_snap)
    if not cfg.build_snapshots and os.path.exists(tr_side) and os.path.exists(te_side):
        train_idx, test_idx = np.load(tr_side), np.load(te_side)
        print(f"  loaded split from {tr_side}, {te_side}")
    else:
        train_idx, test_idx = pivoted_qr_split(
            grid.reshape(-1, N_PARAMS), n_tracers, n_times, cfg.n_train, cfg.n_test, cfg.rng_seed)
        # NaN species break k-means and poison any basis fitted on the cluster. Without the raw file, open_snapshots already drops them.
        if mm is not None:
            bad = _failed_tracers(mm, starts)
            if bad.size:
                print(f"  dropping {bad.size} tracer(s) with failed FOM solves")
                train_idx = train_idx[~np.isin(train_idx, bad)]
                test_idx = test_idx[~np.isin(test_idx, bad)]
    print(f"train tracers: {len(train_idx)},  test tracers: {len(test_idx)}")

    if cfg.build_snapshots:
        write_snapshots(mm, starts, train_idx, rpt, train_snap)
        write_snapshots(mm, starts, test_idx, rpt, test_snap)

    return SimpleNamespace(
        net=net, qoi_indices=qoi_indices,
        cluster_indices=qoi_indices, n_species=n_species,
        param_names=list(PARAM_NAMES),
        dt_hydro=dt_hydro, n_times=n_times, n_steps=n_steps,
        train_indices=train_idx, train_tracers=grid[train_idx],
        test_indices=test_idx, test_tracers=grid[test_idx],
        train_snapshots_path=train_snap, test_snapshots_path=test_snap,
    )
