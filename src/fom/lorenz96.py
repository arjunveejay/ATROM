"""The Lorenz-96 system: the network adapter, the dataset, and the loading.

The system is solved in ``x`` in the affine-quadratic form ``xdot = A x + B(x, x) + c``:

    xdot_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F

so ``A = -I`` and ``B`` are independent of the forcing ``F``, and ``c = F 1``.

Generating the dataset is the CLI at the bottom:

    python -m src.fom.lorenz96 --N 100 --out datasets/lorenz96/N100_F0-2.5
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from .data import N_LEADING, SUBSTEPS, write_snapshots, open_trajectories, read_params

PARAM_NAMES = ["F"]          # the forcing, constant along a trajectory
N_PARAMS = len(PARAM_NAMES)
SPECIES_COL0 = N_LEADING + N_PARAMS   # first state column of a source row

# Fixed for this system. The choices of each experiment live in experiments/lorenz96/<model>/run.py.
DEFAULTS = dict(
    ode_method="RK45",       # not stiff
    scale_method="none",
    qoi_log=False,           # the state is signed
    error_norm="absolute",   # the QoIs cross zero, so a relative error is meaningless
    positivity=False,        # a signed state needs no non-negativity correction
    carry_step=False,        # applies to BDF only
    atol=1e-8,
    rtol=1e-8,
)


class Lorenz96Network:
    """Lorenz-96 in the shape the ROM machinery expects of a network.

    ``N >= 4`` so the stencil offsets +1, -1 and -2 stay distinct modulo N.
    """

    def __init__(self, N):
        N = int(N)
        if N < 4:
            raise ValueError(f"Lorenz-96 needs N >= 4; got {N}")
        self.N = N
        self.species = [f"x{i}" for i in range(N)]
        self.species_map = {name: i for i, name in enumerate(self.species)}
        self._coo = None

    @staticmethod
    def env(p):
        """Parameter p_k -> environment. The only parameter is the forcing."""
        return {"F": float(p[0])}

    def _B_coo(self):
        """Canonical COO arrays of the B tensor, built once.

        ``ROM.reduced_system`` pairs ``B.tocoo().data`` positionally against :meth:`get_B_structure`, so both must come from this one ordering.
        """
        if self._coo is None:
            N = self.N
            i = np.arange(N, dtype=np.int64)
            row = np.concatenate([i, i])
            col_j = np.concatenate([(i + 1) % N, (i - 2) % N])
            col_k = np.concatenate([(i - 1) % N, (i - 1) % N])
            data = np.concatenate([np.ones(N), -np.ones(N)])
            # csr canonicalizes to (row, col) lexicographic order; match it.
            order = np.lexsort((col_j * N + col_k, row))
            self._coo = (row[order], col_j[order], col_k[order], data[order])
        return self._coo

    def get_B_structure(self):
        """Fixed COO indices of B. Topological, independent of the forcing."""
        row, col_j, col_k, _ = self._B_coo()
        return row.copy(), col_j.copy(), col_k.copy()

    def operators(self, env):
        """``(A, B, c)`` at forcing ``env["F"]``: ``A = -I``, ``B`` independent of the forcing, and ``c = F 1``."""
        N = self.N
        A = -sp.identity(N, format="csr")

        row, col_j, col_k, data = self._B_coo()
        B = sp.csr_matrix((data, (row, col_j * N + col_k)), shape=(N, N * N))
        return A, B, np.full(N, float(env["F"]))

    @staticmethod
    def offset(p):
        return None

    def rhs(self, x, F):
        """The right-hand side evaluated directly, for validation."""
        return lorenz96_rhs(0.0, np.asarray(x, dtype=float), F)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_context(cfg):
    """One-time setup: the network, the training/testing partition and the snapshot files.

    Returns the same namespace as the astrochemistry loader, so everything downstream is shared.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    data = read_params(cfg.data_dir)
    grid = data["params"]
    n_tracers, n_times = grid.shape[:2]
    n_steps = n_times - 1
    dt_hydro = float(data["t"][1] - data["t"][0])

    train_snap = os.path.join(cfg.out_dir, "snapshots_train.npy")
    test_snap = os.path.join(cfg.out_dir, "snapshots_test.npy")
    train_side = os.path.splitext(train_snap)[0] + "_indices.npy"
    test_side = os.path.splitext(test_snap)[0] + "_indices.npy"

    # The state size is the width of the trajectories, read from whichever file is opened.
    if cfg.build_snapshots:
        mm, starts = open_trajectories(cfg.data_dir, n_tracers, n_steps)
        N = mm.shape[1] - SPECIES_COL0
    else:
        N = int(np.load(train_snap, mmap_mode="r").shape[1]) - SPECIES_COL0
    net = Lorenz96Network(N)
    qoi_indices = np.unique(np.asarray(cfg.qoi, dtype=int))
    if qoi_indices.min() < 0 or qoi_indices.max() >= N:
        raise ValueError(f"qoi {cfg.qoi} out of range for N={N}")
    print(f"  {n_tracers} forcings, {n_steps} intervals of {dt_hydro:g}")

    if not cfg.build_snapshots and os.path.exists(train_side):
        train_idx, test_idx = np.load(train_side), np.load(test_side)
        print(f"  loaded split from {train_side}, {test_side}")
    else:
        # Random, not pivoted QR: with a single parameter column there is nothing for the QR to pivot on.
        if cfg.n_train + cfg.n_test > n_tracers:
            raise ValueError(f"n_train + n_test > {n_tracers} forcings")
        order = np.random.default_rng(cfg.rng_seed).permutation(n_tracers)
        # Not sorted: this order sets the row order of the snapshot file and therefore affects the k-means++ initialization.
        train_idx = order[:cfg.n_train]
        test_idx = order[cfg.n_train:cfg.n_train + cfg.n_test]
    print(f"train forcings: {len(train_idx)},  test forcings: {len(test_idx)}")

    if cfg.build_snapshots:
        rows = SUBSTEPS * n_steps + 1
        write_snapshots(mm, starts, train_idx, rows, train_snap)
        write_snapshots(mm, starts, test_idx, rows, test_snap)

    return SimpleNamespace(
        net=net, qoi_indices=qoi_indices,
        # The feature vector holds the whole state; the error is measured on the QoIs.
        cluster_indices=np.arange(N), n_species=N,
        param_names=list(PARAM_NAMES),
        dt_hydro=dt_hydro, n_times=n_times, n_steps=n_steps,
        train_indices=train_idx, train_tracers=grid[train_idx],
        test_indices=test_idx, test_tracers=grid[test_idx],
        train_snapshots_path=train_snap, test_snapshots_path=test_snap,
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def lorenz96_rhs(_t, x, F):
    """``xdot_i = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F``.

    It is evaluated directly instead of through the operator form, whose per-call overhead brings no benefit to the full-order solve.
    """
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def default_initial_condition(N, amplitude=1.0, n_excited=1):
    """Sparse initial condition shared by every forcing, so the set of initial states is rank one.

    The excited components break the spatial symmetry that would otherwise pin the trajectory to the uniform state.
    """
    x0 = np.zeros(N, dtype=float)
    x0[:n_excited] = amplitude
    return x0


def generate(N, F_values, x0, t_final, dt, method="RK45", rtol=1e-8,
             atol=1e-10, verbose=True):
    """Integrate one trajectory per forcing, all from the same ``x0``.

    The whole time span is recorded: the ROM has to reproduce the evolution from 0 to ``t_final``, and a per-forcing burn-in would replace the shared ``x0`` with a different starting state in every trajectory.

    Returns ``(X, t_eval, F_values)`` with ``X`` of shape ``(len(F_values) * len(t_eval), SPECIES_COL0 + N)``.
    """
    F_values = np.asarray(F_values, dtype=float)
    if np.any(F_values == 0.0):
        raise ValueError(
            "F = 0 is not allowed: the params column doubles as the validity "
            "flag in open_snapshots, which reads 0 as a failed solve.")

    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (N,):
        raise ValueError(f"x0 must have shape ({N},); got {x0.shape}")

    t_eval = np.arange(0.0, t_final + 0.5 * dt, dt)
    # The snapshots must divide into whole intervals of SUBSTEPS snapshots.
    if (t_eval.size - 1) % SUBSTEPS:
        raise ValueError(
            f"{t_eval.size} time points gives {t_eval.size - 1} intervals, "
            f"not a multiple of SUBSTEPS={SUBSTEPS}; load_context would refuse "
            f"this dataset. Adjust --t-final or --dt.")

    n_times, n_F = t_eval.size, F_values.size
    X = np.zeros((n_F * n_times, SPECIES_COL0 + N), dtype=np.float64)

    for k, F in enumerate(F_values):
        F = float(F)
        sol = solve_ivp(lorenz96_rhs, (0.0, float(t_eval[-1])), x0,
                        t_eval=t_eval, args=(F,), method=method, rtol=rtol,
                        atol=atol)
        if not sol.success:
            raise RuntimeError(f"solve failed at F={F}: {sol.message}")

        rows = slice(k * n_times, (k + 1) * n_times)
        X[rows, 0] = k                 # idx: position in the dataset
        X[rows, 1] = t_eval
        X[rows, N_LEADING] = F         # the one parameter
        X[rows, SPECIES_COL0:] = sol.y.T
        if verbose:
            print(f"  [{k + 1}/{n_F}] F={F:.4g}  nfev={sol.nfev}  "
                  f"|x|_max={np.abs(sol.y).max():.3f}")

    return X, t_eval, F_values


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate the Lorenz-96 dataset: one trajectory per forcing F.")
    p.add_argument("--N", type=int, default=100, help="state dimension")
    p.add_argument("--F-min", type=float, default=0.0125)
    p.add_argument("--F-max", type=float, default=2.5)
    p.add_argument("--n-F", type=int, default=200, help="number of values of the forcing F")
    p.add_argument("--F-values", type=float, nargs="+", default=None,
                   help="explicit forcing list, overriding --F-min/--F-max/--n-F")
    p.add_argument("--t-final", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.0078125,
                   help="snapshot spacing; t_final/dt must be a multiple of SUBSTEPS")
    p.add_argument("--amplitude", type=float, default=1.0,
                   help="value of the excited components in the shared sparse IC")
    p.add_argument("--n-excited", type=int, default=1,
                   help="number of nonzero components in the shared sparse IC")
    p.add_argument("--method", default="RK45")
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--out", required=True,
                   help="output directory for trajectories.npy and params.npz")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)

    F_values = (np.asarray(a.F_values, dtype=float) if a.F_values is not None
                else np.linspace(a.F_min, a.F_max, a.n_F))
    x0 = default_initial_condition(a.N, a.amplitude, a.n_excited)

    print(f"Lorenz-96: N={a.N}, {F_values.size} values of F in "
          f"[{F_values.min():.4g}, {F_values.max():.4g}]")
    X, t_eval, F_values = generate(
        N=a.N, F_values=F_values, x0=x0, t_final=a.t_final, dt=a.dt,
        method=a.method, rtol=a.rtol, atol=a.atol)

    os.makedirs(a.out, exist_ok=True)
    traj_path = os.path.join(a.out, "trajectories.npy")
    params_path = os.path.join(a.out, "params.npz")
    np.save(traj_path, X)
    t = t_eval[::SUBSTEPS]
    np.savez(params_path, params=np.repeat(F_values[:, None, None], t.size, axis=1), t=t)

    print(f"\ntrajectories {X.shape} -> {traj_path}")
    print(f"parameters -> {params_path}")
    print(f"{t.size - 1} intervals of {SUBSTEPS} samples, interval length {a.dt * SUBSTEPS:g}")
    print("\nBuild the snapshot files with:\n"
          "  python experiments/lorenz96/<model>/run.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
