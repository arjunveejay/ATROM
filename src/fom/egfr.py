"""EGFR endocytosis (ODEbase model 1452, BIOMD0000000205) with the initial EGF concentration as the parameter: the network adapter, the dataset, and the loading.

The model is pure mass action of order one and two, so ``dx/dt = A x + B (x kron x)`` holds exactly with constant ``A`` and ``B``. The parameter ``p = EGF(0)`` sets only the initial state, ``x0(p) = x0_base + p e`` with ``e`` the unit vector of EGF. The ROM solves the system in the shifted variable ``z = x - x0(p)``,

    dz/dt = c(p) + A'(p) z + B (z kron z),    z(0) = 0,
    A'(p) = A + B(x0, .) + B(., x0) = A'_0 + p A'_1,    c(p) = A x0 + B(x0 kron x0) = c_0 + p c_1,

whose operators are affine in ``p`` because no reaction consumes two EGF molecules. Every linear invariant of ``z`` is zero, so any basis fitted on ``z`` conserves them exactly. Routing and the non-negativity correction act on ``x = x0(p) + z`` through :meth:`EgfrNetwork.offset`.

Generating the dataset is the CLI at the bottom:

    python -m src.fom.egfr --out datasets/egfr/T1000
"""
from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from .. import REPO_ROOT
from ..rom import apply_B_xy
from .data import N_LEADING, SUBSTEPS, write_snapshots, open_trajectories, read_params

PARAM_NAMES = ["p"]          # p = EGF(0), constant along a trajectory
N_PARAMS = len(PARAM_NAMES)
SPECIES_COL0 = N_LEADING + N_PARAMS   # first state column of a source row
PARAM_SPECIES = "EGF"
FOM_RTOL = 1e-8                        # FOM BDF tolerances of the dataset, also used by the FOM intervals of hybrid
FOM_ATOL = 1e-12
X_FLOOR = 1e-30

# Fixed for this system. The choices of each experiment live in experiments/egfr/<model>/run.py.
DEFAULTS = dict(
    model_file=os.path.join(REPO_ROOT, "networks/odebase_1452/1452.json"),
    ode_method="BDF",        # stiff
    scale_method="pareto",   # diagonal scaling: each species scaled by 1/sqrt(std)
    qoi_log=True,            # cluster on log10 of the QoIs
    error_norm="relative",
    positivity=False,        # non-negativity correction off unless an experiment asks for it
    carry_step=True,         # each interval's solve starts from the previous interval's last step size
    atol=1e-6,               # ROM solves
    rtol=1e-3,
)


# ---------------------------------------------------------------------------
# Worker processes (top-level functions, so they can be sent to worker processes)
# ---------------------------------------------------------------------------

_gen_state = None


def _gen_init(model_file, t_eval, rtol, atol):
    """Pool initializer: compile the right-hand side once per worker."""
    global _gen_state
    model, k, _ = _load_model(model_file)
    f, sparsity = _ode_function(model, k)
    _gen_state = (f, sparsity, EgfrNetwork(model_file), np.asarray(t_eval, dtype=np.float64), rtol, atol)


def _gen_one(p):
    """The trajectory in ``x`` at ``t_eval`` for ``EGF(0) = p``, solved from ``t = 0``."""
    f, sparsity, net, t_eval, rtol, atol = _gen_state
    sol = solve_ivp(f, (0.0, float(t_eval[-1])), net.initial_state(p), method="BDF",
                    t_eval=t_eval, rtol=rtol, atol=atol, jac_sparsity=sparsity)
    if not sol.success:
        raise RuntimeError(f"FOM solve failed at p = {p:g}: {sol.message}")
    return sol.y


# ---------------------------------------------------------------------------
# Main classes
# ---------------------------------------------------------------------------

class EgfrNetwork:
    """ODEbase model 1452 in the shifted variable ``z = x - x0(p)``, with the interface the ROM requires of a network.

    ``operators`` returns ``(A'(p), B, c(p))`` for the parameter ``p_k`` of an interval, and ``offset`` returns ``x0(p)``. The attributes ``A`` and ``B`` are the operators of the original system in ``x``.
    """

    def __init__(self, path):
        model, k, x0 = _load_model(path)
        self.species = _species_names(model)
        if len(set(self.species)) != len(self.species):
            raise ValueError(f"{path}: species names are not unique")
        self.species_map = {name: i for i, name in enumerate(self.species)}
        self.x0_nominal = x0
        S = _matrix(model["reconfigured_stoichiometric_matrix"])
        K = _matrix(model["kinetic_matrix"])
        kappa = np.array([k[int(p["kx"][1:]) - 1] for p in model["local_parameters"]])
        self.A, self.B = _mass_action_operators(S, K, kappa)
        _check_complexes(S, K)
        n = len(self.species)
        coo = self.B.tocoo()
        self._B_structure = (coo.row.astype(np.int64), (coo.col // n).astype(np.int64), (coo.col % n).astype(np.int64))

        self._e = np.zeros(n)
        self._e[self.species_map[PARAM_SPECIES]] = 1.0
        self._x0_base = self.initial_state(0.0)
        if np.any(apply_B_xy(self._e, self._e, self.B)):
            raise ValueError(f"{path}: a reaction between two {PARAM_SPECIES} molecules makes c(p) quadratic in p")
        self._A0 = sp.csr_matrix(self.A + _bilinear_matrix(self.B, self._x0_base))
        self._A1 = _bilinear_matrix(self.B, self._e)
        self._c0 = self.A @ self._x0_base + apply_B_xy(self._x0_base, self._x0_base, self.B)
        self._c1 = (self.A @ self._e + apply_B_xy(self._e, self._x0_base, self.B)
                    + apply_B_xy(self._x0_base, self._e, self.B))

    @staticmethod
    def env(p):
        """The environment for ``p_k``: the EGF concentration ``p``."""
        return {"p": float(p[0])}

    def get_B_structure(self):
        """Fixed COO indices of B."""
        row, bj, bk = self._B_structure
        return row.copy(), bj.copy(), bk.copy()

    def operators(self, env):
        """``(A'(p), B, c(p))`` of the system in ``z`` at ``p = env["p"]``."""
        p = env["p"]
        return sp.csr_matrix(self._A0 + p * self._A1), self.B, self._c0 + p * self._c1

    def offset(self, p):
        """``x0(p) = x0_base + p e`` for the parameter values ``p``, so that ``x = x0(p) + z``."""
        return self._x0_base + np.asarray(p, dtype=np.float64)[..., :1] * self._e

    def initial_state(self, p):
        """The nominal initial state with ``EGF(0) = p``."""
        x0 = self.x0_nominal.copy()
        x0[self.species_map[PARAM_SPECIES]] = float(p)
        return x0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matrix(rows):
    """An ODEbase matrix, stored as one comma-separated string per row."""
    return np.array([[float(v) for v in str(r).split(",")] for r in rows])


def _load_model(path):
    """``(model, k, x0)``: the parsed JSON, the parameter vector and the nominal initial state, converted from the exact rational values stored in the file."""
    with open(path) as fh:
        model = json.load(fh)
    k = np.zeros(int(model["num_all_parameters"]))
    for name, val in model["all_parameters"].items():
        k[int(name[1:]) - 1] = float(Fraction(val))
    x0 = np.zeros(int(model["num_species"]))
    for s in model["species"]:
        x0[int(s["var"][1:]) - 1] = float(Fraction(s["initial_concentration"]))
    return model, k, x0


def _species_names(model):
    """Species names in state order, from ODEbase's ``species_<i> (<name>)`` labels."""
    names = [""] * int(model["num_species"])
    for s in model["species"]:
        names[int(s["var"][1:]) - 1] = re.search(r"\((.*)\)\s*$", s["name"]).group(1)
    return names


def _ode_function(model, k):
    """The right-hand side compiled from the ODE expressions in the model file, and the sparsity pattern of its Jacobian."""
    n = int(model["num_species"])
    lines = ["def f(t, x):", f"    dx = np.empty({n})"]
    sparsity = sp.lil_matrix((n, n), dtype=int)
    for eq in model["species_odes"]:
        lhs, rhs = eq.split("=", 1)
        i = int(re.search(r"Derivative\(x(\d+)", lhs).group(1)) - 1
        for j in re.findall(r"\bx(\d+)\b", rhs):
            sparsity[i, int(j) - 1] = 1
        rhs = re.sub(r"\bx(\d+)\b", lambda m: f"x[{int(m.group(1)) - 1}]", rhs)
        rhs = re.sub(r"\bk(\d+)\b", lambda m: f"k[{int(m.group(1)) - 1}]", rhs)
        lines.append(f"    dx[{i}] = {rhs.strip()}")
    lines.append("    return dx")
    ns = {"np": np, "k": k}
    exec("\n".join(lines), ns)
    return ns["f"], sparsity.tocsr()


def _mass_action_operators(S, K, kappa):
    """``(A, B)`` with ``S (kappa * x^K) = A x + B (x kron x)``; column ``j*n + k`` of ``B`` multiplies ``x_j x_k``."""
    n = S.shape[0]
    order = K.sum(axis=0)
    if not set(np.unique(order)) <= {1.0, 2.0}:
        raise ValueError("every reaction must be of order one or two for the quadratic form")
    lin, quad = np.flatnonzero(order == 1), np.flatnonzero(order == 2)
    A = sp.csr_matrix(S[:, lin] * kappa[lin]) @ sp.csr_matrix(
        (np.ones(lin.size), (np.arange(lin.size), np.argmax(K[:, lin], axis=0))), shape=(lin.size, n))
    pairs = [np.repeat(np.arange(n), K[:, r].astype(int)) for r in quad]
    cols = np.array([p[0] * n + p[1] for p in pairs], dtype=np.int64)
    B = sp.csr_matrix(S[:, quad] * kappa[quad]) @ sp.csr_matrix(
        (np.ones(quad.size), (np.arange(quad.size), cols)), shape=(quad.size, n * n))
    A, B = sp.csr_matrix(A), sp.csr_matrix(B)
    # The entries are put in canonical order, because ROM.reduced_system matches B.tocoo().data to get_B_structure by position.
    A.sum_duplicates()
    B.sum_duplicates()
    return A, B


def _check_complexes(S, K):
    """Refuse a model whose kinetic and stoichiometric matrices do not describe whole numbers of each species in every reactant and product."""
    P = K + S
    if (P < 0).any() or not np.array_equal(P, np.round(P)) or not np.array_equal(K, np.round(K)):
        raise ValueError("kinetic and stoichiometric matrices do not describe integer complexes")


def _bilinear_matrix(B, v):
    """The matrix ``M`` with ``M z = B(v kron z) + B(z kron v)``."""
    n = v.size
    coo = B.tocoo()
    bj, bk = coo.col // n, coo.col % n
    return sp.csr_matrix((np.concatenate([coo.data * v[bj], coo.data * v[bk]]),
                          (np.concatenate([coo.row, coo.row]), np.concatenate([bk, bj]))), shape=(n, n))


def operator_mismatch(net, f, n_states=5, seed=0):
    """Largest relative difference between ``A x + B (x kron x)`` and the ODE expressions of the model file, over random states."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for x in rng.uniform(0.0, 2.0, size=(n_states, len(net.species))):
        ref = f(0.0, x)
        worst = max(worst, float(np.max(np.abs(net.A @ x + apply_B_xy(x, x, net.B) - ref)) / np.max(np.abs(ref))))
    return worst


def _to_deviation(snapshots_path, params, rows_per_tracer, net):
    """Replace the concentrations ``x`` of a snapshot file, in place, by ``z = max(x, X_FLOOR) - x0(p)``, one trajectory at a time."""
    mm = np.load(snapshots_path, mmap_mode="r+")
    for m, p in enumerate(params):
        rows = slice(m * rows_per_tracer, (m + 1) * rows_per_tracer)
        x = np.maximum(np.asarray(mm[rows, SPECIES_COL0:]), X_FLOOR)
        mm[rows, SPECIES_COL0:] = x - net.initial_state(p)
    mm.flush()


def load_context(cfg):
    """One-time setup: the network, the training/testing partition and the snapshot files.

    Returns the same namespace as the astrochemistry loader, so everything downstream is shared. The training and testing values of p are the ones fixed in ``params.npz``. The snapshot files hold ``z = max(x, X_FLOOR) - x0(p)``.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    net = EgfrNetwork(cfg.model_file)
    missing = [q for q in cfg.qoi if q not in net.species_map]
    if missing:
        raise ValueError(f"QoI {missing} are not species of {cfg.model_file}")
    qoi_indices = np.array([net.species_map[q] for q in cfg.qoi], dtype=int)
    n_species = len(net.species)

    data = read_params(cfg.data_dir)
    grid = data["params"]
    p = grid[:, 0, 0]
    idx = {"train": data["train"], "test": data["test"]}
    n_steps = data["t"].size - 1
    dt_hydro = float(data["t"][1] - data["t"][0])

    feat = {"train": os.path.join(cfg.out_dir, "snapshots_train.npy"),
            "test": os.path.join(cfg.out_dir, "snapshots_test.npy")}
    if cfg.build_snapshots:
        src, starts = open_trajectories(cfg.data_dir, p.size, n_steps)
        if src.shape[1] != SPECIES_COL0 + n_species:
            raise ValueError(f"{cfg.data_dir}/trajectories.npy has {src.shape[1]} columns; expected {SPECIES_COL0 + n_species}")
        for split, sel in idx.items():
            write_snapshots(src, starts, sel, SUBSTEPS * n_steps + 1, feat[split])
            _to_deviation(feat[split], p[sel], n_steps * (SUBSTEPS + 1), net)

    print(f"  {idx['train'].size} training and {idx['test'].size} testing values of p, {n_steps} intervals of {dt_hydro:g} s")
    return SimpleNamespace(
        net=net, qoi_indices=qoi_indices,
        cluster_indices=qoi_indices, n_species=n_species,
        param_names=list(PARAM_NAMES),
        dt_hydro=dt_hydro, n_times=n_steps + 1, n_steps=n_steps,
        train_indices=idx["train"], train_tracers=grid[idx["train"]],
        test_indices=idx["test"], test_tracers=grid[idx["test"]],
        train_snapshots_path=feat["train"], test_snapshots_path=feat["test"],
    )


def generate(model_file, out_dir, train_params, test_params, t_eval, rtol=FOM_RTOL, atol=FOM_ATOL, n_workers=1, overwrite=False):
    """Solve the full-order model at every parameter value and write ``trajectories.npy`` and ``params.npz`` to ``out_dir``.

    ``trajectories.npy`` is ``[idx | t | p | species]`` with the concentrations ``x``, one trajectory of ``t_eval.size`` rows per value of p, the training values first. ``params.npz`` holds p at every interval boundary t_k, the boundary times, and the indices of the training and testing values.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_params, test_params = np.asarray(train_params, dtype=np.float64), np.asarray(test_params, dtype=np.float64)
    params = np.concatenate([train_params, test_params])
    path = os.path.join(out_dir, "trajectories.npy")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to regenerate it.")

    net = EgfrNetwork(model_file)
    model, k, _ = _load_model(model_file)
    mismatch = operator_mismatch(net, _ode_function(model, k)[0])
    print(f"A, B against the ODE strings: max relative mismatch {mismatch:.2e}")
    if mismatch > 1e-12:
        raise RuntimeError(f"the quadratic operators disagree with the model's ODEs ({mismatch:.2e})")

    rpt = t_eval.size
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                    shape=(params.size * rpt, SPECIES_COL0 + len(net.species)))
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_gen_init,
                             initargs=(model_file, t_eval, rtol, atol)) as pool:
        for m, y in enumerate(pool.map(_gen_one, params)):
            lo = m * rpt
            out[lo:lo + rpt, 0] = m
            out[lo:lo + rpt, 1] = t_eval
            out[lo:lo + rpt, N_LEADING] = params[m]
            out[lo:lo + rpt, SPECIES_COL0:] = y.T
    out.flush()
    print(f"  wrote {path}  {out.shape}")

    t = t_eval[::SUBSTEPS]
    n_train = train_params.size
    np.savez(os.path.join(out_dir, "params.npz"), params=np.repeat(params[:, None, None], t.size, axis=1), t=t,
             train=np.arange(n_train), test=n_train + np.arange(test_params.size))


def build_parser():
    p = argparse.ArgumentParser(description="Generate the EGFR ensemble over p = EGF(0).")
    p.add_argument("--model-file", default=os.path.join(REPO_ROOT, "networks/odebase_1452/1452.json"))
    p.add_argument("--out", required=True, help="generator output directory")
    p.add_argument("--pmin", type=float, default=5e-3, help="smallest training value of p [uM]")
    p.add_argument("--pmax", type=float, default=1.0, help="largest training value of p [uM]")
    p.add_argument("--n-train", type=int, default=200, help="log-spaced training values of p; the testing values are their log-midpoints")
    p.add_argument("--t-final", type=float, default=1000.0, help="end time [s]")
    p.add_argument("--dt", type=float, default=0.2, help="sampling interval [s]; SUBSTEPS samples make one interval")
    p.add_argument("--rtol", type=float, default=FOM_RTOL)
    p.add_argument("--atol", type=float, default=FOM_ATOL)
    p.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    train = np.logspace(np.log10(a.pmin), np.log10(a.pmax), a.n_train)
    lg = np.log10(train)
    test = 10 ** ((lg[:-1] + lg[1:]) / 2)
    t_eval = np.linspace(0.0, a.t_final, int(round(a.t_final / a.dt)) + 1)
    if (t_eval.size - 1) % SUBSTEPS:
        raise ValueError(f"{t_eval.size - 1} samples do not divide into intervals of SUBSTEPS={SUBSTEPS}")
    print(f"{train.size} training and {test.size} testing values of p in [{a.pmin:g}, {a.pmax:g}] uM, {t_eval.size} samples over t=0-{a.t_final:g} s")
    generate(a.model_file, a.out, train, test, t_eval, a.rtol, a.atol, a.n_workers, a.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
