"""Cost of AT-ROM inference against the FOM for this experiment.

    python experiments/nelson/R0.1_M6.0/timing.py

Every testing trajectory is solved twice: with the AT-ROM at (TAU, ETA), using this experiment's settings, and with the FOM on every interval, as hAT-ROM solves its FOM windows (exact dense Jacobian). For each, the mean per trajectory of the right-hand-side calls, Jacobian evaluations, LU factorizations and wall time is reported, with the speedup and the largest relative L2 QoI error. The calls are counted by wrapping the right-hand-side and Jacobian functions and SciPy's BDF LU factorization.

Needs the AT-ROM at (TAU, ETA) saved by a run of run.py. Writes timing_test.csv next to this file. One trajectory per worker process at a time, so run it on an otherwise idle node.
"""
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import numpy as np
import scipy.integrate._ivp.bdf as bdf

import src.hatrom as hatrom
from src.hatrom import HATROM, HATROMConfig
from src.rom import ROM
from src.run import Experiment

TAU, ETA = 0.01, 1e-5        # the AT-ROM that is timed: split tolerance and residual SVD energy tolerance

count = dict(rhs=0, jac=0, lu=0)
exp = atrom = cfg = None     # set in main before the worker pool starts; the forked workers inherit them


def counted(key, f):
    def wrapper(*args, **kwargs):
        count[key] += 1
        return f(*args, **kwargs)
    return wrapper


def count_calls():
    """Count every right-hand-side call, Jacobian evaluation and BDF LU factorization, of both the AT-ROM and the FOM."""
    reduced_system = ROM.reduced_system

    def counted_reduced_system(self, A, B, c=None):
        rhs, jac = reduced_system(self, A, B, c)
        return counted("rhs", rhs), counted("jac", jac)

    ROM.reduced_system = counted_reduced_system
    rhs, jac = hatrom._quadratic_rhs, hatrom._quadratic_jacobian
    hatrom._quadratic_rhs = lambda *a: counted("rhs", rhs(*a))
    hatrom._quadratic_jacobian = lambda *a: counted("jac", jac(*a))
    bdf.lu_factor = counted("lu", bdf.lu_factor)


def measure(solve, t_ref, y_ref):
    """Run ``solve``; its call counts, wall time and largest relative L2 QoI error (nan if it did not reach the end)."""
    for k in count:
        count[k] = 0
    t0 = time.perf_counter()
    t, y = solve()
    seconds = time.perf_counter() - t0
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != len(t_ref):
        err = float("nan")
    else:
        err = max(np.linalg.norm(y[j] - y_ref[j]) / np.linalg.norm(y_ref[j]) for j in exp.ctx.qoi_indices)
    return dict(count, s=seconds, err=err)


def time_one(i):
    """Solve testing trajectory ``i`` with the AT-ROM and with the FOM."""
    t_ref, y_ref, x_ref = exp.trajectories([i])
    t_ref, y_ref, x0 = t_ref[0], y_ref[0], x_ref[0]
    pt, dt = exp.split_files[1][i], exp.ctx.dt_hydro

    def atrom_solve():
        t, y, _, _ = atrom.solve_tracer(pt, x0, dt, cfg.atol, cfg.rtol, t_eval=t_ref, positivity=cfg.positivity,
                                        method=cfg.ode_method, carry=cfg.carry_step)
        return t, y

    def fom_solve():
        result = HATROM(atrom, cfg, HATROMConfig(positivity=False, force_fom=True)).solve(pt, x0, dt, t_eval=t_ref)
        return result.t, result.y

    return measure(atrom_solve, t_ref, y_ref), measure(fom_solve, t_ref, y_ref)


def main():
    global exp, atrom, cfg
    exp = Experiment(HERE, "test", eta=ETA)
    atrom, cfg = exp.load_atrom(TAU, ETA), exp.cfg
    n, n_workers = len(exp.split_files[2]), os.cpu_count() or 1
    count_calls()
    with ProcessPoolExecutor(n_workers) as pool:
        results = list(pool.map(time_one, range(n), chunksize=4))

    rows = []
    for j, model in enumerate(("AT-ROM", "FOM")):
        err = np.array([r[j]["err"] for r in results])
        rows.append(dict(model=model, rhs_calls=np.mean([r[j]["rhs"] for r in results]),
                         jacobians=np.mean([r[j]["jac"] for r in results]),
                         lu_factorizations=np.mean([r[j]["lu"] for r in results]),
                         wall_seconds=np.mean([r[j]["s"] for r in results]),
                         mean_worst_qoi_error=float(np.nanmean(err)), completed=int(np.isfinite(err).sum())))
    speedup = rows[1]["wall_seconds"] / rows[0]["wall_seconds"]

    print(f"[{exp.case}] tau_split={TAU:g}, eta={ETA:g}: {n} testing trajectories, {exp.ctx.n_steps} intervals, "
          f"{n_workers} worker processes; means per trajectory")
    print(f"{'':8} {'RHS calls':>10} {'Jacobians':>10} {'LUs':>8} {'wall s':>8} {'worst-QoI err':>14} {'complete':>9}")
    for r in rows:
        print(f"{r['model']:8} {r['rhs_calls']:10.0f} {r['jacobians']:10.0f} {r['lu_factorizations']:8.0f} "
              f"{r['wall_seconds']:8.3f} {r['mean_worst_qoi_error']:14.3e} {r['completed']:9d}")
    print(f"speedup (FOM wall time / AT-ROM wall time): {speedup:.2f}x")
    path = os.path.join(HERE, "timing_test.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
