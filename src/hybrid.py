"""hAT-ROM: AT-ROM inference on one trajectory, with backtracking to the FOM.

AT-ROM is solved interval by interval. When the cluster changes on T_k (or k = 0), the relative projection error of the normalized initial state in the new basis is computed. If it exceeds ``sqrt(eta)``, backtracking is triggered. For k >= 1 the solve returns to the interval T_l at which the cluster active on T_{k-1} was first assigned, discards the AT-ROM solution over T_l..T_{k-1}, recomputes it with the FOM in one FOM window, and resumes AT-ROM on T_k without a further check. For k = 0, T_0 is solved with the FOM.

    result = HATROM(atrom, cfg, HATROMConfig()).solve(context.test_tracers[i], x0s[i], context.dt_hydro, t_eval=t_refs[i])

Times are in the solver's own units throughout, matching ``context.dt_hydro``.

The only correction applied is the non-negativity correction.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d

import scipy.sparse as sp
from scipy.integrate import BDF, solve_ivp

from .fom.egfr import FOM_ATOL as EGFR_FOM_ATOL, FOM_RTOL as EGFR_FOM_RTOL, EgfrNetwork
from .fom.lorenz96 import Lorenz96Network

#: The two error measures, in the order they are stored: the largest relative error at any time, and the relative L2 error over the trajectory.
AGGREGATES = ("max-in-time", "l2-in-time")

#: Cluster id recorded for an interval solved by the FOM, which belongs to no cluster.
FOM_INTERVAL = -1

_FLOOR = 1e-30


# ---------------------------------------------------------------------------
# Worker processes (top-level functions, so they can be sent to worker processes)
# ---------------------------------------------------------------------------

_hatrom_state = None


def _hatrom_init(hatrom):
    global _hatrom_state
    _hatrom_state = hatrom


def _hatrom_solve_one(args):
    """Solve and score one trajectory in a worker.

    Returns only the error vectors and scalars; the solution itself is not needed by the caller.
    """
    i, pt, x0, t_fom, y_fom, dt_hydro = args
    hatrom = _hatrom_state
    try:
        result = hatrom.solve(pt, x0, dt_hydro, t_eval=t_fom)
        n_steps = np.asarray(pt).shape[0] - 1
        valid = result.failure is None and result.t.size > 0 and np.isclose(result.t[-1], n_steps * dt_hydro)
        errors = _error_series(result.t, result.y, t_fom, y_fom) if valid else None
        return i, errors, result.diagnostics()
    except Exception as exc:
        return i, None, {"failure": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Main classes
# ---------------------------------------------------------------------------

class HATROM:
    """hAT-ROM: AT-ROM with backtracking to the FOM where the new cluster's basis represents the state poorly.

    Holds the AT-ROM (``atrom``) and the FOM (``fom``). ``cfg`` is the run's settings (its ``eta`` sets the trigger threshold) and ``config`` the :class:`HATROMConfig`.
    """

    def __init__(self, atrom, cfg, config=None):
        self.atrom, self.cfg = atrom, cfg
        self.config = c = HATROMConfig() if config is None else config
        self.fom = FOM(atrom.network, cfg.ode_method, atol=c.fom_atol, rtol=c.fom_rtol,
                             min_scale=c.fom_min_scale, carry_step=c.fom_carry_step)

    def solve(self, pt, x0, dt_hydro, t_eval=None):
        """Solve one trajectory with AT-ROM, backtracking to the FOM when the state entering a new cluster is poorly represented by that cluster's basis.

        ``pt`` has one row per interval boundary t_k, so ``n_steps + 1`` rows. ``t_eval`` is the whole trajectory's time array, normally the FOM reference's own times, which is what keeps the output columns aligned with it.

        A solver failure is recorded in :attr:`HATROMResult.failure` and the solution so far is returned, following ``solve_tracer``'s ``info`` convention rather than raising out of a sweep.
        """
        atrom, cfg, hatrom = self.atrom, self.cfg, self.config
        self.fom.check_supported()
        pt = np.asarray(pt, dtype=np.float64)
        x0 = np.asarray(x0, dtype=np.float64)
        dt = float(dt_hydro)
        n_steps = pt.shape[0] - 1
        if n_steps < 1:
            raise ValueError(f"pt needs at least two rows; got shape {pt.shape}")
        t_eval = (np.arange(n_steps + 1, dtype=np.float64) * dt if t_eval is None
                  else np.asarray(t_eval, dtype=np.float64))
        carry = bool(cfg.carry_step) and cfg.ode_method == "BDF"
        h = None    # last AT-ROM step size, carried into the next AT-ROM interval; reset after a FOM window

        # The trigger matches the basis truncation: a state whose relative projection error exceeds sqrt(eta) is not represented to the accuracy the basis was fitted for.
        thresh = math.sqrt(cfg.eta)

        iv = _Intervals(x0)
        n_backtrack = 0
        trigger_step, trigger_proj_error = [], []
        failure = None
        # Includes AT-ROM intervals a later backtrack discards, which the final path does not show.
        n_rom_attempted = n_rom_solves = n_fom_solves = 0
        rom_seconds = fom_seconds = 0.0

        def full_model(k0, k1):
            """Solve intervals k0..k1 with the FOM and store them; returns the failure message or None."""
            nonlocal n_fom_solves, fom_seconds
            t0 = time.perf_counter()
            tt, yy, fail = self.fom.solve_window(pt, k0, k1, iv.x_entry[k0], dt, t_eval)
            fom_seconds += time.perf_counter() - t0
            if fail is None:
                n_fom_solves += 1
                iv.extend_window(tt, yy, k0, k1, dt)
            return fail

        k = 0
        while k < n_steps:
            x = iv.x_entry[k]
            cid = atrom.route(pt[k], x)
            if cid not in atrom.roms:
                failure = f"interval {k}: no ROM fitted for cluster {cid}"
                break
            # An interval that follows a FOM interval always counts as a cluster change, but its projection error is not checked, because the FOM window ends at T_{k-1}.
            after_fom = k > 0 and iv.fom[k - 1]
            entering = after_fom or cid != iv.last_cluster()
            r = (atrom.roms[cid].basis.projection_error(x)
                 if (entering and not after_fom) else np.nan)

            if hatrom.force_fom:
                failure = full_model(k, k)
                if failure is not None:
                    break
                h = None
                k += 1
                continue

            # ---- backtracking: recompute the preceding intervals in one cluster with the FOM ----
            # The FOM window is T_l to T_{k-1}, where T_l is the interval at which the cluster active on T_{k-1} was first assigned; at k = 0 there is none, so T_0 is solved. AT-ROM then resumes on T_k without a check, so every backtrack advances the solve.
            if entering and r > thresh:
                n_backtrack += 1
                trigger_step.append(k)
                trigger_proj_error.append(r)
                k0 = iv.stay_start(k)
                k1 = max(k0, k - 1)
                iv.truncate(k0)
                failure = full_model(k0, k1)
                if failure is not None:
                    break
                h = None
                k = k1 + 1
                continue

            # ---- ordinary AT-ROM interval --------------------------------------
            n_rom_attempted += 1
            t0 = time.perf_counter()
            step = atrom.solve_interval(pt[k], x, cid, dt, cfg.atol, cfg.rtol,
                                        window_samples(t_eval, k * dt, (k + 1) * dt),
                                        entering=entering, positivity=hatrom.positivity,
                                        method=cfg.ode_method, carry=carry, first_step=h)
            rom_seconds += time.perf_counter() - t0
            if step.failure is not None:
                failure = f"interval {k}: ROM solve failed -- {step.failure}"
                break
            n_rom_solves += 1
            h = step.h
            iv.append(step.t + k * dt, step.y, cid, step.y[:, -1], step.pos_entry, step.pos_endpoint)
            k += 1

        t_out, y_out = iv.concatenate()
        return HATROMResult(
            t=t_out, y=y_out,
            path=np.asarray(iv.path, dtype=int),
            fom_step=np.asarray(iv.fom, dtype=bool),
            n_backtrack=n_backtrack,
            trigger_step=np.asarray(trigger_step, dtype=int),
            trigger_proj_error=np.asarray(trigger_proj_error, dtype=float),
            n_pos_entry=int(sum(iv.pos_entry)),
            n_pos_endpoint=int(sum(iv.pos_endpoint)),
            n_steps=n_steps, dt_hydro=dt, config=hatrom, failure=failure,
            n_fom_solves=n_fom_solves, n_rom_solves=n_rom_solves, n_rom_attempted=n_rom_attempted,
            fom_seconds=fom_seconds, rom_seconds=rom_seconds)

    def solve_pure(self, pt, x0, dt_hydro, t_eval=None, positivity=True):
        """AT-ROM without backtracking on the same trajectory, as a :class:`RomResult`.

        Wraps :meth:`ATROM.solve_tracer` so hAT-ROM and AT-ROM score through the same code.
        """
        atrom, cfg = self.atrom, self.cfg
        pt = np.asarray(pt, dtype=np.float64)
        n_steps = pt.shape[0] - 1
        t, y, path, info = atrom.solve_tracer(
            pt, x0, dt_hydro, cfg.atol, cfg.rtol, t_eval=t_eval,
            positivity=positivity, method=cfg.ode_method)
        t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
        return RomResult(t=t, y=y,
                         path=to_interval_path(path, t, dt_hydro, n_steps),
                         info=info)

    def evaluate(self, tracers, x_eqs, t_foms, y_foms, qoi_indices, qoi_names, dt_hydro, n_workers=1, verbose=True):
        """Solve and score every trajectory: the counts, ``valid_idx`` and the per-trajectory errors of the completed trajectories, as in ``run.Experiment.score``.

        Adds a ``hatrom`` block of per-trajectory diagnostics with no AT-ROM counterpart, over every attempted trajectory rather than only the completed ones, since the failures are needed to assess the backtracks. A run in which no trajectory completes returns empty per-trajectory errors.
        """
        n = len(t_foms)
        args = [(i, np.asarray(tracers[i]), np.asarray(x_eqs[i]),
                 np.asarray(t_foms[i]), np.asarray(y_foms[i]), dt_hydro)
                for i in range(n)]

        errs, diags = [None] * n, [None] * n
        if n_workers <= 1:
            _hatrom_init(self)
            for a in args:
                i, e, d = _hatrom_solve_one(a)
                errs[i], diags[i] = e, d
        else:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_hatrom_init,
                                     initargs=(self,)) as pool:
                futures = {pool.submit(_hatrom_solve_one, a): a[0] for a in args}
                for done, fut in enumerate(as_completed(futures), 1):
                    i = futures[fut]
                    try:
                        idx, e, d = fut.result()
                        errs[idx], diags[idx] = e, d
                    except Exception as exc:   # a failure on one trajectory must not end the sweep
                        diags[i] = {"failure": f"{type(exc).__name__}: {exc}"}
                    if verbose and done % 50 == 0:
                        print(f"  completed {done}/{n}", end="\r")
        if verbose:
            print(f"  completed {n}/{n}")

        valid = [i for i in range(n) if errs[i] is not None]
        failed = [i for i in range(n)
                  if diags[i] is None or diags[i].get("failure") is not None]
        out = {"n_tracers": n, "n_solved": n - len(failed), "n_valid": len(valid),
               "failed_idx": failed, "valid_idx": list(valid),
               "hatrom_config": self.config, "per_tracer": {}}
        for j, tag in enumerate(AGGREGATES):
            x = np.stack([errs[i][j] for i in valid])[:, list(qoi_indices)] if valid else np.empty((0, len(qoi_names)))
            out["per_tracer"][tag] = {name: x[:, k].copy() for k, name in enumerate(qoi_names)}

        out["hatrom"] = {
            k: np.array([(diags[i] or {}).get(k, np.nan) for i in range(n)],
                        dtype=float) for k in HATROMResult.DIAG_KEYS}
        return out


@dataclass(frozen=True)
class HATROMConfig:
    """Settings of the hAT-ROM solve. Validated on construction.

    Backtracking is triggered when the relative projection error in the new cluster's basis exceeds ``sqrt(eta)``, with ``eta`` the residual SVD energy tolerance of the settings passed to :class:`HATROM`. The error is not checked on the interval that follows a FOM window. AT-ROM without backtracking is :meth:`ATROM.solve_tracer`, not a setting of this solve.

    ``force_fom`` solves every interval with the FOM, the reference path for validating the FOM window solver.

    ``fom_atol``, ``fom_rtol`` and ``fom_min_scale`` govern the FOM windows and default to the values with which KIDApy generated the chemistry datasets. ``fom_atol`` applies to the scaled variables ``x / max(x_entry, fom_min_scale)``, not to the abundances. ``fom_carry_step`` starts each interval's FOM BDF solve from the previous interval's last step size.
    """

    positivity: bool = True
    force_fom: bool = False
    fom_atol: float = 1e-6
    fom_rtol: float = 1e-3
    fom_min_scale: float = 1e-22
    fom_carry_step: bool = False

    def __post_init__(self):
        for name in ("fom_atol", "fom_rtol", "fom_min_scale"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0; got {getattr(self, name)!r}")

    def summary(self):
        return (f"HATROMConfig(positivity={self.positivity}, "
                f"force_fom={self.force_fom}, fom_atol={self.fom_atol:g}, "
                f"fom_rtol={self.fom_rtol:g}, "
                f"fom_min_scale={self.fom_min_scale:g}, "
                f"fom_carry_step={self.fom_carry_step})")


@dataclass
class HATROMResult:
    """One hAT-ROM solution and a record of how it was solved.

    ``t``/``y`` are concatenated with shared interval boundaries de-duplicated, so the columns stay aligned with the FOM reference's. ``path`` (the cluster of each interval, :data:`FOM_INTERVAL` for a FOM one) and ``fom_step`` (whether each interval was solved with the FOM) are indexed by interval. ``trigger_step`` and ``trigger_proj_error`` are the interval at which each backtrack was triggered and the projection error that triggered it.
    """

    t: np.ndarray
    y: np.ndarray
    path: np.ndarray
    fom_step: np.ndarray
    n_backtrack: int
    trigger_step: np.ndarray
    trigger_proj_error: np.ndarray
    n_pos_entry: int             # intervals whose entering state got the non-negativity correction
    n_pos_endpoint: int          # intervals whose leaving state got the non-negativity correction
    n_steps: int
    dt_hydro: float
    config: HATROMConfig
    failure: Optional[str] = None
    # An interval count is not a cost: per-interval expense differs between FOM and AT-ROM, and n_rom_attempted exceeds n_rom_solves by the AT-ROM solves a backtrack discarded.
    n_fom_solves: int = 0
    n_rom_solves: int = 0
    n_rom_attempted: int = 0
    fom_seconds: float = 0.0
    rom_seconds: float = 0.0

    #: The entries of :meth:`diagnostics` that become columns of the per-trajectory error table.
    DIAG_KEYS = ("backtracks", "n_fom_intervals", "fom_frac", "pos_entry",
                 "pos_endpoint", "switches", "max_trigger_proj_error",
                 "n_intervals_done", "fom_solves", "rom_solves", "rom_discarded",
                 "fom_seconds", "rom_seconds", "fom_time_frac")

    def fom_windows(self):
        """The FOM windows (contiguous FOM intervals) as inclusive ``(k0, k1)`` pairs."""
        w = np.flatnonzero(self.fom_step)
        if not w.size:
            return []
        edges = np.r_[0, np.flatnonzero(np.diff(w) != 1) + 1, w.size]
        return [(int(w[a]), int(w[b - 1])) for a, b in zip(edges[:-1], edges[1:])]

    def diagnostics(self):
        """Scalar summary of this solve, for one row of the per-trajectory CSV."""
        total_seconds = self.fom_seconds + self.rom_seconds
        return {
            "backtracks": int(self.n_backtrack),
            "n_fom_intervals": int(self.fom_step.sum()),
            "fom_frac": float(self.fom_step.mean()) if self.fom_step.size else 0.0,
            "pos_entry": int(self.n_pos_entry),
            "pos_endpoint": int(self.n_pos_endpoint),
            "switches": count_switches(self.path),
            "max_trigger_proj_error": (float(self.trigger_proj_error.max())
                                       if self.trigger_proj_error.size else float("nan")),
            "n_intervals_done": int(self.path.size),
            # FOM windows solved: neither backtracks (force_fom makes none) nor n_fom_intervals (one window spans several intervals).
            "fom_solves": int(self.n_fom_solves),
            "rom_solves": int(self.n_rom_solves),
            "rom_discarded": int(self.n_rom_attempted - self.n_rom_solves),
            "fom_seconds": float(self.fom_seconds),
            "rom_seconds": float(self.rom_seconds),
            "fom_time_frac": float(self.fom_seconds / total_seconds) if total_seconds > 0 else 0.0,
            "failure": self.failure,
        }

    def summary(self, time_scale=1.0, time_label="s"):
        """Report of the backtracks, non-negativity corrections, cost and FOM windows.

        ``time_scale`` divides the window times; pass ``NSEC * 1e3`` with ``time_label="kyr"`` for the chemistry figures' units.
        """
        d, dt = self.diagnostics(), self.dt_hydro
        lines = [self.config.summary(),
                 f"  backtracks {d['backtracks']}, FOM intervals "
                 f"{d['n_fom_intervals']}/{self.n_steps} ({d['fom_frac']:.1%})",
                 f"  positivity: entry {d['pos_entry']}, endpoint {d['pos_endpoint']}",
                 f"  cluster switches {d['switches']}",
                 f"  cost: {d['fom_solves']} FOM solve(s) {d['fom_seconds']:.2f}s, "
                 f"{d['rom_solves']} ROM solve(s) {d['rom_seconds']:.2f}s "
                 f"({d['rom_discarded']} discarded) -- "
                 f"{d['fom_time_frac']:.1%} of time at full order"]
        if self.trigger_proj_error.size:
            lines.append("  triggers: " + ", ".join(
                f"interval {int(k)} at {v:.3g}"
                for k, v in zip(self.trigger_step, self.trigger_proj_error)))
        for k0, k1 in self.fom_windows():
            lines.append(f"  FOM intervals {k0}-{k1} = "
                         f"{k0 * dt / time_scale:.3g}-"
                         f"{(k1 + 1) * dt / time_scale:.3g} {time_label}")
        if self.failure is not None:
            lines.append(f"  INCOMPLETE after {d['n_intervals_done']}/{self.n_steps} "
                         f"intervals: {self.failure}")
        return "\n".join(lines)


@dataclass
class RomResult:
    """An AT-ROM solution without backtracking, shaped to compare against :class:`HATROMResult`."""

    t: np.ndarray
    y: np.ndarray
    path: np.ndarray
    info: object = None

    @property
    def n_switches(self):
        return count_switches(self.path)


class FOM:
    """The FOM solve over a FOM window, which hAT-ROM backtracks to, for one network.

    For astrochemistry each interval is solved as KIDApy's solve with p_k held constant on each interval generated those datasets: in the coordinates ``z = x / max(x_entry, min_scale)``, rescaled at every interval boundary. For EGFR the window is solved as the EGFR datasets were generated: one solve over the whole window, without rescaling, at ``FOM_RTOL`` and ``FOM_ATOL`` of :mod:`src.fom.egfr`, since the operators are constant along a trajectory. The exact Jacobian is passed as a dense array.

    Lorenz-96 is not supported: its signed state admits no such scaling.

    ``method`` is the solver; ``atol``, ``rtol`` and ``min_scale`` govern the astrochemistry solves (``atol`` applies to the scaled variables ``x / max(x_entry, min_scale)``, not to the abundances). With ``carry_step``, each interval's BDF solve starts from the previous interval's last step size.
    """

    def __init__(self, network, method="BDF", atol=1e-6, rtol=1e-3, min_scale=1e-22, carry_step=False):
        self.network = network
        self.method, self.atol, self.rtol, self.min_scale, self.carry_step = method, atol, rtol, min_scale, carry_step

    def check_supported(self):
        """Raise ``NotImplementedError`` for a network this solve does not cover (Lorenz-96)."""
        if isinstance(self.network, Lorenz96Network):
            raise NotImplementedError("the full-order solve works in coordinates scaled by the positive abundances "
                                      "and is not defined for the signed Lorenz-96 state")

    def solve_window(self, pt, k0, k1, x, dt_hydro, t_eval=None):
        """Solve intervals ``k0`` to ``k1`` (inclusive) from the state ``x``.

        ``pt`` has one parameter row per interval boundary and ``t_eval`` holds the whole trajectory's sample times. Returns ``(t, y, failure)``: the times (absolute) and states, and ``None`` or a message if the solve failed.
        """
        self.check_supported()
        if isinstance(self.network, EgfrNetwork):
            return self._solve_window_unscaled(pt, k0, k1, x, dt_hydro, t_eval)
        net, dt = self.network, float(dt_hydro)
        x = np.asarray(x, dtype=np.float64)
        ts, ys = [], []
        h_last = None
        # Operators are constant within an interval and jump at each boundary, so each interval is a separate solve, scaled by the state entering it.
        for k in range(k0, k1 + 1):
            A, B, c = net.operators(net.env(pt[k]))
            s = np.maximum(x, self.min_scale)
            A_sc, B_sc, c_sc = _scale_operators(A, B, s, c)
            te = window_samples(t_eval, k * dt, (k + 1) * dt)
            if te is not None:
                # A stored interval boundary can land a few ULPs outside the span, which solve_ivp rejects outright.
                te = np.clip(te, 0.0, dt)
            fun, jac = _quadratic_rhs(A_sc, B_sc, c_sc), _quadratic_jacobian(A_sc, B_sc)
            if self.carry_step:
                t_k = te if te is not None else np.array([0.0, dt])
                solver = BDF(fun, 0.0, x / s, dt, rtol=self.rtol, atol=self.atol, jac=jac, first_step=h_last)
                z_k, dense = [], None
                for tj in t_k:
                    while solver.t < tj and solver.status == "running":
                        solver.step()
                        dense = None
                        if solver.t_old is not None and solver.t > solver.t_old:
                            h_last = solver.t - solver.t_old
                    if solver.status == "failed":
                        break
                    if solver.t_old is None:
                        z_k.append(solver.y.copy())
                    else:
                        dense = dense or solver.dense_output()
                        z_k.append(dense(tj))
                failed, message = solver.status == "failed", "BDF step failed"
                y_k = np.column_stack(z_k) if z_k else None
            else:
                sol = solve_ivp(fun, (0.0, dt), x / s, method=self.method, atol=self.atol, rtol=self.rtol,
                                t_eval=te, jac=jac)
                failed, message, t_k, y_k = sol.status != 0, sol.message, sol.t, sol.y
            if failed:
                return None, None, f"window {k0}-{k1}: FOM solve failed in interval {k} -- {message}"
            tt, yy = t_k + k * dt, s[:, None] * y_k
            # Consecutive intervals share their boundary; keep it once.
            if ts and tt.size and np.isclose(tt[0], ts[-1][-1]):
                tt, yy = tt[1:], yy[:, 1:]
            ts.append(tt)
            ys.append(yy)
            x = s * y_k[:, -1]
        return np.concatenate(ts), np.hstack(ys), None

    def _solve_window_unscaled(self, pt, k0, k1, x, dt_hydro, t_eval):
        """Intervals ``k0`` to ``k1`` as one solve without rescaling, for a network whose operators are constant along a trajectory. The output holds every sample in the window and every interval boundary."""
        net, dt = self.network, float(dt_hydro)
        x = np.asarray(x, dtype=np.float64)
        A, B, c = net.operators(net.env(pt[k0]))
        span = (k1 - k0 + 1) * dt
        bounds = np.arange(k1 - k0 + 2) * dt
        te = window_samples(t_eval, k0 * dt, (k1 + 1) * dt)
        # A stored interval boundary can land a few ULPs outside the span, which solve_ivp rejects outright.
        te = np.clip(np.unique(bounds if te is None else np.concatenate([te, bounds])), 0.0, span)
        sol = solve_ivp(_quadratic_rhs(A, B, c), (0.0, span), x, method=self.method,
                        atol=EGFR_FOM_ATOL, rtol=EGFR_FOM_RTOL, t_eval=te, jac=_quadratic_jacobian(A, B))
        if sol.status != 0:
            return None, None, f"window {k0}-{k1}: FOM solve failed -- {sol.message}"
        return sol.t + k0 * dt, sol.y, None


class _Intervals:
    """Per-interval accumulator for a hAT-ROM solution.

    Every list is indexed by interval, ``t``/``y`` included -- one block per interval even when a FOM solve spans many, or :meth:`truncate` would cut in the wrong place. ``x_entry[k]`` is the state entering interval ``k``, so it holds one more element than the rest.
    """

    def __init__(self, x0):
        self.t = []
        self.y = []
        self.path = []
        self.fom = []
        self.pos_entry = []
        self.pos_endpoint = []
        self.x_entry = [np.asarray(x0, dtype=np.float64).copy()]

    def __len__(self):
        return len(self.path)

    def append(self, t, y, cluster, x_end, pos_entry=False, pos_endpoint=False):
        """Record one finished interval: its samples, its cluster (:data:`FOM_INTERVAL` for a FOM one) and its end state."""
        self.t.append(np.asarray(t, dtype=np.float64))
        self.y.append(np.asarray(y, dtype=np.float64))
        self.path.append(int(cluster))
        self.fom.append(cluster == FOM_INTERVAL)
        self.pos_entry.append(bool(pos_entry))
        self.pos_endpoint.append(bool(pos_endpoint))
        self.x_entry.append(np.asarray(x_end, dtype=np.float64))

    def extend_window(self, tt, yy, k0, k1, dt):
        """Store a FOM window over intervals ``k0`` to ``k1`` as one block per interval."""
        tt, yy = np.asarray(tt), np.asarray(yy)
        for kk in range(k0, k1 + 1):
            msk = (tt >= kk * dt) & (tt <= (kk + 1) * dt)
            j = int(np.argmin(np.abs(tt - (kk + 1) * dt)))
            self.append(tt[msk], yy[:, msk], FOM_INTERVAL, yy[:, j])

    def truncate(self, k0):
        """Drop every interval from ``k0`` on, keeping the state entering ``k0``."""
        for lst in (self.t, self.y, self.path, self.fom, self.pos_entry, self.pos_endpoint):
            del lst[k0:]
        del self.x_entry[k0 + 1:]

    def last_cluster(self):
        """The most recent AT-ROM cluster, skipping FOM intervals.

        Comparing against the :data:`FOM_INTERVAL` sentinel instead would make every interval after a FOM window look like a cluster change.
        """
        return next((c for c in reversed(self.path) if c >= 0), FOM_INTERVAL)

    def stay_start(self, k):
        """The interval at which the cluster active on ``k - 1`` was first assigned (start of its unbroken run of AT-ROM intervals); ``k`` itself if ``k - 1`` is a FOM interval or there is none."""
        if k == 0 or self.fom[k - 1]:
            return k
        c, s = self.path[k - 1], k - 1
        while s > 0 and self.path[s - 1] == c:
            s -= 1
        return s

    def concatenate(self):
        """One array pair, with the interval boundary shared by consecutive blocks emitted once."""
        if not self.t:
            return np.array([]), np.empty((self.x_entry[0].size, 0))
        t_out, y_out = [self.t[0]], [self.y[0]]
        for j in range(1, len(self.t)):
            if (self.t[j].size and t_out[-1].size
                    and np.isclose(self.t[j][0], t_out[-1][-1])):
                t_out.append(self.t[j][1:])
                y_out.append(self.y[j][:, 1:])
            else:
                t_out.append(self.t[j])
                y_out.append(self.y[j])
        return np.concatenate(t_out), np.hstack(y_out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_switches(path):
    """Number of cluster occupancies along ``path``, ignoring :data:`FOM_INTERVAL` intervals."""
    p = np.asarray(path)
    p = p[p >= 0]
    return int((np.diff(p) != 0).sum() + 1) if p.size else 0


def to_interval_path(path, t, dt_hydro, n_steps):
    """Per-interval cluster path, obtained from a per-snapshot path if necessary.

    The path from ``solve_tracer`` is already per interval. A per-snapshot path is sampled strictly inside each interval, since at the interval boundary ``int(t/dt)`` gives the index of the preceding interval. An interval without an interior sample is assigned :data:`FOM_INTERVAL`.
    """
    path, t = np.asarray(path), np.asarray(t, dtype=np.float64)
    if len(path) == n_steps:
        return path.astype(int, copy=False)
    out = np.full(n_steps, FOM_INTERVAL, dtype=int)
    for k in range(n_steps):
        inside = np.flatnonzero((t > k * dt_hydro) & (t < (k + 1) * dt_hydro))
        if inside.size:
            out[k] = int(path[min(inside[inside.size // 2], len(path) - 1)])
    return out


def window_samples(t_eval, lo, hi):
    """The times of ``t_eval`` inside ``[lo, hi]``, measured from ``lo``, or ``None`` if there are fewer than two."""
    if t_eval is None:
        return None
    p = np.unique(t_eval[(t_eval >= lo) & (t_eval <= hi)] - lo)
    return p if p.size >= 2 else None


def _scale_operators(A, B, s, c=None):
    """``(A, B, c)`` in the coordinates ``z = x / s``, as KIDApy's ``QuadraticSolver.solve`` forms them; ``c`` stays ``None`` when the system has no constant term."""
    N = s.size
    s_inv = 1.0 / s
    A_sc = sp.diags(s_inv, format="csr") @ A.tocsr() @ sp.diags(s, format="csr")
    c_sc = None if c is None else s_inv * np.asarray(c, dtype=np.float64)
    if B.nnz == 0:
        return A_sc, B, c_sc
    Bcoo = B.tocoo(copy=False)
    bi = Bcoo.row.astype(np.int64, copy=False)
    bj = (Bcoo.col // N).astype(np.int64, copy=False)
    bk = (Bcoo.col % N).astype(np.int64, copy=False)
    B_sc = sp.coo_matrix(
        (Bcoo.data * s[bj] * s[bk] * s_inv[bi], (Bcoo.row, Bcoo.col)),
        shape=B.shape, dtype=np.float64).tocsr()
    return A_sc, B_sc, c_sc


def _quadratic_rhs(A, B, c=None):
    """``z -> c + A z + B (z kron z)``, with ``B``'s index arrays extracted once rather than on every call; ``c`` is omitted when ``None``."""
    N = A.shape[0]
    A = A.tocsr()
    Bcoo = B.tocoo()
    bi, bj, bk, bv = Bcoo.row, Bcoo.col // N, Bcoo.col % N, np.asarray(Bcoo.data, dtype=np.float64)

    def fun(_t, z):
        f = A @ z + np.bincount(bi, weights=bv * z[bj] * z[bk], minlength=N)
        return f if c is None else f + c

    return fun


def _quadratic_jacobian(A, B):
    """``z -> A + B(z, ·) + B(·, z)`` as a dense array, so BDF factorises it with dense LU; the index arrays are extracted once."""
    N = A.shape[0]
    a_flat = A.toarray().ravel()
    Bcoo = B.tocoo()
    bi = Bcoo.row.astype(np.int64)
    bj, bk = (Bcoo.col // N).astype(np.int64), (Bcoo.col % N).astype(np.int64)
    flat = np.concatenate([bi * N + bj, bi * N + bk])
    bv = np.asarray(Bcoo.data, dtype=np.float64)

    def jac(_t, z):
        quad = np.bincount(flat, weights=np.concatenate([bv * z[bk], bv * z[bj]]), minlength=N * N)
        return (a_flat + quad).reshape(N, N)

    return jac


def _error_series(t, y, t_fom, y_fom):
    """Per-species ``(max-in-time, l2-in-time)`` relative errors, in the order of ``AGGREGATES``.

    Same reductions, floor and re-interpolation as ``run.Experiment.score``, so hAT-ROM numbers are comparable with the sweep CSVs.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(y, dtype=float)
    t_fom, y_fom = np.asarray(t_fom, dtype=float), np.asarray(y_fom, dtype=float)
    if t.shape[0] != t_fom.shape[0]:
        y = interp1d(t, y, axis=1, bounds_error=False,
                     fill_value="extrapolate")(t_fom)
    resid = y - y_fom
    err = np.abs(resid) / (np.abs(y_fom) + _FLOOR)
    num = np.sqrt(np.sum(resid ** 2, axis=1))
    den = np.sqrt(np.sum(y_fom ** 2, axis=1)) + _FLOOR
    return err.max(axis=1), num / den
