"""Cluster bases, reduced quadratic operators, and the Adaptive Tree Reduced Order Model (AT-ROM).

The dynamics are quadratic, ``dx/dt = c + A x + B (x kron x)``, with ``c`` (absent for the chemistry), ``A`` and ``B`` rebuilt on each interval ``T_k = [t_k, t_{k+1}]`` from ``p_k = p(t_k)``, the parameter held constant on ``T_k`` (the row ``pt[k]``). Each cluster carries a triple

    encode:  z = U^T D^-1 (x - x_bar)
    decode:  x = x_bar + D U z

with centering state ``x_bar``, diagonal scaling ``D`` and reduced basis ``U``.

A trajectory is solved one interval at a time: assign a cluster, integrate one interval (length ``dt_hydro``) in that cluster's reduced coordinates, and use the end state as the initial condition of the next interval.
"""
from __future__ import annotations

import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np
from scipy.integrate import BDF, solve_ivp
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .cluster import block_starts


# ---------------------------------------------------------------------------
# Worker processes (top-level functions, so they can be sent to worker processes)
# ---------------------------------------------------------------------------

_score_state = None


def _score_init(network, roms, snapshots_path, sp, pslice, dt_hydro, atol, rtol,
                qoi_rows, method, norm):
    """Pool initializer: hold the bases, reopen the memmap read-only per worker."""
    global _score_state
    _score_state = dict(network=network, roms=roms,
                        mm=np.load(snapshots_path, mmap_mode="r"), sp=sp,
                        pslice=pslice, dt_hydro=float(dt_hydro), atol=atol,
                        rtol=rtol, qoi_rows=np.asarray(qoi_rows, dtype=int),
                        method=method, norm=norm)


def _score_block(args):
    """Score one ``(tracer, step)`` block, i.e. one trajectory on one interval: the single-interval QoI error of the cluster's ROM against the FOM."""
    b, rows, teval, label = args
    s = _score_state
    rom = s["roms"].get(int(label))
    if rom is None:
        return b, float("inf")
    y_fom = np.asarray(s["mm"][rows][:, s["sp"]], dtype=np.float64).T
    net = s["network"]
    A, B, c = net.operators(net.env(
        np.asarray(s["mm"][rows[0]][s["pslice"]], dtype=np.float64)))
    try:
        _, y_rom, sol = rom.solve(A, B, y_fom[:, 0], (0.0, s["dt_hydro"]),
                                  s["atol"], s["rtol"], t_eval=teval,
                                  method=s["method"], c=c)
    except Exception:
        return b, float("inf")
    if sol.status != 0 or y_rom.shape[1] != y_fom.shape[1]:
        return b, float("inf")
    return b, qoi_error(y_rom, y_fom, s["qoi_rows"], s["norm"])


_fit_state = None


def _fit_init(snapshots_path, species_slice):
    global _fit_state
    _fit_state = (np.load(snapshots_path, mmap_mode="r"), species_slice)


def _fit_one(args):
    c, idx, var_threshold, scale_method = args
    mm, sp = _fit_state
    return c, Basis.fit(np.asarray(mm[idx][:, sp], dtype=np.float64),
                        var_threshold, scale_method), idx.size


_solve_ensemble = None


def _solve_init(ensemble):
    global _solve_ensemble
    _solve_ensemble = ensemble


def _solve_one(args):
    i, pt, x0, dt_hydro, atol, rtol, t_eval, positivity, method, carry = args
    try:
        return i, _solve_ensemble.solve_tracer(pt, x0, dt_hydro, atol, rtol,
                                               t_eval=t_eval,
                                               positivity=positivity,
                                               method=method, carry=carry)
    except Exception as exc:
        return i, exc


# ---------------------------------------------------------------------------
# Main classes
# ---------------------------------------------------------------------------

class ATROM:
    """AT-ROM: one ROM per leaf cluster, chosen on each interval by descending the cluster tree."""

    def __init__(self, cluster, network, roms):
        self.cluster = cluster
        self.network = network
        self.roms = dict(roms)
        self.summary = {}

    # ---- build ----------------------------------------------------------

    @classmethod
    def build_adaptive(cls, cluster, network, fm, n_steps, dt_hydro, cfg,
                       split_tols, qoi_indices, verbose=True, fit_only=None):
        """Fit one AT-ROM per split tolerance, sharing one base pass.

        Starting from the initial clusters of ``cluster``, the cluster with the largest error above the split tolerance is repeatedly split in two, both children get a new basis, and only the intervals that belonged to the parent are scored again. The refinement stops when every cluster is under the tolerance, when no cluster can be split, or when ``max_clusters`` is reached.

        A cluster's error is its single-interval QoI error: the maximum over its intervals of the error after integrating one interval from the FOM state at its start. Starting from the FOM state leaves out the error carried from earlier intervals; a failed solve counts as ``+inf``, so its cluster is split first.

        An unsplittable cluster over tolerance has its basis rank raised instead when ``cfg.increase_rank`` is set, and is otherwise left unchanged.

        The base fit and first scoring pass do not depend on the tolerance, so they run once and each tolerance starts from that saved state.

        Yields ``(tau, ATROM)`` as soon as each tolerance is finished, so the caller can save it at once; ``cluster`` is not mutated. ``fit_only`` limits the fit to some of ``split_tols`` (e.g. those a stopped job did not finish). Each tolerance's random splits are seeded by its position in the full ``split_tols``, so a tolerance gets the same model whether or not the others are fitted with it.
        """
        mm, sp, pslice = fm.memmap, fm.species_slice, fm.param_slice
        qoi_rows = np.unique(np.asarray(qoi_indices, dtype=int))
        n_w = int(cfg.n_workers)
        var_threshold = cfg.var_threshold
        scale_method, norm = cfg.scale_method, cfg.error_norm

        # ---- group valid rows into blocks, one per trajectory and interval ----
        # The rows of a block belong to one interval, hence to one cluster label, so a block is the unit of the single-interval solve.
        valid = np.flatnonzero(fm.valid_rows)
        # Keyed on (tracer, step), i.e. trajectory and interval, rather than interval alone, matching the blocks on which ``Cluster`` fitted its labels.
        cuts = np.flatnonzero(block_starts(np.asarray(fm.sample_tracer)[valid],
                                           np.asarray(fm.sample_step)[valid]))[1:]
        block_rows, block_teval = [], []
        for rows in np.split(valid, cuts):
            times = np.asarray(mm[rows][:, 1], dtype=np.float64)
            srt = np.argsort(times)
            teval = np.unique(times[srt] - times[srt][0])
            if teval.size < 2:
                continue                       # need >= 2 points to compare
            block_rows.append(rows[srt])
            block_teval.append(teval)
        n_blocks = len(block_rows)

        sample_labels = cluster.labels_for_rows(fm).copy()
        sample_labels[~fm.valid_rows] = -1

        def score(block_ids, roms_subset, out):
            ids = np.asarray(list(block_ids), dtype=np.int64)
            if ids.size == 0:
                return
            args = [(int(b), block_rows[b], block_teval[b], int(block_label[b]))
                    for b in ids]
            init = (network, roms_subset, fm.path, sp, pslice, dt_hydro,
                    cfg.atol, cfg.rtol, qoi_rows, cfg.ode_method, norm)
            if n_w == 1:
                _score_init(*init)
                for a in args:
                    b, e = _score_block(a)
                    out[b] = e
                return
            with ProcessPoolExecutor(max_workers=n_w, initializer=_score_init,
                                     initargs=init) as pool:
                for b, e in pool.map(_score_block, args,
                                     chunksize=max(1, len(args) // (n_w * 8))):
                    out[b] = e

        # ---- 1. one basis per initial cluster ----
        if verbose:
            print(f"  [adaptive] fitting {cluster.n_clusters} base bases ...")
        roms = _fit_bases(fm, sample_labels, cluster.n_clusters,
                          var_threshold, scale_method, n_w, verbose)
        block_label = np.array([sample_labels[rows[0]] for rows in block_rows],
                               dtype=np.int64)
        blocks_of = {}
        for b, lab in enumerate(block_label):
            blocks_of.setdefault(int(lab), []).append(b)

        # ---- 2. one full scoring pass: QoI error of every block ----
        block_err = np.full(n_blocks, np.inf)
        if verbose:
            print(f"  [adaptive] ROM-scoring {n_blocks:,} tracer-steps across "
                  f"{cluster.n_clusters} clusters ...")
        score(range(n_blocks), roms, block_err)

        def cluster_error(c):
            bs = blocks_of.get(c, [])
            return float(np.max(block_err[bs])) if bs and c in roms else None

        base = SimpleNamespace(
            roms=roms, blocks_of=blocks_of, block_label=block_label,
            block_err=block_err, sample_labels=sample_labels,
            errors={c: cluster_error(c) for c in blocks_of},
            sizes={c: int(np.count_nonzero(sample_labels == c))
                   for c in blocks_of},
            template=cluster.clone())

        # Each tolerance gets its own random seed for its splits, independent of the order of the tolerances.
        seeds = (np.random.SeedSequence(cfg.rng_seed).spawn(len(split_tols))
                 if len(split_tols) > 1 else [cfg.rng_seed])

        for ti, tau in enumerate(split_tols):
            if fit_only is not None and not any(np.isclose(tau, t) for t in fit_only):
                continue
            cluster = base.template.clone()
            roms = copy.deepcopy(base.roms)
            blocks_of = {c: list(v) for c, v in base.blocks_of.items()}
            block_label = base.block_label.copy()
            block_err = base.block_err.copy()
            sample_labels = base.sample_labels.copy()
            errors, sizes = dict(base.errors), dict(base.sizes)
            rng = np.random.default_rng(seeds[ti])

            blocked, rank_increases = set(), {}

            def raise_rank(c):
                """Raise cluster ``c``'s basis rank until its QoI error meets ``tau`` (RaiseRank).

                Every rank from ``r0 + 1`` upward is scored by ROM solves on the cluster's blocks, so the rank kept is the lowest that meets ``tau``. The rank with the smallest error is kept if no rank meets ``tau``.
                """
                rom, bs = roms.get(c), blocks_of.get(c, [])
                if rom is None or not bs or rom.basis.rank >= rom.basis.max_rank:
                    return
                r0, basis = rom.basis.rank, rom.basis
                start = r0 + 1

                best_rank, best_err = r0, errors.get(c)
                best_scores = {b: block_err[b] for b in bs}
                for r in range(start, basis.max_rank + 1):
                    basis.set_rank(r)
                    score(bs, {c: rom}, block_err)
                    err = cluster_error(c)
                    if err is not None and (best_err is None or err < best_err):
                        best_rank, best_err = r, err
                        best_scores = {b: block_err[b] for b in bs}
                    if err is not None and err <= tau:
                        break
                basis.set_rank(best_rank)
                block_err[list(best_scores)] = list(best_scores.values())
                errors[c] = best_err
                if best_rank != r0:
                    rank_increases[c] = (r0, best_rank)
                    if verbose:
                        print(f"    cluster {c}: rank {r0} -> {best_rank}, "
                              f"err={best_err:.4g}")

            n_splits, stop_reason = 0, "all clusters under tol"
            while cluster.n_clusters < cfg.max_clusters:
                cands = [(e, c) for c, e in errors.items()
                         if e is not None and e > tau and c not in blocked]
                if not cands:
                    if any(e is not None and e > tau for e in errors.values()):
                        stop_reason = ("clusters over tol but unsplittable "
                                       "(cannot form two children >= "
                                       f"min_cluster_size={cfg.min_cluster_size})")
                    break
                parent_err, c = max(cands)

                new_id = cluster.split_cluster(c, rng, cfg.min_cluster_size)
                if new_id is None:
                    blocked.add(c)
                    if cfg.increase_rank:
                        raise_rank(c)
                    continue

                affected = blocks_of.pop(c, [])
                sample_labels = cluster.labels_for_rows(fm)
                sample_labels[~fm.valid_rows] = -1
                for cid in (c, new_id):
                    idx = np.flatnonzero(sample_labels == cid)
                    sizes[cid] = int(idx.size)
                    if idx.size >= 2:
                        roms[cid] = ROM(Basis.fit(
                            np.asarray(mm[idx][:, sp], dtype=np.float64),
                            var_threshold, scale_method))
                    else:
                        roms.pop(cid, None)
                    blocks_of[cid] = []
                for b in affected:
                    lab = int(sample_labels[block_rows[b][0]])
                    block_label[b] = lab
                    blocks_of.setdefault(lab, []).append(b)

                children = {cid: roms[cid] for cid in (c, new_id) if cid in roms}
                score([b for cid in (c, new_id) for b in blocks_of.get(cid, [])],
                      children, block_err)
                for cid in (c, new_id):
                    errors[cid] = cluster_error(cid)
                n_splits += 1
                if verbose:
                    over = sum(1 for e in errors.values()
                               if e is not None and e > tau)
                    print(f"  [tau={tau:g}] split {n_splits}: {c} "
                          f"(err={parent_err:.4g}) -> {c},{new_id}; "
                          f"{cluster.n_clusters} clusters, {over} over tolerance")
            else:
                stop_reason = f"reached max_clusters={cfg.max_clusters}"
                # Raising the basis rank adds no clusters, so a cluster still over tolerance can improve even though no further split is allowed.
                if cfg.increase_rank:
                    for cid in sorted(c for c, e in errors.items()
                                      if e is not None and e > tau
                                      and c not in blocked):
                        raise_rank(cid)

            finite = [e for e in errors.values() if e is not None]
            over_ids = sorted(c for c, e in errors.items()
                              if e is not None and e > tau)
            atrom = cls(cluster=cluster, network=network, roms=roms)
            atrom.summary = {
                "split_tol": float(tau),
                "n_clusters": int(cluster.n_clusters),
                "n_splits": int(n_splits),
                "n_over_tol": len(over_ids),
                "n_unsplittable": len(blocked),
                "worst_err": float(max(finite)) if finite else float("nan"),
                "stop_reason": stop_reason,
                "rank_increases": {int(c): [int(a), int(b)]
                                   for c, (a, b) in rank_increases.items()},
                "cluster_ranks": {int(c): int(r.basis.rank)
                                  for c, r in roms.items()},
            }
            if verbose:
                print(f"  [tau={tau:g}] done: {n_splits} splits, "
                      f"{cluster.n_clusters} clusters, {len(over_ids)} over tolerance, "
                      f"worst err={atrom.summary['worst_err']:.4g} "
                      f"({stop_reason})")
            yield tau, atrom

    # ---- persistence ----------------------------------------------------

    def save(self, path):
        """Save the cluster tree (the arrays of :meth:`Cluster.state`) and each cluster's triple (the arrays ending in ``_<c>``).

        For cluster ``c``: ``basis_mean_<c>`` is the centering state ``x_bar``, ``basis_scaling_<c>`` the diagonal scaling ``D``, ``basis_modes_<c>`` every POD mode (one row per mode, all of them, so the rank can be changed after loading), ``mode_energy_<c>`` the fraction of the scaled snapshot variance captured by each mode, and ``basis_rank_<c>`` the basis rank in use. The network is not stored; pass the same one to :meth:`load`.
        """
        state = {"cluster_ids": np.array(sorted(self.roms), dtype=int),
                 "summary_json": np.array(json.dumps(self.summary))}
        state.update(self.cluster.state())
        for c, rom in self.roms.items():
            b = rom.basis
            state[f"basis_mean_{c}"] = b._x_bar
            state[f"basis_scaling_{c}"] = b.scaler.scale_
            state[f"basis_modes_{c}"] = b.pca.components_
            state[f"mode_energy_{c}"] = b.pca.explained_variance_ratio_
            state[f"basis_rank_{c}"] = np.array(b.rank)
        np.savez(path, **state)
        print(f"Saved to {path}.npz")

    @classmethod
    def load(cls, path, network):
        """The AT-ROM saved by :meth:`save`, with its cluster tree, for ``network``."""
        from .cluster import Cluster

        d = np.load(path, allow_pickle=True)
        cluster = Cluster.from_state(d)
        roms = {}
        for c in d["cluster_ids"].astype(int):
            b = Basis()
            b.scaler = StandardScaler()
            b.scaler.mean_ = d[f"basis_mean_{c}"]
            b.scaler.scale_ = d[f"basis_scaling_{c}"]
            b.scaler.var_ = b.scaler.scale_ ** 2
            b.pca = PCA()
            b.pca.components_ = d[f"basis_modes_{c}"]
            b.pca.explained_variance_ratio_ = d[f"mode_energy_{c}"]
            b.pca.mean_ = np.zeros_like(b.scaler.mean_)
            b.pca.n_components_ = b.pca.components_.shape[0]
            b.rank = int(d[f"basis_rank_{c}"])
            b._precompute()
            roms[int(c)] = ROM(b)
        atrom = cls(cluster=cluster, network=network, roms=roms)
        atrom.summary = json.loads(str(d["summary_json"]))
        return atrom

    # ---- inference ------------------------------------------------------

    def precompute_B_structure(self):
        """Cache the B sparsity slices on every ROM. Call once after loading, before solving."""
        bi, bj, bk = self.network.get_B_structure()
        for rom in self.roms.values():
            rom.set_B_structure(bi, bj, bk)
        return self

    def mean_rank(self):
        return float(np.mean([r.basis.rank for r in self.roms.values()]))

    def summary_columns(self):
        """The build diagnostics written to ``fits.csv``: the largest basis rank and the summary of the refinement, with ``worst_qoi_err_1_interval`` the largest single-interval QoI error over the clusters at the end of the build."""
        s = self.summary
        return dict(
            max_rank=max(rom.basis.rank for rom in self.roms.values()),
            n_splits=s.get("n_splits"),
            n_over_tol=s.get("n_over_tol"),
            n_unsplittable=s.get("n_unsplittable"),
            worst_qoi_err_1_interval=s.get("worst_err"),
            stop_reason=s.get("stop_reason"),
        )

    def _positivity(self, c, x, x0=None):
        """Non-negativity correction: project the state ``x`` in cluster ``c``'s basis so that ``x0 + x`` is nonnegative, with ``x0`` the network's offset (zero when ``None``).

        The SLSQP solve runs only when ``x0 + x`` is negative somewhere.
        """
        if not np.any((x if x0 is None else x0 + x) < 0.0):
            return x
        basis = self.roms[c].basis
        z, ok = basis.project_nonnegative(basis.encode(x), x0)
        return basis._x_bar + basis._R @ z if ok else x

    def route(self, pt_k, x):
        """The cluster of the state ``x`` at the parameters ``pt_k``, by descent of the cluster tree. Uses ``x0 + x`` when the network defines an offset ``x0``."""
        x0 = self.network.offset(pt_k)
        return self.cluster.predict(pt_k, x if x0 is None else x0 + x)

    def solve_interval(self, pt_k, x, c, dt_hydro, atol, rtol, t_eval=None, *, entering=True,
                       positivity=True, method="BDF", carry=False, first_step=None):
        """Solve one interval in cluster ``c`` from the state ``x``, at the parameters ``pt_k``.

        In order: if ``entering`` the cluster (and with ``positivity``), the start state is projected into the basis and the non-negativity correction is applied; one interval is integrated; with ``positivity`` the end state is corrected the same way. ``t_eval`` holds the sample times inside the interval, measured from its start. With ``carry`` the BDF integration starts from ``first_step``, the last step size of the previous interval.

        Used by :meth:`solve_tracer` and by the hybrid solve, so both solve an interval the same way. Returns ``t`` (from 0), ``y``, ``h`` (the last step size, with ``carry``), ``failure`` (``None`` or a message), and whether the start and end states were corrected.
        """
        x0 = self.network.offset(pt_k)
        basis = self.roms[c].basis
        pos_entry = pos_endpoint = False
        if positivity and entering:
            # The integration starts from decode(encode(x)), which can be negative where x is not.
            x = basis.decode(basis.encode(x))
            pos_entry = bool(np.any((x if x0 is None else x0 + x) < 0.0))
            x = self._positivity(c, x, x0)

        A, B, c_k = self.network.operators(self.network.env(pt_k))
        h, failure = None, None
        if carry:
            t, y, h, ok = self.roms[c].solve_carried(A, B, x, (0.0, dt_hydro), atol, rtol, t_eval=t_eval,
                                                     first_step=first_step, c=c_k)
            if not ok:
                failure = f"BDF failed in cluster {c}"
        else:
            t, y, sol = self.roms[c].solve(A, B, x, (0.0, dt_hydro), atol, rtol, t_eval=t_eval, method=method, c=c_k)
            if sol.status != 0:
                failure = f"{method} failed in cluster {c}: {sol.message}"

        if failure is None and positivity:
            # The corrected end state is stored at the interval boundary and starts the next interval; samples inside the interval are not changed.
            end = y[:, -1]
            pos_endpoint = bool(np.any((end if x0 is None else x0 + end) < 0.0))
            if pos_endpoint:
                y = y.copy()
                y[:, -1] = self._positivity(c, end, x0)
        return SimpleNamespace(t=np.asarray(t), y=np.asarray(y), h=h, failure=failure,
                               pos_entry=pos_entry, pos_endpoint=pos_endpoint)

    def solve_tracer(self, pt, x0, dt_hydro, atol, rtol, t_eval=None,
                     positivity=True, method="BDF", carry=False):
        """ROM solve along one trajectory, one interval at a time.

        On each interval, the state is assigned a cluster from ``(pt[k], x)``, one interval is integrated, and the end state is the initial condition of the next interval. With ``positivity=False`` nothing else is done, and a negative state is passed to the next interval, where the BDF integration then fails.

        With ``positivity=True``, the non-negativity correction is applied at two points, since feasibility depends on the basis:

        * at the start of an interval, when the assigned cluster changes and at ``k = 0``. A state that is nonnegative after projection in the previous basis need not satisfy ``decode_c(encode_c(x)) >= 0`` in the new basis, and the integration starts from that projection; at ``k = 0`` it is the projected initial condition, which can be negative although ``x0`` is not. The correction is skipped when the cluster is unchanged, since it then has no effect.
        * at the end of an interval, on the state passed to the next interval.

        The samples inside an interval are not modified.

        With ``carry``, each interval is still integrated by a new BDF solver, but its initial step size is the last step size of the integration over the preceding interval instead of the estimate BDF computes. Valid for ``method="BDF"`` only.

        Returns ``(t, y, cluster_path, info)``. ``info`` is ``None`` on success, otherwise the failure that ended the solve, with ``t`` and ``y`` covering only the completed intervals.
        """
        pt = np.asarray(pt, dtype=np.float64)
        x = np.asarray(x0, dtype=np.float64).copy()
        n_steps = pt.shape[0] - 1
        if t_eval is not None:
            # Each interval's samples, closed at both ends, located once rather than by scanning every sample per interval.
            t_eval = np.unique(np.asarray(t_eval, dtype=np.float64))
            bounds = np.arange(n_steps + 1) * dt_hydro
            lo_idx = np.searchsorted(t_eval, bounds[:-1], side="left")
            hi_idx = np.searchsorted(t_eval, bounds[1:], side="right")

        all_t, all_y = [], []
        path = np.full(n_steps, -1, dtype=int)
        h = None    # last step size, carried into the next interval

        def done(info):
            if not all_t:
                return np.array([]), np.empty((x.size, 0)), path, info
            # Consecutive intervals share their boundary point.
            t_out, y_out = [all_t[0]], [all_y[0]]
            for j in range(1, len(all_t)):
                if all_t[j].size and all_t[j][0] == t_out[-1][-1]:
                    t_out.append(all_t[j][1:])
                    y_out.append(all_y[j][:, 1:])
                else:
                    t_out.append(all_t[j])
                    y_out.append(all_y[j])
            return np.concatenate(t_out), np.hstack(y_out), path, info

        for k in range(n_steps):
            c = self.route(pt[k], x)
            path[k] = c
            if c not in self.roms:
                return done(KeyError(f"no ROM fitted for cluster {c}"))
            t_eval_step = None
            if t_eval is not None:
                pts = t_eval[lo_idx[k]:hi_idx[k]] - k * dt_hydro
                t_eval_step = pts if pts.size >= 2 else None
            try:
                step = self.solve_interval(pt[k], x, c, dt_hydro, atol, rtol, t_eval_step,
                                           entering=(k == 0 or c != path[k - 1]), positivity=positivity,
                                           method=method, carry=carry, first_step=h)
            except Exception as exc:
                return done(exc)
            if step.failure is not None:
                return done(RuntimeError(f"interval {k}: {step.failure}"))
            h = step.h
            all_t.append(step.t + k * dt_hydro)
            all_y.append(step.y)
            x = step.y[:, -1]

        return done(None)

    def solve_tracers(self, tracers, x0s, dt_hydro, atol, rtol, t_evals,
                      n_workers, positivity=True, method="BDF", verbose=True, carry=False):
        """Solve many trajectories in parallel, one result tuple each.

        A single solve has no time limit: a trajectory stuck inside a compiled scipy call hangs the run, so limit the wall-clock time outside the process.
        """
        if carry and method != "BDF":
            raise ValueError("carry requires method='BDF'")
        tracers = list(tracers)
        n = len(tracers)
        x0s = np.asarray(x0s, dtype=np.float64)
        x0_list = [x0s[i].copy() for i in range(n)]

        def failed(i, info):
            # Same shape as a real result so zip(*results) stays valid.
            return (np.array([]), np.empty((x0_list[i].size, 0)),
                    np.full(np.asarray(tracers[i]).shape[0] - 1, -1, dtype=int),
                    info)

        args = [(i, tracers[i], x0_list[i], dt_hydro, atol, rtol, t_evals[i],
                 positivity, method, carry) for i in range(n)]
        results = [None] * n

        if n_workers <= 1:
            _solve_init(self)
            for a in args:
                i, r = _solve_one(a)
                results[i] = failed(i, r) if isinstance(r, Exception) else r
        else:
            with ProcessPoolExecutor(max_workers=n_workers,
                                     initializer=_solve_init,
                                     initargs=(self,)) as pool:
                futures = {pool.submit(_solve_one, a): a[0] for a in args}
                for done_n, fut in enumerate(as_completed(futures), 1):
                    i = futures[fut]
                    try:
                        idx, r = fut.result()
                        results[idx] = (failed(idx, r) if isinstance(r, Exception)
                                        else r)
                    except Exception as exc:   # worker died, e.g. OOM-killed
                        results[i] = failed(i, exc)
                    if verbose and done_n % 50 == 0:
                        print(f"  completed {done_n}/{n}", end="\r")
        if verbose:
            print(f"  completed {n}/{n}")
        return results

    # ---- the global-basis ROM baseline -----------------------------------

    @classmethod
    def fit_global(cls, fm, network, cfg, qoi_indices, param_names, rank, n_workers=1):
        """The global-basis ROM: one cluster whose basis is fitted on all the training snapshots, truncated to ``rank``.

        The baseline the abundance figure compares against: the same code with one cluster, so the only difference is one basis for every ``p_k``. ``rank`` above the fitted rank is clamped, not an error.
        """
        from .cluster import Cluster

        cluster = Cluster(param_names)
        cluster.fit(fm, qoi_indices, 1, cfg.log_cols, cfg.drop_cols, qoi_log=cfg.qoi_log, offset=network.offset)
        labels = np.where(fm.valid_rows, 0, -1)
        # Through the worker pool like every other fit: the k=1 block is the whole training set, and its memory is released when the pool closes.
        roms = _fit_bases(fm, labels, 1, 1.0, cfg.scale_method, n_workers, verbose=False)
        basis = roms[0].basis
        basis.set_rank(min(int(rank), basis.max_rank))
        print(f"global ROM: one basis over {int(fm.valid_rows.sum()):,} snapshots, "
              f"rank {basis.rank} of {basis.max_rank}")
        return cls(cluster=cluster, network=network, roms=roms).precompute_B_structure()


class ROM:
    """Reduced model of one cluster: one :class:`Basis` plus its reduced system."""

    def __init__(self, basis):
        self.basis = basis
        self._bi = self._bj = self._bk = None
        self._L_bi = self._G = None
        self._xbar_bj = self._xbar_bk = None
        self._cached_rank = None

    def set_B_structure(self, bi, bj, bk):
        """Cache the B-tensor sparsity slices for the fast operator build."""
        self._bi, self._bj, self._bk = bi, bj, bk
        self._slice_B()

    def _slice_B(self):
        """Recompute the rank-dependent slices against the basis's current rank."""
        b = self.basis
        self._L_bi = b._L[:, self._bi]
        R_bj, R_bk = b._R[self._bj], b._R[self._bk]
        # G[n, j*m+k] = R_bj[n,j] * R_bk[n,k] is independent of the rates, so folding those into L_bi and `@ G` makes Bred a single BLAS gemm.
        self._G = (R_bj[:, :, None] * R_bk[:, None, :]).reshape(R_bj.shape[0], -1)
        self._R_bj, self._R_bk = R_bj, R_bk
        self._xbar_bj, self._xbar_bk = b._x_bar[self._bj], b._x_bar[self._bk]
        self._cached_rank = b.rank

    def reduced_system(self, A, B, c=None):
        """The reduced system for one interval: the full quadratic operators, with the constant term ``c`` when the system has one, projected into this cluster's reduced coordinates.

        Returns the right-hand side ``z -> f0 + Ar z + (Bred z) z``, with ``f0``, ``Ar``, ``Bred`` the reduced operators, and its Jacobian, as functions of ``(t, z)`` for the solver.
        """
        b = self.basis
        x_bar, R, L, m = b._x_bar, b._R, b._L, b.rank

        f_bar = np.asarray(A @ x_bar).ravel() + apply_B_xy(x_bar, x_bar, B)
        f0 = L @ (f_bar if c is None else f_bar + c)
        A_R = np.asarray(A @ R)

        if self._bi is not None and B is not None and B.nnz and m:
            if self._cached_rank != m:
                self._slice_B()
            Bcoo = B.tocoo(copy=False)
            # The cached indices are the superset pattern; if B at this ``p_k`` differs, the rates misalign and every reduced operator is silently wrong.
            if Bcoo.data.size != self._bi.size:
                raise RuntimeError(
                    f"B nnz={Bcoo.data.size} != cached structure nnz="
                    f"{self._bi.size}: this network's sparsity pattern is "
                    "parameter-dependent, so precompute_B_structure() is "
                    "invalid for it.")
            bv = Bcoo.data.astype(np.float64)

            mixed = bv[:, None] * (self._xbar_bj[:, None] * self._R_bk
                                   + self._R_bj * self._xbar_bk[:, None])
            B_Ar = np.zeros_like(A_R)
            np.add.at(B_Ar, self._bi, mixed)
            Ar = L @ (A_R + B_Ar)
            Bred = ((self._L_bi * bv[None, :]) @ self._G).reshape(m, m, m)
        else:
            Ar_full = np.empty_like(A_R)
            for j in range(m):
                Rj = R[:, j]
                Ar_full[:, j] = (A_R[:, j] + apply_B_xy(x_bar, Rj, B)
                                 + apply_B_xy(Rj, x_bar, B))
            Ar = L @ Ar_full
            Bred = np.zeros((m, m, m), dtype=np.float64)
            if B is not None and B.nnz and m:
                Bcoo = B.tocoo(copy=False)
                N = b.n_species
                bi = Bcoo.row.astype(np.int64)
                bj, bk = Bcoo.col // N, Bcoo.col % N
                Bred = np.einsum("pn,n,nj,nk->pjk", L[:, bi],
                                 np.asarray(Bcoo.data, dtype=np.float64),
                                 R[bj], R[bk], optimize=True)

        Bsym = Bred + Bred.transpose(0, 2, 1)     # for the Jacobian
        return (lambda _t, z: f0 + Ar @ z + (Bred @ z) @ z), (lambda _t, z: Ar + Bsym @ z)

    def solve(self, A, B, x0, t_span, atol, rtol, t_eval=None,
              method="BDF", c=None):
        """Integrate one interval at fixed ``p_k``.

        Returns ``(t, y, sol)``. A failed solve returns what it completed rather than raising, so check ``sol.status``.
        """
        rhs, jacobian = self.reduced_system(A, B, c)
        z0 = np.asarray(self.basis.encode(x0), dtype=np.float64)
        if t_eval is not None:
            # A stored interval boundary can land a few ULPs outside t_span, which solve_ivp rejects outright; clipping is a no-op otherwise.
            t_eval = np.clip(np.asarray(t_eval, dtype=np.float64),
                             min(t_span), max(t_span))
        # Only the implicit solvers use a Jacobian; with an explicit method SciPy ignores it and issues a warning at every step.
        jac = ({"jac": jacobian}
               if method in ("BDF", "Radau", "LSODA") else {})
        sol = solve_ivp(rhs, t_span, z0,
                        method=method, atol=atol, rtol=rtol, t_eval=t_eval,
                        **jac)
        return sol.t, self.basis.decode(sol.y), sol

    def integrator(self, A, B, x0, t0, t_bound, atol, rtol, first_step=None, c=None):
        """A BDF integrator in this cluster's reduced coordinates from ``x0`` at ``t0``, stepped by the caller.

        Unlike :meth:`solve`, this exposes the size of the last step, which :meth:`solve_carried` returns for use as the initial step size of the next interval.
        """
        rhs, jacobian = self.reduced_system(A, B, c)
        return BDF(rhs, t0, np.asarray(self.basis.encode(x0), dtype=np.float64), t_bound,
                   rtol=rtol, atol=atol, jac=jacobian, first_step=first_step)

    def solve_carried(self, A, B, x0, t_span, atol, rtol, t_eval=None, first_step=None, c=None):
        """Integrate one interval at fixed ``p_k`` with BDF, with initial step size ``first_step``.

        Returns ``(t, y, h, ok)``: the sample times ``t_eval`` (the ends of ``t_span`` when ``None``), the decoded states at those times, the size of the last step, to be used as ``first_step`` of the next interval, and whether the integration succeeded.
        """
        solver = self.integrator(A, B, x0, t_span[0], t_span[1], atol, rtol, first_step=first_step, c=c)
        t = np.asarray(t_span if t_eval is None else t_eval, dtype=np.float64)
        h = first_step
        # Every sample the current step covers is read from its dense output in one call; a finished solver covers the rest, which can lie a few ULPs past t_bound.
        z, j, dense = np.empty((solver.y.size, t.size)), 0, None
        while True:
            cover = solver.t if solver.status == "running" else np.inf
            m = j + int(np.searchsorted(t[j:], cover, side="right"))
            if m > j:
                if solver.t_old is None:
                    z[:, j:m] = solver.y[:, None]
                else:
                    dense = dense or solver.dense_output()
                    z[:, j:m] = dense(t[j:m])
                j = m
            if j == t.size:
                break
            solver.step()
            dense = None
            if solver.status == "failed":
                return t[:j], self.basis.decode(z[:, :j]), h, False
            if solver.t_old is not None and solver.t > solver.t_old:
                h = solver.t - solver.t_old
        return t, self.basis.decode(z), h, True


class Basis:
    """One cluster's triple: centering state, diagonal scaling and reduced basis."""

    def __init__(self):
        self.scaler = None
        self.pca = None
        self.rank = 0
        self.var_threshold = None
        # Precomputed coordinate-space matrices (see _precompute).
        self._scale = None
        self._x_bar = None
        self._Q_full = None    # every PCA row, (n_pca, n_species)
        self._R_full = None    # D[:,None] * Q_full.T
        self._L_full = None    # Q_full / D[None,:]
        self._R = self._L = None   # rank-dependent slices

    def _precompute(self):
        scale = self.scaler.scale_
        self._scale = scale
        self._x_bar = self.scaler.mean_ + scale * self.pca.mean_
        self._Q_full = np.asarray(self.pca.components_, dtype=np.float64)
        self._R_full = scale[:, None] * self._Q_full.T
        self._L_full = self._Q_full / scale[None, :]
        self._slice()

    def _slice(self):
        self._R = self._R_full[:, :self.rank]
        self._L = self._L_full[:self.rank, :]

    def set_rank(self, rank):
        """Change the basis rank; no refit, the PCA is already full-rank."""
        rank = int(rank)
        if not 0 <= rank <= self.pca.components_.shape[0]:
            raise ValueError(f"rank must be in [0, "
                             f"{self.pca.components_.shape[0]}]; got {rank}")
        self.rank = rank
        self._slice()
        return self

    @property
    def max_rank(self):
        return int(self.pca.components_.shape[0])

    @property
    def n_species(self):
        return self.scaler.mean_.size

    def encode(self, x):
        """``z = U^T D^-1 (x - x_bar)``, for a vector or ``(n_species, n)`` block.

        Centres before projecting: the same result as ``L@x - L@x_bar``, without cancelling two large numbers to reach a small coordinate.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            return self._L @ (x - self._x_bar)
        return self._L @ (x - self._x_bar[:, None])

    def decode(self, z):
        """``x = x_bar + D U z``.

        A diverged solve can return an empty or truncated reduced state; return an empty block so the caller can report it via ``sol.status``.
        """
        z = np.asarray(z, dtype=np.float64)
        if z.ndim == 1:
            if z.size < self.rank:
                return np.empty((self.n_species, 0))
            return self._x_bar + self._R @ z
        if z.shape[1] == 0 or z.shape[0] < self.rank:
            return np.empty((self.n_species, 0))
        return self._x_bar[:, None] + self._R @ z

    def projection_error(self, x):
        """Relative error of representing ``x`` in this basis, ``||u - Q_r^T Q_r u|| / ||u||`` with ``u = D^-1 (x - x_bar)``.

        Measured in the normalized coordinates ``u`` in which the POD was fitted, with ``Q_r`` the first ``r`` POD modes; in physical units the error is dominated by the abundant species every basis reconstructs perfectly.
        """
        u = (np.asarray(x, dtype=np.float64) - self._x_bar) / self._scale
        Q = self._Q_full[:self.rank]
        return float(np.linalg.norm(u - Q.T @ (Q @ u)) / (np.linalg.norm(u) + 1e-300))

    def project_nonnegative(self, z0, offset=None):
        """Nearest reduced state to ``z0`` whose decoded state, plus ``offset`` if given, is nonnegative. Returns ``(z, ok)``.

        ``min 0.5||z - z0||^2  s.t.  offset + x_bar + R z >= 0``, via SLSQP.

        Non-negativity is imposed in the normalized coordinates. SLSQP's ``ftol`` is absolute and governs constraint satisfaction, so on a state spanning many decades an unscaled constraint has no effect on trace species: a violation much larger than their magnitude is accepted as feasible, and ``z0`` is returned unchanged.
        """
        x_bar = self._x_bar if offset is None else offset + self._x_bar
        scale = np.asarray(self._scale, dtype=np.float64)
        pos_A, pos_b = self._R / scale[:, None], x_bar / scale
        solution = minimize(lambda z: 0.5 * np.sum((z - z0) ** 2), z0,
                            jac=lambda z: z - z0, method="SLSQP",
                            constraints=[{"type": "ineq",
                                          "fun": lambda z: pos_b + pos_A @ z,
                                          "jac": lambda z: pos_A}],
                            options={"maxiter": 200, "ftol": 1e-14})
        return solution.x, bool(solution.success)

    @classmethod
    def fit(cls, X, var_threshold, scale_method="pareto"):
        """Fit one cluster's triple from its snapshot matrix ``(n_samples, n_species)``.

        A full-rank PCA on the scaled snapshots; the basis rank is the smallest reaching ``var_threshold``.

        ``"pareto"`` (Pareto scaling) divides by ``sqrt(std)``: full z-scoring would weight a near-constant species like one spanning ten decades, and the square root keeps trace species visible without letting them dominate. ``"none"`` leaves the basis orthonormal in physical space, which is what a homogeneous state vector wants.
        """
        scaler = StandardScaler().fit(X)
        scaler.scale_ = (np.sqrt(scaler.scale_) if scale_method == "pareto"
                         else np.ones_like(scaler.scale_))
        scaler.var_ = scaler.scale_ ** 2

        X_scaled = (X - scaler.mean_) / scaler.scale_
        n_comp = min(X.shape[1], X_scaled.shape[0])
        try:
            pca = PCA(n_components=n_comp, svd_solver="full").fit(X_scaled)
        except np.linalg.LinAlgError:
            # A near-rank-deficient block can make the data-matrix SVD fail to converge; the covariance eigendecomposition is robust and equivalent for n_samples >> n_features.
            pca = PCA(n_components=n_comp, svd_solver="covariance_eigh").fit(X_scaled)

        # Fix the sign of each POD mode: U and -U decode identically, but sklearn picks the sign from memory layout, so stored operators would otherwise differ run to run.
        comp = pca.components_
        if comp.size:
            lead = np.argmax(np.abs(comp), axis=1)
            sign = np.sign(comp[np.arange(comp.shape[0]), lead])
            sign[sign == 0] = 1.0
            pca.components_ = comp * sign[:, None]

        cumvar = np.cumsum(np.nan_to_num(pca.explained_variance_ratio_))
        rank = (n_comp if var_threshold >= 1
                else min(int(np.searchsorted(cumvar, var_threshold)) + 1, n_comp))

        basis = cls()
        basis.scaler, basis.pca = scaler, pca
        basis.rank, basis.var_threshold = int(rank), float(var_threshold)
        basis._precompute()
        return basis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_B_xy(x, y, B):
    """``out[i] = sum_jk B[i,j,k] x[j] y[k]``, with ``B`` stored as ``(N, N*N)``."""
    x = np.asarray(x, dtype=np.float64)
    if B is None or not B.nnz:
        return np.zeros_like(x)
    N = x.size
    Bcoo = B.tocoo(copy=False)
    out = np.zeros(N, dtype=np.float64)
    np.add.at(out, Bcoo.row,
              Bcoo.data * x[Bcoo.col // N] * np.asarray(y)[Bcoo.col % N])
    return out


def qoi_error(y_rom, y_fom, qoi_rows, norm="relative"):
    """QoI error between two ``(n_species, n)`` blocks: the largest L2-in-time error over the QoIs.

    Taking the largest over QoIs means a cluster meets the tolerance only when every QoI does.

    ``"relative"`` divides by ``||y_fom||``, so the tolerance reads as a fraction. ``"absolute"`` is the RMS-in-time error in state units, for a signed state whose norm can be near zero on a block where a QoI stays near zero; the maximum over QoIs would then keep splitting toward a tolerance that cannot be met.
    """
    d = np.asarray(y_rom)[qoi_rows] - np.asarray(y_fom)[qoi_rows]
    if norm == "absolute":
        return float(np.max(np.sqrt(np.mean(d ** 2, axis=1))))
    num = np.sqrt(np.sum(d ** 2, axis=1))
    den = np.sqrt(np.sum(np.asarray(y_fom)[qoi_rows] ** 2, axis=1)) + 1e-300
    return float(np.max(num / den))


def _fit_bases(fm, sample_labels, n_clusters, var_threshold, scale_method, n_workers, verbose):
    """Fit one basis per cluster label, in parallel across clusters.

    Each worker reads its block straight from the memmap. Full-rank PCA already uses multithreaded BLAS, so keep ``n_workers`` below the core count.
    """
    args = []
    for c in range(int(n_clusters)):
        idx = np.flatnonzero(sample_labels == c)
        if idx.size >= 2:
            args.append((c, idx, var_threshold, scale_method))
        elif verbose:
            print(f"  [cluster {c}] skipped ({idx.size} snapshot(s))")

    roms = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_fit_init,
                             initargs=(fm.path, fm.species_slice)) as pool:
        for fut in as_completed([pool.submit(_fit_one, a) for a in args]):
            c, basis, n_snap = fut.result()
            roms[c] = ROM(basis)
            if verbose:
                print(f"  [cluster {c}] fit on {n_snap:,} snapshots "
                      f"-> rank {basis.rank}")
    return roms
