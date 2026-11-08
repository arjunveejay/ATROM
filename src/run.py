"""Run one experiment from its settings file, ``experiments/<case>/<model>/run.py``.

Each model is run over every set (testing, then training trajectories) before the next:

    1. AT-ROM:           at every (tau_split, eta) pair; fitted on the first set's pass, reused after
    2. Hybrid:          AT-ROM with backtracking to the FOM, using the AT-ROMs from step 1
    3. global-basis ROM: one basis for all data, at each basis rank in ``ranks``
    4. figures for each set

Results are written next to the settings file. A job that stops partway can be restarted: work whose results are already on disk is skipped (``--overwrite`` redoes everything).

Three classes:

    Settings     the settings of a run: the system's fixed values plus the experiment's choices
    Experiment   one experiment on one set of trajectories: its data, its models, the sweeps
    Results      the result files: their names, writing them as a sweep goes, reading them back
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import math
import os
from functools import cached_property
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import interp1d

from . import REPO_ROOT
from .cluster import Cluster
from .fom.data import load_trajectories, open_snapshots
from .hybrid import AGGREGATES, Hybrid, HybridConfig
from .rom import ATROM


def main(experiment_file):
    """Run the experiment whose settings file is ``experiment_file``; results go in that file's folder."""
    p = argparse.ArgumentParser(description="Run an experiment")
    p.add_argument("--splits", nargs="+", choices=["test", "train"], default=["test", "train"])
    p.add_argument("--overwrite", action="store_true", help="redo everything instead of skipping finished work")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--n-workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(experiment_file))
    Experiment(out_dir, n_workers=args.n_workers).generate_dataset()
    shared = {}
    exps = [Experiment(out_dir, split, args.overwrite, shared, n_workers=args.n_workers) for split in args.splits]
    # The snapshot files are rebuilt, if the settings ask for it, by the first split only.
    exps[0].cfg = exps[0].cfg.replace(build_snapshots=bool(exps[0].s.get("build_snapshots")))
    s = exps[0].s
    print(f"[{exps[0].case}] {len(s['etas'])} eta x {len(s['taus'])} tau_split, sets {args.splits} -> {out_dir}", flush=True)

    # Each model over every set before the next model. The AT-ROMs are fitted on the first set's pass and reused after.
    for exp in exps:
        exp.atrom_sweep()
    if s.get("hybrid_positivity"):
        for exp in exps:
            exp.hybrid_sweep()
    for exp in exps:
        exp.global_rom_sweep()
    if not args.no_figures:
        exps[0].figures()
    print("done", flush=True)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(SimpleNamespace):
    """All settings of one run, read as attributes (``cfg.atol``).

    They come from two places: the system's fixed values (``src/fom/astrochemistry.py``, ``egfr.py``, ``lorenz96.py``) and the experiment's settings file. Runs and notebooks both read them through :meth:`load`, so they cannot drift apart.
    """

    #: Entries of an experiment's settings that steer the run rather than the model.
    RUN_KEYS = {"build_snapshots", "taus", "etas", "ranks", "n_eval", "splits", "generate", "positivity_modes",
                "hybrid_positivity", "figure_qoi"}

    def __init__(self, **values):
        super().__init__(**values)
        if values.get("eta") is not None and not 0.0 < values["eta"] < 1.0:
            raise ValueError(f"eta must be between 0 and 1; got {values['eta']!r}")
        if values.get("max_clusters", 1) < values.get("n_clusters", 1):
            raise ValueError("max_clusters is smaller than n_clusters")

    @property
    def var_threshold(self) -> float:
        """Share of the variance a basis keeps."""
        return 1.0 - self.eta

    def replace(self, **over) -> "Settings":
        return Settings(**{**vars(self), **over})

    @classmethod
    def load(cls, out_dir, **over):
        """The settings of the run in ``out_dir``: the system's fixed values, then the experiment's settings file, then ``over``."""
        case, experiment = cls.read_experiment(out_dir)
        n_eval = experiment.get("n_eval", 0)
        values = dict(out_dir=out_dir, build_snapshots=False, n_workers=os.cpu_count() or 1,
                      n_eval=n_eval["test"] if isinstance(n_eval, dict) else n_eval)
        values.update(cls.system_defaults(case))
        values.update({k: v for k, v in experiment.items() if k not in cls.RUN_KEYS})
        values.update(over)
        if "data_dir" in values:
            values["data_dir"] = os.path.join(REPO_ROOT, values["data_dir"])
        for key in ("qoi", "log_cols", "drop_cols"):
            if key in values:
                values[key] = list(values[key])
        return cls(**values)

    @staticmethod
    def read_experiment(out_dir):
        """``(CASE, SETTINGS)`` of the experiment's settings file, ``<out_dir>/run.py``."""
        path = os.path.join(out_dir, "run.py")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found; each run folder holds the run.py with its settings")
        spec = importlib.util.spec_from_file_location("experiment_settings", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CASE, dict(module.SETTINGS)

    @staticmethod
    def system_defaults(case):
        """The fixed values of ``case``'s system."""
        if case in ("nelson", "osu"):
            from .fom.astrochemistry import DEFAULTS
            return dict(DEFAULTS[case])
        if case == "lorenz96":
            from .fom.lorenz96 import DEFAULTS
        elif case == "egfr":
            from .fom.egfr import DEFAULTS
        else:
            raise ValueError(f"unknown case {case!r}")
        return dict(DEFAULTS)


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """One experiment on one data split: its settings, its data, its models and the three sweeps.

    ``out_dir`` is the experiment's folder, which holds its ``run.py``. ``over`` overrides model settings, e.g. ``n_workers=1`` or ``eta=1e-6``. The data are loaded when first used, so opening an experiment in a notebook is cheap.
    """

    def __init__(self, out_dir, split="test", overwrite=False, shared=None, **over):
        self.out_dir = os.path.abspath(out_dir)
        self.case, self.s = Settings.read_experiment(self.out_dir)
        self.split, self.overwrite = split, overwrite
        n_eval = self.s.get("n_eval", 0)
        if isinstance(n_eval, dict):
            over.setdefault("n_eval", n_eval[split])
        self.cfg = Settings.load(self.out_dir, **over)
        self.sweep_dir = os.path.join(self.out_dir, "sweep")
        self.shared = {} if shared is None else shared   # what the sets of one run share, e.g. the initial clusters
        self.tag = Results.suffix(split)

    # --- data ---------------------------------------------------------------

    @cached_property
    def ctx(self):
        """The system's network and dataset."""
        if self.case == "lorenz96":
            from .fom.lorenz96 import load_context
        elif self.case == "egfr":
            from .fom.egfr import load_context
        else:
            from .fom.astrochemistry import load_context
        return load_context(self.cfg)

    @property
    def split_files(self):
        """``(snapshot file path, parameter trajectories p(t), global ids)`` of this split."""
        ctx = self.ctx
        if self.split == "train":
            return ctx.train_snapshots_path, ctx.train_tracers, ctx.train_indices
        if self.split == "test":
            return ctx.test_snapshots_path, ctx.test_tracers, ctx.test_indices
        raise ValueError(f"split must be 'test' or 'train'; got {self.split!r}")

    @cached_property
    def train(self):
        """The training snapshot file, memory-mapped: what the models are fitted on."""
        ctx = self.ctx
        return open_snapshots(ctx.train_snapshots_path, len(ctx.train_indices), ctx.n_steps, ctx.n_species)

    @cached_property
    def ref(self):
        """The FOM solutions the models are scored against: the first ``cfg.n_eval`` trajectories of this split."""
        _, tracers, ids = self.split_files
        n = int(min(self.cfg.n_eval, len(ids)))
        t, y, x0s = self.trajectories(n)
        return SimpleNamespace(tracers=tracers[:n], x0s=x0s, t=t, y=y, ids=ids)

    def trajectories(self, n):
        """``(times, states, initial states)`` of the FOM solutions of the first ``n`` trajectories of this split (or those at the positions in ``n``)."""
        return load_trajectories(self.split_files[0], n, self.ctx.n_species)

    @property
    def qoi_names(self):
        return [self.ctx.net.species[j] for j in self.ctx.qoi_indices]

    def generate_dataset(self):
        """Create the dataset with the system's generator, if the settings ask for one and ``data_dir`` holds none."""
        options = self.s.get("generate")
        if options is None or os.path.exists(os.path.join(self.cfg.data_dir, "params.npz")):
            return
        if self.case == "lorenz96":
            from .fom.lorenz96 import main as generate
        elif self.case == "egfr":
            from .fom.egfr import main as generate
            options = [*options, "--n-workers", str(self.cfg.n_workers)]
        else:
            raise ValueError(f"{self.case} has no dataset generator; its data are provided separately")
        print(f"[{self.case}] generating the dataset in {self.cfg.data_dir}", flush=True)
        generate([*options, "--out", self.cfg.data_dir])

    # --- models -------------------------------------------------------------

    def load_atrom(self, tau, eta):
        """The AT-ROM saved for one (tau_split, eta), ready to solve. Its clusters are ``atrom.cluster``."""
        atrom = ATROM.load(os.path.join(self.sweep_dir, f"atrom_{Results.label(tau, eta)}.npz"), self.ctx.net)
        return atrom.precompute_B_structure()

    def initial_clusters(self):
        """The initial clusters (k-means on the training feature vectors), fitted on first use and shared by every set of the run.

        They do not depend on tau_split or eta, and they are needed only when an AT-ROM has to be fitted.
        """
        if "clusters" not in self.shared:
            ctx, cfg = self.ctx, self.cfg
            base = Cluster(ctx.param_names)
            base.fit(self.train, ctx.cluster_indices, cfg.n_clusters, cfg.log_cols, cfg.drop_cols,
                     qoi_log=cfg.qoi_log, seed=cfg.base_seed, offset=ctx.net.offset)
            self.shared["clusters"] = base
        return self.shared["clusters"]

    def fit_or_load_atroms(self, eta, taus):
        """The AT-ROMs at one eta for each tau_split in ``taus``, handed over one at a time as ``(tau, atrom)``: loaded where saved, otherwise built by adaptive cluster refinement and saved as soon as each tau is finished."""
        path = lambda tau: os.path.join(self.sweep_dir, f"atrom_{Results.label(tau, eta)}")
        saved = [] if self.overwrite else [tau for tau in taus if os.path.exists(path(tau) + ".npz")]
        for tau in saved:
            print(f"[AT-ROM] {Results.label(tau, eta)}: reusing the saved model", flush=True)
            yield tau, ATROM.load(path(tau) + ".npz", self.ctx.net)
        missing = [tau for tau in taus if tau not in saved]
        if not missing:
            return
        print(f"[AT-ROM] eta={eta:g}: fitting tau = {[float(t) for t in missing]}", flush=True)
        ctx = self.ctx
        # Seeded by the experiment's full list of taus, so each tau gets the same model however many are fitted together.
        fits = ATROM.build_adaptive(self.initial_clusters(), ctx.net, self.train, ctx.n_steps, ctx.dt_hydro, self.cfg.replace(eta=eta),
                                    self.s["taus"], ctx.qoi_indices, fit_only=missing)
        for tau, atrom in fits:
            atrom.save(path(tau))
            yield tau, atrom

    # --- scoring ------------------------------------------------------------

    def score(self, model, **over):
        """Solve ``model`` on the reference trajectories, with ``over`` applied to the settings, and measure its error.

        Two measures per QoI: ``max-in-time``, the largest relative error at any time, and ``l2-in-time``, the relative L2 error ``||y_rom - y_fom|| / ||y_fom||`` over the trajectory. Only trajectories that completed every interval are scored.
        """
        cfg, ref, ctx = self.cfg.replace(**over), self.ref, self.ctx
        n = len(ref.t)
        results = model.solve_tracers(ref.tracers[:n], ref.x0s, ctx.dt_hydro, cfg.atol, cfg.rtol, list(ref.t),
                                      cfg.n_workers, positivity=cfg.positivity, method=cfg.ode_method,
                                      verbose=False, carry=cfg.carry_step)
        t_roms, y_roms, _paths, infos = map(list, zip(*results))

        expected_end = (np.asarray(ref.tracers[0]).shape[0] - 1) * ctx.dt_hydro
        valid = [i for i in range(n)
                 if infos[i] is None and t_roms[i].size and np.isclose(t_roms[i][-1], expected_end)]
        out = {"n_tracers": n, "n_valid": len(valid), "valid_idx": list(valid),
               "n_solved": sum(1 for i in infos if i is None), "per_tracer": {}}
        # No trajectory completed: report empty errors rather than failing, so one (tau, eta) pair cannot end a sweep.
        if not valid:
            out["per_tracer"] = {tag: {name: np.empty(0) for name in self.qoi_names} for tag in AGGREGATES}
            return out

        max_err, l2_err = [], []
        for i in valid:
            t_fom, y_fom, y_rom = ref.t[i], ref.y[i], y_roms[i]
            if t_roms[i].shape[0] != t_fom.shape[0]:
                y_rom = interp1d(t_roms[i], y_rom, axis=1, bounds_error=False, fill_value="extrapolate")(t_fom)
            resid = y_rom - y_fom
            max_err.append((np.abs(resid) / (np.abs(y_fom) + 1e-30)).max(axis=1))
            l2_err.append(np.sqrt(np.sum(resid ** 2, axis=1)) / (np.sqrt(np.sum(y_fom ** 2, axis=1)) + 1e-30))

        for tag, err in (("max-in-time", max_err), ("l2-in-time", l2_err)):
            x = np.stack(err)[:, ctx.qoi_indices]
            out["per_tracer"][tag] = {name: x[:, j].copy() for j, name in enumerate(self.qoi_names)}
        return out

    def error_rows(self, ev, positivity, extra=None):
        """One row per attempted trajectory: position, global id, ``positivity``, whether it completed, and its error per (measure, QoI); nan if it did not complete.

        ``extra`` maps column names to values over all attempted trajectories, such as the hybrid diagnostics.
        """
        completed = {int(p): k for k, p in enumerate(ev["valid_idx"])}
        rows = []
        for i in range(ev["n_tracers"]):
            row = dict(position=i, global_id=int(self.ref.ids[i]), positivity=_on_off(positivity),
                       completed=int(i in completed))
            for measure, per_qoi in ev["per_tracer"].items():
                for q, values in per_qoi.items():
                    row[f"{measure}|{q}"] = float(values[completed[i]]) if i in completed else float("nan")
            for name, values in (extra or {}).items():
                row[name] = float(values[i])
            rows.append(row)
        return rows

    @staticmethod
    def summary_row(tau, eta, positivity, ev, atrom):
        """One row of a (tau, eta) sweep's summary table."""
        return dict(tau_split=float(tau), eta=float(eta), positivity=_on_off(positivity),
                    n_tracers=int(ev["n_tracers"]), n_completed=int(ev["n_valid"]),
                    n_clusters=int(atrom.cluster.n_clusters), mean_rank=atrom.mean_rank(), **atrom.summary_columns())

    # --- the run ------------------------------------------------------------

    def run(self, figures=True):
        """All three sweeps on this set, then its figures: AT-ROM, hybrid, global-basis ROM."""
        self.atrom_sweep()
        if self.s.get("hybrid_positivity"):
            self.hybrid_sweep()
        self.global_rom_sweep()
        if figures:
            self.figures()

    def global_rom_sweep(self):
        """Global-basis ROM (one basis, no clusters) at each basis rank, without the non-negativity correction.

        Writes ``global_fits_<split>.csv`` (one row per rank) and ``global_errors_<split>.csv`` (one row per trajectory and rank).
        """
        results = Results(self._path(f"global_fits{self.tag}.csv"), self._path(f"global_errors{self.tag}.csv"),
                          key=lambda row: int(float(row["rank"])), overwrite=self.overwrite)
        max_rank = int(float(results.rows[0]["rho"])) if results.rows else None
        todo = [int(r) for r in self.s["ranks"] if not results.done(int(r)) and (max_rank is None or int(r) <= max_rank)]
        if not todo:
            print("[global ROM] all sizes already done", flush=True)
            return

        ctx, cfg = self.ctx, self.cfg
        rom = ATROM.fit_global(self.train, ctx.net, cfg, ctx.cluster_indices, ctx.param_names, ctx.n_species, cfg.n_workers)
        basis = next(iter(rom.roms.values())).basis
        energy = np.nan_to_num(basis.pca.explained_variance_ratio_)
        for r in todo:
            if r > basis.max_rank:
                print(f"[global ROM] size {r} is larger than the basis ({basis.max_rank}); skipped", flush=True)
                continue
            basis.set_rank(r)
            ev = self.score(rom, positivity=False)
            summary = dict(rank=r, rho=basis.max_rank, residual_energy=float(max(1.0 - energy[:r].sum(), 0.0)),
                           n_tracers=int(ev["n_tracers"]), n_completed=int(ev["n_valid"]))
            errors = []
            for row in self.error_rows(ev, False):
                row.pop("positivity")
                errors.append(dict(position=row.pop("position"), global_id=row.pop("global_id"), rank=r, **row))
            results.add(r, [summary], errors)
            print(f"[global ROM] size {r}: {ev['n_valid']}/{ev['n_tracers']} completed", flush=True)

    def atrom_sweep(self):
        """AT-ROM at every (tau_split, eta), each evaluated with every setting in ``positivity_modes``.

        Each (tau, eta) model is saved in ``sweep/atrom_<tau_eta>.npz`` as soon as it is fitted, then scored at once, writing ``sweep/errors_<tau_eta>_<split>.csv`` and updating the summary ``fits_<split>.csv``. Later splits reuse the saved models.
        """
        os.makedirs(self.sweep_dir, exist_ok=True)
        results = self._pair_results("fits", "errors")

        for eta in self.s["etas"]:
            todo = [tau for tau in self.s["taus"] if not results.done(Results.label(tau, eta))]
            if not todo:
                continue
            for tau, atrom in self.fit_or_load_atroms(eta, todo):
                label = Results.label(tau, eta)
                atrom.precompute_B_structure()
                summary, errors = [], []
                for positivity in self.s["positivity_modes"]:
                    ev = self.score(atrom, eta=eta, positivity=positivity)
                    errors += self.error_rows(ev, positivity)
                    summary.append(self.summary_row(tau, eta, positivity, ev, atrom))
                    print(f"[AT-ROM] {label} positivity {_on_off(positivity)}: {ev['n_valid']}/{ev['n_tracers']} completed, "
                          f"{atrom.cluster.n_clusters} clusters", flush=True)
                results.add(label, summary, errors)

    def hybrid_sweep(self):
        """Hybrid at every (tau_split, eta), using the AT-ROMs saved by :meth:`atrom_sweep`; nothing is refitted.

        Writes ``sweep/hybrid_errors_<tau_eta>_<split>.csv`` per pair; a pair whose file exists is skipped.
        """
        for eta in self.s["etas"]:
            for tau in self.s["taus"]:
                label = Results.label(tau, eta)
                path = os.path.join(self.sweep_dir, f"hybrid_errors_{label}{self.tag}.csv")
                if os.path.exists(path) and not self.overwrite:
                    continue
                ref, ctx = self.ref, self.ctx
                atrom = self.load_atrom(tau, eta)
                errors = []
                for positivity in self.s["hybrid_positivity"]:
                    cfg = self.cfg.replace(eta=eta, positivity=positivity)
                    ev = Hybrid(atrom, cfg, HybridConfig(positivity=positivity)).evaluate(
                        ref.tracers, ref.x0s, ref.t, ref.y, ctx.qoi_indices, self.qoi_names, ctx.dt_hydro,
                        n_workers=cfg.n_workers, verbose=False)
                    errors += self.error_rows(ev, positivity, extra=ev["hybrid"])
                    fom = ev["hybrid"]["fom_frac"]
                    fom = float(np.nanmean(fom)) if np.isfinite(fom).any() else float("nan")
                    print(f"[hybrid] {label} positivity {_on_off(positivity)}: {ev['n_valid']}/{ev['n_tracers']} completed, "
                          f"full model used {fom:.1%} of the time", flush=True)
                Results._write(path, errors)

    def figures(self):
        """The run's figures, in ``figures/``: the error panel over both sets, and the AT-ROM and hybrid violins."""
        from .plots import draw_figures
        fig_dir = draw_figures(self.case, self.out_dir, self.s.get("figure_qoi") or self.qoi_names, self.s["taus"], self.s["etas"])
        print(f"[{self.case}] figures -> {fig_dir}", flush=True)

    def _path(self, name):
        return os.path.join(self.out_dir, name)

    def _pair_results(self, summary_name, errors_name):
        """The result files of a (tau_split, eta) sweep: ``<summary_name>_<split>.csv`` and one ``sweep/<errors_name>_<pair>_<split>.csv`` per pair."""
        return Results(self._path(f"{summary_name}{self.tag}.csv"),
                       lambda label: os.path.join(self.sweep_dir, f"{errors_name}_{label}{self.tag}.csv"),
                       key=lambda row: Results.label(float(row["tau_split"]), float(row["eta"])),
                       overwrite=self.overwrite)


def _on_off(positivity):
    return "on" if positivity else "off"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class Results:
    """The result files of a run: how they are named, writing them as a sweep goes, and reading them back.

    An instance holds one sweep's files: a summary CSV with a row per finished piece of work, and the per-trajectory errors, in one file (``errors_path`` a path) or one file per piece (``errors_path`` a function of its key). Both are rewritten after every :meth:`add`, so a stopped job loses at most the piece it was on, and a restarted job skips what is already done. ``key`` maps a summary row to its piece of work, e.g. its basis rank or its (tau_split, eta) label.

    Rows are written as dicts and read back (:meth:`read`) as numpy structured arrays: ``tab["eta"]`` is a column and ``tab[tab["qoi"] == "CO"]`` a subset of rows.
    """

    def __init__(self, summary_path, errors_path, key, overwrite=False):
        self.summary_path, self.errors_path, self.key = summary_path, errors_path, key
        self.rows, self.errors = [], []
        if not overwrite:
            if callable(errors_path):
                # A piece counts as done only if its errors file was written too.
                self.rows = [row for row in self._read_rows(summary_path) if os.path.exists(errors_path(key(row)))]
            else:
                self.rows = self._read_rows(summary_path)
                done = {key(row) for row in self.rows}
                self.errors = [row for row in self._read_rows(errors_path) if key(row) in done]
        self.finished = {key(row) for row in self.rows}
        if self.rows:
            print(f"  resuming {summary_path}: {len(self.finished)} finished piece(s) kept", flush=True)

    def done(self, key):
        return key in self.finished

    def add(self, key, summary_rows, error_rows):
        """Record one finished piece of work and rewrite the files."""
        if callable(self.errors_path):
            self._write(self.errors_path(key), error_rows)
        else:
            self.errors += error_rows
            self._write(self.errors_path, self.errors)
        self.rows += summary_rows
        self._write(self.summary_path, self.rows)
        self.finished.add(key)

    # --- names --------------------------------------------------------------

    @staticmethod
    def label(tau, eta):
        """The label ``tol<tau>_eta<eta>`` shared by the files of one (tau, eta) pair, e.g. ``tol0.01_eta1e-6``."""
        exp = math.log10(eta)
        eta_part = f"eta1e{int(round(exp))}" if abs(exp - round(exp)) < 1e-12 else f"eta{eta:g}"
        return f"tol{tau:g}_{eta_part}"

    @staticmethod
    def suffix(split):
        """File-name suffix of the training or testing set: ``"_train"`` or ``"_test"``."""
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test'; got {split!r}")
        return f"_{split}"

    # --- reading ------------------------------------------------------------

    @staticmethod
    def read(path):
        """A CSV as a structured array, one field per column: floats where every value parses, strings otherwise; an empty numeric cell is nan."""
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            raw = list(reader)
        if not raw:
            raise ValueError(f"{path} has a header but no rows")
        columns, dtypes = [], []
        for j, name in enumerate(header):
            vals = [r[j] for r in raw]
            try:
                col = np.array([float(v) if v != "" else np.nan for v in vals])
            except ValueError:
                col = np.array(vals, dtype=str)
            columns.append(col)
            dtypes.append((name, col.dtype))
        tab = np.empty(len(raw), dtype=dtypes)
        for name, col in zip(header, columns):
            tab[name] = col
        return tab

    @staticmethod
    def where(tab, **equals):
        """Rows of ``tab`` matching every ``column=value`` pair."""
        mask = np.ones(tab.size, dtype=bool)
        for col, val in equals.items():
            mask &= tab[col] == val
        return tab[mask]

    @staticmethod
    def columns(tab):
        """Column names of a table from :meth:`read`."""
        return tab.dtype.names

    # --- writing ------------------------------------------------------------

    @staticmethod
    def _write(path, rows):
        """Write row dicts to ``path``; the first row fixes the column order."""
        rows = list(rows)
        if not rows:
            raise ValueError(f"no rows to write to {path}")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    @staticmethod
    def _read_rows(path):
        """The rows of a CSV as dicts of strings, as :meth:`_write` takes them; empty if the file does not exist."""
        if not os.path.exists(path):
            return []
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
