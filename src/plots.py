"""The figures of a run and the helpers its notebook uses, all drawn from results a run already wrote.

``draw_figures``           what a run draws: the error panel over both sets, and the AT-ROM and hybrid violins
``sweep_panel_splits``     the error panel, training set above testing set
``sweep_partition_panel``  cluster counts, mean basis rank and hybrid FOM use versus eta
``violin_panel``           violin plots of the per-trajectory errors versus eta
``abundance_panel``, ``trajectory_panel``, ``qoi_panels_pair``   one-trajectory figures used by the notebooks

The error and violin panels need only the result tables; the one-trajectory figures need that trajectory solved.
File names (``Results.label``) live in :mod:`run`, so the solvers never import matplotlib.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
from numpy.lib import recfunctions
import logging
import matplotlib
matplotlib.use("Agg")
# Missing math fonts fall back to DejaVu Sans; the warning for each one is noise.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, SymmetricalLogLocator

from .fom.astrochemistry import NSEC
from .hybrid import FOM_INTERVAL
from .run import Results


LIGHT = {0.005: "#b39ddb", 0.01: "#7fc4f2", 0.05: "#c9b3e6", 0.10: "#f5a86a",
         0.20: "#f2b9c4", 0.40: "#a9d6a9"}

DARK = {0.005: "#4a148c", 0.01: "#1f77b4", 0.05: "#6a3d9a", 0.10: "#d9601b",
        0.20: "#c2185b", 0.40: "#2e7d32"}

TOL_PAIRS = [("#7fc4f2", "#1f77b4"),        # blue
             ("#f5a86a", "#d9601b"),        # orange
             ("#a9d6a9", "#2e7d32"),        # green
             ("#c9b3e6", "#6a3d9a"),        # purple
             ("#f2b9c4", "#c2185b"),        # pink
             ("#bdbdbd", "#525252")]

HYBRID_MARKER = "*"      # Hybrid points on the AT-ROM error panel, in the colour of their tau_split

ETA_MIN = 1e-16

FAIL_COLOR = "tab:red"

GLOBAL_COLOR = "k"

SPLIT_CAPTION = {"test": "Testing", "train": "Training"}

SPLIT_WORD = {"test": "testing", "train": "training"}

PAPER_STYLE = {
    "text.usetex": False,
    "font.size": 18,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
}

# Cluster bands of the QoI panels: one alpha for the data panel and the strip below it.
BAND_ALPHA = 0.30

# Cluster colours, as in abundance_panel.
BAND_CMAP = plt.cm.tab20

# QoI line colours (Okabe-Ito, colour-blind safe, without its yellow).
LINE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
               "#56B4E9", "#000000", "#8B4513")

#: Cluster band alpha of trajectory_panel.
TRAJ_BAND_ALPHA = 0.12

#: FOM solid, AT-ROM dashed, global-basis ROM dotted.
TRAJ_COLORS = {"reference": "0.15", "clustered": "#009f47", "global": "#ff0800"}

TRAJ_STYLE = {"reference": ("-", 2.2), "clustered": ("--", 1.8),
              "global": (":", 1.8)}

# Per-system settings: a signed state has no non-negativity correction, and astrochemistry times are in kyr.
CASE_FIGURES = {
    "nelson": dict(positivity_label="on", time_scale=NSEC * 1e3, time_label="Time [kyr]", hybrid_positivity=True),
    "osu": dict(positivity_label="on", time_scale=NSEC * 1e3, time_label="Time [kyr]", hybrid_positivity=True),
    "lorenz96": dict(positivity_label="off", time_scale=1.0, time_label="Time", hybrid_positivity=False),
    "egfr": dict(positivity_label="on", time_scale=1.0, time_label="Time [s]", hybrid_positivity=False),
}


# ---------------------------------------------------------------------------
# What a run draws
# ---------------------------------------------------------------------------

def draw_figures(case, out_dir, qoi_names, taus, etas):
    """Draw a finished run's figures into ``<out_dir>/figures`` and return that folder.

    The error panel (``panel_errors.pdf``, only when both sets were run), the cluster counts and mean basis rank versus eta (``panel_partition.pdf``), and the testing-set violins, ``violin_atrom.pdf`` and, when hybrid was run, ``violin_hybrid.pdf``.
    """
    shown = CASE_FIGURES[case]["positivity_label"]   # one setting only, so runs with and without the correction are not averaged
    fig_dir, sweep_dir = os.path.join(out_dir, "figures"), os.path.join(out_dir, "sweep")
    os.makedirs(fig_dir, exist_ok=True)
    taus, etas = sorted(taus), sorted(etas, reverse=True)
    has = lambda prefix, split: (bool(glob.glob(os.path.join(sweep_dir, f"hybrid_errors_*{Results.suffix(split)}.csv"))) if prefix
                                 else os.path.exists(os.path.join(out_dir, f"fits{Results.suffix(split)}.csv")))

    splits = [s for s in ("train", "test") if has("", s) and os.path.exists(os.path.join(out_dir, f"global_fits{Results.suffix(s)}.csv"))]
    sweep = {s: load_fits(out_dir, shown, s) for s in splits}
    rank = {s: load_global_fits(out_dir, s) for s in splits}
    hybrid = {s: load_fits(out_dir, shown, s, prefix="hybrid_") for s in splits}
    n_eval = {s: int(np.max(sweep[s]["n_tracers"])) for s in splits}
    if len(splits) == 2:
        sweep_panel_splits(sweep["train"], rank["train"], sweep["test"], rank["test"], qoi_names,
                           n_eval_train=n_eval["train"], n_eval_test=n_eval["test"], show_tols=tuple(taus),
                           twin_tol=min(taus), hybrid_train=hybrid["train"], hybrid_test=hybrid["test"],
                           savepath=os.path.join(fig_dir, "panel_errors.pdf"))
    else:
        print("  the error panel needs both the training and testing results; skipped", flush=True)

    if "test" in sweep:
        # Cluster counts, mean basis rank and, where hybrid was run, its FOM share, all versus eta.
        fig, _ = sweep_partition_panel(sweep["test"], sweep_dir, show_tols=tuple(taus), positivity=shown, eval_split="test",
                                       max_eta_exp=int(round(-np.log10(min(etas)))), fom=has("hybrid_", "test"))
        fig.savefig(os.path.join(fig_dir, "panel_partition.pdf"), dpi=200, bbox_inches="tight")

    violins = dict(etas=etas, positivity=shown, eval_split="test", figsize=(5 * len(taus) + 2, 5 * len(qoi_names)))
    if has("", "test"):
        violin_panel(sweep_dir, taus[::-1], qoi_names, savepath=os.path.join(fig_dir, "violin_atrom.pdf"), **violins)
    if has("hybrid_", "test"):
        violin_panel(sweep_dir, taus[::-1], qoi_names, prefix="hybrid_",
                     savepath=os.path.join(fig_dir, "violin_hybrid.pdf"), **violins)
    plt.close("all")
    return fig_dir


def sweep_panel_splits(df_train, sc_train, df_test, sc_test, qoi, n_eval_train, n_eval_test,
                       show_tols=(0.01, 0.1, 0.4), twin_tol=0.1, ylim=(5e-05, 0.4), figsize=(15, 10),
                       savepath=None, hybrid_train=None, hybrid_test=None, row_gap=0.0, legend_ncol=None,
                       legend_y=0.93, sharex=True):
    """Mean relative L2 error versus eta: global-basis ROM (left) and AT-ROM (right), training row above testing row.

    ``n_eval_train`` and ``n_eval_test`` are the trajectory counts behind the failure percentages. ``twin_tol`` picks the tau_split whose mean basis rank labels the top axis. ``row_gap`` adds space (figure fraction) between the rows; ``sharex`` links each column's eta axis.
    """
    use_paper_style()
    qoi = list(qoi)
    tols = [t for t in sorted(np.unique(df_test["tau_split"])) if t in set(show_tols)]
    if not tols:
        raise ValueError(f"none of show_tols={show_tols} present in the test "
                         f"sweep (have {sorted(np.unique(df_test['tau_split']))})")
    missing = [t for t in tols if t not in set(df_train["tau_split"])]
    if missing:
        raise ValueError(f"train sweep is missing tau_split {missing}, which the test sweep has")
    dark = {t: pair[1] for t, pair in tol_colors(tols).items()}

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    any_fail = False
    for row, split, sc_d, d, n_ev, hyb in ((axes[0], "train", sc_train, df_train, n_eval_train, hybrid_train),
                                           (axes[1], "test", sc_test, df_test, n_eval_test, hybrid_test)):
        cap = SPLIT_CAPTION[split]
        any_fail |= _panel_global(row[0], sc_d, qoi, n_ev, split, cap)
        _panel_atrom(row[1], d, tols, dark, qoi, n_ev, twin_tol, ylim, hyb, cap)

    # One y range per column; the AT-ROM column gets headroom for its top-left caption.
    for col, pad in ((0, 1.0), (1, 3.0)):
        a, b = axes[0][col], axes[1][col]
        lo = min(a.get_ylim()[0], b.get_ylim()[0])
        hi = max(a.get_ylim()[1], b.get_ylim()[1]) * pad
        a.set_ylim(lo, hi)
        b.set_ylim(lo, hi)

    if sharex:
        # Linked after drawing: each panel inverts its own eta axis, and two inversions of a shared axis cancel.
        for top, bottom in zip(axes[0], axes[1]):
            bottom.sharex(top)
            top.tick_params(axis="x", labelbottom=False)
            top.set_xlabel("")

    for ax in axes.ravel():
        ax.tick_params(axis="x", labelsize=20)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.86,
                        wspace=0.20, hspace=0.45)
    if row_gap:
        _widen_first_gap(fig, axes, row_gap)

    handles = [Line2D([], [], color=GLOBAL_COLOR, marker="o", ls="-", lw=2, label="Global basis ROM")]
    handles += [Line2D([], [], color=dark[t], marker="o", ls="-", lw=2, label=_tau_label(t)) for t in tols]
    if any(h is not None and len(h) for h in (hybrid_train, hybrid_test)):
        handles.append(Line2D([], [], marker=HYBRID_MARKER, ls="none", label="Hybrid",
                              color=dark.get(twin_tol, TOL_PAIRS[0][1]), ms=20))
    if any_fail:
        handles.append(Line2D([], [], color=FAIL_COLOR, ls="none", marker=r"$\%$", ms=12, label="Failures"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, legend_y),
               ncol=legend_ncol or -(-len(handles) // 2), frameon=False)
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"saved figure -> {savepath}")
    return fig, axes


def sweep_partition_panel(df, sweep_dir, show_tols=(0.01, 0.1, 0.4), positivity='on', eval_split='test',
                          max_eta_exp=8, bar_headroom=1.9, figsize=(22, 6), fom=True):
    """One row versus eta: AT-ROM cluster counts, AT-ROM mean basis rank and the hybrid's share of FOM intervals.

    The first two describe the fit, so ``df`` may be either set's table. The FOM share is read from the ``hybrid_errors`` files in ``sweep_dir``; ``fom=False`` leaves it out.
    """
    use_paper_style()
    tols = [t for t in sorted(np.unique(df["tau_split"])) if t in set(show_tols)]
    if not tols:
        raise ValueError(f"none of show_tols={show_tols} present in the sweep "
                         f"(have {sorted(np.unique(df['tau_split']))})")
    pal = tol_colors(tols)
    dark = {t: pal[t][1] for t in tols}
    light = {t: pal[t][0] for t in tols}
    etas = sorted(np.unique(df["eta"]), reverse=True)

    fig, axes = plt.subplots(1, 3 if fom else 2, figsize=figsize)
    ax_bar, ax_rank = axes[:2]
    _panel_clusters(ax_bar, df, tols, dark, light, max_eta_exp, bar_headroom)
    _panel_rank(ax_rank, df, tols, dark, etas)
    if fom:
        _panel_fom_fraction(axes[2], sweep_dir, tols, dark, etas, positivity, eval_split)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=20)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.86, wspace=0.28)

    handles = [Line2D([], [], color=dark[t], marker="o", ms=6, lw=2, label=_tau_label(t)) for t in tols]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.88),
               ncol=len(handles), frameon=False)
    ax_bar.legend(handles=[Patch(facecolor="none", edgecolor="0.35", lw=1.8, label="unsplittable")],
                  loc="upper right", frameon=False)
    return fig, axes


def violin_panel(sweep_dir, taus, qoi, etas=None, positivity="on",
                 eval_split="test", agg="l2-in-time", floor=1e-16, ylim=None,
                 figsize=(15, 20), fs=28, eta_fs=24, stride=2,
                 err_label=r"$L^2$ error", savepath=None, rank_df=None,
                 rank_label="Mean basis size", rank_labelpad=12, prefix="", labels=None):
    """Violins of the log10 per-trajectory error versus eta: one row per QoI, one column per tau_split.

    Read from the sweep's ``<prefix>errors_*.csv`` files (``prefix="hybrid_"`` for hybrid). White dot: median; thick bar: 25th to 75th percentile; thin line: up to 1.5 times that range. Red text: percentage of trajectories that failed.
    ``rank_df`` (a table with ``mean_rank``) adds a top axis with the mean basis rank. ``labels`` renames QoIs on the row labels. Every ``stride``-th eta is labelled.
    """
    taus = list(taus)
    qoi = list(qoi)
    etas = ([float(e) for e in etas] if etas
            else etas_on_disk(sweep_dir, taus, eval_split))
    if not etas:
        raise FileNotFoundError(
            f"no {eval_split} errors file shared by tau={taus} under {sweep_dir}")

    use_paper_style()
    fig, axes = plt.subplots(len(qoi), len(taus), figsize=figsize,
                             sharex=True, sharey="row", squeeze=False)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(etas)))
    any_failed = False

    for j, tau in enumerate(taus):
        frames = {}
        for eta in etas:
            path = errors_path(sweep_dir, tau, eta, eval_split, prefix)
            if os.path.exists(path):
                frames[eta] = Results.where(Results.read(path), positivity=positivity)
        for i, q in enumerate(qoi):
            ax = axes[i][j]
            data, pos, failed = [], [], []
            for kk, eta in enumerate(etas):
                df = frames.get(eta)
                if df is None:
                    continue
                col = f"{agg}|{q}"
                if col not in Results.columns(df):
                    raise KeyError(
                        f"{col} not in {errors_path(sweep_dir, tau, eta, eval_split, prefix)}; "
                        f"QoIs there: {[c.split('|')[1] for c in Results.columns(df) if '|' in c]}")
                # A failed trajectory has NaN errors: left out.
                e = np.asarray(df[col], dtype=float)
                e = e[np.isfinite(e) & (e > 0)]
                if e.size == 0:
                    continue
                data.append(np.log10(np.maximum(e, floor)))
                pos.append(kk + 1)
                failed.append((int(np.sum(df["completed"] == 0)), df.size))
            if not data:
                continue
            parts = ax.violinplot(data, positions=pos, widths=0.85,
                                  showextrema=False, showmedians=False)
            for body, kk in zip(parts["bodies"], [x - 1 for x in pos]):
                body.set_facecolor(colors[kk])
                body.set_alpha(0.8)
                body.set_edgecolor("none")
            for d, x in zip(data, pos):
                q1, med, q3 = np.percentile(d, [25, 50, 75])
                reach = 1.5 * (q3 - q1)
                ax.vlines(x, d[d >= q1 - reach].min(), d[d <= q3 + reach].max(), color="0.2", lw=1.2, zorder=3)
                ax.vlines(x, q1, q3, color="0.2", lw=6, zorder=4)
                ax.scatter(x, med, s=14, color="white", zorder=5)
            for d, x, (n, n_eval) in zip(data, pos, failed):
                if n:
                    any_failed = True
                    ax.annotate(_fail_label(n, n_eval), xy=(x, d.max()), xytext=(0, 4),
                                textcoords="offset points", ha="center", va="bottom",
                                fontsize=eta_fs, color=FAIL_COLOR, annotation_clip=False)
            if i == 0:
                ax.text(0.97, 0.95, rf"$\tau={tau:g}$", transform=ax.transAxes,
                        ha="right", va="top", fontsize=fs, zorder=6)
            if j == 0:
                ax.set_ylabel(f"{(labels or {}).get(q, q)}\n" + rf"$\log_{{10}}$ {err_label or agg}")
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", alpha=0.25)

    # Labels on the kept etas; every violin keeps a minor tick.
    keep = list(range(0, len(etas), stride))
    if keep[-1] != len(etas) - 1:
        keep.append(len(etas) - 1)
    axes[0][0].set_xlim(0.4, len(etas) + 0.6)
    for ax in axes[-1]:
        ax.set_xticks([k + 1 for k in keep])
        ax.set_xticklabels([rf"$10^{{{int(round(np.log10(etas[k])))}}}$" for k in keep])
        ax.set_xticks(np.arange(1, len(etas) + 1), minor=True)
        ax.set_xlabel(r"$\eta$", fontsize=fs)
        ax.tick_params(axis="x", which="major", labelsize=eta_fs)
        ax.tick_params(axis="x", which="minor", length=3)
    twins = []
    if rank_df is not None:
        for tau, host in zip(taus, axes[0]):
            eta_r, rank_r = rank_series(rank_df, tau)

            def _label(eta):
                hit = np.isclose(eta_r, eta, rtol=1e-6, atol=0.0)
                r = rank_r[hit][0] if hit.any() else np.nan
                return f"{r:.1f}" if np.isfinite(r) else ""

            tw = host.twiny()
            tw.set_xlim(host.get_xlim())
            tw.set_xticks([k + 1 for k in keep])
            tw.set_xticklabels([_label(etas[k]) for k in keep])
            tw.set_xticks(np.arange(1, len(etas) + 1), minor=True)
            tw.set_xlabel(rank_label, fontsize=eta_fs, labelpad=rank_labelpad)
            tw.tick_params(axis="x", which="major", labelsize=eta_fs)
            tw.tick_params(axis="x", which="minor", length=3)
            tw.grid(False)
            twins.append(tw)
    for ax in axes.ravel():
        ax.tick_params(axis="y", labelsize=fs)
        ax.yaxis.label.set_fontsize(fs)

    fig.tight_layout()
    handles = [Line2D([], [], ls="none", marker="o", ms=6, mfc="white", mec="0.2"),
               Line2D([], [], color="0.2", lw=6),
               Line2D([], [], color="0.2", lw=1.2)]
    names = ["median", "IQR", "1.5 IQR"]
    if any_failed:
        handles.append(Line2D([], [], ls="none", marker=r"$\%$", ms=12, color=FAIL_COLOR))
        names.append("Failures")
    # Legend just above the top row (top axes included).
    renderer = fig.canvas.get_renderer()
    top = max(ax.get_tightbbox(renderer).y1 for ax in [*axes[0], *twins])
    top = fig.transFigure.inverted().transform((0.0, top))[1]
    fig.legend(handles, names, ncol=len(names), frameon=False,
               loc="lower right", bbox_to_anchor=(1.0, top), fontsize=eta_fs)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {savepath}")
    return fig, axes


# ---------------------------------------------------------------------------
# Used by the notebooks: tables, paper styling, per-trajectory figures
# ---------------------------------------------------------------------------

def load_tables(out_dir, positivity):
    """The AT-ROM, global-basis ROM and hybrid tables of both sets, with mean errors added; a set without hybrid maps to None."""
    splits = ("train", "test")
    sweep = {s: load_fits(out_dir, positivity, s) for s in splits}
    rank = {s: load_global_fits(out_dir, s) for s in splits}
    hybrid = {s: load_fits(out_dir, positivity, s, prefix="hybrid_") for s in splits}
    return sweep, rank, hybrid


def load_fits(out_dir, positivity="on", eval_split="test", prefix=""):
    """The rows of ``fits<split>.csv`` at one positivity setting, with the mean error over completed trajectories added.

    With ``prefix="hybrid_"``, the mean errors are hybrid's, for the pairs hybrid was run on; None if it was not run.
    """
    path = os.path.join(out_dir, f"fits{Results.suffix(eval_split)}.csv")
    fits = Results.where(Results.read(path), positivity=positivity)
    if not fits.size:
        raise ValueError(f"{path} has no rows with positivity {positivity}")
    sweep_dir = os.path.join(out_dir, "sweep")
    if prefix:
        fits = fits[[os.path.exists(errors_path(sweep_dir, tau, eta, eval_split, prefix))
                     for tau, eta in zip(fits["tau_split"], fits["eta"])]]
        if not fits.size:
            return None
    rows = []
    for tau, eta in zip(fits["tau_split"], fits["eta"]):
        errors = Results.read(errors_path(sweep_dir, tau, eta, eval_split, prefix))
        rows.append(_mean_errors(Results.where(errors, positivity=positivity)))
    return _append_columns(fits, rows)


def load_global_fits(out_dir, eval_split='test'):
    """The rows of ``global_fits<split>.csv`` with the mean errors added, without the full-basis row."""
    split_tag = Results.suffix(eval_split)
    fits = Results.read(os.path.join(out_dir, f"global_fits{split_tag}.csv"))
    errors = Results.read(os.path.join(out_dir, f"global_errors{split_tag}.csv"))
    rows = [_mean_errors(Results.where(errors, rank=r)) for r in fits["rank"]]
    return drop_full_basis(_append_columns(fits, rows))


def use_paper_style():
    plt.rcParams.update(PAPER_STYLE)


def set_font_sizes(fig, label=24, tick=22, xtick=26, legend=24):
    """Set the font sizes of every label, tick label, legend and text in ``fig``.

    ``xtick`` is for the bottom tick labels; top-axis labels are 22, eta labels 32 and in-panel text 20.
    """
    for ax in fig.axes:
        top = ax.xaxis.get_label_position() == "top"
        ax.xaxis.label.set_fontsize(22 if top else 32 if ax.get_xlabel() == r"$\eta$" else label)
        ax.yaxis.label.set_fontsize(label)
        ax.tick_params(axis="y", labelsize=tick)
        ax.tick_params(axis="x", labelsize=tick if top else xtick)
        for t in ax.texts:
            t.set_fontsize(20)
    for leg in [*fig.legends, *(ax.get_legend() for ax in fig.axes if ax.get_legend())]:
        for t in leg.get_texts():
            t.set_fontsize(legend)


def place_captions(axes, moves):
    """Move failure captions, each found by panel, text, colour and eta.

    ``moves`` maps ``(row, col)`` to ``{(text, color, eta): (dx, dy, ha, va[, y])}``: offsets in points, and optionally a new height ``y`` in data units. Raises if a caption is not found.
    """
    for (i, j), table in moves.items():
        texts = [t for t in axes[i][j].texts if hasattr(t, "xy")]
        for (text, color, eta), (dx, dy, ha, va, *y) in table.items():
            # eta within 0.05 decade: global-basis ROM captions sit at non-round etas.
            hit = [t for t in texts if t.get_text() == text and to_hex(t.get_color()) == color
                   and abs(np.log10(t.xy[0]) - np.log10(eta)) < 0.05]
            if len(hit) != 1:
                raise KeyError(f"panel {(i, j)}: {len(hit)} captions match {(text, color, eta)}")
            if y:
                hit[0].xy = (hit[0].xy[0], y[0])
            hit[0].set_position((dx, dy))
            hit[0].set_ha(ha)
            hit[0].set_va(va)


def stack_captions(ax, order):
    """Stack each eta's failure captions above the highest point at that eta, bottom to top in the colour order ``order``."""
    groups = {}
    for t in ax.texts:
        if hasattr(t, "xy"):
            groups.setdefault(round(float(np.log10(t.xy[0])), 3), []).append(t)
    for key, group in groups.items():
        tops = []
        for line in ax.lines:
            at = np.isclose(np.log10(np.asarray(line.get_xdata(), dtype=float)), key, atol=1e-3)
            if at.any():
                tops.append(np.nanmax(np.asarray(line.get_ydata(), dtype=float)[at]))
        top = max(tops)
        group.sort(key=lambda t: order.index(to_hex(t.get_color())) if to_hex(t.get_color()) in order else len(order))
        for k, t in enumerate(group):
            t.xy = (10.0 ** key, top)
            t.set_position((0, 4 + k * 24))
            t.set_ha("center")
            t.set_va("bottom")


def move_corner_label(ax):
    """Move the set caption ("Training", "Testing") to the top right of ``ax``."""
    for t in ax.texts:
        if not hasattr(t, "xy"):
            t.set_x(0.98)
            t.set_ha("right")


def thin_tick_labels(fig, column, keep):
    """Blank every x tick label except the positions in ``keep``, on the axes of ``column`` and any axes drawn over them.

    Axes that share an eta axis share its formatter, so each formatter is relabelled once.
    """
    targets, formatters = [], set()
    for ax in column:
        if id(ax.xaxis.get_major_formatter()) not in formatters:
            formatters.add(id(ax.xaxis.get_major_formatter()))
            targets.append(ax)
    targets += [a for ax in column for a in fig.axes
                if a not in column and np.allclose(a.get_position().bounds, ax.get_position().bounds)]
    for a in targets:
        ticks = list(a.get_xticks())
        fmt = a.xaxis.get_major_formatter()
        # Read from the formatter: tick labels are empty until the figure is drawn.
        fmt.set_locs(ticks)
        labels = [fmt(v, i) for i, v in enumerate(ticks)]
        a.set_xticks(ticks)
        a.set_xticklabels([s if i in keep else "" for i, s in enumerate(labels)])


def set_partition_titles(fig, axes):
    """Title the panels of :func:`sweep_partition_panel` and move its legend above the titles."""
    for ax, title in zip(axes, ("AT-ROM clusters", "AT-ROM basis rank", "Hybrid FOM use")):
        ax.set_title(title, fontsize=24)
    fig.legends[0].set_bbox_to_anchor((0.5, 0.95), transform=fig.transFigure)


def abundance_panel(d, d_global, species, qoi_indices, figsize=(20, 13), ensemble=None):
    """Every species along one trajectory: FOM (circles), AT-ROM (solid) and global-basis ROM (dashed), over the AT-ROM's cluster route (bands).

    Four panels: the QoIs, then the other species in three groups by the AT-ROM's error, smallest to largest. Values are shown at interval boundaries.
    ``d`` comes from :func:`tracer_data`, ``d_global`` from the same call on a global-basis ROM (or None). Returns ``(fig, axes, info)``; ``info["route"]`` summarises the cluster route (:func:`route_stats`, with basis ranks when ``ensemble`` is given).
    """
    use_paper_style()
    plt.rcParams.update({"font.size": 26, "axes.titlesize": 26,
                         "axes.labelsize": 26, "xtick.labelsize": 26,
                         "ytick.labelsize": 26, "legend.fontsize": 22})

    peak = np.asarray(d["peak"])
    qoi = list(qoi_indices)
    non_qoi = [i for i in range(len(species)) if i not in set(qoi)]

    rom_key = "y_rom_pos" if "y_rom_pos" in d else "y_rom"
    rom_tkey = "tyr_rom_pos" if "y_rom_pos" in d else "tyr_rom"
    dex = _dex_error(d, peak, 1e-12, rom_key, rom_tkey)

    ranked = sorted(non_qoi, key=lambda j: -dex[j])
    n = len(ranked)
    hi, mid, lo = ranked[:n // 3], ranked[n // 3:2 * n // 3], ranked[2 * n // 3:]

    # (values, times, line style, line width, label, marker); the FOM has markers only.
    variants = [("y_fom", "tyr", "none", 0.0, "FOM", "o"),
                (rom_key, rom_tkey, "-", 1.8, "AT-ROM", None)]

    # Bands follow the AT-ROM's own cluster route.
    rom_seg = d.get("rom_pos", d["rom"])
    cc = {c: plt.cm.tab20(i % 20) for i, c in enumerate(np.unique(rom_seg["cp"]))}
    dt_step = d.get("dt_hydro_yr")
    colors = {j: plt.cm.tab20(j % 20) for j in range(len(species))}

    def xy(t, y):
        t = np.asarray(t)
        if dt_step:
            m = boundary_mask(t, dt_step)
            t, y = t[m], np.asarray(y)[m]
        return t / 1e3, y                          # yr -> kyr

    band_t = np.asarray(rom_seg["te"]) / 1e3
    panels = [(qoi, "QoIs"), (lo, "Non-QoIs, smallest error"),
              (mid, "Non-QoIs, intermediate error"),
              (hi, "Non-QoIs, largest error")]

    fig, axes2d = plt.subplots(2, 2, figsize=figsize, sharex=True)
    for ax, (group, title) in zip(axes2d.flat, panels):
        if not group:
            ax.set_visible(False)
            continue
        for j in group:
            for ykey, tkey, ls, lw, _, mk in variants:
                t_, y_ = xy(d[tkey], d[ykey][j])
                ax.plot(t_, y_, ls=ls, lw=lw, color=colors[j], alpha=0.85,
                        zorder=5, marker=mk, ms=6 if mk else 0, mfc="none",
                        mew=1.4, markevery=max(1, len(t_) // 30))
            if d_global is not None:
                gk = "y_rom_pos" if "y_rom_pos" in d_global else "y_rom"
                gt = "tyr_rom_pos" if "y_rom_pos" in d_global else "tyr_rom"
                t_, y_ = xy(d_global[gt], d_global[gk][j])
                ax.plot(t_, y_, ls="--", lw=1.6, color=colors[j], alpha=0.85, zorder=4)

        # Symlog per panel, so small abundances are not flattened.
        peaks = [peak[j] for j in group if peak[j] > 0]
        lt = max(min(peaks) * 1e-3, 1e-30) if peaks else 1e-30
        ax.set_yscale("symlog", linthresh=lt)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        loc = SymmetricalLogLocator(linthresh=lt, base=10)
        loc.numticks = 6
        ax.yaxis.set_major_locator(loc)
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda y, pos, c=3 * lt: "" if y != 0 and abs(y) < c else _logfmt(y, pos)))
        for k in range(len(rom_seg["seg"])):
            ax.axvspan(band_t[k], band_t[k + 1], color=cc[rom_seg["seg"][k]], alpha=0.12, lw=0, zorder=0)
        ax.set_title(title)

    for row in axes2d:
        if row[0].get_visible():
            row[0].set_ylabel("Fractional abundance")
    for ax in axes2d[-1]:
        if ax.get_visible():
            ax.set_xlabel("Time [kyr]")

    # Each panel has a species legend below it, 4 columns wide, so the row gap grows with the largest group.
    rows_needed = max(int(np.ceil(len(g) / 4)) for g, _ in panels if g)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.87, bottom=0.10,
                        wspace=0.20, hspace=0.38 + 0.22 * (rows_needed - 1))

    # Cluster key in the order the route visits the clusters.
    visited = list(dict.fromkeys(np.asarray(rom_seg["seg"]).tolist()))
    cluster_key = Patch(label="Clusters")
    entries = variants + ([("g", "g", "--", 1.6, "Global basis ROM", None)] if d_global is not None else [])
    fig.legend(handles=[Line2D([], [], color="0.25", ls=ls, lw=lw, label=lab,
                               marker=mk, ms=6, mfc="none", mew=1.4)
                        for _, _, ls, lw, lab, mk in entries] + [cluster_key],
               handler_map={cluster_key: _HandlerBands([cc[c] for c in visited if c in cc], alpha=0.45)},
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.005), frameon=True)

    # Species legends below each panel; the bottom row's clear its tick labels and x label.
    for r, row in enumerate(axes2d):
        for ax, (group, _t) in zip(row, panels[2 * r:2 * r + 2]):
            if not group or not ax.get_visible():
                continue
            bb = ax.get_position()
            off = (0.09 if r == len(axes2d) - 1 else 0.02) + 0.012 * (rows_needed - 1)
            fig.legend(handles=[Line2D([], [], color=colors[j], lw=3, label=species[j]) for j in group],
                       loc="upper center", bbox_to_anchor=(bb.x0 + bb.width / 2, bb.y0 - off),
                       bbox_transform=fig.transFigure, ncol=min(len(group), 4), frameon=False)

    return fig, axes2d, dict(dex_error=dex, groups=dict(hi=hi, mid=mid, lo=lo),
                             route=route_stats(d, ensemble=ensemble))


def tracer_data(ensemble, tracer, x0, t_fom, y_fom, dt_hydro, atol, rtol,
                method="BDF", positivity=(True, False), time_scale=1.0):
    """Solve one trajectory with the AT-ROM once per entry of ``positivity``, in the form :func:`abundance_panel` reads.

    With the non-negativity correction the keys end in ``_pos`` (``y_rom_pos``, ``tyr_rom_pos``, ``rom_pos``); without it they do not. ``time_scale`` divides every time; ``dt_hydro`` stays in solver units.
    """
    out = {"tyr": np.asarray(t_fom) / time_scale,
           "y_fom": np.asarray(y_fom),
           "peak": np.asarray(y_fom).max(axis=1),
           "dt_hydro_yr": dt_hydro / time_scale}

    for on in positivity:
        tag = "_pos" if on else ""
        t, y, path, info = ensemble.solve_tracer(
            tracer, x0, dt_hydro, atol, rtol, t_eval=t_fom, positivity=on, method=method)
        out[f"tyr_rom{tag}"] = t / time_scale
        out[f"y_rom{tag}"] = y
        out[f"ok{tag}"] = info is None
        if t.size == 0:
            raise RuntimeError(
                f"the positivity-{'on' if on else 'off'} solve returned no "
                f"trajectory for this tracer: {info!r}")
        # Each solve keeps its own cluster route, which ends where that solve stopped.
        out[f"rom{tag}"] = route_segments(steps_to_snapshots(path, t, dt_hydro), t / time_scale)

    # The keys without the correction must exist even when only the corrected solve ran.
    if "rom" not in out:
        for k in ("rom", "y_rom", "tyr_rom", "ok"):
            out[k] = out[f"{k}_pos"]
    return out


def trajectory_panel(t, series, route, qoi, figsize=(13, 7.5)):
    """One panel per QoI: several solutions of one trajectory, over bands showing its cluster route.

    ``series`` maps ``"reference"`` (FOM), ``"clustered"`` (AT-ROM) or ``"global"`` (global-basis ROM) to an ``(n_species, n_times)`` array, or to None for a failed solve. ``route`` has one cluster per interval (one shorter than ``t``).
    """
    t = np.asarray(t, dtype=float)
    qoi = list(qoi)
    cc = {c: plt.cm.tab20(i % 20) for i, c in enumerate(sorted(set(route)))}

    ncols = 2
    nrows = int(np.ceil(len(qoi) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax, q in zip(axes, qoi):
        for c, a, b in _runs(route):
            ax.axvspan(t[a], t[min(b + 1, t.size - 1)], color=cc[c], alpha=TRAJ_BAND_ALPHA, lw=0, zorder=0)
        for z, (key, y) in enumerate(series.items()):
            if y is None:
                continue
            ls, lw = TRAJ_STYLE.get(key, ("-", 1.8))
            ax.plot(t, np.asarray(y)[q], ls, lw=lw, color=TRAJ_COLORS[key], zorder=2 + z)
        ax.set_ylabel(rf"$x_{{{q}}}$")
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)
        ax.margins(y=0.12)
    for ax in axes[len(qoi):]:
        ax.set_visible(False)
    axes[0].set_xlim(t[0], t[-1])              # shared, so this sets them all
    for ax in axes[max(0, len(qoi) - ncols):len(qoi)]:
        ax.set_xlabel("$t$")

    band_key = Patch(label="Clusters")
    handles = [Line2D([], [], color=TRAJ_COLORS[k], ls=TRAJ_STYLE[k][0], lw=TRAJ_STYLE[k][1], label=lab)
               for k, lab in (("reference", "FOM"), ("clustered", "AT-ROM"), ("global", "Global basis ROM"))
               if series.get(k) is not None] + [band_key]
    fig.legend(handles=handles, handler_map={band_key: _HandlerBands([cc[c] for c in sorted(cc)], alpha=0.45)},
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))     # room at the top for the legend
    return fig, axes


def qoi_panels_pair(hybrid_args, nonnegativity_args):
    """:func:`hybrid_qoi_panel` (left) and :func:`nonnegativity_qoi_panel` (right) side by side, with one QoI legend on top.

    Each argument is a dict of that function's arguments. Returns ``(fig, axes)``: row 0 the data panels, row 1 the route strips.
    """
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(26, 8.5), squeeze=False, sharey="row",
                             gridspec_kw=dict(height_ratios=[1.0, 0.09], hspace=0.45, wspace=0.05))
    hybrid_qoi_panel(**hybrid_args, axes=(axes[0][0], axes[1][0]), fontsize=24)
    nonnegativity_qoi_panel(**nonnegativity_args, axes=(axes[0][1], axes[1][1]), qoi_legend=False, fontsize=24)
    axes[0][1].set_ylabel("")
    axes[0][1].tick_params(axis="y", labelleft=False)
    # Right-align a tick label at the right end of the time axis, so it stays inside its panel.
    for ax in axes[0]:
        lo, hi = ax.get_xlim()
        ticks = [t for t in ax.get_xticks() if lo <= t <= hi]
        ax.set_xticks(ticks)
        ax.set_xlim(lo, hi)     # set_xticks can widen the view
        if np.isclose(ticks[-1], hi):
            ax.get_xticklabels()[-1].set_ha("right")

    # Move the left panel's QoI legend to span the figure.
    leg = axes[0][0].get_legend()
    handles = list(getattr(leg, "legend_handles", None) or leg.legendHandles)
    names = [t.get_text() for t in leg.get_texts()]
    leg.remove()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.95),
               ncol=len(handles), frameon=False, fontsize=24)
    return fig, axes


def hybrid_qoi_panel(result, rom, t_ref, y_ref, *, qoi_indices, species, title=None, labels=None,
                     n_markers=40, t_min=None, ylim=None, time_label="Time [s]", time_scale=1.0,
                     time_unit="s", figsize=(13, 6.5), fontsize=16, xscale="log", axes=None,
                     qoi_legend=True, normalize=True, savepath=None):
    """The QoIs of one trajectory: hybrid (lines) against the FOM (circles), with the AT-ROM's cluster route in a strip below.

    The bands are the hybrid's cluster route, white boxes its FOM windows and red triangles its triggers. The strip ends where the AT-ROM fails.
    Shared options are described in :func:`_qoi_start` and :func:`_qoi_finish`. Returns ``(fig, (ax, strip_ax))``.
    """
    dt = result.dt_hydro / time_scale
    fig, ax, strip_ax, t_ref, y_ref, n_steps, t_end, t_min, marks = _qoi_start(
        t_ref, y_ref, dt, result.n_steps, time_scale, t_min, xscale, n_markers, axes, figsize)
    qi, names = _qoi_names(qoi_indices, species, labels)
    handles = _qoi_curves(ax, [(result.t, result.y, "-", dict(lw=1.8, zorder=4))],
                          t_ref, y_ref, marks, qi, names, normalize, time_scale)

    cmap = _band_colors(result.path, rom.path)
    for a in (ax, strip_ax):
        a.set_xscale(xscale)
        a.set_xlim(t_min, t_end)
    _route_bands(ax, result.path, cmap, dt, n_steps)
    for k0, k1 in result.fom_windows():
        ax.axvspan(k0 * dt, (k1 + 1) * dt, facecolor="w", edgecolor="0.35", lw=1.5, zorder=1)
    for k in np.asarray(result.trigger_step, dtype=int):
        ax.plot(k * dt, 1.0, "v", color=FAIL_COLOR, ms=9, transform=ax.get_xaxis_transform(),
                clip_on=False, zorder=6)

    t_fail = _route_until_failure(strip_ax, rom, cmap, dt, n_steps, time_scale)
    if t_fail < t_end - 0.5 * dt:
        _mark_failure(strip_ax, [strip_ax], t_fail, t_min, t_end, f"AT-ROM fails at t = {t_fail:.3g} {time_unit}", fontsize)

    route_key = Patch(label="Hybrid cluster route")
    keys = [Line2D([], [], color="0.35", lw=1.8, label="Hybrid"),
            route_key,
            Line2D([], [], color=FAIL_COLOR, marker="v", ms=9, ls="none", label="Hybrid trigger"),
            Patch(facecolor="w", edgecolor="0.35", lw=1.5, label="FOM window")]
    return _qoi_finish(fig, ax, strip_ax, handles, keys, route_key, _route_swatch(result.path, cmap, BAND_ALPHA),
                       "AT-ROM route", y_ref[qi], normalize, ylim, fontsize, time_unit, time_label, title,
                       qoi_legend, savepath)


def nonnegativity_qoi_panel(rom_on, rom_off, t_ref, y_ref, *, qoi_indices, species, dt_hydro,
                            title=None, labels=None, n_markers=40, t_min=None, ylim=None,
                            time_label="Time [s]", time_scale=1.0, time_unit="s", figsize=(13, 6.5),
                            fontsize=16, xscale="log", axes=None, qoi_legend=True, normalize=True,
                            savepath=None):
    """The QoIs of one trajectory: AT-ROM with (solid) and without (dotted) the non-negativity correction, against the FOM (circles).

    The bands are the cluster route with the correction; the strip below is the route without it, ending where that solve fails.
    Shared options are described in :func:`_qoi_start` and :func:`_qoi_finish`. Returns ``(fig, (ax, strip_ax))``.
    """
    dt = float(dt_hydro) / time_scale
    fig, ax, strip_ax, t_ref, y_ref, n_steps, t_end, t_min, marks = _qoi_start(
        t_ref, y_ref, dt, None, time_scale, t_min, xscale, n_markers, axes, figsize)
    qi, names = _qoi_names(qoi_indices, species, labels)
    # The two solves agree until the uncorrected one fails, so its dotted line is thicker and on top.
    handles = _qoi_curves(ax, [(rom_on.t, rom_on.y, "-", dict(lw=1.8, alpha=0.6, zorder=4)),
                               (rom_off.t, rom_off.y, ":", dict(lw=3.0, alpha=1.0, zorder=6))],
                          t_ref, y_ref, marks, qi, names, normalize, time_scale)

    cmap = _band_colors(rom_on.path, rom_off.path)
    for a in (ax, strip_ax):
        a.set_xscale(xscale)
        a.set_xlim(t_min, t_end)
    _route_until_failure(ax, rom_on, cmap, dt, n_steps, time_scale)
    t_fail = _route_until_failure(strip_ax, rom_off, cmap, dt, n_steps, time_scale)
    if t_fail < t_end - 0.5 * dt:
        _mark_failure(strip_ax, [ax, strip_ax], t_fail, t_min, t_end, f"fails at t = {t_fail:.3g} {time_unit}",
                      fontsize, zorder=3)

    route_key = Patch(label="AT-ROM cluster route")
    keys = [Line2D([], [], color="0.35", lw=1.8, label="AT-ROM, non-negativity on"),
            Line2D([], [], color="0.35", lw=3.0, ls=":", label="AT-ROM, non-negativity off"),
            route_key]
    return _qoi_finish(fig, ax, strip_ax, handles, keys, route_key, _route_swatch(rom_on.path, cmap, 0.6),
                       "AT-ROM route, non-negativity off", y_ref[qi], normalize, ylim, fontsize, time_unit,
                       time_label, title, qoi_legend, savepath)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def boundary_mask(t_yr, dt_yr):
    """Which entries of ``t_yr`` lie on an interval boundary."""
    frac = (np.asarray(t_yr, dtype=np.float64) / dt_yr) % 1.0
    return (frac < 1e-6) | (frac > 1 - 1e-6)


def steps_to_snapshots(step_path, t_axis, dt_hydro):
    """Expand one cluster per interval to one per sample on ``t_axis``."""
    step_path = np.asarray(step_path)
    step = np.clip((np.asarray(t_axis) / dt_hydro).astype(np.int64), 0, step_path.size - 1)
    return step_path[step]


def route_segments(cluster_path, t_yr):
    """Split a cluster route into runs of one cluster: the route ``cp``, each run's cluster ``seg`` and the run boundaries ``te``."""
    cp = np.asarray(cluster_path)
    t_yr = np.asarray(t_yr)
    if cp.size == 0:                      # a solve that failed on its first interval
        return dict(cp=cp, seg=cp[:0], te=t_yr[:0])
    edges = np.r_[0, np.flatnonzero(np.diff(cp)) + 1, cp.size]
    return dict(cp=cp, seg=cp[edges[:-1]], te=t_yr[np.clip(edges, 0, cp.size - 1)])


def route_stats(d, ensemble=None, dims=None):
    """The clusters one trajectory's AT-ROM cluster route visits: counts, time in each and, given ``ensemble`` or ``dims``, basis ranks.

    ``d`` comes from :func:`tracer_data`. ``mean_dim`` weights each cluster by its count on the route, ``mean_dim_time`` by time spent in it.
    """
    seg = d.get("rom_pos", d["rom"])
    cp = np.asarray(seg["cp"])
    clusters, counts = np.unique(cp, return_counts=True)
    clusters = [int(c) for c in clusters]
    te = np.asarray(seg["te"], dtype=float)
    dur = {c: 0.0 for c in clusters}
    for c, t0, t1 in zip(np.asarray(seg["seg"]), te[:-1], te[1:]):
        dur[int(c)] += float(t1 - t0)

    out = dict(
        which="rom",
        clusters=clusters,
        n_clusters=len(clusters),
        counts={c: int(n) for c, n in zip(clusters, counts)},
        fractions={c: float(n) / cp.size for c, n in zip(clusters, counts)},
        time={c: dur[c] for c in clusters},
        n_switches=int(len(np.asarray(seg["seg"])) - 1),
    )

    if dims is None and ensemble is not None:
        dims = {int(c): int(ensemble.roms[int(c)].basis.rank) for c in clusters}
    if dims is None:
        return out

    dims = {int(c): int(dims[int(c)]) for c in clusters}
    w = np.array([out["counts"][c] for c in clusters], dtype=float)
    wt = np.array([out["time"][c] for c in clusters], dtype=float)
    v = np.array([dims[c] for c in clusters], dtype=float)
    out.update(
        dims=dims,
        mean_dim=float(np.average(v, weights=w)),
        mean_dim_time=float(np.average(v, weights=wt)) if wt.sum() > 0 else float("nan"),
        mean_dim_unweighted=float(v.mean()),
        min_dim=int(v.min()),
        max_dim=int(v.max()),
        total_dim=int(v.sum()),
    )
    return out


def errors_path(sweep_dir, tau, eta, eval_split="test", prefix=""):
    """The ``<prefix>errors_<tau_eta><split>.csv`` file of one (tau_split, eta)."""
    name = f"{prefix}errors_{Results.label(tau, eta)}{Results.suffix(eval_split)}.csv"
    return os.path.join(sweep_dir, name)


def _mean_errors(errors):
    """``{column: mean}`` of each error column over the completed trajectories."""
    completed = errors[errors["completed"] == 1]
    columns = [c for c in Results.columns(errors) if "|" in c]
    if not completed.size:
        return {c: np.nan for c in columns}
    return {c: float(np.mean(completed[c])) for c in columns}


def _append_columns(tab, rows):
    """``tab`` with the columns of ``rows`` (one dict per row) added."""
    names = list(rows[0])
    values = [np.array([row[n] for row in rows]) for n in names]
    return recfunctions.append_fields(tab, names, values, usemask=False)


def drop_full_basis(rank_df):
    """Drop global-basis ROM rows with zero residual energy (the full basis, which is the FOM and has no place on a log-eta axis)."""
    keep = rank_df["residual_energy"] > 0
    if keep.all():
        return rank_df
    dropped = sorted({int(r) for r in rank_df["rank"][~keep]})
    print(f"  dropping rank {dropped} from the rank sweep: residual energy is "
          f"0 (full basis), which has no place on a log-eta axis")
    return rank_df[keep]


def tol_colors(tols):
    """``{tau_split: (light, dark)}``: ``LIGHT``/``DARK`` when they cover every tolerance, otherwise ``TOL_PAIRS`` in ascending order."""
    tols = list(tols)
    if all(t in DARK for t in tols):
        return {t: (LIGHT[t], DARK[t]) for t in tols}
    order = sorted(tols)
    return {t: TOL_PAIRS[order.index(t) % len(TOL_PAIRS)] for t in tols}


def _tau_label(t):
    return r"$\tau_\mathrm{split} = $" + rf"{t:g}"


def _pow10(v):
    return rf"$10^{{{np.log10(v):.0f}}}$"


def _qoi_mean(rows, qoi, agg):
    """The mean error of each row over the QoIs, each weighted equally."""
    return np.mean([rows[f"{agg}|{q}"] for q in qoi], axis=0)


def mean_series(df, tau, qoi, agg="l2-in-time"):
    """(eta, mean error) for one tau_split."""
    rows = Results.where(df, tau_split=tau)
    rows = rows[np.argsort(rows["eta"])]
    return rows["eta"], _qoi_mean(rows, qoi, agg)


def rank_series(df, tau):
    """(eta, mean basis rank) for one tau_split."""
    rows = Results.where(df, tau_split=tau)
    rows = rows[np.argsort(rows["eta"])]
    if not np.isfinite(rows["mean_rank"]).any():
        raise ValueError(f"no mean_rank values for tau_split={tau}")
    return rows["eta"], rows["mean_rank"]


def global_rank_series(sc_df, qoi, agg="l2-in-time"):
    """(basis rank, mean error) for the global-basis ROM."""
    rows = sc_df[np.argsort(sc_df["rank"])]
    return rows["rank"], _qoi_mean(rows, qoi, agg)


def _fail_label(failed, n_eval):
    """Failure percentage, never rounded down to 0: one decimal below 1%, and a count below 0.05%."""
    pct = 100.0 * failed / max(n_eval, 1)
    if pct >= 1.0:
        return f"{pct:.0f}%"
    if pct >= 0.05:
        return f"{pct:.1f}%"
    return f"{int(failed)}/{int(n_eval)}"


def _annotate_failures(ax, x, y, failed, n_eval, color, dy_pad, others):
    """Write the failure percentage next to each point of a curve with failed trajectories.

    The caption goes on the side with more room from the other curves (``others``, their values at the same etas), otherwise above a peak and below a valley, and never off the axes. ``dy_pad`` adds points of offset per point. Call after the y limits are set.
    """
    fontsize = 18
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = np.asarray(failed, dtype=float)
    n = x.size
    pad_pts = np.broadcast_to(np.asarray(dy_pad, dtype=float), (n,))
    ylo, yhi = ax.get_ylim()
    if ax.get_yscale() == "log":
        to_frac = lambda v: (np.log10(v) - np.log10(ylo)) / max(np.log10(yhi) - np.log10(ylo), 1e-12)
    else:
        to_frac = lambda v: (v - ylo) / max(yhi - ylo, 1e-12)
    for i in range(n):
        if not np.isfinite(f[i]) or f[i] <= 0:
            continue
        prev = y[i - 1] if i > 0 else None
        nxt = y[i + 1] if i < n - 1 else None
        if prev is not None and nxt is not None:
            if y[i] >= prev and y[i] >= nxt:
                above = True
            elif y[i] <= prev and y[i] <= nxt:
                above = False
            else:
                above = y[i] < prev
        else:
            above = True
        frac = to_frac(y[i]) if np.isfinite(y[i]) and y[i] > 0 else 0.5
        # Height of the caption with its offset, as a fraction of the axes.
        h_pts = ax.get_window_extent().height / ax.figure.dpi * 72.0
        pad = (6 + pad_pts[i] + fontsize) / max(h_pts, 1.0)
        gap_up, gap_dn = 1.0 - frac, frac
        for o in np.asarray(others, dtype=float):
            v = o[i]
            if not np.isfinite(v) or v <= 0:
                continue
            d = to_frac(v) - frac
            if abs(d) < 0.3 * pad:          # on top of this point: blocks neither side
                continue
            if d > 0:
                gap_up = min(gap_up, d)
            else:
                gap_dn = min(gap_dn, -d)
        if max(gap_up, gap_dn) > pad and abs(gap_up - gap_dn) > 0.25 * pad:
            above = gap_up > gap_dn
        if above and frac + pad > 0.97:
            above = False
        elif not above and frac - pad < 0.03:
            above = True
        va, dy = ("bottom", 6 + pad_pts[i]) if above else ("top", -6 - pad_pts[i])
        ax.annotate(_fail_label(f[i], n_eval), xy=(x[i], y[i]), xytext=(0, dy),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=fontsize, color=color, clip_on=False)


def _eta_axis(ax, ylog=True):
    """A log eta axis, largest eta on the left."""
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\eta$")
    ax.grid(alpha=0.3, which="major")


def _eta_ticks(ax, ticks, labels):
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(plt.NullLocator())


def _top_axis(ax, ticks, labels, title):
    """A top axis over ``ax`` with ``labels`` at the eta ``ticks``."""
    top = ax.twiny()
    top.set_xscale("log")
    top.set_xlim(ax.get_xlim())
    top.set_xticks(ticks)
    top.set_xticklabels(labels)
    top.set_xlabel(title)
    top.tick_params(axis="x", labelsize=18)
    top.grid(False)
    return top


def _panel_global(ax, sc_df, qoi, n_eval, split, corner_label):
    """Global-basis ROM mean error versus eta, with the basis rank on the top axis. Returns whether any trajectory failed."""
    # The eta of a rank is its residual SVD energy; ranks with none (past the data's rank) are dropped.
    rank_sc, y_sc = global_rank_series(sc_df, qoi)
    m = dict(zip(sc_df["rank"], sc_df["residual_energy"]))
    eta_sc = np.array([max(float(m[r]), ETA_MIN) for r in rank_sc])
    ok = np.isfinite(eta_sc) & (eta_sc > 0)
    if not ok.all():
        rank_sc, y_sc, eta_sc = np.asarray(rank_sc)[ok], np.asarray(y_sc)[ok], eta_sc[ok]
    if eta_sc.size == 0:
        raise ValueError("no rank in the sweep has a finite residual energy; "
                         "there is nothing to place on the eta axis")

    ax.plot(eta_sc, y_sc, marker="o", ls="-", lw=2, ms=6, color=GLOBAL_COLOR)
    _eta_axis(ax)
    ax.set_ylabel(f"Mean {SPLIT_WORD[split]} error")
    ax.text(0.98, 0.97, corner_label, transform=ax.transAxes, ha="right", va="top", fontsize=20, color="0.25", zorder=6)
    # Label a tick only if it is 16% of the axis span from the last label kept, left to right.
    e = np.log10(eta_sc)
    span = max(e.max() - e.min(), 1e-12)
    keep, last = np.zeros(e.size, dtype=bool), None
    for i in np.argsort(eta_sc)[::-1]:
        if last is None or abs(e[i] - last) >= 0.16 * span:
            keep[i] = True
            last = e[i]
    _eta_ticks(ax, eta_sc, [rf"$10^{{{v:.0f}}}$" if k else "" for v, k in zip(e, keep)])
    top = _top_axis(ax, eta_sc, [f"{int(r)}" if k else "" for r, k in zip(rank_sc, keep)], "Global basis rank")
    top.xaxis.set_minor_locator(plt.NullLocator())

    # Failure captions: above a peak or a rising point, below a valley or a falling one, away from the axes' edges.
    failed = dict(zip(sc_df["rank"], sc_df["n_tracers"] - sc_df["n_completed"]))
    eta_of = dict(zip(rank_sc, eta_sc))
    y_of = dict(zip(rank_sc, y_sc))
    seq = [rank_sc[i] for i in np.argsort(eta_sc)[::-1]]      # left to right
    pos = {int(r): k for k, r in enumerate(seq)}
    ys = np.array([y_of[r] for r in seq], dtype=float)
    ylo, yhi = ax.get_ylim()
    span = np.log10(yhi) - np.log10(ylo)
    any_fail = False
    for r in sorted(failed):
        f, r = failed[r], int(r)
        if f <= 0 or r not in pos:
            continue
        any_fail = True
        k = pos[r]
        left = ys[k - 1] if k > 0 else ys[k]
        right = ys[k + 1] if k < len(ys) - 1 else ys[k]
        if ys[k] >= left and ys[k] >= right:
            above = True
        elif ys[k] <= left and ys[k] <= right:
            above = False
        else:
            above = left < ys[k]
        frac = (np.log10(ys[k]) - np.log10(ylo)) / span
        if above and frac > 0.88:
            above = False
        elif not above and frac < 0.12:
            above = True
        ax.annotate(_fail_label(f, n_eval), xy=(eta_of[r], ys[k]), xytext=(0, 13 if above else -17),
                    textcoords="offset points", ha="center", va="bottom" if above else "top",
                    color=FAIL_COLOR, fontsize=18, annotation_clip=False)
    return any_fail


def _panel_atrom(ax, df, tols, dark, qoi, n_eval, twin_tol, ylim, hybrid_df, corner_label):
    """AT-ROM mean error versus eta, one curve per tau_split, with hybrid stars and the mean basis rank of ``twin_tol`` on the top axis."""
    series, fails = [], []
    for t in tols:
        x, y = mean_series(df, t, qoi)
        series.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        ax.plot(x, y, marker="o", ls="-", lw=2, ms=6, color=dark[t])
        rows = df[np.isclose(df["tau_split"], t)]
        done = dict(zip(rows["eta"], rows["n_completed"]))
        fails.append((t, x, y, np.array([n_eval - float(done[v]) if v in done else 0.0 for v in x])))
    _eta_axis(ax)
    if ylim is not None:
        # A minimum range, widened to fit the data.
        lo, hi = ylim
        v = np.concatenate([s[1] for s in series]) if series else np.empty(0)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            lo, hi = min(lo, v.min() / 3.0), max(hi, v.max() * 3.0)
        ax.set_ylim(lo, hi)

    # Captions at the same eta are stacked 19 points apart, in legend order.
    stack = {}
    for i_t, (t, x, y, fail) in enumerate(fails):
        pads = np.zeros(np.size(x))
        for j, (xj, fj) in enumerate(zip(np.asarray(x), fail)):
            if np.isfinite(fj) and fj > 0:
                k = round(float(np.log10(xj)), 6)
                pads[j] = 19 * stack.get(k, 0)
                stack[k] = stack.get(k, 0) + 1
        others = []
        for k, (xk, yk) in enumerate(series):
            if k != i_t:
                lut = dict(zip(np.round(np.log10(xk), 6), yk))
                others.append([lut.get(round(float(np.log10(xj)), 6), np.nan) for xj in np.asarray(x)])
        _annotate_failures(ax, x, y, fail, n_eval, dark[t], pads, others)

    # Hybrid stars; the smallest tau_split on top.
    if hybrid_df is not None and len(hybrid_df):
        for k, t in enumerate(tols):
            if not (hybrid_df["tau_split"] == t).any():
                continue
            hx, hy = mean_series(hybrid_df, t, qoi)
            if np.size(hx):
                ax.plot(hx, hy, ls="none", marker=HYBRID_MARKER, ms=18, mfc=dark[t], mec="white", mew=1.2,
                        zorder=5 + (len(tols) - k) / (len(tols) + 1))

    ticks, rank = rank_series(df, twin_tol)
    _eta_ticks(ax, ticks, [_pow10(v) for v in ticks])
    color = dark.get(twin_tol, TOL_PAIRS[0][1])
    top = _top_axis(ax, ticks, [f"{r:.1f}" for r in rank], "AT-ROM basis rank (mean)")
    top.xaxis.label.set_color(color)
    top.tick_params(axis="x", colors=color)
    ax.text(0.02, 0.97, corner_label, transform=ax.transAxes, ha="left", va="top", fontsize=20, color="0.25", zorder=6)


def _panel_clusters(ax, df, tols, dark, light, max_eta_exp, bar_headroom):
    """Bars of the number of clusters per (eta, tau_split), largest eta on the left; the filled part is the splittable clusters."""
    etas = np.unique(np.asarray(df["eta"], dtype=float))
    etas = np.sort(etas[etas >= 10.0 ** (-max_eta_exp)])[::-1]
    pos = np.arange(len(etas))
    width = 0.8 / max(len(tols), 1)

    def values(t, col):
        rows = Results.where(df, tau_split=t)
        lut = dict(zip(rows["eta"], rows[col]))
        return np.array([float(lut[e]) if e in lut else np.nan for e in etas])

    tallest = 0.0
    for i, t in enumerate(tols):
        off = (i - (len(tols) - 1) / 2) * width
        n = values(t, "n_clusters")
        tallest = max(tallest, float(np.nanmax(n)) if np.isfinite(n).any() else 0.0)
        u = np.nan_to_num(values(t, "n_unsplittable"))
        ax.bar(pos + off, n, width=width, facecolor="none", edgecolor=dark[t], lw=1.8, zorder=3)
        ax.bar(pos + off, n - u, width=width, facecolor=light[t], edgecolor="none", zorder=2)

    ax.set_xticks(pos)
    ax.set_xticklabels([_pow10(v) for v in etas])
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Number of clusters")
    ax.grid(alpha=0.3, axis="y", which="major")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(tallest, 1.0) * bar_headroom)


def _panel_rank(ax, df, tols, dark, etas):
    """AT-ROM mean basis rank versus eta, one curve per tau_split."""
    for t in tols:
        x, y = rank_series(df, t)
        ax.plot(x, y, marker="o", ls="-", lw=2, ms=6, color=dark[t])
    _eta_axis(ax, ylog=False)
    ax.set_ylabel("Basis rank (mean)")
    _eta_ticks(ax, etas, [_pow10(v) for v in etas])


def _panel_fom_fraction(ax, sweep_dir, taus, dark, etas, positivity="on", eval_split="test"):
    """Hybrid percentage of FOM intervals versus eta, one curve per tau_split, averaged over completed trajectories."""
    for tau in taus:
        x, y = [], []
        for eta in sorted(etas, reverse=True):
            path = errors_path(sweep_dir, tau, eta, eval_split, prefix="hybrid_")
            if not os.path.exists(path):
                continue
            rows = Results.where(Results.read(path), positivity=positivity)
            rows = rows[rows["completed"] == 1]
            if rows.size:
                x.append(eta)
                y.append(100.0 * float(np.mean(rows["fom_frac"])))
        ax.plot(x, y, marker="o", ls="-", lw=2, ms=6, color=dark[tau])
    _eta_axis(ax, ylog=False)
    _eta_ticks(ax, list(etas), [rf"$10^{{{int(round(np.log10(e)))}}}$" for e in etas])
    ax.set_ylabel("FOM intervals [%]")
    ax.set_ylim(0, 100)


def _widen_first_gap(fig, axes, gap):
    """Add ``gap`` (figure fraction) between the first two rows of ``axes``, taken equally from each row's height; top axes move with their row."""
    n = axes.shape[0]
    rows = [axes[r, 0].get_position() for r in range(n)]
    h = rows[0].height - gap / n
    if h <= 0:
        raise ValueError(f"row_gap={gap} leaves no height for the panels")
    y0 = [0.0] * n
    y0[-1] = rows[-1].y0
    for r in range(n - 2, -1, -1):
        orig = rows[r].y0 - rows[r + 1].y1
        y0[r] = y0[r + 1] + h + orig + (gap if r == 0 else 0.0)
    for ax in fig.axes:
        b = ax.get_position()
        for r, row in enumerate(rows):
            if np.isclose(b.y0, row.y0) and np.isclose(b.height, row.height):
                ax.set_position([b.x0, y0[r], b.width, h])
                break


def _logfmt(y, _pos):
    """Power-of-ten tick label for a symlog axis."""
    if y == 0:
        return "0"
    e = int(np.round(np.log10(abs(y))))
    return f"$-10^{{{e}}}$" if y < 0 else f"$10^{{{e}}}$"


def _dex_error(d, peak, floor=1e-12, rom_key="y_rom_pos", t_key="tyr_rom_pos"):
    """RMS of log10(ROM) - log10(FOM) per species, each clipped at ``floor`` times the species' FOM peak."""
    y_fom = np.asarray(d["y_fom"])
    t_fom = np.asarray(d["tyr"])
    y_rom = np.asarray(d[rom_key])
    t_rom = np.asarray(d[t_key])
    if t_rom.shape[0] != t_fom.shape[0]:
        # A solve that failed early: interpolated onto the FOM times.
        y_rom = np.stack([np.interp(t_fom, t_rom, y_rom[j]) for j in range(y_rom.shape[0])])
    flo = (np.where(peak > 0, peak, 1.0) * floor)[:, None]
    dex = np.log10(np.clip(y_rom, flo, None)) - np.log10(np.clip(y_fom, flo, None))
    return np.sqrt(np.mean(dex ** 2, axis=1))


class _HandlerBands(HandlerBase):
    """One legend key showing every cluster colour, edge to edge."""

    def __init__(self, band_colors, alpha=None, **kw):
        super().__init__(**kw)
        self.band_colors = list(band_colors)
        self.alpha = alpha

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width,
                       height, fontsize, trans):
        n = max(len(self.band_colors), 1)
        w = width / n
        return [Rectangle((xdescent + i * w, -ydescent), w, height,
                          facecolor=c, alpha=self.alpha, lw=0, transform=trans)
                for i, c in enumerate(self.band_colors)]


def _over_white(color, alpha):
    """``color`` at ``alpha`` over white, as an opaque RGB."""
    r, g, b = to_rgb(color)
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def _band_colors(path, other_path):
    """``{cluster: colour}`` over the clusters of both routes, so a cluster has one colour in the data panel and the strip; FOM intervals are white."""
    ids = np.r_[np.asarray(path, dtype=int), np.asarray(other_path, dtype=int)]
    clusters = [int(c) for c in np.unique(ids) if int(c) != FOM_INTERVAL]
    cmap = {c: BAND_CMAP(i % BAND_CMAP.N) for i, c in enumerate(clusters)}
    cmap[int(FOM_INTERVAL)] = "w"
    return cmap


def _runs(route):
    """Runs of one value as ``(value, first, last)``."""
    route = list(route)
    out, k0 = [], 0
    for k in range(1, len(route) + 1):
        if k == len(route) or route[k] != route[k0]:
            out.append((route[k0], k0, k - 1))
            k0 = k
    return out


def etas_on_disk(sweep_dir, taus, eval_split="test"):
    """Every eta with an errors file for all of ``taus``, largest first."""
    split_tag = Results.suffix(eval_split)
    per_tau = []
    for tau in taus:
        pattern = re.compile(rf"errors_tol{re.escape(f'{tau:g}')}_eta([0-9.e+-]+){split_tag}\.csv")
        found = set()
        for path in glob.glob(os.path.join(sweep_dir, f"errors_tol{tau:g}_eta*.csv")):
            m = pattern.fullmatch(os.path.basename(path))
            if m:
                found.add(float(m.group(1)))
        per_tau.append(found)
    return sorted(set.intersection(*per_tau), reverse=True) if per_tau else []


def _qoi_names(qoi_indices, species, labels):
    """QoI indices and their names, renamed by ``labels``."""
    qi = list(qoi_indices)
    return qi, [(labels or {}).get(str(species[j]), str(species[j])) for j in qi]


def _qoi_start(t_ref, y_ref, dt, n_steps, time_scale, t_min, xscale, n_markers, axes, figsize):
    """Set up a QoI panel: a data panel over a route strip.

    ``time_scale`` divides every time (``NSEC * 1e3`` for kyr). ``t_min`` is the left end of the time axis (default: the first FOM time after 0 on a log axis, 0 on a linear one). ``n_markers`` FOM circles are spread evenly on the ``xscale`` axis. ``axes`` is a ``(data, strip)`` pair to draw into instead of a new figure of size ``figsize``. With ``n_steps`` None, it is the number of intervals in ``t_ref``.
    """
    use_paper_style()
    t_ref = np.asarray(t_ref, dtype=float) / time_scale
    y_ref = np.asarray(y_ref, dtype=float)
    if n_steps is None:
        n_steps = int(round(t_ref[-1] / dt))
    t_end = n_steps * dt
    log_t = xscale == "log"
    if t_min is None:
        t_min = float(t_ref[t_ref > 0][0]) if log_t else 0.0
    if axes is None:
        fig, (ax, strip_ax) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw=dict(height_ratios=[1.0, 0.09], hspace=0.45))
    else:
        ax, strip_ax = axes
        fig = ax.figure
    t_mark = (np.geomspace if log_t else np.linspace)(t_min, t_end, n_markers)
    marks = np.unique(np.searchsorted(t_ref, t_mark).clip(0, t_ref.size - 1))
    return fig, ax, strip_ax, t_ref, y_ref, n_steps, t_end, t_min, marks


def _qoi_curves(ax, models, t_ref, y_ref, marks, qi, names, normalize, time_scale):
    """Draw each QoI of each model ``(t, y, line format, plot options)`` and the FOM circles; return the QoI legend handles."""
    handles = []
    for i, (name, j) in enumerate(zip(names, qi)):
        c = LINE_COLORS[i % len(LINE_COLORS)]
        scale = (np.max(np.abs(y_ref[j])) or 1.0) if normalize else 1.0
        for t, y, fmt, kw in models:
            y = np.asarray(y, dtype=float)
            if y.size:
                ax.plot(np.asarray(t, dtype=float) / time_scale, y[j] / scale, fmt, color=c, **kw)
        ax.plot(t_ref[marks], y_ref[j, marks] / scale, "o", color=c, ms=6, mfc="none", mew=1.4, zorder=5)
        handles.append(Line2D([], [], color=c, lw=1.8, label=name))
    handles.append(Line2D([], [], color="0.35", marker="o", ms=6, mfc="none", mew=1.4, ls="none", label="FOM"))
    return handles


def _route_bands(ax, path, cmap, dt, n):
    """Shade the first ``n`` intervals of a cluster route, skipping FOM intervals."""
    path = np.asarray(path, dtype=int)
    for k in range(min(n, path.size)):
        if path[k] != int(FOM_INTERVAL):
            ax.axvspan(k * dt, (k + 1) * dt, color=cmap[int(path[k])], alpha=BAND_ALPHA, lw=0, zorder=0)


def _route_until_failure(ax, rom, cmap, dt, n_steps, time_scale):
    """Shade an AT-ROM's cluster route up to its last solved time, and return that time."""
    t_last = float(np.asarray(rom.t)[-1]) / time_scale if np.asarray(rom.t).size else 0.0
    _route_bands(ax, rom.path, cmap, dt, min(n_steps, int(np.ceil(t_last / dt - 1e-9))))
    return t_last


def _mark_failure(strip_ax, line_axes, t_fail, t_min, t_end, caption, fontsize, **line_kw):
    """Mark a failure time: a red line on ``line_axes``, and a cross and ``caption`` on the strip."""
    for a in line_axes:
        a.axvline(t_fail, color=FAIL_COLOR, lw=1.5, **line_kw)
    strip_ax.plot(t_fail, 0.5, "X", color=FAIL_COLOR, ms=10, transform=strip_ax.get_xaxis_transform())
    # On the side of the cross with more room.
    right = (t_fail - t_min) / (t_end - t_min) < 0.5
    strip_ax.annotate(caption, xy=(t_fail, 0.5), xycoords=strip_ax.get_xaxis_transform(),
                      xytext=(12 if right else -12, 0), textcoords="offset points", color=FAIL_COLOR,
                      ha="left" if right else "right", va="center", fontsize=fontsize - 1)


def _route_swatch(path, cmap, alpha):
    """Colours of the clusters a route visits, in visiting order, as opaque colours for the legend key."""
    visited = list(dict.fromkeys(int(c) for c in np.asarray(path) if int(c) != int(FOM_INTERVAL)))
    return [_over_white(cmap[c], alpha) for c in visited] or ["0.85"]


def _qoi_finish(fig, ax, strip_ax, handles, keys, route_key, swatch, strip_text, y_qoi, normalize, ylim,
                fontsize, time_unit, time_label, title, qoi_legend, savepath):
    """Axes, titles and legends of a QoI panel.

    ``normalize`` divides each QoI by its FOM maximum; otherwise values are on a log axis. ``ylim`` defaults to (0, 1.12), or a decade beyond the FOM range. ``qoi_legend=False`` leaves out the QoI legend above the panel. The key legend sits under the strip.
    """
    _qoi_yaxis(ax, y_qoi, normalize, ylim, fontsize, "abundance" if time_unit == "kyr" else "concentration")
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.set_xlabel(time_label, fontsize=fontsize + 2)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=fontsize + 6)
    strip_ax.set_ylim(0.0, 1.0)
    strip_ax.set_yticks([])
    strip_ax.set_xticks([])
    strip_ax.xaxis.set_minor_locator(plt.NullLocator())
    strip_ax.text(0.5, -0.25, strip_text, transform=strip_ax.transAxes, ha="center", va="top", fontsize=fontsize)
    if qoi_legend:
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=4,
                  frameon=False, fontsize=fontsize)
    box = strip_ax.get_position()
    fig.legend(handles=keys, handler_map={route_key: _HandlerBands(swatch)}, loc="upper center",
               bbox_to_anchor=(box.x0 + box.width / 2, 0.0), ncol=2, frameon=False, fontsize=fontsize,
               columnspacing=1.0, handlelength=1.5, handletextpad=0.5)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        print(f"wrote {savepath}")
    return fig, (ax, strip_ax)


def _qoi_yaxis(ax, y_qoi, normalize, ylim, fontsize, quantity):
    """The y axis of a QoI panel: ``quantity`` over its FOM maximum, or the fractional abundance on a log axis."""
    if normalize:
        ax.set_ylim(*(ylim or (0.0, 1.12)))
        ax.set_ylabel(f"{quantity.capitalize()} / max", fontsize=fontsize + 2)
        return
    ax.set_yscale("log")
    if ylim is None:
        v = np.asarray(y_qoi, dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        ylim = (v.min() / 10.0, v.max() * 10.0) if v.size else None
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel("Fractional abundance", fontsize=fontsize + 2)
