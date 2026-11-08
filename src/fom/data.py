"""Reading the FOM snapshot data, shared by every system.

Every dataset folder holds two files:

- ``trajectories.npy``: every trajectory, one row per snapshot in ``[idx | t | params | state]`` order, each trajectory a contiguous block of ``SUBSTEPS * n_steps + 1`` rows.
- ``params.npz``: ``params``, the parameters at every interval boundary t_k, ``(n_trajectories, n_steps + 1, n_params)``, and ``t``, the boundary times. A dataset whose training and testing trajectories are fixed also stores their indices as ``train`` and ``test``.

The training and testing snapshot files are built from ``trajectories.npy`` once and kept in ``out_dir``, so that file is read only when ``build_snapshots`` is set.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np

N_LEADING = 2                # idx and t, ahead of the parameters in every row
SUBSTEPS = 5                 # snapshots per interval: t_k and 4 uniformly spaced internal time points


# ---------------------------------------------------------------------------
# Snapshot files
# ---------------------------------------------------------------------------

def _snapshot_paths(out_dir):
    return (os.path.join(out_dir, "snapshots_train.npy"),
            os.path.join(out_dir, "snapshots_test.npy"))


def _idx_path(snapshots_path):
    return os.path.splitext(snapshots_path)[0] + "_indices.npy"


def write_snapshots(src_mm, starts, sel, rpt, out_path):
    """Restack the selected trajectories onto a fixed per-interval grid.

    Interval ``k`` becomes source rows ``[SUBSTEPS*k .. SUBSTEPS*k + SUBSTEPS]``, sharing its last snapshot, at t_{k+1}, with the next interval, so a basis is fitted over whole intervals rather than the first 80% of each.

    ``idx`` is rewritten to the local position; global ids go to an ``_indices.npy`` sidecar.
    """
    sel = np.asarray(sel, dtype=np.int64)
    n_steps = (int(rpt) - 1) // SUBSTEPS
    tp = SUBSTEPS + 1
    rows_out = n_steps * tp
    np.save(_idx_path(out_path), sel)

    seg = np.arange(n_steps) * SUBSTEPS
    row_off = (seg[:, None] + np.arange(tp)[None, :]).ravel()

    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float64,
        shape=(sel.size * rows_out, int(src_mm.shape[1])))
    for pos, g in enumerate(sel):
        lo = pos * rows_out
        out[lo:lo + rows_out] = src_mm[int(starts[g]) + row_off]
        out[lo:lo + rows_out, 0] = pos
    out.flush()
    print(f"  wrote {out_path}  {out.shape}")


def open_snapshots(snapshots_path, n_tracers, n_steps, n_species):
    """Memmap a snapshot file and regenerate its row provenance.

    Everything follows from the file shape, since blocks are contiguous. A trajectory whose FOM solve failed left a zero-filled block, spotted by ``nH == 0`` since real densities are strictly positive.
    """
    mm = np.load(snapshots_path, mmap_mode="r")
    n_rows, n_cols = mm.shape
    n_params = int(n_cols) - N_LEADING - int(n_species)
    if n_params <= 0:
        raise ValueError(f"{snapshots_path}: {n_cols} columns is too few for "
                         f"{n_species} species. Wrong network?")
    tp, rem = divmod(n_rows, n_tracers * int(n_steps))
    if rem or tp < 1:
        raise ValueError(
            f"{snapshots_path}: {n_rows} rows is not divisible by n_tracers*n_steps "
            f"= {n_tracers}*{n_steps}; it does not match this dataset.")

    rows_per_tracer = int(n_steps) * tp
    sample_step = np.tile(
        np.repeat(np.arange(int(n_steps), dtype=np.int64), tp), n_tracers)
    sample_tracer = np.repeat(np.arange(n_tracers, dtype=np.int64), rows_per_tracer)
    first_rows = np.arange(n_tracers, dtype=np.int64) * rows_per_tracer
    valid_rows = np.repeat(np.asarray(mm[first_rows, N_LEADING]) != 0.0,
                           rows_per_tracer)

    print(f"Attached {snapshots_path}: {mm.shape} (samples/step={tp}), "
          f"{int(valid_rows[first_rows].sum())}/{n_tracers} tracers valid")
    return SimpleNamespace(
        memmap=mm, path=snapshots_path, sample_step=sample_step,
        sample_tracer=sample_tracer,
        valid_rows=valid_rows, n_species=int(n_species), n_params=n_params,
        param_slice=slice(N_LEADING, N_LEADING + n_params),
        species_slice=slice(N_LEADING + n_params, N_LEADING + n_params + n_species),
    )


# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------

def read_params(data_dir):
    """The contents of ``<data_dir>/params.npz`` as a dict: ``params``, ``t`` and, if present, ``train`` and ``test``."""
    path = os.path.join(data_dir, "params.npz")
    with np.load(path) as z:
        out = {k: np.asarray(z[k]) for k in z.files}
    out["params"] = out["params"].astype(np.float64)
    if out["params"].ndim != 3 or out["params"].shape[1] != out["t"].size:
        raise ValueError(f"{path}: params has shape {out['params'].shape}, expected (n_trajectories, {out['t'].size}, n_params)")
    if not np.isfinite(out["params"]).all():
        raise ValueError(f"{path}: params is not finite")
    print(f"  parameters: {path}")
    return out


def open_trajectories(data_dir, n_trajectories, n_steps):
    """Memory-map ``<data_dir>/trajectories.npy`` and return it with the first row of each trajectory."""
    path = os.path.join(data_dir, "trajectories.npy")
    mm = np.load(path, mmap_mode="r")
    rows = SUBSTEPS * n_steps + 1
    if mm.shape[0] != n_trajectories * rows:
        raise ValueError(f"{path} has {mm.shape[0]} rows, expected {n_trajectories} trajectories of {rows} rows")
    return mm, np.arange(n_trajectories, dtype=np.int64) * rows


# ---------------------------------------------------------------------------
# FOM trajectories
# ---------------------------------------------------------------------------

def _drop_repeated_boundaries(t, y):
    """Drop the interval-end snapshots the snapshot file repeats at the start of the next interval.

    The last snapshot of each interval, at t_{k+1}, is the same instant as the next interval's first; left in, every time-integrated norm counts it twice.
    """
    tp = SUBSTEPS + 1
    keep = np.ones(t.size, dtype=bool)
    keep[tp - 1:-1:tp] = False
    return t[keep], y[:, keep]


def load_trajectories(snapshots_path, n, n_species):
    """FOM trajectories by position in the dataset: the first ``n``, or exactly the positions listed if ``n`` is a sequence.

    Returns ``(t_foms, y_foms, x_eqs)``, the last stacking the initial states, in the order requested.
    """
    mm = np.load(snapshots_path, mmap_mode="r")
    n_params = int(mm.shape[1]) - N_LEADING - n_species
    positions = range(int(n)) if np.ndim(n) == 0 else [int(p) for p in n]
    # Read the position column once rather than once per trajectory: it is 75 MB for 2048 OSU trajectories, so a prefix load of 1311 would otherwise scan 96 GB.
    pos_col = np.asarray(mm[:, 0])
    t_foms, y_foms = [], []
    for pos in positions:
        lo, hi = np.searchsorted(pos_col, [pos, pos + 1])
        # Copy, not view: a view keeps a mapping of the whole file alive, and one per trajectory adds up fast.
        block = np.array(mm[lo:hi])
        if block.shape[0] == 0 or block[0, N_LEADING] == 0.0:
            raise ValueError(f"{snapshots_path}: tracer {pos} has no stored FOM "
                             f"trajectory; the block is invalid.")
        t, y = _drop_repeated_boundaries(
            block[:, 1],
            block[:, N_LEADING + n_params:N_LEADING + n_params + n_species].T)
        t_foms.append(t)
        y_foms.append(y)
    x_eqs = np.stack([y[:, 0].copy() for y in y_foms])
    return t_foms, y_foms, x_eqs
