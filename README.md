# AT-ROM

Code for the paper *AT-ROM: Adaptive Tree Reduced Order Model for Parametric Quadratic Chemical Kinetics* by Arjun Vijaywargiya, Stella Offner and George Biros.

AT-ROM is a tree-based reduced-order model (ROM) for quadratic dynamical systems whose operators depend on a time-varying parameter. It clusters a state- and parameter-informed feature space built, gather snapshots and fits a local POD basis for each cluster, and iteratively refines the base clustering based on single-interval prediction errors in prescribed quantities of interest. During inference, predictions are made one time interval at a time, by choosing an active basis through tree-descent, and integrating a POD-Galerkin ROM.

In the `Hybrid` approach, the method also allows a fallback to the full-order model (FOM) when the state enters a cluster whose basis represents it poorly.

This repository provides code for reproducing the numerical results in our paper.

## Installation

Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

The notebooks in `experiments/` also need Jupyter.

## Layout

```text
src/
  rom.py          AT-ROM training and inference, POD bases, POD-Galerkin ROM
  cluster.py      clustering
  hybrid.py       AT-ROM with FOM backtracking
  run.py          runs an experiment
  plots.py        figures
  fom/            full-order models and data loading
networks/         reaction networks, initial abundances, shielding tables
experiments/
  <system>/<case>/
    run.py        experiment settings; run this file to run the experiment
    run.slurm     Slurm script for run.py
    <system>.ipynb  notebook for generating the paper's figures
    timing.py     AT-ROM vs FOM cost (Nelson and OSU only)
```

| Folder | System |
|---|---|
| `experiments/lorenz96/N100_F0-2.5` | Lorenz-96, d = 100, forcing p in [0.0125, 2.5] |
| `experiments/nelson/R0.1_M6.0` | Nelson network, 14 species |
| `experiments/osu/R0.1_M6.0` | OSU network, 29 species |
| `experiments/egfr/T1000` | BIOMD0000000205, 194 species, T = 1000 s |

## Data

**Lorenz-96 and EGFR.** The datasets are generated on the first run, into `datasets/lorenz96/N100_F0-2.5` and `datasets/egfr/T1000`, by `src/fom/lorenz96.py` and `src/fom/egfr.py`.

**Nelson and OSU.** The datasets are computed with [KIDApy](https://github.com/NSF-Simons-CosmicAI-Institute/KIDApy), along parameter trajectories obtained from a hydrodynamic simulation of a gravitational core collapse. They are not included in this repository; download them from [huggingface.co/datasets/arjunveejay/atrom-data](https://huggingface.co/datasets/arjunveejay/atrom-data) (folder `grav_collapse/`).

Place them as

```text
datasets/grav_collapse/nelson/params.npz
datasets/grav_collapse/nelson/trajectories.npy
datasets/grav_collapse/osu_09_2008/params.npz
datasets/grav_collapse/osu_09_2008/trajectories.npy
```

**Data credit:** Arjun Vijaywargiya, Luke Smith, Stella Offner, and CosmicAI.

Each dataset folder holds two files:

- `trajectories.npy`: every trajectory, one row per time sample in the order `[index | t | parameters | state]`. Each trajectory is a contiguous block of rows: the interval boundaries and 4 interior samples per interval.
- `params.npz`: `params`, the parameters at every interval boundary, of shape `(trajectories, intervals + 1, parameters)`, and `t`, the boundary times.

A run reads these files once to write its own training and testing files (`snapshots_train.npy`, `snapshots_test.npy`) into the experiment folder.

## Running an experiment

```bash
python experiments/nelson/R0.1_M6.0/run.py
```

| Option | Effect |
|---|---|
| `--splits test train` | the sets to evaluate (default: both) |
| `--overwrite` | redo everything instead of skipping finished work |
| `--no-figures` | skip the figures |
| `--n-workers N` | number of worker processes (default: every core) |

A run does the following:

1. trains AT-ROM on the training set at every pair of split tolerance `tau_split` and residual SVD energy tolerance `eta` in the settings, and runs inference;
2. runs hybrid inference with the AT-ROMs from step 1 (not for Lorenz-96);
3. trains the global-basis ROM at every basis rank in the settings, and runs inference.

Results are saved as generated.


### Settings

Each experiment's `run.py` has a `SETTINGS` dictionary with the QoIs, the chosen number of training and testing trajectories, the clustering features, the `tau_split`, `eta` and rank values, and whether to use the non-negativity correction. System settings such as solver tolerances and the reaction network are in `DEFAULTS` in `src/fom/<system>.py`.

### Output

Each run writes these files to the experiment folder. `<set>` is `test` or `train`, and `<pair>` is a (tau_split, eta) pair, e.g. `tol0.01_eta1e-5`.

```text
fits_<set>.csv                    AT-ROM cluster counts and basis ranks for each pair
global_fits_<set>.csv             global-basis ROM summary for each rank
global_errors_<set>.csv           global-basis ROM QoI errors for each trajectory and rank
sweep/
  atrom_<pair>.npz                trained AT-ROM
  errors_<pair>_<set>.csv         AT-ROM QoI errors for each trajectory
  hybrid_errors_<pair>_<set>.csv  hybrid QoI errors and FOM use for each trajectory
figures/
  panel_errors.pdf                mean QoI error vs eta
  panel_partition.pdf             cluster count, mean basis rank and hybrid FOM use vs eta
  violin_atrom.pdf                AT-ROM testing error distributions
  violin_hybrid.pdf               hybrid testing error distributions
```

## Timing

`timing.py` in the Nelson and OSU folders compares the cost of AT-ROM inference with the FOM over the testing set: right-hand-side evaluations, Jacobian evaluations, LU factorizations and wall time per trajectory.

```bash
python experiments/nelson/R0.1_M6.0/timing.py
```

It needs the AT-ROM at `tau_split = 0.01`, `eta = 1e-5`, which the run saves, and writes `timing_test.csv`. Run it on an otherwise idle node.

## Acknowledgments

This work was supported by the National Science Foundation under Cooperative Agreement 2421782 and the Simons Foundation award MPS-AI-00010515 (NSF-Simons AI Institute for Cosmic Origins—CosmicAI, https://www.cosmicai.org/).

