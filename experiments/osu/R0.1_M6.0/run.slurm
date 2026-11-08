#!/bin/bash
#SBATCH -J osu
#SBATCH -o run_%j.out                    # written next to this script
#SBATCH -e run_%j.err
#SBATCH -N 1
#SBATCH -n 1                             # one launching task; Python distributes the work over the cores
#SBATCH -p gg                            # Vista Grace-Grace CPU partition
#SBATCH -t 24:00:00
#SBATCH -A <ALLOCATION>                  # your Slurm account

# Submit from this folder: cd experiments/osu/R0.1_M6.0 && sbatch run.slurm [run.py options, e.g. --splits test]
# Runs run.py in this folder, which holds every setting of this experiment; results are written here.

set -euo pipefail

[[ -e run.py ]] || { echo "submit from the experiment folder that holds run.py" >&2; exit 2; }
source "$SCRATCH/cosmicai/bin/activate" # Replace with python venv path

# Keep the math libraries single-threaded per worker, so the process pool rather than BLAS owns the cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "host=$(hostname) started $(date)"
python -u run.py --n-workers "$(nproc --all)" "$@"
echo "finished $(date)"
