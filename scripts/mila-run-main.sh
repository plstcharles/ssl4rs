#!/bin/bash
# Mila SLURM job script for the 'main' partition type (6 CPUs, 32GB RAM, 48GB VRAM, 48h)
#SBATCH --job-name=ssl4rs
#SBATCH --wckey=ssl4rs
#SBATCH --partition=main
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:48gb:1
#SBATCH --output=/network/projects/ai4h-disa/india/slurm-logs/%j.log.out
#SBATCH --error=/network/projects/ai4h-disa/india/slurm-logs/%j.log.err

export SRC_DIR="$HOME/dev/ssl4rs"
LAUNCH_SCRIPT_PATH=$(dirname "$0")
LAUNCH_SCRIPT_PATH=$(cd "$LAUNCH_SCRIPT_PATH" && pwd)

echo    "Script:      $LAUNCH_SCRIPT_PATH"
echo    "Arguments:   $*"
echo -n "Date:        "; date
echo    "JobId:       $SLURM_JOBID"
echo    "JobName:     $SLURM_JOB_NAME"
echo    "Node:        $HOSTNAME"
echo    "Nodelist:    $SLURM_JOB_NODELIST"
echo    "Source:      $SRC_DIR"
echo    "CUDA:        $CUDA_VISIBLE_DEVICES"

export PYTHONUNBUFFERED=1
module purge
module load miniconda/3
# shellcheck disable=SC1090
source "$CONDA_ACTIVATE"
conda activate ssl4rs

cd "$SRC_DIR" || exit

echo "Launching ssl4rs experiment ($(date +%Y.%m.%d.%H.%M.%S))"
"$CONDA_PREFIX/bin/python" "$@"
