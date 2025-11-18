#!/bin/bash
#SBATCH --job-name=RoadGraph
#SBATCH --partition=gpu
#SBATCH --output=logs/RG_eval/output_trainer_%j.log
#SBATCH --error=logs/RG_eval/error_trainer_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=45:00:00
#SBATCH --mem=32G

module avail hpcx-mpi

module load miniconda3/23.11.0s
source /oscar/runtime/software/external/miniconda3/23.11.0/etc/profile.d/conda.sh
conda activate csci2952_mocov3

module load cudnn cuda
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('cuda version', torch.version.cuda)"

python eval.py