#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH --job-name=image_gen_mit_preemptable_h200
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH -t 2-00:00:00
#SBATCH --requeue
#SBATCH -o slurm_logs/%j/%j.out
#SBATCH -e slurm_logs/%j/%j.err

set -euo pipefail

module load cuda/12.4
module load miniforge/24.3.0-0
conda activate vg-env

cd /home/eeshan/video-gen/image-gen/semanticist

# Make sure the job log dir exists for any extra files you might write there
mkdir -p "slurm_logs/${SLURM_JOB_ID}"

export SLURM_JOB_ID=${SLURM_JOB_ID}

# HF cache on pool (persistent)
export HF_HOME="/home/eeshan/orcd/pool/.cache/huggingface"
mkdir -p "$HF_HOME"

# torch.compile cache on scratch (unique per job)
export TORCHINDUCTOR_CACHE_DIR="/home/eeshan/orcd/scratch/torchinductor_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# TF32 only affects FP32 matmuls; safe speed-up for FP32 leftovers on A100
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvidia-smi

# Use accelerate so distributed process group is initialized properly
accelerate launch \
  --num_processes=4 \
  --multi_gpu \
  --mixed_precision=bf16 \
  --num_machines=1 \
  --machine_rank=0 \
  --main_process_port=29500 \
  train_net.py --cfg configs/tokenizer_cluster_h200.yaml