#!/bin/bash
#SBATCH --job-name=batch_gpu_job
#SBATCH --partition=short-unkillable
#SBATCH --cpus-per-task=24
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G
#SBATCH --gpus=a100l:4
#SBATCH --time=00:30:00
#SBATCH --output=job_output_%j.txt

# Your commands to run go here
echo "Job is running on GPU"
nvidia-smi
