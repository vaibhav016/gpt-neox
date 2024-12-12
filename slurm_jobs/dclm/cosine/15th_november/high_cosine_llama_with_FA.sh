#!/bin/bash
#SBATCH --output=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/output/slurm-%j.out
#SBATCH --error=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/error/slurm-%j.err
#SBATCH --mail-user=vaibhavsinghfcos@gmail.com
#SBATCH -A bif151
#SBATCH -J dc.cs.h
#SBATCH -t 24:00:00
#SBATCH -p extended
#SBATCH -N 16
#SBATCH --mail-type=BEGIN,END,FAIL


# Recompute the host file
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12803
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/vaibhav_016/neox/gpt-neox/hostfiles/hosts_$SLURM_JOBID

# Setup script
source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/setup.sh

cd /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox

# Run the training script
python deepy.py train.py --conf_dir configs llama2/410M_flash.yml dataset_config/dclm/dclm_config_scratch.yml schedulers/cos/adam_cosine_lr3e-4_3e-5_wu-0.01.yml load_checkpoints/dclm/cosine/load_flash.yml

exit 0
