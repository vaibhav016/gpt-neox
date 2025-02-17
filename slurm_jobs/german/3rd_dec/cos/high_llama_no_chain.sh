#!/bin/bash
#SBATCH --output=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/german/cos/high/output/slurm-%j.out            
#SBATCH --error=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/german/cos/high/error/slurm-%j.err
#SBATCH --mail-user=vaibhavsinghfcos@gmail.com
#SBATCH -A bif151
#SBATCH -J gr.cs.hi
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 32
#SBATCH --mail-type=BEGIN,END,FAIL


SECONDS_BEFORE_EXIT=7020  # 5 hour 57 minutes (for a 6-hour job)
# SECONDS_BEFORE_EXIT=240  # 1 hour 57 minutes (for a 2-hour job)

# Recompute the host file
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12803
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/vaibhav_016/neox/gpt-neox/hostfiles/hosts_$SLURM_JOBID


# Run the training script
python deepy.py train.py --conf_dir configs llama2/410M.yml dataset_config/german/german_config_scratch.yml schedulers/cos/adam_cosine_lr1e-3_3e-5_wu-0.01 load_checkpoints/dclm/3rd_dec/cos/load_high.yml
exit 0
