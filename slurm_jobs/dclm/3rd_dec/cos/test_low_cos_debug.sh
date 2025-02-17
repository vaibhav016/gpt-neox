#!/bin/bash
#SBATCH --output=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/dclm/cos/low/output/slurm-%j.out            
#SBATCH --error=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/dclm/cos/low/error/slurm-%j.err
#SBATCH --mail-user=vaibhavsinghfcos@gmail.com
#SBATCH -A bif151
#SBATCH -J dc.cs.lo
#SBATCH -t 2:00:00
#SBATCH -q debug
#SBATCH -N 32
#SBATCH --mail-type=BEGIN,END,FAIL


SECONDS_BEFORE_EXIT=7020  # 1 hour 57 minutes (for a 2-hour job)

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

# Define variables
CHECKPOINT_DIR="checkpoints_3rd_dec/dclm/cos/low"

# Get the most recent checkpoint path
LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/JOB-* | tail -n 1)

# If there’s a previous checkpoint, use it. Otherwise, start from scratch.
if [[ -n "$LATEST_CHECKPOINT" ]]; then
  CHECKPOINT_PATH="${LATEST_CHECKPOINT}"
else
  CHECKPOINT_PATH="${CHECKPOINT_DIR}"
fi

# Update load_low.yml with the checkpoint path
sed -i "s|\"load\":.*|\"load\": \"${CHECKPOINT_PATH}\",|" configs/load_checkpoints/dclm/3rd_dec/cos/load_low.yml

# Run the training script
python deepy.py train.py --conf_dir configs llama2/410M.yml dataset_config/dclm/dclm_config_scratch.yml schedulers/cos/adam_cosine_lr3e-4_3e-5_wu-0.01.yml load_checkpoints/dclm/3rd_dec/cos/load_low.yml

exit 0

