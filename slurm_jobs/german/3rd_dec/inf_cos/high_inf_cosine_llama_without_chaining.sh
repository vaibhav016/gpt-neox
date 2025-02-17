#!/bin/bash
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
CHECKPOINT_DIR="checkpoints_3rd_dec/german/inf_cos/high"

# Get the most recent checkpoint path
LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/JOB-* | tail -n 1)

# If there’s a previous checkpoint, use it. Otherwise, start from scratch.
if [[ -n "$LATEST_CHECKPOINT" ]]; then
  CHECKPOINT_PATH="${LATEST_CHECKPOINT}"
else
  CHECKPOINT_PATH="${CHECKPOINT_DIR}"
fi

# Update load_low.yml with the checkpoint path
sed -i "s|\"load\":.*|\"load\": \"${CHECKPOINT_PATH}\",|" configs/load_checkpoints/german/3rd_dec/inf_cos/load_high.yml

# Run the training script
python deepy.py train.py --conf_dir configs llama2/410M.yml dataset_config/german/german_config_scratch.yml schedulers/const_inf_cos/adam_constant_lr5e-4_wu-0_high.yml load_checkpoints/german/3rd_dec/inf_cos/load_high.yml                         