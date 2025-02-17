#!/bin/bash
#SBATCH --output=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/code/inf_cos/ablations/high/const_4/output/slurm-%j.out            
#SBATCH --error=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/code/inf_cos/ablations/high/const_4//error/slurm-%j.err
#SBATCH --mail-user=vaibhavsinghfcos@gmail.com
#SBATCH -A bif151
#SBATCH -J cd.ih.4
#SBATCH -t 2:00:00
#SBATCH -q debug
#SBATCH -N 32
#SBATCH --mail-type=BEGIN,END,FAIL

SECONDS_BEFORE_EXIT=7020  # 2 hour 57 minutes (for a 2-hour job)

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
CHECKPOINT_DIR="checkpoints_3rd_dec/code/inf_cos/ablations/high/const_4"

LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/JOB-* | tail -n 1)

# If there’s a previous checkpoint, use it. Otherwise, start from scratch.
if [[ -n "$LATEST_CHECKPOINT" ]]; then
  CHECKPOINT_PATH="${LATEST_CHECKPOINT}"
else
  CHECKPOINT_PATH="${CHECKPOINT_DIR}"
fi

# Update load_low.yml with the checkpoint path
sed -i "s|\"load\":.*|\"load\": \"${CHECKPOINT_PATH}\",|" configs/load_checkpoints/code/3rd_dec/inf_cos/ablations/high/const_4/load_high.yml

# Run the training script
python deepy.py train.py --conf_dir configs llama2/410M.yml dataset_config/code/code_config_scratch.yml schedulers/const_inf_cos/code/annealing/ablation_const_high/adam_constant_lr1e-3_3e-5_wu-0_4.yml load_checkpoints/code/3rd_dec/inf_cos/ablations/high/const_4/load_high.yml                

sed -i "s|\"finetune\":.*|\"finetune\": false,|" configs/load_checkpoints/code/3rd_dec/inf_cos/ablations/high/const_4/load_high.yml


exit 0

