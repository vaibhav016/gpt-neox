#!/bin/bash

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
python deepy.py train.py --conf_dir configs 2nd_dec_testing_flash/410M_pythia_default_fp16_with_flash.yml dataset_config/dclm/dclm_config_scratch.yml schedulers/inf_cos/adam_infcos_lr1e-3_3e-5_wu-001_const_5.yml load_checkpoints/dclm/inf_cos/load_flash.yml                     







