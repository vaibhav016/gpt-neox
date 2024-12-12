#!/bin/bash
#SBATCH --output=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/output/slurm-%j.out
#SBATCH --error=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/error/slurm-%j.err
#SBATCH --mail-user=vaibhavsinghfcos@gmail.com
#SBATCH -A bif151
#SBATCH -J german_inf_cos_low
#SBATCH -t 12:00:00
#SBATCH -p batch
#SBATCH -N 300
#SBATCH --mail-type=BEGIN,END,FAIL

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12803
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/vaibhav_016/gpt-neox/hostfiles/hosts_$SLURM_JOBID

### Setup ####
source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/setup.sh

### Running the training script ###
cd /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/gpt-neox
python deepy.py train.py --conf_dir configs llama2/410M.yml dataset_config/german/german_config_scratch.yml schedulers/inf_cos/adam_infcos_lr3e-4_3e-5_wu-001_const_4.yml                                  

exit 0
