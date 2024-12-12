#!/bin/bash
#SBATCH -A bif151
#SBATCH -t 2:00:00
#SBATCH -N 8
#SBATCH -p extended
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vaibhavsinghfcos@gmail.com
#SBATCH --mail-type=all
#SBATCH --output=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/output/slurm-%j.out
#SBATCH --error=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/error/slurm-%j.err

srun pkill python

source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/setup.sh

export HOSTNAMES=scontrol show hostnames "$SLURM_JOB_NODELIST"
export MASTERADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export COUNT_NODE=scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l


source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/vaibhav_016/gpt-neox/hostfiles/hosts_$SLURM_JOBID
cd /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/gpt-neox
# rm -rf ./megatron/fused_kernels/build/lock
# rm -rf /autofs/nccs-svm1_home2/gopeshh/.cache/torch_extensions/py312_cpu/fused_adam/lock


# python ./deepy.py train_new.py ./configs/new_local_setup.yml
python deepy.py train.py --conf_dir configs local_setup.yml slurm_125M.yml
# python deepy.py train.py --conf_dir configs local_setup.yml 350M.yml
# python deepy.py train.py --conf_dir configs local_setup.yml bf16_125M.yml


exit 0