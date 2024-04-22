#!/bin/bash
#SBATCH --job-name="ger_tr_inf_cos_low_val_ger"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a100l:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=6:00:00 
#SBATCH --output=ger_tr_inf_cos_low_val_ger-%j.out
#SBATCH --error=ger_tr_inf_cos_low_val_ger-%j.err


# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

## Your hostfile creation script from above
source write_hostfile.sh
# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=/home/mila/p/paria.mehrbod/scratch/resetbranch/gpt-neox/hostfiles/hosts_$SLURM_JOBID

module load anaconda/3
conda activate llm_project
module load cuda/12.1.1 
# export SLURM_TMPDIR=/Tmp/slurm.$SLURM_JOB_ID.0
# export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

mkdir $SLURM_TMPDIR/output
echo "save_path: $SLURM_TMPDIR/output"
sd=$SLURM_TMPDIR/output

# Launch training
# python3 deepy.py train.py ./configs/49M_local_test.yml ./configs/local_setup_pile_train.yml ./configs/schedules/adam_infinv_lr3e-4_3e-5_wu-001.yml --save "$sd/checkpoints" --tensorboard_dir "$sd/tensorboard"

# python3 deepy.py train.py ./configs/49M_local_test.yml ./configs/local_setup_pile_train.yml ./configs/schedules/adam_infinv_lr3e-4_3e-5_wu-001.yml

python3 deepy.py train.py ./configs/49M_local_test_finetune.yml ./configs/german_schedules/german_valid/local_setup_german_val_ger_low.yml ./configs/schedules/inf_cosine_schedules/adam_infcos_lr_constant_german_low.yml

cp -r $SLURM_TMPDIR/output $SCRATCH 

exit 0