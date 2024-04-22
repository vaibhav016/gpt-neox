#!/bin/bash
#SBATCH --job-name="pile_training_cosine_decay_val_pile"
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a100l:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=80G
#SBATCH --time=3:00:00 
#SBATCH --output=pile_training_cosine_decay_val_pile-%j.out
#SBATCH --error=pile_training_cosine_decay_val_pile-%j.err


# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Your hostfile creation script from above

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

python3 deepy.py train.py ./configs/49M_local_test_finetune.yml ./configs/german_schedules/pile_valid/local_setup_german_val_pile_cosine_decay.yml ./configs/schedules/adam_cosine_lr3e-3_3e-6_wu-001.yml

cp -r $SLURM_TMPDIR/output $SCRATCH 

exit 0
