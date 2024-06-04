#!/bin/bash
#SBATCH --job-name="cosine_decay"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a100l:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00 
#SBATCH --output=cosine_decay_lg_%j.out
#SBATCH --error=cosine_decay_lg_%j.err


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

# mkdir $SLURM_TMPDIR/output
# echo "save_path: $SLURM_TMPDIR/output"
# sd=$SLURM_TMPDIR/output

# pile training
# python3 deepy.py train.py ./configs/49M_local_test.yml ./configs/pile_schedules/train/local_setup_pile_train.yml ./configs/schedules/cosine_decay_schedules/adam_cosine_lr3e-3_3e-6_wu-0.01.yml

# german training
python3 deepy.py train.py ./configs/49M_local_test_finetune.yml ./configs/german_schedules/pile_valid/local_setup_german_val_gr_and_pile_cosine_decay_replay.yml ./configs/schedules/cosine_decay_schedules/adam_cosine_lr3e-3_3e-6_wu-0.01.yml

# cp -r $SLURM_TMPDIR/output $SCRATCH 

exit 0
