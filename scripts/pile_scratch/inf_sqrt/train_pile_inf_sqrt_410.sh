#!/bin/bash
#BSUB -nnodes 92
#BSUB -W 12:00
#BSUB -q batch
#BSUB -o /gpfs/alpine2/csc565/scratch/vaibhav_016/test/training_logs/gpt_neox_out.%J
#BSUB -e /gpfs/alpine2/csc565/scratch/vaibhav_016/test/training_logs/gpt_neox_err.%J
#BSUB -J 410_pile_sqrt
#BSUB -alloc_flags gpudefault
#BSUB -P csc565
#BSUB -N vaibhav.singh@mila.quebec
#BSUB -B vaibhav.singh@mila.quebec

# clean up nodes
# jsrun pkill python

# deactivate any conda enviroment
/gpfs/alpine2/csc565/scratch/vaibhav_016/miniconda3/bin/activate
conda deactivate

SETUP_PROJECT_DIR='csc565'
SETUP_USERNAME=$(whoami)

# enter the home directory
cd /gpfs/alpine2/$SETUP_PROJECT_DIR/scratch/$SETUP_USERNAME

source setup.sh
source write_hostfile.sh

# Set up the environment
# source /gpfs/alpine/csc499/scratch/btherien/setup.sh

# export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/btherien/latest_install/cache
# rm -r $TORCH_EXTENSIONS_DIR

# Move to the gpt-neox install
# export TRAIN_PATH=/gpfs/alpine/csc499/scratch/btherien/gpt-neox
# cd $TRAIN_PATH

cd test/gpt-neox/
# rm megatron_config_3*.json


# rm core.*
# rm megatron_config_3*.json

# Kill previous job and setup next job pickup
# bkill 3178176
# python /gpfs/alpine/csc499/scratch/btherien/experiments_phase_2/7-1B_future_launch.py --job-id $LSB_JOBID --sleep-time 9 &

python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs pythia/410M.yml summit_setup.yml vaibhav_schedules/inf_inverse_sqrt_schedules/adam_infinv_lr3e-4_3e-5_wu-0.01.yml

# Write the hostfile for this job
# bash /gpfs/alpine/csc499/scratch/btherien/write_hostfile.sh
# export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/btherien/hostfiles/$LSB_JOBID-hosts


# python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs pythia_410m_llama_setup_resume.yml iclr_models/7_1B.yml datasets/val/pile_slimp.yml datasets/train/pile_train.yml load/resume_1-2e-4_001_7-1B_pile_PT.yml schedules/adam_cosine_lr1-2e-4_1-2e-5_wu-001.yml
exit 0
