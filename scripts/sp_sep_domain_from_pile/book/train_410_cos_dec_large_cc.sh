#!/bin/bash
#BSUB -nnodes 92
#BSUB -W 2:30
#BSUB -q batch
#BSUB -o /gpfs/alpine2/csc565/scratch/vaibhav_016/test/training_logs/gpt_neox_out.%J
#BSUB -e /gpfs/alpine2/csc565/scratch/vaibhav_016/test/training_logs/gpt_neox_err.%J
#BSUB -J sp_sep_cc
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

cd test/gpt-neox/

python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs pythia/410M.yml vaibhav_sp_sep_dom/book.yml vaibhav_schedules/cosine_decay_schedules/adam_cosine_lr3e-3_3e-5_wu-0.01.yml checkpoint_paths/sp_sep_dom_cos_high.yml

exit 0
