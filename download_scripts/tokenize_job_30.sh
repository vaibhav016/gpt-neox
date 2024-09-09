#!/bin/bash
#BSUB -nnodes 1
#BSUB -W 2:00
#BSUB -q batch
#BSUB -o /gpfs/alpine2/csc565/scratch/vaibhav_016/test/tokenize_logs/gpt_neox_out.%J
#BSUB -e /gpfs/alpine2/csc565/scratch/vaibhav_016/test/tokenize_logs/gpt_neox_err.%J
#BSUB -J tokenize_30
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

# Set initial values
start=30

endd=40

# end=$((start + increment))

python download_scripts/tokenize_the_dataset.py \
        --input-path data/oscar_german_2301 \
        --output-path data/tokenized_oscar_german_2301 \
        --start $start --end $endd

# Loop until end reaches 100
# while [ $end -le 100 ]; do
#     # Run the jsrun command
#     jsrun -n1 -c1 python download_scripts/tokenize_the_dataset.py \
#         --input-path data/oscar_german_2301 \
#         --output-path data/tokenized_oscar_german_2301 \
#         --start $start --end $end  & 

#     # Update start and end values
#     start=$end
#     end=$((start + increment))
# done 

# wait

# python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs pythia_410m_llama_setup_resume.yml iclr_models/7_1B.yml datasets/val/pile_slimp.yml datasets/train/pile_train.yml load/resume_1-2e-4_001_7-1B_pile_PT.yml schedules/adam_cosine_lr1-2e-4_1-2e-5_wu-001.yml
exit 0




