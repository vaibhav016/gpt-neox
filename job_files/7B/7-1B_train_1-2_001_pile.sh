#!/bin/bash
#BSUB -nnodes 276
#BSUB -W 12:00
#BSUB -q batch
#BSUB -o /gpfs/alpine/csc499/scratch/btherien/training_logs/gpt_neox_out.%J
#BSUB -e /gpfs/alpine/csc499/scratch/btherien/training_logs/gpt_neox_err.%J
#BSUB -J pile_1-2e-4_001_7-1B_PT_276
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499
#BSUB -N btherien@uwaterloo.ca
#BSUB -B btherien@uwaterloo.ca

# clean up nodes
jsrun pkill python

# Set up the environment
source /gpfs/alpine/csc499/scratch/btherien/setup.sh
/gpfs/alpine/csc499/scratch/btherien/miniconda3/bin/activate


export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/btherien/latest_install/cache
# rm -r $TORCH_EXTENSIONS_DIR

# Move to the gpt-neox install
export TRAIN_PATH=/gpfs/alpine/csc499/scratch/btherien/gpt-neox
cd $TRAIN_PATH

rm core.*
rm megatron_config_3*.json

# Kill previous job and setup next job pickup
bkill 3178176
python /gpfs/alpine/csc499/scratch/btherien/experiments_phase_2/7-1B_future_launch.py --job-id $LSB_JOBID --sleep-time 9 &

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/btherien/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/btherien/hostfiles/$LSB_JOBID-hosts


python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py --conf_dir $TRAIN_PATH/configs pythia_410m_llama_setup_resume.yml iclr_models/7_1B.yml datasets/val/pile_slimp.yml datasets/train/pile_train.yml load/resume_1-2e-4_001_7-1B_pile_PT.yml schedules/adam_cosine_lr1-2e-4_1-2e-5_wu-001.yml
