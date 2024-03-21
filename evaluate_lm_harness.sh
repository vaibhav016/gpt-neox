#!/bin/bash
#BSUB -nnodes 276
#BSUB -W 0:15
#BSUB -q debug
#BSUB -o eval_gpt_neox_out.%J
#BSUB -e eval_gpt_neox_err.%J
#BSUB -J eval_gpt_neox
#BSUB -alloc_flags gpudefault
#BSUB -P CSC499

source /gpfs/alpine/csc499/scratch/$(whoami)/setup.sh

export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/$(whoami)/cache

# Move to the gpt-neox install
TRAIN_PATH=/gpfs/alpine/csc499/scratch/$(whoami)/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/$(whoami)/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/$(whoami)/hostfiles/$LSB_JOBID-hosts
export HF_DATASETS_CACHE=/gpfs/alpine/csc499/proj-shared/hf_eval_datasets
# export HF_DATASETS_CACHE=/gpfs/alpine/csc499/scratch/$(whoami)/DATACACHE


# List eval tasks here, separated with whitespaces. E.g. EVAL_TASKS="hellaswag arc_easy"
EVAL_TASKS="hellaswag"

export HF_DATASETS_OFFLINE=0
python check_and_download_eval_tasks.py --eval_tasks $EVAL_TASKS #hellaswag #arc_challenge truthfulqa_mc hendrycksTest-*
export HF_DATASETS_OFFLINE=1



python $TRAIN_PATH/deepy.py $TRAIN_PATH/evaluate.py \
        --conf_dir $TRAIN_PATH/configs \
                pythia_410m_llama_setup_resume.yml \
                iclr_models/7_1B.yml \
                datasets/val/pile_slimp.yml \
                datasets/train/pile+slim_pajama_300B_each.yml \
                load/resume_3e-4_001_7-1B_pile_PT.yml \
                schedules/adam_cosine_lr1-2e-4_1-2e-5_wu-001.yml
        --eval_tasks $EVAL_TASKS #hellaswag #arc_challenge truthfulqa_mc hendrycksTest-*
