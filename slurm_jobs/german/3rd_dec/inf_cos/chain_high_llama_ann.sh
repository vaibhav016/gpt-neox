#!/bin/bash
#SBATCH --output=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/german/inf_cos/high/output/slurm-%j.out            
#SBATCH --error=/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/slurm_logs/german/inf_cos/high/error/slurm-%j.err
#SBATCH --mail-user=vaibhavsinghfcos@gmail.com
#SBATCH -A bif151
#SBATCH -J gr.ic.hi
#SBATCH -t 2:00:00
#SBATCH -q debug
#SBATCH -N 32
#SBATCH --mail-type=BEGIN,END,FAIL

SECONDS_BEFORE_EXIT=7020  # 1 hour 57 minutes (for a 2-hour job)

# Recompute the host file
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12803
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/write_hostfile.sh
export DLTS_HOSTFILE=/lustre/orion/bif151/scratch/vaibhav_016/neox/gpt-neox/hostfiles/hosts_$SLURM_JOBID

# Setup script
source /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/setup.sh

cd /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox

# Define variables
CHECKPOINT_DIR="checkpoints_3rd_dec/german/inf_cos/high"

# Get the most recent checkpoint path
LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/JOB-* | tail -n 1)

# If there’s a previous checkpoint, use it. Otherwise, start from scratch.
if [[ -n "$LATEST_CHECKPOINT" ]]; then
  CHECKPOINT_PATH="${LATEST_CHECKPOINT}"
else
  CHECKPOINT_PATH="${CHECKPOINT_DIR}"
fi



# # Get the most recent checkpoint path
# LATEST_CHECKPOINT="${CHECKPOINT_DIR}/JOB-2935455__it-47684_wu-0_mxlr-0.0005_mnlr-0.0_sch-constant_tr-german-train_resume"

# CHECKPOINT_PATH="${LATEST_CHECKPOINT}"


# Update load_low.yml with the checkpoint path
sed -i "s|\"load\":.*|\"load\": \"${CHECKPOINT_PATH}\",|" configs/load_checkpoints/german/3rd_dec/inf_cos/load_high.yml

# Run the training script
python deepy.py train.py --conf_dir configs llama2/410M.yml dataset_config/german/german_config_scratch.yml schedulers/const_inf_cos/annealing/adam_constant_lr5e-4_wu-0_high.yml load_checkpoints/german/3rd_dec/inf_cos/load_high.yml &                        
PYTHON_PID=$!

# Preemption timer to send SIGUSR1 a few minutes before job end time
(sleep $SECONDS_BEFORE_EXIT && kill -SIGUSR1 $PYTHON_PID) &

# Wait for the Python process to finish
wait $PYTHON_PID
PYTHON_EXIT_STATUS=$?

# Echo the exit status of the Python script
echo "Python script exit status: $PYTHON_EXIT_STATUS"
sed -i "s|\"finetune\":.*|\"finetune\": false,|" configs/load_checkpoints/german/3rd_dec/inf_cos/load_high.yml

Check if the script finished successfully or needs resubmission
if [ $PYTHON_EXIT_STATUS -ne 0 ]; then
  echo "Resubmitting job..."
  sbatch $0  # Re-submit this script to chain jobs
else
  echo "Training completed."
fi

exit 0

