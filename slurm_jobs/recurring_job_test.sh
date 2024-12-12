#!/bin/bash
#SBATCH --output=/path/to/slurm_logs/output/slurm-%j.out
#SBATCH --error=/path/to/slurm_logs/error/slurm-%j.err
#SBATCH -J checkpoint_test
#SBATCH -q debug
#SBATCH -t 0:02:00  # Short runtime for testing
#SBATCH --mail-type=END,FAIL
#SBATCH -N 1
#SBATCH -A bif151

# Define the checkpoint directory
CHECKPOINT_DIR="/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/checkpoints"

# Count the number of existing JOB-* directories
JOB_COUNT=$(ls -d ${CHECKPOINT_DIR}/JOB-* 2>/dev/null | wc -l)

# Exit if this is the fourth job (as we're only running 3 times for testing)
if [[ $JOB_COUNT -ge 3 ]]; then
  echo "Max job runs reached. Exiting."
  exit 0
fi

# Create a new JOB-* directory with a simulated job ID and parameters
NEW_JOB_ID=$((JOB_COUNT + 1))  # Increment job ID for each run
NEW_CHECKPOINT_PATH="${CHECKPOINT_DIR}/JOB-${SLURM_JOB_ID}_410M-Llama2_it-190735_wu-0.01_mxlr-2e-06_mnlr-8e-08_sch-cosine_tr-dclm-train_resume"

mkdir -p "${NEW_CHECKPOINT_PATH}"
echo "Created checkpoint directory: ${NEW_CHECKPOINT_PATH}"

# Update load_ckpt.yml to load from the latest checkpoint
sed -i "s|\"load\":.*|\"load\": \"${NEW_CHECKPOINT_PATH}\",|" /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/configs/load_checkpoints/dclm/cosine/load_scratch.yml
echo "Updated load_ckpt.yml to load from: ${NEW_CHECKPOINT_PATH}"

# Print the contents of load_ckpt.yml for verification
cat /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/configs/load_checkpoints/dclm/cosine/load_scratch.yml

# Resubmit the job if less than 3 runs have completed
if [[ $JOB_COUNT -lt 2 ]]; then  # Resubmit until the third job completes
  sbatch $0
else
  echo "Test completed after 3 runs."
fi
