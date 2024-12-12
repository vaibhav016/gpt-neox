#!/bin/bash

CHECKPOINT_DIR="/ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/gpt-neox/checkpoints"

# Get the most recent checkpoint path
LATEST_CHECKPOINT=$(ls -d ${CHECKPOINT_DIR}/JOB-* | tail -n 1)

# If there’s a previous checkpoint, use it. Otherwise, start from scratch.
if [[ -n "$LATEST_CHECKPOINT" ]]; then
  CHECKPOINT_PATH="${LATEST_CHECKPOINT}"
else
  CHECKPOINT_PATH="${CHECKPOINT_DIR}"
fi

# Update load_ckpt.yml with the checkpoint path
sed -i "s|\"load\":.*|\"load\": \"${CHECKPOINT_PATH}\",|" load_scratch.yml

