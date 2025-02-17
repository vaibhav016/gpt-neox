#!/bin/bash

# Infinite loop to submit jobs
while true; do
    echo "Running debug partition..."
    job_id=$(sbatch test_high_7_debug.sh | awk '{print $NF}')
    echo "Job submitted with ID: $job_id"

    # Wait for the job to complete
    echo "Waiting for job $job_id to complete..."
    while squeue -j "$job_id" &> /dev/null; do
        sleep 1  # Check every 10 seconds
    done
    echo "Job $job_id completed."
    sleep 5
done
