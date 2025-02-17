#!/bin/bash

# Infinite loop to submit jobs
counter=0
while true; do
    echo "Running debug partition..."
    job_id=$(sbatch chain_high_llama_const_3_ann_debug.sh | awk '{print $NF}')
    echo "Job submitted with ID: $job_id"

    # Wait for the job to complete
    echo "Waiting for job $job_id to complete..."
    while squeue -j "$job_id" &> /dev/null; do
        sleep 1  
    done
    echo "Job $job_id completed."
    sleep 5
    counter=$((counter+1))
    if [ $counter -eq 1 ]; then
        break
    fi
done

echo "******** Done with 3 *********"

counter=0
while true; do
    echo "Running debug partition..."
    job_id=$(sbatch chain_high_llama_const_4_ann_debug.sh | awk '{print $NF}')
    echo "Job submitted with ID: $job_id"

    # Wait for the job to complete
    echo "Waiting for job $job_id to complete..."
    while squeue -j "$job_id" &> /dev/null; do
        sleep 1  
    done
    echo "Job $job_id completed."
    sleep 5
    counter=$((counter+1))
    if [ $counter -eq 2 ]; then
        break
    fi
done
echo "******** Done with 4 *********"

counter=0
while true; do
    echo "Running debug partition..."
    job_id=$(sbatch chain_high_llama_const_6_ann_debug.sh | awk '{print $NF}')
    echo "Job submitted with ID: $job_id"

    # Wait for the job to complete
    echo "Waiting for job $job_id to complete..."
    while squeue -j "$job_id" &> /dev/null; do
        sleep 1  
    done
    echo "Job $job_id completed."
    sleep 5
    counter=$((counter+1))
    if [ $counter -eq 2 ]; then
        break
    fi
done

echo "******** Done with 6 *********"

counter=0
while true; do
    echo "Running debug partition..."
    job_id=$(sbatch chain_high_llama_const_7_ann_debug.sh | awk '{print $NF}')
    echo "Job submitted with ID: $job_id"

    # Wait for the job to complete
    echo "Waiting for job $job_id to complete..."
    while squeue -j "$job_id" &> /dev/null; do
        sleep 1  
    done
    echo "Job $job_id completed."
    sleep 5
    counter=$((counter+1))
    if [ $counter -eq 2 ]; then
        break
    fi
done

echo "******** Done with 7 *********"

