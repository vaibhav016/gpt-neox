#!/bin/bash

# Infinite loop to submit jobs
counter=0

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


echo "******** Done with 3 *********"

counter=0

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
echo "******** Done with 4 *********"



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

echo "******** Done with 6 *********"


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

echo "******** Done with 7 *********"

