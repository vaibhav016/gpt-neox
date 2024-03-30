module load anaconda/3
conda activate llm_project
module load cuda/12.1.1 
export SLURM_TMPDIR=/Tmp/slurm.$SLURM_JOB_ID.0
export CUDA_VISIBLE_DEVICES=0