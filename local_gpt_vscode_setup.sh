export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

source write_hostfile.sh

module load StdEnv/2020  gcc/9.3.0  cuda/11.7 python/3.9 arrow/13.0.0
source ~/llm_proj_cuda117/bin/activate

echo $CUDA_VISIBLE_DEVICES
# export SLURM_TMPDIR=/Tmp/slurm.$SLURM_JOB_ID.0
# export CUDA_VISIBLE_DEVICES=0 Uncomment this if on a single ampere GPU 
