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

python deepy.py train.py --conf_dir configs llama2/410M_testing.yml dataset_config/code/code_config_scratch.yml schedulers/cos/code/adam_cosine_lr1e-3_3e-5_wu-0.01 load_checkpoints/code/3rd_dec/cos/load_high.yml 