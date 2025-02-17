###### cosine ######
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/cos/chain_high_llama.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/cos/chain_low_llama.sh

###### inf_cos ######
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/chain_high_llama.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/chain_low_llama.sh



####### High ablations #######
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/high/chain_high_llama_const_3.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/high/chain_high_llama_const_4.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/high/chain_high_llama_const_6.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/high/chain_high_llama_const_7.sh

###### Low ablations #######
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/low/chain_low_llama_const_1p.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/low/chain_low_llama_const_1p25.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/low/chain_low_llama_const_1p75.sh
sbatch /ccs/home/vaibhav_016/bif151/scratch/vaibhav_016/neox/gpt-neox/slurm_jobs/dclm/3rd_dec/inf_cos/ablations/low/chain_low_llama_const_2p.sh

