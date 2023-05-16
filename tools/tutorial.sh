# /bin/bash




CUDA_VISIBLE_DEVICES=0 python tools/convert_hf_to_sequential.py \
    --hf-model-name pythia-410m-v0 \
    --revision 143000 \
    --output-dir /gpfs/alpine/csc499/scratch/btherien/neox_converted/mp1_pp1/pythia/410m \
    --cache-dir /gpfs/alpine/csc499/proj-shared/hf_checkpoints \
    --config configs/pythia/410M.yml configs/local_setup.yml \
    --test