import os


for m in ['pythia-410m-v0','pythia-1b-v0','pythia-2.8b-v0']:
    for r in range(10):
        r = 143000 - r * 1000
        cmd = "CUDA_VISIBLE_DEVICES=0 python tools/convert_hf_to_sequential.py     --output-dir /gpfs/alpine/csc499/scratch/btherien/neox_converted/mp1_pp1/pythia/410m     --cache-dir /gpfs/alpine/csc499/proj-shared/hf_checkpoints     --config configs/pythia/410M.yml configs/local_setup.yml     --test     --download     --hf-model-name {}     --revision {}".format(
            m, r
        )
        print(cmd)
        os.system(cmd)