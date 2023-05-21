# Set up the environment
source /gpfs/alpine/csc499/scratch/kshitijkg/setup.sh
source activate continued_training
# The default cache location is read-only on Summit. Redirect it to somewhere in your scratch dir
export TORCH_EXTENSIONS_DIR=/gpfs/alpine/csc499/scratch/kshitijkg/cache

# Move to the gpt-neox install
TRAIN_PATH=/gpfs/alpine/csc499/scratch/kshitijkg/continued_training/gpt-neox
cd $TRAIN_PATH

# Write the hostfile for this job
bash /gpfs/alpine/csc499/scratch/kshitijkg/write_hostfile.sh
export DLTS_HOSTFILE=/gpfs/alpine/csc499/scratch/kshitijkg/hostfiles/$LSB_JOBID-hosts


python $TRAIN_PATH/deepy.py $TRAIN_PATH/train.py \
         --conf_dir $TRAIN_PATH/configs llama/410M.yml pythia_llama_setup.yml
