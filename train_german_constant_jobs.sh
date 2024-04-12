#!/bin/bash

####### Medium lr######
sbatch sbatch_train_german_constant_medium_val_pile.sh 
sbatch sbatch_train_german_constant_medium_val_ger.sh 

###### Low LR ########
sbatch sbatch_train_german_constant_low_val_pile.sh 
sbatch sbatch_train_german_constant_low_val_ger.sh 

###### High LR ######
sbatch sbatch_train_german_constant_high_val_pile.sh 
sbatch sbatch_train_german_constant_high_val_ger.sh 