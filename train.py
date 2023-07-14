# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train"""
from megatron.neox_arguments import NeoXArgs
from megatron.training import pretrain

import os
import numpy as np

if __name__ == "__main__":
    neox_args = NeoXArgs.consume_neox_args()
    neox_args.configure_distributed_args()
    neox_args.build_tokenizer()  # tokenizer needs to be build in training in order to set the padding vocab

    dir_str = "JOB-{}_{}-iters-{}_warmup-{}_max-lr-{}_min-lr-{}_{}".format(
        os.environ['LSB_JOBID'],
        neox_args.identifier_string,
        neox_args.train_iters, 
        neox_args.warmup, 
        neox_args.optimizer['params']['lr'], 
        neox_args.min_lr, 
        "finetune" if neox_args.finetune else "pretrain")
    
    if 'pile' in neox_args.train_data_paths[0]:
        dir_str += "_pile_"
    elif 'red_pajama' in neox_args.train_data_paths[0]:
        dir_str += "_red_pajama_"
    elif 'tokenized300B' in neox_args.train_data_paths[0]:
        dir_str += "_slim_pajama_"

    if neox_args.load.split('/')[-1].startswith('JOB'):
        dir_str += 'resume'
    else:
        dir_str += neox_args.load.split('/')[-1]
    print(dir_str)
    
    # exit(0)

    
    neox_args.tensorboard_dir = os.path.join(neox_args.tensorboard_dir, dir_str)
    neox_args.save = os.path.join(neox_args.save, dir_str)
    print("NEOX ARGS tensorboard_dir: ", neox_args.tensorboard_dir)
    print("NEOX ARGS save: ", neox_args.save)
    # exit(0)
    neox_args.initialize_tensorboard_writer()  # is initialized if tensorboard directory is defined

    pretrain(neox_args=neox_args)
    
