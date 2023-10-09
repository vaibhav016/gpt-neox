from megatron.data.indexed_dataset import MMapIndexedDataset

import numpy as np
import json
import time
import os
import sys
import multiprocessing
from tqdm import tqdm

class SilenceStdout:
    def __enter__(self):
        self.saved_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, *args):
        sys.stdout = self.saved_stdout

def get_num_tokens(path_to_dataset):
    with SilenceStdout():
        ds = MMapIndexedDataset(path_to_dataset)
        return int(np.sum(ds._index._sizes))

def process_file(f):
    dataset_path = os.path.join(data_path, dir_name, f[:-len('.bin')])
    return dataset_path, get_num_tokens(dataset_path)

data_path = '/gpfs/alpine/csc499/scratch/btherien/data/pile_replay_shards/tokenized_splits'

dirs = os.listdir(data_path)
dataset_map = {}

for dir_name in dirs:
    start_time = time.time()  # start timing
    print("Starting dir:", dir_name, "...")
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    files = [x for x in os.listdir(os.path.join(data_path, dir_name)) if x.endswith('.bin')]
    inner_map = dict(pool.map(process_file, files))
    pool.close()
    pool.join()
    dataset_map[dir_name] = inner_map

    end_time = time.time()  # end timing
    print(f"Time for {dir_name}: {end_time - start_time} seconds")

print("done Looping")
print(dataset_map)

# save the dictionary to a json file
with open('dataset_map_pile_replay.json', 'w') as json_file:
    json.dump(dataset_map, json_file)