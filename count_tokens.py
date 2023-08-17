from megatron.data.indexed_dataset import MMapIndexedDataset
import numpy as np
import json
import sys
import os
import argparse
from tqdm import tqdm
import os.path as osp

print("Starting Count...")

class SilenceStdout:
    def __enter__(self):
        self.saved_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
    def __exit__(self, *args):
        sys.stdout = self.saved_stdout


# get the first argument as a file name, and an output file
parser = argparse.ArgumentParser()
parser.add_argument("--outer-dir", 
                    help="the file name to read", 
                    default='/gpfs/alpine/csc499/scratch/btherien/data/SlimPajama_split/tokenized_train')

parser.add_argument("--jsonl-file-name", 
                    help="the file name to read", 
                    default='/gpfs/alpine/csc499/scratch/btherien/data/SlimPajama_split/test/ArXiv/data_0_time1690516559_slimpajama0.jsonl')

parser.add_argument("--output-file", 
                    help="the file name to write", 
                    default='token_counts.txt')

args = parser.parse_args()

total_size = 0
for dir in os.listdir(args.outer_dir):
    if not osp.isdir(osp.join(args.outer_dir, dir)):
        continue
    dir_size = 0
    print("Starting dir:", dir, "...")
    for f in tqdm(os.listdir(os.path.join(args.outer_dir, dir))):

        with SilenceStdout():
            if f.endswith('.bin'):
                try:
                    ds = MMapIndexedDataset(os.path.join(args.outer_dir, dir,f[:-len('.bin')]))
                    dir_size += np.sum(ds._index._sizes)
                except AttributeError as e:
                    print(e)
                    print("Above exception occured for file:", os.path.join(args.outer_dir, dir,f[:-len('.bin')]))

    total_size += dir_size
    print("Count:",dir, dir_size/1e9, "B")

print("total", total_size/1e9, "B")

# print(ds)
# print(np.sum(ds._index._sizes))
# print(np.sum(ds._index._sizes)/1e9,"B")


# with open(args.jsonl_file_name) as f:
#     total_chars, total_nonw, total_words = 0, 0, 0
#     for line in f:
#         data = json.loads(line)
#         text = data['text']
#         total_chars += len(text)
#         total_nonw += len(text.replace(" ", ""))
#         total_words += len(text.split(' '))

# print("total_chars:",total_chars)
# print("total_char_no_whitespace:", total_nonw )
# print("total_words:",total_words )


