from datasets import Dataset
import json
import os.path as osp
import os
import sys

from tqdm import tqdm


# arrow_path = 'PATH TO EXISTING DIRECTORY CONTAINING DATA IN .ARROW FORMAT'
# jsonl_path = 'NEW DIRECTORY FOR THE JSONL DATA'


# Accepting arguments from the command line
arrow_path = sys.argv[1]  # First argument: path to .arrow files
jsonl_path = sys.argv[2]  # Second argument: path to output .jsonl files

# Ensure the output directory exists
os.makedirs(jsonl_path, exist_ok=True)
arrow_files = os.listdir(arrow_path)

arrow_files = [x for x in arrow_files if x.endswith('.arrow')]
import os.path as osp
for arr_name in tqdm(arrow_files):
    arrow_filepath = osp.join(arrow_path,arr_name)
    dataset = Dataset.from_file(arrow_filepath)
    output_file = osp.join(jsonl_path,arr_name.replace('.arrow', '.jsonl'))
    with open(output_file, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')


print("************* Conversion done ***********")