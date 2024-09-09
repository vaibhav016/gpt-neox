from datasets import Dataset
import json
import os.path as osp
import os
import sys

from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed



# arrow_path = 'PATH TO EXISTING DIRECTORY CONTAINING DATA IN .ARROW FORMAT'
# jsonl_path = 'NEW DIRECTORY FOR THE JSONL DATA'

def process_arrow_file(arr_name, arrow_path, jsonl_path):
    arrow_filepath = osp.join(arrow_path, arr_name)
    dataset = Dataset.from_file(arrow_filepath)
    output_file = osp.join(jsonl_path, arr_name.replace('.arrow', '.jsonl'))
    with open(output_file, 'w') as f:
        for example in dataset:
            f.write(json.dumps(example) + '\n')

def main(arrow_path, jsonl_path, max_workers=4):
    # Ensure the output directory exists
    os.makedirs(jsonl_path, exist_ok=True)
    arrow_files = os.listdir(arrow_path)
    arrow_files = [x for x in arrow_files if x.endswith('.arrow')]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_arrow_file, arr_name, arrow_path, jsonl_path) for arr_name in arrow_files]
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

if __name__ == "__main__":
    # Accepting arguments from the command line
    arrow_path = sys.argv[1]  # First argument: path to .arrow files
    jsonl_path = sys.argv[2]  # Second argument: path to output .jsonl files

    # Ensure the output directory exists
    os.makedirs(jsonl_path, exist_ok=True)
    arrow_files = os.listdir(arrow_path)

    max_workers = os.cpu_count()  # Use all available CPUs
    print(" CPU WORKERS -> ", max_workers)
    main(arrow_path, jsonl_path, max_workers)

    print("************* Conversion done ***********")