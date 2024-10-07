import glob
import argparse
import os
from tqdm import tqdm
from multiprocessing import Pool, Lock

lock = Lock()

# Function to combine JSONL files
# def combine_jsonl_files(input_directory, output_file):
#     # Path to collect all .jsonl files in the directory
#     jsonl_files_path = os.path.join(input_directory, "*.jsonl")

#     # Combine all files
#     with open(output_file, 'w') as outfile:
#         for jsonl_file in tqdm(glob.glob(jsonl_files_path)):
#             with open(jsonl_file, 'r') as infile:
#                 for line in infile:
#                     outfile.write(line)


# Function to combine all jsonl files
def combine_jsonl_files(input_directory, output_file):
    global output_file  # Make the output file path accessible globally
    # Path to collect all .jsonl files in the directory
    jsonl_files_path = os.path.join(input_directory, "*.jsonl")
    jsonl_files = glob.glob(jsonl_files_path)

    # Ensure the output file is empty before we start appending to it
    open(output_file, 'w').close()

    # Use multiprocessing Pool to process files in parallel
    with Pool() as pool:
        pool.map(read_and_write_jsonl, jsonl_files)


# Function to read and write a single jsonl file
def read_and_write_jsonl(jsonl_file):
    with open(jsonl_file, 'r') as infile:
        lines = infile.readlines()

    # Write the lines to the output file with thread-safe locking
    with lock:
        with open(output_file, 'a') as outfile:
            outfile.writelines(lines)


# Main function to handle command line arguments
def main():
    parser = argparse.ArgumentParser(description="Combine multiple JSONL files into one.")
    parser.add_argument("input_directory", type=str, help="Directory containing JSONL files")
    parser.add_argument("output_file", type=str, help="Output JSONL file path")
    
    args = parser.parse_args()

    # Combine JSONL files from the input directory
    combine_jsonl_files(args.input_directory, args.output_file)
    print("***************** Combined all jsonl files into a single file ******************")

if __name__ == "__main__":
    main()