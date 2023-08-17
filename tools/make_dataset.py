
import numpy as np
import argparse
from tqdm import tqdm

import json 

import os
import sys
import argparse

import os.path as osp

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)

from megatron.data import indexed_dataset


def get_dataset_of_mixture(dataset_map, mixture_map, size):
    """
    dataset_map: mapping from dataset to subdatasets and their sizes
    mixture_map: the percentage of this dataset that should be sampled from each source
    size: the total size of the dataset in billions of tokens
    """
    assert set(dataset_map.keys()) ==  set(mixture_map.keys()), "Keys don't match between the dataset and mixture"
    assert sum(list(mixture_map.values())) == 100.0, "mixture does not sum to 100"
    
    files_to_use = {}
    for domain, files in dataset_map.items():
        file_names = np.array(list(files.keys()))
        np.random.shuffle(file_names)
        file_values = [files[x] for x in file_names]
        
        required_tokens = ( mixture_map[domain] / 100.0 ) * size
        # print("Collecting {}B tokens for {}".format(required_tokens, domain))

        arr = np.array(file_values)
        cs = np.cumsum(arr) - required_tokens * 1e9
        gz = np.where( cs >= 0)[0]
        if len(gz) == 0:
            # need all 
            files_to_use[domain] = (np.sum(arr)/1e9,file_names)
        elif np.sum(arr[:gz[0]]) == required_tokens:
            # if required tokens was found
            temp_files = list(file_names[:gz[0]])
            files_to_use[domain] = (required_tokens/1e9,temp_files,)
        else:
            # if below amount
            temp_files = list(np.array(file_names)[:gz[0]])
            temp_count = np.sum(arr[:gz[0]])
            idx = np.argmin( np.abs( (arr[gz[0]:] + temp_count) - required_tokens) )
            if np.abs( (arr[gz[0]+idx] + temp_count) - required_tokens) < required_tokens - temp_count:
                files_to_use[domain] = (arr[gz[0]+idx] + temp_count/1e9, list(temp_files) + [file_names[gz[0]+idx]],) 
            else:
                files_to_use[domain] = (temp_count/1e9,temp_files,)
                
    return files_to_use


def copy_and_merge(dataset,old_dataset_path,new_dataset_path):
    for dir,v in dataset.items():
        for file in dataset.keys():
            old_file = osp.join(old_dataset_path,dir,file)
            new_file = osp.join(new_dataset_path,dir,file)
            os.system("rsync {} {}".format(old_file,new_file))


def merge_to_new_dataset(prefixes, input_dir, output_dir):
    builder = None
    for prefix in sorted(prefixes):
        if builder is None:
            dataset = indexed_dataset.make_dataset(
                os.path.join(input_dir, prefix), "infer"
            )

            if isinstance(dataset, indexed_dataset.MMapIndexedDataset):
                builder = indexed_dataset.MMapIndexedDatasetBuilder(
                    output_dir + ".bin", dtype=dataset._index.dtype
                )
            else:
                builder = indexed_dataset.IndexedDatasetBuilder(
                    output_dir + ".bin"
                )

            del dataset

        builder.merge_file_(os.path.join(input_dir, prefix))

    builder.finalize(output_dir + ".idx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig-dataset-dir",type=str)
    parser.add_argument("--new-dataset-dir",type=str)
    parser.add_argument("--size",type=int)
    parser.add_argument("--dryrun",action='store_true')
    parser.add_argument("--all",action='store_true')
    args = parser.parse_args()

    
    with open('/gpfs/alpine/csc499/scratch/btherien/gpt-neox/dataset_map.json') as f:
        dataset_map = json.load(f)


    if args.all:
        dataset_size = {k:sum(list(v.values()))/1e9 for k,v in dataset_map.items()}
        new_dataset = {k:(dataset_size[k],list(v.keys())) for k,v in dataset_map.items()}

        if args.dryrun:
            for k,v in dataset_size.items():
                print(k,v)
            total_size = sum(list(dataset_size.values()))
            print("Total size: {}B".format(total_size))
            exit(0)

        args.size = int(sum(list(dataset_size.values())))
        args.new_dataset_dir = args.new_dataset_dir + f"_{args.size}B"

    else:
        a = {k:sum(list(v.values()))/1e9 for k,v in dataset_map.items()}
        mixture_map = {k:100*v/sum(a.values()) for k,v in a.items()}

        obj_size = 5
        obj_dist = 1
        iteration = 0
        min_dist = 1
        min_size = 5
        print("running random search for dataset...")
        while(obj_size > 1 and obj_dist > 0.02):
            new_dataset = get_dataset_of_mixture(dataset_map, mixture_map, size=args.size)
            dataset_size = {k:v[0] for k,v in new_dataset.items()}
            total_size = sum(list(dataset_size.values()))
            obj_size = np.abs(total_size - args.size)
            dist = {k:v/total_size for k,v in dataset_size.items()}
            obj_dist = sum([np.abs(dist[k] - mixture_map[k]/100) for k in dist.keys()])
            iteration += 1
            min_dist = min(min_dist, obj_dist)
            min_size = min(min_size, obj_size)
            if iteration % 1000 == 0:
                print("Iteration: {}, min_dist: {}, min_size: {}".format(iteration,min_dist, min_size))

        if args.dryrun:
            for k,v in dataset_size.items():
                print(k,v)
            total_size = sum(list(dataset_size.values()))
            print("Total size: {}B".format(total_size))
            print("Objective size:", obj_size)
            print("Objective dist:", obj_dist)
            exit(0)

    
    os.makedirs(args.new_dataset_dir, exist_ok=True)
    with open(osp.join(args.new_dataset_dir,'dataset_size_and_paths.json'),'w') as f:
        json.dump(new_dataset,f)

    with open(osp.join(args.new_dataset_dir,'dataset_size.json'),'w') as f:
        json.dump(dataset_size,f)


    for dir,v in new_dataset.items():
        output_dir = osp.join(args.new_dataset_dir,dir,dir)
        os.makedirs(output_dir, exist_ok=True)
        merge_to_new_dataset(prefixes=[x.split("/")[-1] for x in v[1]], 
                             input_dir=osp.join(args.orig_dataset_dir,dir), 
                             output_dir=output_dir)
        


"""
python tools/make_dataset.py \
--orig-dataset-dir /gpfs/alpine/csc499/scratch/btherien/data/SlimPajama_split/tokenized_train \
--new-dataset-dir /gpfs/alpine/csc499/proj-shared/incite_datasets/slim_pajama_split/tokenized_train_75B \
--size 75 \
--dryrun

python tools/make_dataset.py --orig-dataset-dir /gpfs/alpine/csc499/scratch/btherien/data/SlimPajama_split/tokenized_train \
--new-dataset-dir /gpfs/alpine/csc499/proj-shared/incite_datasets/slim_pajama_split/tokenized_train_150B \
--size 150 \

"""