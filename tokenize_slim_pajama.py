import os
import argparse

#get args for min and max file number
parser = argparse.ArgumentParser()

parser.add_argument('--filepath', type=str, default="/gpfs/alpine/csc499/scratch/btherien/data/SlimPajama_split/test")
parser.add_argument('--tok-filepath', type=str, default="/gpfs/alpine/csc499/scratch/btherien/data/SlimPajama_split/tokenized_test")
parser.add_argument('--dryrun', action='store_true')
parser.add_argument('--start', type=float,)
parser.add_argument('--end', type=float,)
parser.add_argument('--merge', action='store_true')
parser.add_argument('--dir', type=str, default=None)
args = parser.parse_args()

if args.dir is not None:
    jsonl_files = sorted([x for x in os.listdir(os.path.join(args.filepath, args.dir)) if x.endswith('.jsonl')])
    save_dir = os.path.join(args.tok_filepath,args.dir)
    
    
    
    print("Making dir: {}".format(save_dir))
    if not args.dryrun:
        os.makedirs(save_dir, exist_ok=True)


    print(len(jsonl_files))

    start = int(args.start / 100 * len(jsonl_files))
    end = int(args.end / 100 * len(jsonl_files))
    print("using start and end: {} {}".format(start, end))
    
    for file in jsonl_files[start:end]:
        command = 'python tools/preprocess_data.py \
            --input {} \
            --output-prefix {} \
            --vocab ./data/20B_tokenizer.json \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --jsonl-keys text \
            --workers 64'.format(
                    os.path.join(args.filepath, args.dir, file),
                    os.path.join(save_dir,file[:-len('.jsonl')])
                )
        print(command)
        if not args.dryrun:
            os.system(command)

    exit(0)


for dir in os.listdir(args.filepath):
    jsonl_files = sorted([x for x in os.listdir(os.path.join(args.filepath, dir)) if x.endswith('.jsonl')])
    save_dir = os.path.join(args.tok_filepath,dir)
    
    print("Making dir: {}".format(save_dir))
    if not args.dryrun:
        os.makedirs(save_dir, exist_ok=True)


    if args.merge:
        command = 'python tools/preprocess_data.py \
            --input {} \
            --output-prefix {} \
            --vocab ./data/20B_tokenizer.json \
            --tokenizer-type HFTokenizer \
            --append-eod \
            --jsonl-keys text \
            --workers 64'.format(
                    ",".join([os.path.join(args.filepath, dir, file) for file in jsonl_files]),
                    os.path.join(save_dir,"tokenized_{}".format(dir.lower()))
                )
        print(command)
        if not args.dryrun:
            os.system(command)
    else:
        start = int(args.start / 100 * len(jsonl_files))
        end = int(args.end / 100 * len(jsonl_files))
        print("using start and end: {} {}".format(start, end))

        for file in jsonl_files[start:end]:
            command = 'python tools/preprocess_data.py \
                --input {} \
                --output-prefix {} \
                --vocab ./data/20B_tokenizer.json \
                --tokenizer-type HFTokenizer \
                --append-eod \
                --jsonl-keys text \
                --workers 64'.format(
                        os.path.join(args.filepath, dir, file),
                        os.path.join(save_dir,file[:-len('.jsonl')])
                    )
            print(command)
            if not args.dryrun:
                os.system(command)

            


exit(0)