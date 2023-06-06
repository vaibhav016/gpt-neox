import os
import argparse

#get args for min and max file number
parser = argparse.ArgumentParser()
parser.add_argument('--min', type=int, default=0)
parser.add_argument('--max', type=int, default=29)
parser.add_argument('--dryrun', action='store_true')
args = parser.parse_args()

data_path = './data'

files = os.listdir(os.path.join(data_path, 'pile/train'))
files = [x for x in files if x.endswith('.jsonl')]
files = sorted(files,key=lambda x: int(x.split('.')[0]))
print(files)


for x in range(args.min, args.max):
    save_dir = os.path.join(data_path, 'pile/tokenized_train_debug/shard_{}'.format(x))
    command = 'python tools/preprocess_data.py \
--input {} \
--output-prefix {} \
--vocab ./data/20B_tokenizer.json \
--tokenizer-type HFTokenizer \
--append-eod \
--jsonl-keys text \
--workers 32'.format(
        os.path.join(data_path, 'pile/train', files[x]),
        save_dir
    )
    print(command)
    if not args.dryrun:
        os.system(command)

exit(0)