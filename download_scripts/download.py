import os
import sys
from datasets import load_dataset, Features, Value

if len(sys.argv) != 2:
    print("Usage: python download_dataset.py <dataset_name>")
    sys.exit(1)

dataset_name = sys.argv[1]

print(f"Starting download for dataset: {dataset_name}...")

TRAIN_PATH = "/gpfs/alpine2/csc565/scratch/vaibhav_016/test/gpt-neox"

cache_dir = TRAIN_PATH + "/data/cache_dclm"

# dataset_name = "bigcode/the-stack-dedup"
# dataset_name = mlfoundations/dclm-baseline-1.0
# dataset_name = "oscar-corpus/OSCAR-2301"
os.makedirs(cache_dir, exist_ok=True)
max_workers = os.cpu_count()//2
print("***** CPU WORKERS****", max_workers)

if dataset_name == "oscar-corpus/OSCAR-2301": 
  ds = load_dataset(  # mlfoundations/dclm-baseline-1.0, 
                    dataset_name, "de",
                    # split="train",
                    cache_dir=cache_dir, 
                    num_proc=max_workers,
                  #   use_auth_token=True,
                    # language="de"
                    )
else:
  features = Features({
        'bff_contained_ngram_count_before_dedupe': Value('int64'),
        'language_id_whole_page_fasttext': {
            'en': Value('float64')
        },
        'metadata': {
           'abcd': Value('string'),
            'Content-Length': Value('string'),
            'Content-Type': Value('string'),
            'WARC-Block-Digest': Value('string'),
            'WARC-Concurrent-To': Value('string'),
            'WARC-Date': Value('timestamp[s]'),
            'WARC-IP-Address': Value('string'),
            # 'WARC-Identified-Payload-Type': Value('string'),
            'WARC-Payload-Digest': Value('string'),
            'WARC-Record-ID': Value('string'),
            'WARC-Target-URI': Value('string'),
            'WARC-Type': Value('string'),
            'WARC-Warcinfo-ID': Value('string'),
            'WARC-Truncated': Value('string')
        },
        'previous_word_count': Value('int64'),
        'text': Value('string'),
        'url': Value('string'),
        'warcinfo': Value('string'),
        'fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob': Value('float64')
    })
  ds = load_dataset(  # mlfoundations/dclm-baseline-1.0, 
                    dataset_name,
                    split="train",
                    cache_dir=cache_dir, 
                    num_proc=max_workers,
                    features=features
                  #   use_auth_token=True,
                    # language="de"
                    )

print("\n \n ********************* DONE *********************")