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

"""GPT2 style dataset."""

import os
import time

import numpy as np
import torch

from megatron import mpu, print_rank_0


class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=True,
        use_shared_fs=True,
        index_mapping_paths=None,
        index_offset=0,
        reshuffle_when_loading=True,
    ):

        self.name = name
        self.indexed_dataset = indexed_dataset

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]
        if not build_index_mappings:
            assert index_mapping_paths, "If not building index mappings, the path to existing ones must be provided."

        if build_index_mappings:
            # Build index mappings.
            self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                num_samples,
                seq_length,
                seed,
                use_shared_fs=use_shared_fs,
            )

        else:
            # If not building the index mappings, we need to load them
            self.doc_idx, self.sample_idx, self.shuffle_idx = _load_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                num_samples,
                seq_length,
                seed,
                use_shared_fs=use_shared_fs,
                index_mapping_paths=index_mapping_paths,
                index_offset=index_offset,
                reshuffle_when_loading=reshuffle_when_loading,
            )


        self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
        self.sample_idx_len = self.sample_idx.shape[0] - 1

        if self.shuffle_idx_len != self.sample_idx_len:
            print(
                f"WARNING: shuffle index length ({self.shuffle_idx_len}) is not equal to sample index length ({self.sample_idx_len})"
            )


    def __len__(self):
        return min(self.shuffle_idx_len, self.sample_idx_len)

    def __getitem__(self, idx):
        try:
            # Get the shuffled index.
            idx = self.shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_index_f = self.sample_idx[idx][0]
            doc_index_l = self.sample_idx[idx + 1][0]
            offset_f = self.sample_idx[idx][1]
            offset_l = self.sample_idx[idx + 1][1]
            # If we are within the same document, just extract the chunk.
            if doc_index_f == doc_index_l:
                sample = self.indexed_dataset.get(
                    self.doc_idx[doc_index_f],
                    offset=offset_f,
                    length=offset_l - offset_f + 1,
                )
            else:
                # Otherwise, get the rest of the initial document.
                sample_list = [
                    self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
                ]
                # Loop over all in between documents and add the entire document.
                for i in range(doc_index_f + 1, doc_index_l):
                    sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
                # And finally add the relevant portion of last document.
                sample_list.append(
                    self.indexed_dataset.get(
                        self.doc_idx[doc_index_l], length=offset_l + 1
                    )
                )
                sample = np.concatenate(sample_list)

            return {"text": np.array(sample, dtype=np.int64)}
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


# Warning: only implemented with replay in mind, some issues may arise when dealing with more than 1 epoch over the dataset
def _load_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    index_mapping_paths,
    use_shared_fs=True,
    index_offset=0,
    reshuffle_when_loading=True,
):
    """Build doc-idx, sample-idx, and shuffle-idx from ones loaded.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)
    is_replay = name.split("_")[0] == "replay"


    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"


    if not use_shared_fs:
        should_process_dataset = int(os.environ["LOCAL_RANK"]) == 0
    else:
        should_process_dataset = torch.distributed.get_rank() == 0

    # Build the indexed mapping if not exist.
    if should_process_dataset:
        start_time = time.time()
        print_rank_0(" > loading shuffle-idx mapping from {}".format(index_mapping_paths["shuffle_idx_path"]))
        shuffle_idx = np.load(index_mapping_paths["shuffle_idx_path"], allow_pickle=True, mmap_mode="r")
        print_rank_0(
            "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
        )

        idx_prefix = index_mapping_paths["shuffle_idx_path"].split("_")[:-2]

        ## restrict to samples seen during original pretraining if this is replay
        ## careful, this is hardcoded based on the number on idx filenames looking like dataset_train_4_indexmap_5781ns_2048sl_1234s_doc_idx.npy

        # remove 1.005 buffer that was added when estimating numbers of samples in get_normalized_weights_and_num_samples(). Note that of course
        # this may be missing a few of the seen samples.
        num_samples_originally_seen = np.int64(np.int64(idx_prefix[-3][:-2]) / 1.005)
        seq_length_originally_seen = int(idx_prefix[-2][:-2])
        assert seq_length_originally_seen == seq_length, "Current seq len {} does not match seq len the indices were built for ({}).".format(
            seq_length,
            seq_length_originally_seen,
        )
        # Useful if we want to support extending the idx if more than 1 epoch is necessary, but for now will assert 1 epoch
        num_epochs_in_replay = _num_epochs(tokens_per_epoch, seq_length, num_samples_originally_seen)
        assert num_epochs_in_replay == 1, "Enough samples from replay dataset {} for more than one epoch; this is currently\
            untested.".format(data_prefix)
        if is_replay:
            shuffle_idx = shuffle_idx[:num_samples_originally_seen]
            # get a sufficient length if needed


        # apply offset, but add back the removed elements.
        # TODO Do we want to shuffle again the shuffle_idx[:index_offset] term ?
        index_offset = index_offset % len(shuffle_idx)
        if reshuffle_when_loading:
            # Numpy can throw errors if the array isn't writeable, which it is not apparently when it is loaded
            if not shuffle_idx.flags.writeable:
                try:
                    shuffle_idx.setflags(write=True)
                except:
                    # copy trick if we couldn't set it to writeable
                    print("Loaded shuffle_idx at {} is not writeable, need to copy it as workaround...".format("_".join(idx_prefix)))
                    temp = shuffle_idx.copy()
                    shuffle_idx = temp
                    assert shuffle_idx.flags.writeable, "Failed to make shuffle_idx writeable, the shuffling will not work."
                    print("succesfully copied !")
            # For some reason this is faster than shuffling the copied array directly. I'm sure there is a good reason for it.
            temp_random_idx = np.array(range(len(shuffle_idx)))
            np_rng.shuffle(temp_random_idx)
            shuffle_idx = shuffle_idx[temp_random_idx]
            del temp_random_idx
            # np_rng.shuffle(shuffle_idx)
        shuffle_idx = np.concatenate([shuffle_idx[index_offset:], shuffle_idx[:index_offset]])
        np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
        print_rank_0(
            " > elapsed time to build and save shuffle-idx mapping"
            " (seconds): {:4f}".format(time.time() - start_time)
        )

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_io_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_io_parallel_group()
    )

    # Load mappings.
    start_time = time.time()
    print_rank_0(" > loading doc-idx mapping from {}".format(index_mapping_paths["doc_idx_path"]))
    doc_idx = np.load(index_mapping_paths["doc_idx_path"], allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading sample-idx mapping from {}".format(index_mapping_paths["sample_idx_path"]))
    sample_idx = np.load(index_mapping_paths["sample_idx_path"], allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
    )
    print_rank_0("    total number of samples: {}".format(shuffle_idx.shape[0] + 1))
    print_rank_0("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    use_shared_fs=True,
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"

    if not use_shared_fs:
        should_process_dataset = int(os.environ["LOCAL_RANK"]) == 0
    else:
        should_process_dataset = torch.distributed.get_rank() == 0

    # Build the indexed mapping if not exist.
    if should_process_dataset:
        if (
            (not os.path.isfile(doc_idx_filename))
            or (not os.path.isfile(sample_idx_filename))
            or (not os.path.isfile(shuffle_idx_filename))
        ):
            print_rank_0(
                " > WARNING: could not find index map files, building "
                "the indices on rank 0 ..."
            )
            # doc-idx.
            start_time = time.time()
            doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
            np.save(doc_idx_filename, doc_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save doc-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # sample-idx.
            start_time = time.time()
            # Use C++ implementation for speed.
            from megatron.data import helpers

            assert doc_idx.dtype == np.int32
            assert sizes.dtype == np.int32

            num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length
            if 2 * (num_samples + 1) < np.iinfo(np.int32).max:
                sample_idx = helpers.build_sample_idx_int32(
                    sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
                )
            else:
                sample_idx = helpers.build_sample_idx_int64(
                    sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
                )
            np.save(sample_idx_filename, sample_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save sample-idx mapping "
                "(seconds): {:4f}".format(time.time() - start_time)
            )
            # shuffle-idx.
            start_time = time.time()
            # -1 is due to data structure used to retrieve the index:
            #    sample i --> [sample_idx[i], sample_idx[i+1])
            shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(
                " > elapsed time to build and save shuffle-idx mapping"
                " (seconds): {:4f}".format(time.time() - start_time)
            )

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_io_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_io_parallel_group()
    )

    # Load mappings.
    start_time = time.time()
    print_rank_0(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading sample-idx mapping from {}".format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_0(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
    )
    print_rank_0("    total number of samples: {}".format(sample_idx.shape[0]))
    print_rank_0("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""
    doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx


def _build_sample_idx(sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int64)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Beginning offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += remaining_seq_length + doc_length - 1
                remaining_seq_length = 0
            else:
                # Otherwise, start from the beginning of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(size, np_rng):
    """Build the range [0, size) and shuffle."""
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx
