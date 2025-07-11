# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

# Modified from our curriculum paper: https://github.com/brineylab/curriculum-paper

import math
import random
from functools import partial

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import Dataset

__all__ = ["MixedProbDataset", "process_mixed_dataset"]


def curriculum_prob(current_step, train_steps, vals):
    t = current_step / train_steps
    prob = 1 / (1 + math.exp(-vals["k"] * (t - vals["shift"])))
    prob = (-vals["A"] * prob) + vals["B"]
    return prob


# Samples based on the probability of unpaired data
# Probability is updated the MixedProbCallback (if not constant)
class MixedProbDataset(Dataset):
    def __init__(
        self,
        tokenized_data: dict,  # expects dict containing both 'unpaired' and 'paired' dataset (dataset can be 'None')
        num_training_steps: int,
        constant_prob: float = None,
        curriculum_prob: dict = None,  # dict must contain keys "k", "shift", "A", "B"
        seed: int = 42,
    ):
        super().__init__()

        # handle probablities
        if (constant_prob is not None) == (curriculum_prob is not None):
            raise ValueError("Specify exactly one of constant_prob or curriculum_prob.")

        self.constant_prob = None
        self.curriculum_fn = None
        self.current_step = 0

        if curriculum_prob is not None:
            # check dict for keys
            required_keys = {"k", "shift", "A", "B"}
            if not required_keys.issubset(curriculum_prob):
                raise ValueError(f"curriculum_prob must contain keys {required_keys}")

            self.curriculum_fn = partial(
                curriculum_prob,
                train_steps=num_training_steps,
                vals=curriculum_prob,
            )
        elif constant_prob is not None:
            self.constant_prob = float(constant_prob)

        # seperate unpaired and paired data
        self.unpaired_data = tokenized_data["unpaired"]
        self.paired_data = tokenized_data["paired"]
        self.unpaired_count = 0
        self.paired_count = 0

        # generator for random sampling
        seed = seed + Accelerator().process_index
        self.generator = torch.Generator().manual_seed(seed)

    @property
    def current_prob(self):
        if not self.constant_prob:
            return self.curriculum_fn(self.current_step)
        return self.constant_prob

    def set_current_step(self, step):
        self.current_step = step

    # when training, ignores idx provided by dataloader (b/c it has no awareness of the two datasets)
    # instead randomly selects idx from the dataset
    def __getitem__(self, idx):
        if (self.current_prob == 0) or (random.random() > self.current_prob):
            self.paired_count += 1
            random_idx = torch.randint(
                high=len(self.paired_data), size=(1,), generator=self.generator
            ).item()
            return self.paired_data[random_idx]
        else:
            self.unpaired_count += 1
            random_idx = torch.randint(
                high=len(self.unpaired_data), size=(1,), generator=self.generator
            ).item()
            return self.unpaired_data[random_idx]

    def __len__(self):
        # account for one of the datasets potentially being 'None'
        unpaired_len = len(self.unpaired_data) if self.unpaired_data is not None else 0
        paired_len = len(self.paired_data) if self.paired_data is not None else 0
        return unpaired_len + paired_len


def process_mixed_dataset(
    data_files,
    tokenizer,
    num_training_steps,
    max_len,
    constant_prob: float = None,
    curriculum_prob: bool = None,
    seed=42,
    num_proc=128,
    cache_dir="./.cache/",
):

    # check dict for keys
    required_keys = {"paired_train", "unpaired_train", "paired_eval", "unpaired_eval"}
    if not required_keys.issubset(data_files):
        raise ValueError(f"data_files must contain keys {required_keys}")

    # check for None values and remove from dict
    none_keys = [k for k in data_files if data_files[k] is None]
    data_files = {k: v for k, v in data_files.items() if k not in none_keys}

    # load
    dataset = load_dataset(
        "parquet", data_files=data_files, num_proc=num_proc, cache_dir=cache_dir
    )

    # tokenize
    # be careful with this caching method
    # if you change the tokenization, you should delete
    # the cache manually or it seems to reuse the previous cache
    tokenized_dataset = dataset.map(
        lambda seq: tokenizer(
            seq["sequence"],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_special_tokens_mask=True,
        ),
        cache_file_names={k: f"{cache_dir}/{str(k)}.arrow" for k in dataset},
        num_proc=num_proc,
        remove_columns=["sequence", "sequence_id"],
    )

    # add back None keys
    tokenized_dataset.update({key: None for key in none_keys})

    # format
    train_dataset = MixedProbDataset(
        {
            "paired": tokenized_dataset["paired_train"],
            "unpaired": tokenized_dataset["unpaired_train"],
        },
        num_training_steps=num_training_steps,
        constant_prob=constant_prob,
        curriculum_prob=curriculum_prob,
        seed=seed,
    )
    eval_dataset = {
        "paired": tokenized_dataset["paired_eval"],
        "unpaired": tokenized_dataset["unpaired_eval"],
    }

    return train_dataset, eval_dataset
