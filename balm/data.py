# Copyright (c) 2025 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import polars as pl
import torch
from datasets import Dataset, DatasetDict
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
    pad_without_fast_tokenizer_warning,
)
from transformers.tokenization_utils_base import BatchEncoding

__all__ = [
    "load_dataset"
]


def load_dataset(
    file_format: str,
    data_files: Union[str, Dict[str, Union[str, List[str]]], List[str]],
    preprocess_fn: Optional[Callable] = None,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Dataset, DatasetDict]:
    """
    A variant of the load_dataset function that produces identical outputs but adds the option
    for a preprocess callable that will be applied to each input example before building the dataset.

    For text files:
        `preprocess` is given a single line (string) and should return a processed string.
        The resulting dataset will have one column named "text".

    For CSV/TSV/Parquet files:
        `preprocess` is given a dictionary representing the column names and values of one row
        and should return a processed dictionary (with possibly modified keys/values).

    If `preprocess` is None, the dataset is loaded identically to load_dataset (no changes made).
    If multiple files or splits are provided, a DatasetDict is returned. Otherwise, a single Dataset is returned.

    Args:
        path_or_paths: Path to a file or a dictionary of splits to file(s) similar to `data_files` in load_dataset.
        file_format: Optional string to specify the format ("text", "csv", "tsv", "parquet").
                     If None, inferred by file extension.
        preprocess_fn: A callable to preprocess each example.

    Returns:
        A Dataset or DatasetDict with processed data.

    """

    if isinstance(data_files, str) or isinstance(data_files, list):
        data_files = {"train": data_files}
    file_format = file_format.lower()
    preprocess_kwargs = preprocess_kwargs or {}

    def load_and_preprocess_single_file(file_path):
        if file_format == "text":
            return _load_and_preprocess_text_file(
                file_path=file_path,
                preprocess_fn=preprocess_fn,
                preprocess_kwargs=preprocess_kwargs,
            )
        elif file_format in ["csv", "tsv"]:
            separator = "\t" if file_format == "tsv" else ","
            return _load_and_preprocess_csv_file(
                file_path=file_path,
                separator=separator,
                preprocess_fn=preprocess_fn,
                preprocess_kwargs=preprocess_kwargs,
            )
        elif file_format == "parquet":
            return _load_and_preprocess_parquet_file(
                file_path=file_path,
                preprocess_fn=preprocess_fn,
                preprocess_kwargs=preprocess_kwargs,
            )
        # elif file_format == "dataframe":
        #     return _load_and_preprocess_dataframe(
        #         df=file_path,
        #         preprocess_fn=preprocess_fn,
        #         preprocess_kwargs=preprocess_kwargs,
        #     )
        else:
            # fallback to text
            return _load_and_preprocess_text_file(
                file_path=file_path,
                preprocess_fn=preprocess_fn,
                preprocess_kwargs=preprocess_kwargs,
            )

    dataset_dict = {}
    for split_name, file_or_files in data_files.items():
        if isinstance(file_or_files, str):
            file_or_files = [file_or_files]

        all_examples = []
        for fp in file_or_files:
            all_examples.extend(load_and_preprocess_single_file(fp))

        if len(all_examples) == 0:  # empty dataset
            dataset = Dataset.from_dict({})
        else:
            all_keys = set(all_examples[0].keys())
            columns = {k: [ex[k] for ex in all_examples] for k in all_keys}
            dataset = Dataset.from_dict(columns)
        dataset_dict[split_name] = dataset

    if len(dataset_dict) == 1 and "train" in dataset_dict:
        return dataset_dict["train"]
    else:
        return DatasetDict(dataset_dict)


def _load_and_preprocess_text_file(
    file_path: str,
    preprocess_fn: Optional[Callable] = None,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if preprocess_fn is not None:
                line = preprocess_fn(line, **preprocess_kwargs)
            examples.append({"text": line})
    return examples


def _load_and_preprocess_dataframe(
    df: pl.DataFrame,
    preprocess_fn: Optional[Callable] = None,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    examples = []
    for row in df.iter_rows(named=True):
        if preprocess_fn is not None:
            row = preprocess_fn(row, **preprocess_kwargs)
        examples.append(row)
    return examples


def _load_and_preprocess_csv_file(
    file_path: str,
    separator: str = ",",
    preprocess_fn: Optional[Callable] = None,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    df = pl.read_csv(file_path, separator=separator)
    return _load_and_preprocess_dataframe(
        df=df, preprocess_fn=preprocess_fn, preprocess_kwargs=preprocess_kwargs
    )


def _load_and_preprocess_parquet_file(
    file_path: str,
    preprocess_fn: Optional[Callable] = None,
    preprocess_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    df = pl.read_parquet(file_path)
    return _load_and_preprocess_dataframe(
        df=df, preprocess_fn=preprocess_fn, preprocess_kwargs=preprocess_kwargs
    )
