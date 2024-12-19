# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import polars as pl
import torch
from datasets import Dataset, DatasetDict
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import BatchEncoding

__all__ = [
    "DataCollatorForLanguageModeling",
    "load_dataset",
]


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator that applies dynamic masking for MLM training.
    It extends the standard HuggingFace DataCollatorForLanguageModeling to allow
    passing a unique list/tensor of per-position masking probabilities for each input.

    If `mask_probabilities` is provided within the batch, it must be a tensor of shape [batch_size, seq_length]
    where each element is the probability of masking that particular token.

    If `mask_probabilities` is not provided, a uniform masking at `mlm_probability` (default 15%) is applied.

    This data collator is fully compatible with the HuggingFace Trainer and the models described above.
    """

    tokenizer: Any
    mlm: bool = True
    mlm_probability: float = 0.15
    return_tensors: str = "pt"

    def torch_call(
        self, examples: List[Union[List[int], Dict[str, Any], BatchEncoding]]
    ) -> Dict[str, Any]:
        # convert to dict of lists
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = {}
            for k in examples[0].keys():
                batch[k] = [e[k] for e in examples]
        else:
            batch = {"input_ids": examples}

        input_ids = torch.tensor(batch["input_ids"], dtype=torch.long)
        labels = input_ids.clone()

        # check if per-position mask_probabilities are provided
        mask_probabilities = batch.get("mask_probabilities", None)
        if mask_probabilities is not None:
            mask_probabilities = torch.tensor(mask_probabilities, dtype=torch.float)
            assert (
                mask_probabilities.shape == input_ids.shape
            ), "mask_probabilities must match input_ids shape."
        # if not, use uniform mlm_probability
        else:
            mask_probabilities = torch.full(
                input_ids.shape, self.mlm_probability, dtype=torch.float
            )

        # special tokens mask
        special_tokens_mask = None
        if "special_tokens_mask" in batch:
            special_tokens_mask = torch.tensor(
                batch["special_tokens_mask"], dtype=torch.bool
            )
        else:
            special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # add padding tokens to special tokens mask
        if self.tokenizer.pad_token_id is not None:
            padding_mask = input_ids == self.tokenizer.pad_token_id
            special_tokens_mask = special_tokens_mask | padding_mask

        # select tokens to mask
        rand = torch.rand(input_ids.shape)
        final_mask = (rand < mask_probabilities) & ~special_tokens_mask
        probability_matrix = final_mask.float()

        # replace with mask token 80% of the time
        masked_indices = final_mask
        indices_replaced = torch.rand_like(labels.float()) < 0.8 * probability_matrix
        indices_replaced = indices_replaced & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # replace with a random non-special token 10% of the time
        # the final 10% are left unchanged
        indices_random = torch.rand_like(labels.float()) < 0.1 * probability_matrix
        indices_random = indices_random & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # ignore labels for non-masked tokens by setting to -100
        labels[~masked_indices] = -100

        batch["input_ids"] = input_ids
        if "attention_mask" in batch:
            # invert attention mask to match behavior of the 🤗 DataCollatorForLanguageModeling
            # 1 should now indicate non-padding tokens and 0 should indicate padding tokens
            batch["attention_mask"] = 1 - torch.tensor(
                batch["attention_mask"], dtype=torch.int
            )
        if "token_type_ids" in batch:
            batch["token_type_ids"] = torch.tensor(
                batch["token_type_ids"], dtype=torch.int
            )

        batch["labels"] = labels
        if "mask_probabilities" in batch:
            del batch["mask_probabilities"]

        return batch


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


# class Dataset:
#     def __init__(self, data: Union[Dict[str, Iterable[str]], pl.DataFrame]):
#         if isinstance(data, pd.DataFrame):
#             data = pl.from_pandas(data)
#         elif not isinstance(data, pl.DataFrame):
#             data = pl.DataFrame(data)
#         self.data = data

#     def __getitem__(self, index: Union[int, str]):
#         if isinstance(index, str) and index in self.data.columns:
#             return self.data[index].to_list()
#         elif isinstance(index, int):
#             d = self.data[index].to_dict(as_series=False)
#             return {k: v[0] for k, v in d.items()}
#         else:
#             raise ValueError(f"Index {index} is not valid")

#     def __setitem__(self, index: str, value: Any):
#         if len(value) != self.num_rows:
#             raise ValueError(
#                 f"Value length {len(value)} does not match dataset length {self.num_rows}"
#             )
#         self.data = self.data.with_columns(pl.Series(name=index, values=value))

#     def __len__(self):
#         return self.data.shape[0]

#     @property
#     def shape(self) -> Tuple[int, int]:
#         """
#         Returns the shape of the dataset.
#         """
#         return self.data.shape

#     @property
#     def num_rows(self) -> int:
#         """
#         Returns the number of rows in the dataset.
#         """
#         return self.data.shape[0]

#     def remove_columns(self, columns: Union[str, Iterable[str]]):
#         """
#         Removes the specified columns from the dataset.
#         """
#         if isinstance(columns, str):
#             columns = [columns]
#         self.data = self.data.drop(columns)

#     def rename_columns(self, columns: Dict[str, str]):
#         """
#         Renames the specified columns in the dataset.
#         """
#         self.data = self.data.rename(columns)

#     def clone(self):
#         """
#         Returns a clone of the dataset.
#         """
#         return self.__class__(self.data.clone())


# class DatasetDict:
#     def __init__(self, data: Dict[str, Dataset]):
#         self.dataset_dict = data

#     def __getitem__(self, index: str):
#         return self.dataset_dict[index]

#     def __repr__(self):
#         repr = "DatasetDict\n-----------\n"
#         for name, dataset in self.dataset_dict.items():
#             repr += f"  {name}\n"
#             repr += f"    num_rows: {dataset.num_rows}\n"
#             repr += f"    columns: {dataset.data.columns}\n"
#         return repr

#     def __str__(self):
#         return self.__repr__()

#     def keys(self):
#         return self.dataset_dict.keys()

#     def values(self):
#         return self.dataset_dict.values()

#     def items(self):
#         return self.dataset_dict.items()

#     def map(
#         self,
#         func: Callable,
#         remove_columns: Optional[Union[str, Iterable[str]]] = None,
#         rename_columns: Optional[Dict[str, str]] = None,
#         preprocess_fn: Optional[Callable] = None,
#         preprocess_kwargs: Optional[Dict[str, Any]] = None,
#     ) -> "DatasetDict":
#         """
#         Maps a function to the dataset.

#         Parameters
#         ----------
#         func : Callable
#             The function to map to the dataset.

#         remove_columns : Optional[Union[str, Iterable[str]]]
#             The columns to remove from the dataset.

#         rename_columns : Optional[Dict[str, str]]
#             The columns to rename in the dataset.

#         preprocess_fn : Optional[Callable]
#             A function to preprocess the dataset. The function must accept a single dataset (torch.Tensor or dict of torch.Tensor)
#             as input, and return a processed version of the dataset.

#         preprocess_kwargs : Optional[Dict[str, Any]]
#         Keyword arguments to pass to the preprocessing function.

#         Returns
#         -------
#         DatasetDict
#             The processed dataset.

#         """
#         cloned_data = {k: v.clone() for k, v in self.items()}
#         dataset_dict = self.__class__(cloned_data)
#         for dataset in dataset_dict.values():
#             if preprocess_fn is not None:
#                 preprocess_kwargs = preprocess_kwargs or {}
#                 dataset = preprocess_fn(dataset, **preprocess_kwargs)
#             result = func(dataset)
#             if not isinstance(result, (dict, BatchEncoding)):
#                 raise ValueError(
#                     f"Mapping function returned an object of type {type(result)}, but must return either a dictionary or BatchEncoding object"
#                 )
#             for key, value in result.items():
#                 dataset[key] = value
#             if remove_columns is not None:
#                 if isinstance(remove_columns, str):
#                     remove_columns = [remove_columns]
#                 dataset.remove_columns(remove_columns)
#             if rename_columns is not None:
#                 dataset.rename_columns(rename_columns)
#         return dataset_dict


# class DataCollator:
#     def __init__(
#         self,
#         tokenizer: TokenizerBase,
#         mlm: bool = True,
#         mlm_probability: float = 0.15,
#     ):
#         self.tokenizer = tokenizer
#         self.mlm = mlm
#         self.mlm_probability = mlm_probability

#     def __call__(
#         self,
#         examples: List[Union[List[int], Any, Dict[str, Any]]],
#         mask_probs: Optional[str] = None,
#     ) -> Dict[str, Any]:
#         """
#         Collate a batch of examples.

#         Parameters
#         ----------
#         examples : List[Union[List[int], Any, Dict[str, Any]]]
#             The batch of examples to collate.

#         mask_probs : Optional[str]
#             The name of a key in the examples dictionary that contains the masking probabilities
#             for MLM training. If not specified, examples will be uniformly masked at ``self.mlm_probability``.

#         Returns
#         -------
#         Dict[str, Any]
#             The collated batch.

#         """
#         # convert to tensors if necessary
#         if isinstance(examples, dict):
#             batch = {k: torch.tensor(v) for k, v in examples.items()}
#         elif isinstance(examples, torch.Tensor):
#             batch = {"input_ids": examples}
#         else:
#             if isinstance(examples[0], (list, tuple, np.ndarray)):
#                 examples = [torch.tensor(e, dtype=torch.long) for e in examples]
#             # batch = {"input_ids": torch.stack(examples)}
#             batch = {"input_ids": torch.stack([d["input_ids"] for d in examples])}

#         # MLM masking
#         if self.mlm:
#             # TODO: implement mask_probs
#             # not sure how they'll be formatted -- as a stacked tensor, a list of tensors, or a list of CIGAR-like strings
#             batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"])
#         # else:
#         #     labels = batch["input_ids"].clone()
#         #     if self.tokenizer.pad_token_id is not None:
#         #         labels[labels == self.tokenizer.pad_token_id] = -100
#         #     batch["labels"] = labels

#         # key padding mask
#         kp_mask = torch.zeros_like(batch["input_ids"])  # 1 for non-pad tokens
#         kp_mask[batch["input_ids"] == self.tokenizer.pad_idx] = 1  # 0 for pad tokens
#         batch["key_padding_mask"] = kp_mask.bool()
#         return batch

#     def mask_tokens(
#         self,
#         inputs: torch.Tensor,
#         mask_probs: Optional[str] = None,
#         special_tokens_mask: Optional[Any] = None,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         labels = inputs.clone()
#         # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
#         probability_matrix = torch.full(labels.shape, self.mlm_probability)
#         if special_tokens_mask is None:
#             special_tokens_mask = [
#                 self.tokenizer.get_special_tokens_mask(val) for val in labels.tolist()
#             ]
#         special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

#         probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#         masked_indices = torch.bernoulli(probability_matrix).bool()
#         labels[~masked_indices] = -100  # We only compute loss on masked tokens

#         # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#         indices_replaced = (
#             torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#         )
#         inputs[indices_replaced] = self.tokenizer.mask_idx

#         # 10% of the time, we replace masked input tokens with random word
#         indices_random = (
#             torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
#             & masked_indices
#             & ~indices_replaced
#         )

#         random_tokens = np.random.choice(
#             self.tokenizer.all_nonspecial_tokens_idx,
#             size=inputs.numel(),
#             replace=True,
#         )
#         random_tokens = torch.tensor(random_tokens, dtype=torch.long).view(inputs.shape)
#         # random_tokens = torch.multinomial(
#         #     self.tokenizer.all_nonspecial_tokens_idx,
#         #     labels.numel,
#         #     replacement=True,
#         #     dtype=torch.long,
#         # ).view(labels.shape)
#         inputs[indices_random] = random_tokens[indices_random]

#         # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#         return inputs, labels


# class SequentialMaskingDataCollator:
#     def __init__(self, tokenizer: TokenizerBase):
#         self.tokenizer = tokenizer

#     def __call__(self, examples):
#         # convert to tensors if necessary
#         if isinstance(examples, dict):
#             batch = examples
#         elif isinstance(examples, torch.Tensor):
#             batch = {"input_ids": examples}
#         else:
#             if isinstance(examples[0], (list, tuple, np.ndarray)):
#                 examples = [torch.tensor(e, dtype=torch.long) for e in examples]
#             # batch = {"input_ids": torch.stack(examples)}
#             batch = {"input_ids": torch.stack([d["input_ids"] for d in examples])}
#         input_ids = batch["input_ids"]

#         # Prepare masks for each sequence
#         masks = []
#         labels = []
#         for seq in input_ids:
#             seq_masks = []
#             seq_labels = []
#             for i in range(len(seq)):
#                 mask = torch.zeros_like(seq)
#                 mask[i] = 1
#                 seq_masks.append(mask)

#                 label = torch.full_like(seq, -100)
#                 label[i] = seq[i]
#                 seq_labels.append(label)
#             masks.append(torch.stack(seq_masks))
#             labels.append(torch.stack(seq_labels))

#         # Stack masks and labels to create a batch
#         batch["input_ids"] = input_ids.unsqueeze(1).expand(-1, input_ids.size(1), -1)
#         batch["labels"] = labels
#         # key padding mask
#         kp_mask = torch.zeros_like(batch["input_ids"])  # 1 for non-pad tokens
#         kp_mask[batch["input_ids"] == self.tokenizer.pad_idx] = 1  # 0 for pad tokens
#         batch["key_padding_mask"] = kp_mask.bool()

#         return batch


# def load_dataset(
#     path: str,
#     data_files: Dict[str, str],
#     strip_lines: bool = True,
#     preprocess_fn: Optional[Callable] = None,
#     preprocess_kwargs: Optional[Dict[str, Any]] = None,
# ):
#     """
#     Loads a dataset.

#     Parameters
#     ----------
#     path : str
#         The type of dataset to load. Options are:
#             - "text": load a text dataset
#             - "csv": load a CSV dataset
#             - "tsv": load a TSV dataset
#             - "df": load a pandas or polars ``DataFrame`` dataset
#             - "json": load a JSON dataset
#             - "parquet": load a Parquet dataset

#     data_files : Dict[str, str]
#         A ``dict`` mapping dataset names to a file or directory.
#         If a directory is provided, the dataset will be loaded from all
#         files in the directory.

#     strip_lines : bool
#         Whether to strip lines. Used only for text datasets.

#     preprocess_fn : Optional[Callable]
#         A function to preprocess the dataset. Optional.
#         For text datasets, this function should accept and return a single
#         line of the text file (as a string). For CSV and TSV datasets,
#         this function should accept and return a single row of the CSV or
#         TSV file (as a dictionary mapping column names to values).

#     preprocess_kwargs : Optional[Dict[str, Any]]
#         Keyword arguments to pass to the preprocessing function.

#     Returns
#     -------
#     DatasetDict
#         The dataset.
#     """
#     path = path.lower()
#     if path == "text":
#         return _load_text_dataset(
#             path=path,
#             data_files=data_files,
#             strip_lines=strip_lines,
#             preprocess_fn=preprocess_fn,
#             preprocess_kwargs=preprocess_kwargs,
#         )
#     elif path == "csv":
#         return _load_tabular_dataset(
#             data_files=data_files,
#             preprocess_fn=preprocess_fn,
#             preprocess_kwargs=preprocess_kwargs,
#             sep=",",
#         )
#     elif path == "tsv":
#         return _load_tabular_dataset(
#             data_files=data_files,
#             preprocess_fn=preprocess_fn,
#             preprocess_kwargs=preprocess_kwargs,
#             sep="\t",
#         )
#     elif path in ["df", "dataframe"]:
#         return _load_dataframe_dataset(
#             dataframes=data_files,
#             preprocess_fn=preprocess_fn,
#             preprocess_kwargs=preprocess_kwargs,
#         )
#     elif path == "json":
#         return _load_json_dataset(
#             data_files=data_files,
#             preprocess_fn=preprocess_fn,
#             preprocess_kwargs=preprocess_kwargs,
#         )
#     elif path == "parquet":
#         return _load_parquet_dataset(
#             data_files=data_files,
#             preprocess_fn=preprocess_fn,
#             preprocess_kwargs=preprocess_kwargs,
#         )
#     else:
#         raise ValueError(f"Invalid dataset type: {path}")


# def _load_text_dataset(
#     path: str,
#     data_files: Dict[str, str],
#     strip_lines: bool = True,
#     preprocess_fn: Optional[Callable] = None,
#     preprocess_kwargs: Optional[Dict[str, Any]] = None,
# ):
#     dataset_dict = {}
#     for name, files in data_files.items():
#         if os.path.isdir(files):
#             files = [os.path.join(files, f) for f in os.listdir(files)]
#         elif os.path.isfile(files):
#             files = [files]
#         else:
#             raise ValueError(f"Invalid file or directory: {files}")
#         data = []
#         for file in files:
#             with open(file, "r") as f:
#                 file_data = f.readlines()
#                 if strip_lines:
#                     file_data = [line.strip() for line in file_data]
#                 if preprocess_fn is not None:
#                     preprocess_kwargs = preprocess_kwargs or {}
#                     file_data = [
#                         preprocess_fn(line, **preprocess_kwargs) for line in file_data
#                     ]
#                 data.extend(file_data)
#         dataset_dict[name] = Dataset({path: data})
#     return DatasetDict(dataset_dict)


# def _load_tabular_dataset(
#     data_files: Dict[str, str],
#     preprocess_fn: Optional[Callable] = None,
#     preprocess_kwargs: Optional[Dict[str, Any]] = None,
#     sep: str = ",",
# ):
#     dataset_dict = {}
#     for name, files in data_files.items():
#         if os.path.isdir(files):
#             files = [os.path.join(files, f) for f in os.listdir(files)]
#         elif os.path.isfile(files):
#             files = [files]
#         else:
#             raise ValueError(f"Invalid file or directory: {files}")
#         data = []
#         for file in files:
#             with open(file, "r") as f:
#                 reader = csv.DictReader(f, delimiter=sep)
#                 for row in reader:
#                     if preprocess_fn is not None:
#                         preprocess_kwargs = preprocess_kwargs or {}
#                         row = preprocess_fn(row, **preprocess_kwargs)
#                     data.append(row)
#         dataset_dict[name] = Dataset(data)
#     return DatasetDict(dataset_dict)


# def _load_dataframe_dataset(
#     dataframes: Dict[str, str],
#     preprocess_fn: Optional[Callable] = None,
#     preprocess_kwargs: Optional[Dict[str, Any]] = None,
# ):
#     dataset_dict = {}
#     for name, df in dataframes.items():
#         if isinstance(df, pl.DataFrame):
#             is_pandas = False
#             iter_kwargs = {"named": True}
#         elif isinstance(df, pd.DataFrame):
#             is_pandas = True
#             iter_kwargs = {}
#         else:
#             raise ValueError(f"Invalid dataframe type: {type(df)}")
#         if preprocess_fn is None:
#             dataset_dict[name] = Dataset(df)
#         else:
#             data = []
#             for row in df.iterrows(**iter_kwargs):
#                 if is_pandas:
#                     _, row = row
#                 if preprocess_fn is not None:
#                     preprocess_kwargs = preprocess_kwargs or {}
#                     row = preprocess_fn(row, **preprocess_kwargs)
#                 data.append(row)
#             dataset_dict[name] = Dataset(df)
#     return DatasetDict(dataset_dict)


# def _load_json_dataset(
#     data_files: Dict[str, str],
#     preprocess_fn: Optional[Callable] = None,
#     preprocess_kwargs: Optional[Dict[str, Any]] = None,
# ):
#     dataset_dict = {}
#     for name, files in data_files.items():
#         if os.path.isdir(files):
#             files = [os.path.join(files, f) for f in os.listdir(files)]
#         elif os.path.isfile(files):
#             files = [files]
#         else:
#             raise ValueError(f"Invalid file or directory: {files}")
#         data = []
#         for file in files:
#             with open(file, "r") as f:
#                 json_data = json.load(f)
#                 for json_elem in json_data:
#                     if preprocess_fn is not None:
#                         preprocess_kwargs = preprocess_kwargs or {}
#                         json_elem = preprocess_fn(json_elem, **preprocess_kwargs)
#                     data.append(json_elem)
#         dataset_dict[name] = Dataset(data)
#     return DatasetDict(dataset_dict)


# def _load_parquet_dataset(
#     data_files: Dict[str, str],
#     preprocess_fn: Optional[Callable] = None,
#     preprocess_kwargs: Optional[Dict[str, Any]] = None,
# ):
#     dataset_dict = {}
#     for name, files in data_files.items():
#         if os.path.isdir(files):
#             files = [os.path.join(files, f) for f in os.listdir(files)]
#         elif os.path.isfile(files):
#             files = [files]
#         else:
#             raise ValueError(f"Invalid file or directory: {files}")
#         df = pl.read_parquet(files)
#         if preprocess_fn is None:
#             dataset_dict[name] = Dataset(df)
#         else:
#             data = []
#             for row in df.iterrows(named=True):
#                 if preprocess_fn is not None:
#                     preprocess_kwargs = preprocess_kwargs or {}
#                     row = preprocess_fn(row, **preprocess_kwargs)
#                 data.append(row)
#             dataset_dict[name] = Dataset(data)
#     return DatasetDict(dataset_dict)
