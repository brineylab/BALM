# Copyright (c) 2024 brineylab @ scripps
# Distributed under the terms of the MIT License.
# SPDX-License-Identifier: MIT

import os
import re
from typing import Optional

from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

__all__ = [
    "BalmTokenizer",
    "DEFAULT_VOCAB",
]


class BalmTokenizer(PreTrainedTokenizerFast):
    """
    A tokenizer class mirroring the ESM-2 tokenizer's vocabulary and style, but without the <null_1> token.
    This results in a vocabulary size of 32 (a `multiple of 8`_) instead of ESM-2's 33.

    Parameters
    ----------
    vocab_file: str, optional
        Path to the vocabulary file. If not provided, the default vocabulary is used.

    bos_token: str, optional
        Beginning of sequence token. Default is "<cls>".

    eos_token: str, optional
        End of sequence token. Default is "<eos>".

    unk_token: str, optional
        Unknown token. Default is "<unk>".

    pad_token: str, optional
        Padding token. Default is "<pad>".

    mask_token: str, optional
        Mask token. Default is "<mask>".

    Any additional keyword arguments are passed to the `PreTrainedTokenizerFast` constructor.


    .. _multiple of 8:
        https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/#:~:text=Simply%20padding%20the%20vocabulary%20size,performance%20of%20the%20projection%20layer.

    """

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        bos_token: str = "<cls>",
        eos_token: str = "<eos>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        **kwargs,
    ):
        # parse vocab
        if vocab_file is not None and os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = [line.strip() for line in f if line.strip()]
        else:
            vocab = DEFAULT_VOCAB
        vocab_dict = {token: i for i, token in enumerate(vocab)}

        # create tokenizer
        tokenizer = Tokenizer(
            WordLevel(
                vocab=vocab_dict,
                unk_token=unk_token,
            )
        )

        # pre-tokenization
        if "sep_token" in locals():
            special_start_char = sep_token[0]
            special_end_char = sep_token[-1]
        elif "bos_token" in locals():
            special_start_char = bos_token[0]
            special_end_char = bos_token[-1]
        else:
            special_start_char = Regex("|".join(["<", "["]))
            special_end_char = Regex("|".join([">", "]"]))
        nonspecial_tokens = [tok for tok in vocab if len(tok) == 1]
        pattern = "|".join([re.escape(tok) for tok in nonspecial_tokens])
        tokenizer.pre_tokenizer = Sequence(
            [
                Split(special_start_char, behavior="merged_with_next"),
                Split(special_end_char, behavior="merged_with_previous"),
                Split(Regex(pattern), behavior="isolated"),
            ]
        )

        # post-processing (add bos and eos tokens)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{bos_token} $A {eos_token}",
            pair=f"{bos_token} $A $B {eos_token}",
            special_tokens=[
                (bos_token, vocab_dict[bos_token]),
                (eos_token, vocab_dict[eos_token]),
            ],
        )

        # initialize PreTrainedTokenizerFast
        super().__init__(
            tokenizer_object=tokenizer,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )


DEFAULT_VOCAB = [
    "<cls>",
    "<pad>",
    "<eos>",
    "<unk>",
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "O",
    "Z",
    ".",
    "-",
    "<mask>",
]


# DEFAULT_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "vocab.json")


# class TokenizerBase:
#     """
#     Base class for tokenizers.
#     """

#     def __init__(self, vocab: Union[str, Dict[str, int]]):
#         self.vocab = self._process_vocab(vocab)

#     def _process_vocab(self, vocab: Union[str, Dict[str, int]]) -> Dict[str, int]:
#         if isinstance(vocab, str):
#             if os.path.isfile(vocab):
#                 return json.load(open(vocab, "r"))
#             elif os.path.isdir(vocab):
#                 return json.load(open(os.path.join(vocab, "vocab.json"), "r"))
#             else:
#                 raise ValueError(
#                     f"If vocab is a string, it must be a file or directory. {vocab} does not exist"
#                 )
#         elif isinstance(vocab, dict):
#             return vocab
#         else:
#             raise ValueError("Vocab must be a string or dictionary")


# class Tokenizer(TokenizerBase):
#     """
#     Simple tokenizer class. Provides basic methods for tokenization and encoding.
#     """

#     def __init__(
#         self,
#         vocab: Union[str, Dict[str, int]] = DEFAULT_VOCAB_PATH,
#         model_max_length: int = 320,
#         cls_token: str = "<cls>",
#         pad_token: str = "<pad>",
#         # sep_token: str = "<sep>",
#         # bos_token: str = "<bos>",
#         eos_token: str = "<eos>",
#         unk_token: str = "<unk>",
#         mask_token: str = "<mask>",
#         additional_special_tokens: Optional[Iterable[str]] = None,
#     ):
#         """
#         Initialize the tokenizer.

#         Parameters
#         ----------
#         vocab : Union[str, Dict[str, int]]
#             The vocabulary to use. If a string, it must be a file or directory.

#         model_max_length : int, optional
#             The maximum length of the sequence. Default is 320.

#         cls_token : str, optional
#             The beginning of sequence token. Default is "<cls>".

#         pad_token : str, optional
#             The padding token. Default is "<pad>".

#         eos_token : str, optional
#             The end of sequence token. Default is "<eos>".

#         unk_token : str, optional
#             The unknown token. Default is "<unk>".

#         mask_token : str, optional
#             The mask token. Default is "<mask>".

#         additional_special_tokens : Optional[Iterable[str]], optional
#             Additional special tokens to use when building the special tokens mask.
#             Default is None.
#         """
#         super().__init__(vocab)
#         self.max_length = model_max_length
#         self.tok_to_idx = self.vocab
#         self.idx_to_tok = {v: k for k, v in self.vocab.items()}

#         self.unk_token = unk_token
#         self.pad_token = pad_token
#         self.cls_token = cls_token
#         self.mask_token = mask_token
#         self.eos_token = eos_token
#         self.all_special_tokens = [
#             eos_token,
#             unk_token,
#             pad_token,
#             cls_token,
#             mask_token,
#         ]
#         if additional_special_tokens is not None:
#             self.all_special_tokens.extend(additional_special_tokens)
#         self.all_nonspecial_tokens = [
#             tok for tok in self.vocab.keys() if tok not in self.all_special_tokens
#         ]

#         self.unk_idx = self.tok_to_idx[unk_token]
#         self.pad_idx = self.get_idx(pad_token)
#         self.cls_idx = self.get_idx(cls_token)
#         self.mask_idx = self.get_idx(mask_token)
#         self.eos_idx = self.get_idx(eos_token)
#         self.all_special_tokens_idx = [self.get_idx(t) for t in self.all_special_tokens]
#         self.all_nonspecial_tokens_idx = [
#             self.get_idx(t) for t in self.all_nonspecial_tokens
#         ]

#         self.all_toks = list(self.vocab.keys())
#         self.unique_no_split_tokens = self.all_toks

#     def __len__(self):
#         return len(self.all_toks)

#     def __call__(
#         self,
#         text: Union[str, Iterable[str]],
#         padding: Union[bool, str] = False,
#         add_special_tokens: bool = True,
#         truncation: Union[bool, str] = True,
#         max_length: int = 320,
#         return_attention_mask: bool = False,
#         return_special_tokens_mask: bool = False,
#         as_dict: bool = False,
#         name: Optional[str] = None,
#         verbose: bool = False,
#         **kwargs,
#     ) -> "BatchEncoding":
#         """
#         Encodes a string or list of strings into a sequence of tokens, using the tokenizer.

#         Parameters
#         ----------
#         text : Union[str, Iterable[str]]
#             The sequence to be encoded. Can be a string (single example) or an iterable of strings (batch of examples).

#         padding : bool, optional
#             Padding strategy. Can be either:
#                 * ``True`` or ``'longest'``: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
#                 * ``'max_length'``: Pad to a maximum length specified with the argument `max_length`. If the `max_length` argument is not provided,
#                 sequences will be padded to the longest sequence in the batch (functioning as if this argument was set to ``True``).
#                 * ``False`` or ``'do_not_pad'`` (default): No padding (i.e., can output a batch with sequences of different lengths).

#         add_special_tokens : bool, optional
#             Whether to add the special tokens to the sequence. Default is True.

#         truncation : bool, optional
#             Whether to truncate the sequence to the maximum length (as supplied by `max_length`). Default is True, however, if
#             `max_length` is not provided, truncation will not occur.

#         max_length : int, optional
#             The maximum length of the sequence. Default is 320.

#         return_tensors : str, optional
#             The type of tensor to return. Can be either 'pt' (PyTorch) or 'np' (NumPy). Default is 'pt'.

#         return_attention_mask : bool, optional
#             Whether to return the attention mask. Default is True.

#         return_special_tokens_mask : bool, optional
#             Whether to return the special tokens mask. Default is False.

#         as_dict : bool, optional
#             Whether to return the encoded tokens as a dictionary. Default is False.

#         name : str, optional
#             The name of the dataset to be tokenized. Default is None.

#         verbose : bool, optional
#             Whether to print the number of special tokens added. Default is False.

#         Any additional kwargs will be passed directly to ``encode()`` or ``batch_encode()``, depending upon whether `text` is
#         a single string or an iterable of strings.


#         Returns
#         -------
#         BatchEncoding
#             A BatchEncoding object, with the following attributes:
#             with the following fields:
#               * input_ids — List of token ids to be fed to a model.
#               * attention_mask — List of indices specifying which tokens should be attended to by the model (when `return_attention_mask` is True or if “attention_mask” is in self.model_input_names).
#               * special_tokens_mask — List of 0s and 1s, with 1 specifying added special tokens and 0 specifying regular sequence tokens (when add_special_tokens=True and return_special_tokens_mask=True).
#         """
#         all_kwargs = {
#             "padding": padding,
#             "add_special_tokens": add_special_tokens,
#             "truncation": truncation,
#             "max_length": max_length,
#             "return_attention_mask": return_attention_mask,
#             "return_special_tokens_mask": return_special_tokens_mask,
#             "as_dict": as_dict,
#             "name": name,
#             "verbose": verbose,
#         }
#         all_kwargs.update(kwargs)
#         return self.encode(text, **all_kwargs)

#     @property
#     def vocab_size(self):
#         return len(self.all_toks)

#     def get_idx(self, tok):
#         return self.tok_to_idx.get(tok, self.unk_idx)

#     def get_tok(self, ind):
#         return self.all_toks[ind]

#     def to_dict(self):
#         return self.tok_to_idx.copy()

#     def to_json(self, output: Optional[str] = None):
#         """
#         Save the tokenizer to a JSON file.

#         Parameters
#         ----------
#         output : str, optional
#             The path to the JSON file to save the tokenizer to. If None, the tokenizer is returned as a JSON string.
#         """
#         json_string = json.dumps(self.tok_to_idx)
#         if output is not None:
#             with open(output, "w") as f:
#                 f.write(json_string)
#         else:
#             return json_string

#     def _tokenize(self, text: str) -> Iterable[str]:
#         return text.split()

#     def tokenize(self, text: str) -> Iterable[str]:
#         """
#         Converts a string into a sequence of tokens by iteratively splitting on the tokenizer's list of tokens.

#         Inspired by the ``tokenize`` method in the `HuggingFace Tokenizer`_ and `ESM Alphabet`_ classes.

#         Parameters
#         ----------
#         text : str
#             The sequence to be encoded.

#         Returns
#         -------
#         Iterable[str]
#             The list of tokens.

#         .. _HuggingFace Tokenizer:
#             https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py

#         .. _ESM's Alphabet:
#             https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91
#         """

#         def split_on_token(tok, text):
#             result = []
#             split_text = text.split(tok)
#             for i, spl_text in enumerate(split_text):
#                 spl_text = spl_text.strip()
#                 if i == 0 and not spl_text:
#                     # if the first item in the split is an empty string,
#                     # we should add the token used to split, because:
#                     # "ABC".split("A") --> ["", "B", "C"]
#                     result.append(tok)
#                 elif i == len(split_text) - 1:
#                     if spl_text:
#                         result.append(spl_text)
#                     else:
#                         # edge case with a single input token that matches the splitting token:
#                         # "A".split("A") --> ["", ""]
#                         # since we already added the token used to split due to the initial
#                         # empty string, we don't need to add it again
#                         pass
#                 else:
#                     if spl_text:
#                         result.append(spl_text)
#                     result.append(tok)
#             return result

#         def split_on_tokens(tok_list, text):
#             if not text.strip():
#                 return []
#             tokenized_text = []
#             text_list = [text]
#             for tok in tok_list:
#                 tokenized_text = []
#                 for sub_text in text_list:
#                     if sub_text not in self.unique_no_split_tokens:
#                         tokenized_text.extend(split_on_token(tok, sub_text))
#                     else:
#                         tokenized_text.append(sub_text)
#                 text_list = tokenized_text
#             return list(
#                 itertools.chain.from_iterable(
#                     (
#                         token.split()
#                         if token not in self.unique_no_split_tokens
#                         else [token]
#                         for token in tokenized_text
#                     )
#                 )
#             )

#         no_split_token = self.unique_no_split_tokens
#         tokenized_text = split_on_tokens(no_split_token, text)
#         return tokenized_text

#     def encode(
#         self,
#         text: Union[str, Iterable[str]],
#         padding: Union[bool, str] = True,
#         add_special_tokens: bool = True,
#         truncation: Union[bool, str] = True,
#         max_length: int = 320,
#         return_attention_mask: bool = False,
#         return_special_tokens_mask: bool = False,
#         as_dict: bool = False,
#         name: Optional[str] = None,
#         verbose: bool = False,
#         **kwargs,
#     ) -> Dict[str, torch.Tensor]:
#         all_kwargs = {
#             "padding": padding,
#             "add_special_tokens": add_special_tokens,
#             "truncation": truncation,
#             "max_length": max_length,
#             "return_attention_mask": return_attention_mask,
#             "return_special_tokens_mask": return_special_tokens_mask,
#             "verbose": verbose,
#         }
#         all_kwargs.update(kwargs)
#         # single input
#         if isinstance(text, str):
#             encoded = [self._encode(text, **all_kwargs)]
#         # batched inputs
#         else:
#             n_procs = min(len(text), mp.cpu_count())
#             # init progress bar before launching ProcessPoolExecutor
#             progress_bar = tqdm(
#                 total=len(text),
#                 desc="Encoding" if name is None else f"Encoding ({name})",
#             )
#             # async_results = []
#             # p = mp.Pool(processes=n_procs)
#             # for txt in text:
#             #     ar = p.apply_async(
#             #         self._encode,
#             #         args=(txt,),
#             #         kwds=all_kwargs,
#             #         callback=lambda x: progress_bar.update(1),
#             #     )
#             #     async_results.append(ar)
#             # p.close()
#             # p.join()
#             # encoded = [ar.get() for ar in async_results]
#             with ProcessPoolExecutor(max_workers=n_procs) as executor:
#                 futures = {
#                     executor.submit(self._encode, txt, **all_kwargs): txt
#                     for txt in text
#                 }
#                 encoded = []
#                 for future in as_completed(futures):
#                     result = future.result()
#                     encoded.append(result)
#                     progress_bar.update(1)
#             progress_bar.close()

#         encoded = [torch.tensor(e) for e in encoded]
#         results_dict = {
#             "input_ids": encoded,
#         }
#         # attention mask
#         if return_attention_mask:
#             attn_mask = []
#             for e in encoded:
#                 amask = torch.ones_like(e)  # 1 for non-pad tokens
#                 amask[e == self.pad_idx] = 0  # 0 for pad tokens
#                 attn_mask.append(amask)
#             results_dict["attention_mask"] = attn_mask
#         # special tokens mask
#         if return_special_tokens_mask:
#             special_tokens_mask = []
#             for e in encoded:
#                 stmask = torch.zeros_like(e)  # 0 for non-special tokens
#                 stmask[e.isin(self.all_special_tokens_idx)] = 1  # 1 for special tokens
#                 special_tokens_mask.append(stmask)
#             results_dict["special_tokens_mask"] = special_tokens_mask

#         if as_dict:
#             return results_dict
#         else:
#             return BatchEncoding(results_dict)

#     def decode(
#         self, tokens: Union[Iterable[int], torch.Tensor, int]
#     ) -> Union[str, List[str]]:
#         """
#         Decodes a sequence of tokens into a string.

#         Parameters
#         ----------
#         tokens : Union[Iterable[int], torch.Tensor, int]
#             The token(s) to decode. Can be an iterable of integers, a PyTorch tensor, or a single integer.

#         Returns
#         -------
#         Union[str, List[str]]
#             The decoded token(s). If a single integer is provided, a single string is returned.
#             If an iterable of integers is provided, a list of strings is returned.
#             If a PyTorch tensor is provided, a list of strings is returned.

#         """
#         if isinstance(tokens, int):
#             return self.idx_to_tok[tokens]
#         if isinstance(tokens, torch.Tensor):
#             # single token as a tensor
#             if not tokens.shape:
#                 return self.idx_to_tok[tokens.item()]
#             # multiple tokens as a tensor
#             tokens = tokens.tolist()
#         return [self.idx_to_tok[tok] for tok in tokens]

#     def _encode(
#         self,
#         text: str,
#         padding: Union[bool, str] = True,
#         add_special_tokens: bool = True,
#         truncation: Union[bool, str] = True,
#         max_length: int = 320,
#         **kwargs,
#     ):
#         """
#         Encodes a string in a sequence of tokens, using the tokenizer.

#         Parameters
#         ----------
#         text : str
#             The sequence to be encoded.

#         Returns
#         -------
#         List[int]
#             The list of token indices.
#         """
#         encoded = [self.tok_to_idx[tok] for tok in self.tokenize(text)]

#         if truncation:
#             trunc_length = max_length - 2 if add_special_tokens else max_length
#             if len(encoded) > trunc_length:
#                 encoded = encoded[:trunc_length]

#         # prepend bos token, if necessary
#         if add_special_tokens:
#             encoded = [self.cls_idx] + encoded + [self.eos_idx]

#         # pad and add eos token if necessary
#         if padding and len(encoded) < max_length:
#             encoded += [self.pad_idx] * (max_length - len(encoded))
#         return encoded

#     def get_special_tokens_mask(self, tokens: Iterable[int]) -> List[int]:
#         return [1 if tok in self.all_special_tokens_idx else 0 for tok in tokens]


# class BatchEncoding:
#     """
#     BatchEncoding class. Provides a way to store and manipulate batches of encoded sequences.
#     """

#     def __init__(self, data: Optional[Dict[str, Iterable[torch.Tensor]]] = None):
#         self.data = data

#     def __getitem__(self, key: str) -> torch.Tensor:
#         return self.data[key]

#     def __setitem__(self, key: str, value: Iterable[torch.Tensor]):
#         self.data[key] = value

#     def __len__(self):
#         return len(self.data)

#     def __add__(self, other):
#         self.append(other)

#     def keys(self):
#         return self.data.keys()

#     def values(self):
#         return self.data.values()

#     def items(self):
#         return self.data.items()

#     def append(self, new_data: Dict[str, Iterable[torch.Tensor]]):
#         """
#         Append new data to the batch encoding. Distinct from ``update``, as this method
#         will not add keys that do not already exist in the batch encoding.

#         Parameters
#         ----------
#         new_data : Dict[str, Iterable[torch.Tensor]]
#             The new data to add to the batch encoding.
#         """
#         if self.data is None:
#             self.data = new_data
#         else:
#             for key in self.data.keys():
#                 if key in new_data:
#                     self.data[key].extend(new_data[key])

#     def update(self, new_data: Dict[str, Iterable[torch.Tensor]]):
#         """
#         Update the batch encoding with new data. Distinct from ``append``, as this method
#         will add keys that do not already exist in the batch encoding.

#         Parameters
#         ----------
#         new_data : Dict[str, Iterable[torch.Tensor]]
#             The new data to add to the batch encoding.

#         """
#         if self.data is None:
#             self.data = new_data
#         else:
#             for key in self.data.keys():
#                 if key in new_data:
#                     self.data[key].extend(new_data[key])
#                 else:
#                     self.data[key] = new_data[key]

#     def to(self, device: Union[str, int, "torch.device"]) -> "BatchEncoding":
#         """
#         Send all values to device by calling ``BatchEncoding.to(device)``

#         Parameters
#         ----------

#         device : Union[str, torch.device]
#             The device to put the tensors on.


#         Returns
#         -------

#         BatchEncoding
#             The same instance after modification.
#         """
#         self.data = {k: v.to(device=device) for k, v in self.data.items()}


# import itertools
# import json
# import os
# from typing import Dict, Iterable, Optional, Union

# PROTEIN_TOKS = {
#     "roberta": list("ACDEFGHIKLMNPQRSTVWY"),
#     "esm": list("LAGVSERTIDPKQNFYMHWCXBUZO.-"),
# }


# class TokenizerMixin:
#     """
#     Mixin class for tokenizers. Provides methods for tokenization and encoding.
#     """

#     def _tokenize(self, text) -> str:
#         return text.split()

#     def tokenize(self, text) -> Iterable[str]:
#         """
#         Converts a string into a sequence of tokens by iteratively splitting on the tokenizer's list of tokens.

#         Inspired by the ``tokenize`` method in the `HuggingFace Tokenizer`_ and `ESM Alphabet`_ classes.

#         Parameters
#         ----------
#         text : str
#             The sequence to be encoded.

#         Returns
#         -------
#         Iterable[str]
#             The list of tokens.

#         .. _HuggingFace Tokenizer:
#             https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py

#         .. _ESM's Alphabet:
#             https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91
#         """

#         def split_on_token(tok, text):
#             result = []
#             split_text = text.split(tok)
#             for i, spl_text in enumerate(split_text):
#                 # strip white space from left and right sides of the splits
#                 spl_text = spl_text.strip()
#                 # parse text split
#                 if i == 0 and not spl_text:
#                     # if the first item in the split is an empty string,
#                     # we should add the token used to split:
#                     #
#                     # "ABC".split("A") --> ["", "B", "C"]
#                     result.append(tok)
#                 elif i == len(split_text) - 1:
#                     if spl_text:
#                         result.append(spl_text)
#                     else:
#                         # edge case with a single token input that matches the splitting token:
#                         #
#                         # "A".split("A") --> ["", ""]
#                         #
#                         # this could result in duplication of he splitting token, since we've
#                         # already replaced the first empty string:
#                         pass
#                 else:
#                     if spl_text:
#                         result.append(spl_text)
#                     result.append(tok)
#             return result

#         def split_on_tokens(tok_list, text):
#             if not text.strip():
#                 return []
#             tokenized_text = []
#             text_list = [text]
#             for tok in tok_list:
#                 tokenized_text = []
#                 for sub_text in text_list:
#                     if sub_text not in self.unique_no_split_tokens:
#                         tokenized_text.extend(split_on_token(tok, sub_text))
#                     else:
#                         tokenized_text.append(sub_text)
#                 text_list = tokenized_text
#             return list(
#                 itertools.chain.from_iterable(
#                     (
#                         self._tokenize(token)
#                         if token not in self.unique_no_split_tokens
#                         else [token]
#                         for token in tokenized_text
#                     )
#                 )
#             )

#         no_split_token = self.unique_no_split_tokens
#         tokenized_text = split_on_tokens(no_split_token, text)
#         return tokenized_text

#     def encode(
#         self,
#         text: str,
#         max_length: int = 320,
#         pad_to_max_length: bool = True,
#     ):
#         """
#         Encodes a string in a sequence of tokens, using the tokenizer.

#         Parameters
#         ----------
#         text : str
#             The sequence to be encoded.

#         max_length : int, optional
#             The maximum sequence length. Default is 320.

#         pad_to_max_length : bool, optional
#             Whether to pad the sequence to the maximum length. Default is True.

#         threads : int, optional
#             The number of threads to use. Default is 1.

#         Returns
#         -------
#         List[int]
#             The list of token indices.
#         """
#         encoded = [self.tok_to_idx[tok] for tok in self.tokenize(text)]

#         # prepend bos token, if necessary
#         if self.prepend_bos and encoded[0] != self.cls_idx:
#             encoded = [self.cls_idx] + encoded

#         # truncate and add eos token if necessary
#         if len(encoded) >= max_length:
#             encoded = encoded[:max_length]
#             if self.append_eos:
#                 encoded = encoded[:-1] + [self.eos_idx]

#         # pad and add eos token if necessary
#         else:
#             if self.append_eos:
#                 if pad_to_max_length:
#                     encoded += [self.padding_idx] * (max_length - len(encoded) - 1)
#                     encoded.append(self.eos_idx)
#                 else:
#                     encoded = encoded.append(self.eos_idx)
#             else:
#                 if pad_to_max_length:
#                     encoded += [self.padding_idx] * (max_length - len(encoded))
#         return encoded


# class Tokenizer(TokenizerMixin):
#     """
#     Tokenizer class, heavily inspired by HuggingFace's Tokenizer_ class.
#     Provides methods for loading a vocab, sequence tokenization, and encoding.

#     Parameters
#     ----------
#     vocab : str, dict
#         The vocabulary to be used. Can be either a JSON-formatted file,
#         a directory containing a vocab.json file, or a ``dict``.

#     prepend_bos : bool, optional
#         Whether to prepend the beginning of sequence token. Default is True.

#     append_eos : bool, optional
#         Whether to append the end of sequence token. Default is False.

#     cls_token : str, optional
#         The beginning of sequence token. Default is "<cls>".

#     pad_token : str, optional
#         The padding token. Default is "<pad>".

#     eos_token : str, optional
#         The end of sequence token. Default is "<eos>".

#     unk_token : str, optional
#         The unknown token. Default is "<unk>".

#     mask_token : str, optional
#         The mask token. Default is "<mask>".

#     .. _Tokenizer:
#         https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
#     """

#     def __init__(
#         self,
#         vocab: Union[str, Dict[str, int]],
#         prepend_bos: bool = True,
#         append_eos: bool = True,
#         cls_token: str = "<cls>",
#         pad_token: str = "<pad>",
#         eos_token: str = "<eos>",
#         unk_token: str = "<unk>",
#         mask_token: str = "<mask>",
#     ):
#         self.vocab = self._process_vocab(vocab)
#         self.prepend_bos = prepend_bos
#         self.append_eos = append_eos

#         self.all_toks = list(self.vocab.keys())

#         self.tok_to_idx = self.vocab

#         self.unk_idx = self.tok_to_idx[unk_token]
#         self.padding_idx = self.get_idx(pad_token)
#         self.cls_idx = self.get_idx(cls_token)
#         self.mask_idx = self.get_idx(mask_token)
#         self.eos_idx = self.get_idx(eos_token)
#         self.all_special_tokens = [
#             eos_token,
#             unk_token,
#             pad_token,
#             cls_token,
#             mask_token,
#         ]
#         self.unique_no_split_tokens = self.all_toks

#     def __len__(self):
#         return len(self.all_toks)

#     def __call__(self, text: str):
#         return self.encode(text)

#     def get_idx(self, tok):
#         return self.tok_to_idx.get(tok, self.unk_idx)

#     def get_tok(self, ind):
#         return self.all_toks[ind]

#     def to_dict(self):
#         return self.tok_to_idx.copy()

#     def _process_vocab(self, vocab: Union[str, Dict[str, int]]) -> Dict[str, int]:
#         if isinstance(vocab, str):
#             if os.path.isfile(vocab):
#                 return json.load(open(vocab, "r"))
#             elif os.path.isdir(vocab):
#                 return json.load(open(os.path.join(vocab, "vocab.json"), "r"))
#             else:
#                 raise ValueError(
#                     f"If vocab is a string, it must be a file or directory. {vocab} does not exist"
#                 )
#         elif isinstance(vocab, dict):
#             return vocab
#         else:
#             raise ValueError("Vocab must be a string or dictionary")

#     @classmethod
#     def from_pretrained(
#         cls,
#         vocab: Union[str, Dict[str, int]],
#         prepend_bos: bool = True,
#         append_eos: bool = True,
#         cls_token: str = "<cls>",
#         pad_token: str = "<pad>",
#         eos_token: str = "<eos>",
#         unk_token: str = "<unk>",
#         mask_token: str = "<mask>",
#     ) -> "Tokenizer":
#         """
#         Loads a pretrained tokenizer from a vocab.

#         Parameters
#         ----------
#         vocab : str, dict
#             The vocabulary to be used. Can be either the name of a built-in
#             vocab (e.g., "esm" or "roberta"), a JSON-formatted vocab file,
#             a directory containing a vocab.json file, or a ``dict``.

#         prepend_bos : bool, optional
#             Whether to prepend the beginning of sequence token. Default is True.

#         append_eos : bool, optional
#             Whether to append the end of sequence token. Default is False.

#         cls_token : str, optional
#             The beginning of sequence token. Default is "<cls>".

#         pad_token : str, optional
#             The padding token. Default is "<pad>".

#         eos_token : str, optional
#             The end of sequence token. Default is "<eos>".

#         unk_token : str, optional
#             The unknown token. Default is "<unk>".

#         mask_token : str, optional
#             The mask token. Default is "<mask>".
#         """
#         if isinstance(vocab, str):
#             if vocab.lower() == "esm":
#                 toks = ["<cls>", "<pad>", "<eos>", "<unk>"]
#                 toks += list(PROTEIN_TOKS["esm"])
#                 toks += ["<mask>"]
#                 return cls(
#                     vocab={t: i for i, t in enumerate(toks)},
#                     prepend_bos=True,
#                     append_bos=True,
#                     cls_token="<cls>",
#                     pad_token="<pad>",
#                     eos_token="<eos>",
#                     unk_token="<unk>",
#                     mask_token="<mask>",
#                 )
#             elif vocab.lower() == "roberta":
#                 toks = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
#                 toks += list(PROTEIN_TOKS["roberta"])
#                 return cls(
#                     vocab={t: i for i, t in enumerate(toks)},
#                     prepend_bos=True,
#                     append_bos=True,
#                     cls_token="<s>",
#                     pad_token="<pad>",
#                     eos_token="</s>",
#                     unk_token="<unk>",
#                     mask_token="<mask>",
#                 )
#             elif os.path.isdir(vocab):
#                 return cls(
#                     vocab=json.load(open(os.path.join(vocab, "vocab.json"), "r")),
#                     prepend_bos=prepend_bos,
#                     append_eos=append_eos,
#                     cls_token=cls_token,
#                     pad_token=pad_token,
#                     eos_token=eos_token,
#                     unk_token=unk_token,
#                     mask_token=mask_token,
#                 )

#             elif os.path.isfile(vocab):
#                 return cls(
#                     vocab=json.load(open(vocab, "r")),
#                     prepend_bos=prepend_bos,
#                     append_eos=append_eos,
#                     cls_token=cls_token,
#                     pad_token=pad_token,
#                     eos_token=eos_token,
#                     unk_token=unk_token,
#                     mask_token=mask_token,
#                 )
#             else:
#                 raise ValueError(f"Unknown vocab: {vocab}")
#         return cls(
#             vocab=vocab,
#             prepend_bos=prepend_bos,
#             append_eos=append_eos,
#             cls_token=cls_token,
#             pad_token=pad_token,
#             eos_token=eos_token,
#             unk_token=unk_token,
#             mask_token=mask_token,
#         )


# class Alphabet(TokenizerMixin):
#     """
#     Tokenizer class, heavily inspired by ESM's Alphabet_ class.
#     Provides methods for loading pre-defined alphabets, sequence tokenization, and encoding.

#     Parameters
#     ----------
#     standard_toks : Sequence[str], optional
#         Standard tokens to be used. Default is PROTEIN_TOKS["esm"].

#     prepend_toks : Sequence[str], optional
#         Tokens to be prepended. Default is ("<cls>", "<pad>", "<eos>", "<unk>").

#     append_toks : Sequence[str], optional
#         Tokens to be appended. Default is ("<mask>",).

#     prepend_bos : bool, optional
#         Whether to prepend the beginning of sequence token. Default is True.

#     append_eos : bool, optional
#         Whether to append the end of sequence token. Default is False.

#     cls_token : str, optional
#         The beginning of sequence token. Default is "<cls>".

#     pad_token : str, optional
#         The padding token. Default is "<pad>".

#     eos_token : str, optional
#         The end of sequence token. Default is "<eos>".

#     unk_token : str, optional
#         The unknown token. Default is "<unk>".

#     mask_token : str, optional
#         The mask token. Default is "<mask>".

#     .. _Alphabet:
#         https://github.com/facebookresearch/esm/blob/2b369911bb5b4b0dda914521b9475cad1656b2ac/esm/data.py#L91
#     """

#     def __init__(
#         self,
#         standard_toks: Iterable[str] = PROTEIN_TOKS["esm"],
#         prepend_toks: Iterable[str] = ("<cls>", "<pad>", "<eos>", "<unk>"),
#         append_toks: Iterable[str] = ("<mask>",),
#         prepend_bos: bool = True,
#         append_eos: bool = False,
#         cls_token: str = "<cls>",
#         pad_token: str = "<pad>",
#         eos_token: str = "<eos>",
#         unk_token: str = "<unk>",
#         mask_token: str = "<mask>",
#     ):
#         self.standard_toks = list(standard_toks)
#         self.prepend_toks = list(prepend_toks)
#         self.append_toks = list(append_toks)
#         self.prepend_bos = prepend_bos
#         self.append_eos = append_eos

#         self.all_toks = list(self.prepend_toks)
#         self.all_toks.extend(self.standard_toks)
#         # for i in range((8 - (len(self.all_toks) % 8)) % 8):
#         #     self.all_toks.append(f"<null_{i  + 1}>")
#         self.all_toks.extend(self.append_toks)

#         self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

#         self.unk_idx = self.tok_to_idx[unk_token]
#         self.padding_idx = self.get_idx(pad_token)
#         self.cls_idx = self.get_idx(cls_token)
#         self.mask_idx = self.get_idx(mask_token)
#         self.eos_idx = self.get_idx(eos_token)
#         self.all_special_tokens = [
#             eos_token,
#             unk_token,
#             pad_token,
#             cls_token,
#             mask_token,
#         ]
#         self.unique_no_split_tokens = self.all_toks

#     def __len__(self):
#         return len(self.all_toks)

#     def __call__(self, text: str):
#         return self.encode(text)

#     def get_idx(self, tok):
#         return self.tok_to_idx.get(tok, self.unk_idx)

#     def get_tok(self, ind):
#         return self.all_toks[ind]

#     def to_dict(self):
#         return self.tok_to_idx.copy()

#     @classmethod
#     def from_architecture(cls, name: str) -> "Alphabet":
#         if name.lower() in ("balm", "roberta"):
#             standard_toks = PROTEIN_TOKS["roberta"]
#             prepend_toks: Iterable[str] = ("<s>", "</s>", "<pad>", "<unk>", "<mask>")
#             append_toks: Iterable[str] = ()
#             cls_token = "<s>"
#             pad_token = "<pad>"
#             eos_token = "</s>"
#             unk_token = "<unk>"
#             mask_token = "<mask>"
#             prepend_bos = True
#             append_eos = True
#         elif name.lower() in ("esm"):
#             standard_toks = PROTEIN_TOKS["esm"]
#             prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
#             append_toks = ("<mask>",)
#             cls_token = "<cls>"
#             pad_token = "<pad>"
#             eos_token = "<eos>"
#             unk_token = "<unk>"
#             mask_token = "<mask>"
#             prepend_bos = True
#             append_eos = True
#         else:
#             raise ValueError("Unknown architecture selected")
#         return cls(
#             standard_toks,
#             prepend_toks,
#             append_toks,
#             cls_token,
#             pad_token,
#             eos_token,
#             unk_token,
#             mask_token,
#             prepend_bos,
#             append_eos,
#         )
