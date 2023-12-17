#!/usr/bin/python
# filename: balm.py

#
# Copyright (c) 2023 Bryan Briney
# License: GNU General Public License, version 3.0 (http://opensource.org/licenses/gpl-3-0/)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


from typing import Optional

# import lightning as L
import torch
import torch.nn as nn

from ..config import BALMConfig
from ..embedding import RelativePositionalEmbedding
from ..modules import MaskedLMHead, TransformerLayer


class BalmModel(nn.Module):
    """
    Base BALM model.
    """

    def __init__(self, config: BALMonfig):
        super().__init__()
        self.config = config
        self.position_embedding_type = config.position_embedding_type
        # token embedding
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        # position embedding
        if config.position_embedding_type in ["relative_key", "relative_key_query"]:
            self.position_embeddings = RelativePositionalEmbedding(config.hidden_size)
        else:  # absolute
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        # layers
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_layers)]
        )
        self.embedding_dropout = nn.Dropout(config.embedding_dropout_rate)
        self.final_norm = nn.LayerNorm(config.embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Parameters:
        -----------

        input_ids: torch.LomgTensor
            Tokenized input IDs

        attention_mask: torch.BoolTensor
            Attention mask

        output_attentions: bool
            Whether to output attention weights

        output_hidden_states: bool
            Whether to output hidden states

        return_dict: bool
            Whether to return a dictionary of outputs (returns a tuple by default)


        Returns:
        --------
        output (tuple or dict):
            If `return_dict` is ``True``, the output is a ``dict`` of outputs:
                - last_hidden_state (torch.FloatTensor): last hidden state
                - attentions (torch.FloatTensor): attention weights
                - hidden_states (torch.FloatTensor): hidden states
            If `return_dict` is ``False``, the output is a ``tuple`` with the same three elements.
        """
        # init
        attn_weights = []
        hidden_states = {}

        # embeddings
        x = self.embed_scale * self.embed_tokens(input_ids)
        if self.position_embeddings_type in ["relative_key", "relative_key_query"]:
            x = self.position_embeddings(x)
        else:  # absolute embeddings
            position_ids = torch.arange(
                input_ids.size(1), dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeds = self.position_embeddings(position_ids)
            x = x + position_embeds
        x = self.embedding_dropout(x)

        # encoder
        x = x.transpose(0, 1)
        for layer_idx, layer in enumerate(self.layers, 1):
            x = layer(x, attention_mask, need_weights=output_attentions)
            if output_attentions:
                x, attn = x
                attn_weights.append(attn)
            else:
                x = x
            if output_hidden_states:
                hidden_states[layer_idx] = x.transpose(0, 1)
        x = self.final_norm(x)
        x = x.transpose(0, 1)

        # results
        outputs = [x]
        result = {
            "last_hidden_state": x,
            "router_z_loss": z_loss,
            "router_aux_loss": aux_loss,
        }
        if output_attentions:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            outputs.append(attentions)
        if output_hidden_states:
            result["hidden_states"] = hidden_states
            outputs.append(hidden_states)
        if return_dict:
            return result
        return tuple(outputs)


class BalmForMaskedLM(nn.Module):
    """
    BALM for Masked Language Modeling.
    """

    def __init__(self, config: BALMMoEConfig):
        super().__init__()
        self.config = config
        self.balm = Balm(config)
        self.lm_head = MaskedLMHead(
            embed_dim=self.balm.embed_dim,
            output_dim=self.balm.vocab_size,
            weight=self.balm.embed_tokens.weight,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """
        Args:
            input_ids (torch.LongTensor): tokenized input IDs
            attention_mask (torch.BoolTensor): attention mask
            return_dict (bool): return a dictionary of outputs
        """
        # encoder
        encoder_outputs = self.balm_moe(
            input_ids,
            attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        x = encoder_outputs["last_hidden_state"]

        # LM head
        lm_logits = self.lm_head(x)

        # loss
        if labels is not None:
            loss_func = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device
            labels = labels.to(lm_logits.device)
            loss = loss_func(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            encoder_outputs["loss"] = loss

        # return
        if return_dict:
            encoder_outputs["logits"] = lm_logits
            return encoder_outputs
        else:
            outputs = [lm_logits]
            if output_attentions:
                outputs.append(encoder_outputs["attentions"])
            if output_hidden_states:
                outputs.append(encoder_outputs["hidden_states"])
            return tuple(outputs)


# import sys
# from typing import Union

# from transformers import RobertaConfig, RobertaForMaskedLM


# __all__ = ["get_model", "BALM_PARAMS"]


# def get_model(model_config: Union[str, RobertaConfig, None] = None):
#     """
#     Retrieves a BALM model, given a model configuration.

#      Parameters
#      ----------
#      model_config : Union[str, RobertaConfig, None], default=None
#          Can be either a ``RobertaConfig`` object, or the name of a built-in
#          model. Built-in model options are:
#              * ``"balm"``: default BALM model size, based on RoBERTa-large
#              * ``"balm-small"``: reduced size BALM model, based on RoBERTa
#              * ``"balm-large"``: based on LLaMA 7B
#              * ``"balm-xlarge"``: based on LLaMA 13B
#              * ``"balm-huge"``: based on LLaMA 33B
#              * ``"balm-gigantic"``: based on LLaMA 65B
#          Default is ``"balm"``.

#      Returns
#      -------
#      RobertaConfig
#          Model configuration
#     """
#     if isinstance(model_config, RobertaConfig):
#         config = model_config
#     else:
#         if model_config is None:
#             model_config = "balm"
#         if model_config not in BALM_PARAMS:
#             err = "\nERROR: Invalid model name. Options are:\n  -"
#             err += "\n  - ".join(BALM_PARAMS.keys())
#             err += "\n\n"
#             print(err)
#             sys.exit(1)
#         params = BALM_PARAMS[model_config.lower()]
#         config = RobertaConfig(**params)
#     return RobertaForMaskedLM(config)


# BALM_PARAMS = {
#     # based on RoBERTa-large
#     "balm": {
#         "vocab_size": 25,
#         "hidden_size": 1024,
#         "intermediate_size": 4096,
#         "max_position_embeddings": 320,
#         "num_hidden_layers": 24,
#         "num_attention_heads": 16,
#         "type_vocab_size": 2,
#     },
#     # based on RoBERTa
#     "balm-small": {
#         "vocab_size": 25,
#         "hidden_size": 768,
#         "intermediate_size": 3072,
#         "max_position_embeddings": 320,
#         "num_hidden_layers": 12,
#         "num_attention_heads": 12,
#         "type_vocab_size": 2,
#     },
#     # based on LLaMA 7B
#     "balm-large": {
#         "vocab_size": 25,
#         "hidden_size": 1024,
#         "intermediate_size": 4096,
#         "max_position_embeddings": 320,
#         "num_hidden_layers": 32,
#         "num_attention_heads": 32,
#         "type_vocab_size": 2,
#     },
#     # based on LLaMA 13B
#     "balm-xlarge": {
#         "vocab_size": 25,
#         "hidden_size": 1280,
#         "intermediate_size": 5120,
#         "max_position_embeddings": 320,
#         "num_hidden_layers": 40,
#         "num_attention_heads": 40,
#         "type_vocab_size": 2,
#     },
#     # based on LLaMA 33B
#     "balm-huge": {
#         "vocab_size": 25,
#         "hidden_size": 1664,
#         "intermediate_size": 6656,
#         "max_position_embeddings": 320,
#         "num_hidden_layers": 52,
#         "num_attention_heads": 60,
#         "type_vocab_size": 2,
#     },
#     # based on LLaMA 65B
#     "balm-gigantic": {
#         "vocab_size": 25,
#         "hidden_size": 2048,
#         "intermediate_size": 8192,
#         "max_position_embeddings": 320,
#         "num_hidden_layers": 64,
#         "num_attention_heads": 80,
#         "type_vocab_size": 2,
#     },
# }