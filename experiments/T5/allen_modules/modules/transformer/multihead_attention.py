from typing import Tuple, Callable
import math
import warnings

import torch
import torch.nn.functional as F

from allennlp.common.checks import ExperimentalFeatureWarning
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.common import Registrable
from allennlp.nn import util


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.BoolTensor = None,
    dropout: Callable = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, util.min_value_of_dtype(scores.dtype))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class CrossMultiHeadedAttention(torch.nn.Module):
    def __init__(self, num_heads: int, input_dim: int, encoder_output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be a multiple of num_heads"
        # We assume d_v always equals d_k
        self.d_k = input_dim // num_heads
        self.num_heads = num_heads
        # These linear layers are
        #  [query_projection, key_projection, value_projection, concatenated_heads_projection]
        self.linears = util.clone(torch.nn.Linear(input_dim, input_dim), 4)
        self.linears[1] = torch.nn.Linear(encoder_output_dim, input_dim)
        self.linears[2] = torch.nn.Linear(encoder_output_dim, input_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        if mask is not None:
            # Same mask applied to all h heads.
            # Shape (batch_size, num_heads, timesteps, timesteps)
            mask = mask.unsqueeze(1).expand([-1, self.num_heads, -1, -1])

        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            layer(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for layer, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)