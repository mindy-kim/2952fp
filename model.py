import torch
import torch.nn as nn
from typing import Callable, Union

class Transformer(nn.Module):
    def __init__(self, size: int, dropout: float, nhead: int, dim_ff: int, timesteps: int):
        super(Transformer, self).__init__()

        self.size = size
        self.dim_ff = dim_ff
        self.timesteps = timesteps

        self.self_attn = MultiHeadAttention(size, nhead)
        self.sublayer1 = SublayerConnection(size, dropout)
        self.sublayer2 = SublayerConnection(size, dropout)

        self.out = nn.Linear(size, size) # fix dims

    def forward(self, x: torch.Tensor):
        for _ in range(self.timesteps):
            x = self.sublayer1(x, lambda x: self.self_attn(x, x, x))
            x = self.sublayer2(x, self.feed_forward)

        x = self.out(x)

        return x

class AttentionHead(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionHead, self).__init__()
        self.K = nn.Parameter(torch.randn(input_size, output_size))
        self.Q = nn.Parameter(torch.randn(input_size, output_size))
        self.V = nn.Parameter(torch.randn(input_size, output_size))

    def forward(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        K = torch.tensordot(inputs_for_keys, self.K, 1)
        V = torch.tensordot(inputs_for_values, self.V, 1)
        Q = torch.tensordot(inputs_for_queries, self.Q, 1)

        scores = torch.matmul(Q, K.T) # double check if this is right
        weighted = scores @ V
        return weighted


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.heads = []

        for _ in range(self.num_heads):
            self.heads.append(AttentionHead(emb_size*self.num_heads, emb_size))

        # self.combine_heads = nn.Linear(emb_size, emb_size) # is this needed..?

    def forward(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        out = torch.zeros_like(inputs_for_keys) # don't know if this is right dim

        for i in range(self.num_heads):
            out += self.heads[i](inputs_for_keys, inputs_for_values, inputs_for_queries)

        # out = self.combine_heads(out)
        return out


class SublayerConnection(nn.Module):
    '''Applies a residual connection followed by a layer norm to any sublayer'''
    def __init__(self, size: int, dropout: float):
        """
        Initializes a SublayerConnection module

        Parameters
        ----------
        size : int
            size of the expected input to the module
        dropout : float
            dropout value to be used after the sublayer
        """
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Union[nn.Module, Callable]) -> torch.Tensor:
        """
        Forward pass of the model. Normalizes the input, applies the sublayer,
        performs a dropout, and then performs a residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))
    
class PositionwiseFeedForward(nn.Module):
    ''' Implements the two-layer feedforward neural network used in the transformer.'''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a PositionwiseFeedForward module
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. Normalizes the input, applies the sublayer,
        performs a dropout, and then performs a residual connection.
        """
        return self.w_2(self.dropout(self.w_1(x).relu()))