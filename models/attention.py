import torch
import torch.nn as nn
from typing import Callable, Union

class TransformerBlock(nn.Module):
    def __init__(self, dmodel: int, dropout: float, nhead: int, dim_ff: int, use_mlp: bool):
        super(TransformerBlock, self).__init__()

        self.dmodel = dmodel
        self.dim_ff = dim_ff
        self.use_mlp = use_mlp

        self.self_attn = MultiHeadAttention(dmodel, nhead)
        self.sublayer1 = SublayerConnection(dmodel, dropout)

        if use_mlp:
            self.feed_forward = PositionwiseFeedForward(dmodel, dim_ff, dropout)
            self.sublayer2 = SublayerConnection(dmodel, dropout)

    def forward(self, x: torch.Tensor):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x[:,:-1,:], x[:,:-1,:]))
        if self.use_mlp:
            x = self.sublayer2(x, self.feed_forward)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_key=None, d_val=None):
        super(MultiHeadAttention, self).__init__()
        if not d_key:
            d_key = d_model
        if not d_val:
            d_val = d_model

        self.n_head = n_head
        self.w_q = nn.Linear(d_model, d_model * n_head)
        self.w_k = nn.Linear(d_model, d_model * n_head)
        self.w_v = nn.Linear(d_model, d_model * n_head)
        self.w_p = nn.Linear(d_model * n_head, d_model)

    def forward(self, q, k, v, is_concat=False):
        Bq, Tq, Dq = q.size()
        Bv, Tv, Dv = v.size()
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(Bq, Tq, self.n_head, Dq)
        k = k.view(Bv, Tv, self.n_head, Dv)
        v = v.view(Bv, Tv, self.n_head, Dv)
        
        attn_weights = torch.einsum('...thd,...Thd->...htT', q, k)
        attn_out = torch.einsum('...htT,...Thd->...thd', attn_weights, v)
        attn_out = attn_out.reshape([Bq, Tq, -1])

        out = self.w_p(attn_out)

        return out

    def attention(self, q, k, v):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        k_t = k.transpose(2, 3)
        score = q @ k_t
        weighted = score @ v

        return weighted

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


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