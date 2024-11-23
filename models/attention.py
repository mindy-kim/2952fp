import torch
import torch.nn as nn
from typing import Callable, Union

class LinearSelfAttention(nn.Module):
    def __init__(self):
        super(LinearSelfAttention, self).__init__()

    def forward(self, q, k, v):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        k_t = k.transpose(2, 3)
        score = q @ k_t
        weighted = score @ v

        return weighted


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_out, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.self_attention = LinearSelfAttention()
        self.w_q = nn.Linear(d_model, d_out)
        self.w_k = nn.Linear(d_model, d_out - 1)
        self.w_v = nn.Linear(d_model, d_out - 1)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, is_concat=True):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        out = self.self_attention(q, k, v)
        
        if is_concat:
            out = self.concat(out)
        else:
            out = torch.sum(out, dim=1)

        out = self.w_concat(out)

        return out

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