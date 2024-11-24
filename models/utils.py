import torch
import torch.nn as nn
from typing import Callable, Union

class LinearCustom(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearCustom, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.xavier_normal_(self.linear.weight, gain=0.002)
    
    def forward(self, x):
        return self.linear(x)
    

class PositionwiseFeedForward(nn.Module):
    ''' Implements the two-layer feedforward neural network used in the transformer.'''
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Initializes a PositionwiseFeedForward module
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = LinearCustom(d_model, d_ff)
        self.w_2 = LinearCustom(d_ff, d_model)
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
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Union[nn.Module, Callable]) -> torch.Tensor:
        """
        Forward pass of the model. Normalizes the input, applies the sublayer,
        performs a dropout, and then performs a residual connection.
        """
        return x + self.dropout(sublayer(x))
