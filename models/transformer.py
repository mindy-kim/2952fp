import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Callable, Union
from attention import MultiHeadAttention, PositionwiseFeedForward, SublayerConnection

class Transformer(pl.LightningModule):
    def __init__(self, size: int, dropout: float, nhead: int, dim_ff: int, timesteps: int):
        super(Transformer, self).__init__()

        self.size = size
        self.dim_ff = dim_ff
        self.timesteps = timesteps

        self.self_attn = MultiHeadAttention(size, nhead)
        self.feed_forward = PositionwiseFeedForward(size, dim_ff, dropout)
        self.sublayer1 = SublayerConnection(size, dropout)
        self.sublayer2 = SublayerConnection(size, dropout)

        self.out = nn.Linear(size, size) # fix dims

    def forward(self, x: torch.Tensor):
        for _ in range(self.timesteps):
            x = self.sublayer1(x, lambda x: self.self_attn(x, x, x))
            x = self.sublayer2(x, self.feed_forward)
            x = self.out(x)

        return x