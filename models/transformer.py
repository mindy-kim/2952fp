import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Callable, Union
from models.attention import TransformerBlock

class Transformer(pl.LightningModule):
    def __init__(self, *, 
                 num_layers: int, 
                 mlp_layers: list[int], 
                 dim_x: int,
                 dim_y: int, 
                 dropout: float, 
                 nhead: int, 
                 dim_ff = None,
                 proj_out: bool):
        super().__init__()
        
        
        dmodel = dim_x + dim_y
        self.d_model, self.N_x, self.N_y = dmodel, dim_x, dim_y
        if not dim_ff:
            dim_ff = 4 * dmodel
        self.proj_out = proj_out

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            use_mlp = True if i in mlp_layers else False
            self.layers.append(TransformerBlock(dmodel, dropout, nhead, dim_ff, use_mlp=use_mlp))

        self.out = nn.Linear(dmodel, dmodel) if proj_out else nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        xs, ys, weight = batch
        # mask y_true for query token
        mask = torch.ones_like(ys)
        mask[:, -1, :] = 0

        embs = torch.cat([xs, ys * mask], dim=-1)
        pred = self.forward(embs)

        if self.proj_out:
            y_pred = self.out(pred[:, -1, -self.N_y:])
        else:    
            y_pred = -pred[:, -1, -self.N_y:]
        y_true = ys[:, -1, :]

        loss = F.mse_loss(y_pred, y_true)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def parameters(self):
        for layer in self.layers:
            yield from layer.parameters()

        yield from self.out.parameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer