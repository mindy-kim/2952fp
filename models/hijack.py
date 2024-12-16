import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import cycle
from collections import defaultdict

class Hijack(nn.Module):
    def __init__(self, num_steps, batch_sz, num_batches, lr, num_tokens=1):
        super().__init__()
        self.model = None
        self.num_batches = num_batches
        self.num_steps = num_steps
        self.batch_sz = batch_sz
        self.lr = lr
        self.num_tokens = num_tokens
        self.metrics = defaultdict(str)

        
    def forward(self, x, inds, tokens):
        x[:, inds, 1] = tokens
        out = self.model(x)
        out = self.model.read_y(out)
        return out
    
    def train(self, model, dataset):
        self.model = model
        self.metrics["loss/retain_loss"] = defaultdict(int)
        self.metrics["loss/forget_loss"] = defaultdict(int)
        # self.metrics["loss/total_loss"] = defaultdict(int)

        for _ in range(self.num_batches):
            out_dict = {}
            out_dict['xs'], out_dict['ys'], out_dict['weights'] = dataset.sample_dr(self.batch_sz)
            out_dict['xsF'], out_dict['ysF'], out_dict['weightsF'] = dataset.sample_df(self.batch_sz)
            self.training_steps(out_dict)

        return self.metrics
    
    def training_steps(self, batch):
        xs, ys, weights = batch['xs'], batch['ys'], batch['weights']
        xsF, ysF, weightsF = batch['xsF'], batch['ysF'], batch['weightsF']

        # mask y_true for query token
        mask = torch.ones_like(ys)
        mask[:, -1, :] = 0

        embs = torch.cat([xs, ys * mask], dim=-1)
        embsF = torch.cat([xsF, ysF * mask], dim=-1)

        hijacked_inds = np.array([np.random.choice(range(embs.shape[1] - 1), 
                                                   size=self.num_tokens, 
                                                   replace=False) for _ in range(embs.shape[0])])

        # hijacked_inds = np.random.randint(low=0, high=embs.shape[1] - 1, size=(embs.shape[0], self.num_tokens))
        hijacked_tokens = torch.tensor(np.random.rand(embs.shape[0], self.num_tokens, 2), dtype=torch.float32)
        hijacked_tokens.requires_grad = True
        optimizer = torch.optim.Adam([hijacked_tokens], lr=self.lr)

        for step in range(self.num_steps):
            y_pred = self.forward(embs, hijacked_inds, hijacked_tokens[:, :, 0])
            y_true = torch.zeros_like(ys[:, -1, :])

            y_predF = self.forward(embsF, hijacked_inds, hijacked_tokens[:, :, 1])
            y_trueF = ysF[:,-1,:]

            loss = F.mse_loss(y_pred, y_true)
            lossF = F.mse_loss(y_predF, y_trueF)
            train_loss = self.model.lam1 * loss + self.model.lam2 * lossF

            self.metrics["loss/retain_loss"][step] += (loss.item() / self.num_batches)
            self.metrics["loss/forget_loss"][step] += (lossF.item() / self.num_batches)
            # self.metrics["loss/total_loss"][step] += (train_loss.item() / self.num_batches)

            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()

        print(train_loss.item())