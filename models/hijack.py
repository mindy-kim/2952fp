import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Hijack():
    def __init__(self, model, num_steps, batch_sz, lr):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.batch_sz = batch_sz
        self.lr = lr
        # self.logger = logger

    def forward(self, x, inds, tokens):
        x[:, inds, 1] = tokens
        out = self.model(x)
        out = self.model.read_y(out)
        return out
    
    def train(self, dataloader):
        total_loss = 0
        print('start training')

        for batch_idx, batch in enumerate(dataloader):
            total_loss += self.training_steps(batch)

            # self.logger.log("hijack_avg_loss", total_loss / (batch_idx + 1), on_step=False, on_epoch=True, prog_bar=True)

        print(total_loss / len(dataloader))
        return total_loss / len(dataloader)
    
    def training_steps(self, batch):
        xs, ys, weights = batch['xs'], batch['ys'], batch['weights']
        xsF, ysF, weightsF = batch['xsF'], batch['ysF'], batch['weightsF']

        # mask y_true for query token
        mask = torch.ones_like(ys)
        mask[:, -1, :] = 0

        embs = torch.cat([xs, ys * mask], dim=-1)
        embsF = torch.cat([xsF, ysF * mask], dim=-1)

        hijacked_inds = np.random.randint(low=0, high=embs.shape[1] - 1, size=(embs.shape[0],))
        hijacked_tokens = torch.tensor(np.random.rand(embs.shape[0]), dtype=torch.float32)
        hijacked_tokens.requires_grad = True
        optimizer = torch.optim.Adam([hijacked_tokens], lr=self.lr)

        for step in range(self.num_steps):
            print(step)
            y_pred = self.forward(embs, hijacked_inds, hijacked_tokens)
            y_true = torch.zeros_like(ys[:, -1, :])

            y_predF = self.forward(embsF, hijacked_inds, hijacked_tokens)
            y_trueF = ysF[:,-1,:]

            loss = F.mse_loss(y_pred, y_true)
            lossF = F.mse_loss(y_predF, y_trueF)
            train_loss = self.model.lam1 * loss + self.model.lam2 * lossF

            metrics = {
                "hijack_loss/retain_loss": loss, 
                "hijack_loss/forget_loss": lossF,
                "hijack_train_loss": train_loss
            }

            # self.logger.log_metrics(metrics, step=step)

            optimizer.zero_grad()
            train_loss.backward(retain_graph=True)
            optimizer.step()

        # hijacked_tokens.requires_grad_(False)
        print(metrics)

        return train_loss