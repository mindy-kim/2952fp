import torch
import torch.nn as nn
import torch.nn.functional as F

class Hijack():
    def __init__(self, model, num_steps, batch_sz, lr):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.batch_sz = batch_sz
        self.lr = lr

    def forward(self, x, inds, tokens):
        x[:, inds, 1] = tokens
        out = self.model(x)
        out = self.read_y(out)
        return out
    
    def train(self, dataloader):
        total_loss = 0

        for batch_idx, batch in range(dataloader):
            total_loss += self.training_steps(batch)

            self.model.log("hijack_avg_loss", total_loss / (batch_idx + 1), on_step=False, on_epoch=True, prog_bar=True)

        return total_loss / len(dataloader)
    
    def training_steps(self, batch):
        xs, ys, weights = batch['xs'], batch['ys'], batch['weights']
        xsF, ysF, weightsF = batch['xsF'], batch['ysF'], batch['weightsF']

        # mask y_true for query token
        mask = torch.ones_like(ys)
        mask[:, -1, :] = 0

        embs = torch.cat([xs, ys * mask], dim=-1)
        embsF = torch.cat([xsF, ysF * mask], dim=-1)

        hijacked_inds = torch.randint(low=0, high=embs.shape[1] - 1, size=(embs.shape[0],))
        hijacked_tokens = nn.Parameter(torch.rand(embs.shape[0]))
        optimizer = torch.optim.Adam(hijacked_tokens, lr=self.lr)

        for _ in range(self.num_steps):
            y_pred = self.forward(embs, hijacked_inds, hijacked_tokens)
            y_true = torch.zeros_like(ys[:, -1, :])

            y_predF = self.forward(embsF, hijacked_inds, hijacked_tokens)
            y_trueF = ysF[:,-1,:]

            loss = F.mse_loss(y_pred, y_true)
            lossF = F.mse_loss(y_predF, y_trueF)
            train_loss = self.lam1 * loss + self.lam2 * lossF

            self.model.log_dict(
                {"hijack_loss/retain_loss": loss, "hijack_loss/forget_loss": lossF},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.model.log("hijack_train_loss", train_loss, on_step=False, on_epoch=True, prog_bar=True)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        return train_loss