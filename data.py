import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math
from scipy.special import factorial


def gamma(x):
    return math.exp(torch.lgamma())

class MULData(Dataset):
    def __init__(self, N, C, Nx, Ny, Nf=1, epsW=0.05):
        super().__init__()
        self.N, self.Nf, self.C = N, Nf, C
        self.Nx, self.Ny = Nx, Ny
        self.epsW = epsW

        # full dataset
        self.weights = torch.randn((N, Ny, Nx))
        self.xs = torch.rand((N, C, Nx)) * 2.0 - 1.0
        self.ys = torch.einsum('ijk,ick->icj', self.weights, self.xs)

        # initialize forget tasks
        self.weightsF = torch.randn((Nf, Ny, Nx))
        dim = Nx * Ny
        self.EN = 1 / (2 ** (-1/2) * dim * factorial((dim + 1) / 2) / factorial((dim + 2) / 2)) # normalizing factor for generation
    
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs, self.ys, self.weights
    
    def sample_df(self, N):
        """
        Get N samples from the forget set
        """
        tIdx = torch.randint(0, self.Nf, (N,))
        orig_weights = self.weightsF[tIdx] 

        noise = self.EN * torch.randn_like(orig_weights)
        while True:
            mask = F.norm(noise, dim=(-2,-1), keepdim=True) > 1.0
            if torch.any(mask):
                noise = torch.where(mask, self.EN * torch.randn_like(orig_weights), noise)
                continue
            
            break

        weights = orig_weights + self.epsW * noise
        xs = torch.rand((N, self.C, self.Nx)) * 2.0 - 1.0
        ys = torch.einsum('ijk,ick->icj', weights, xs)
        return xs, ys, weights
    
    def sample_dr(self, N):
        """
        Get N samples from the retain set
        """
        weights = torch.randn((N, self.Ny, self.Nx))
        while True:
            mask = torch.zeros((N,1,1))
            for i in range(self.Nf):
                mask = torch.logical_or(F.norm(weights-self.weightsF[i], dim=(-2,-1), keepdim=True) < self.epsW, mask)
            
            if torch.any(mask):
                weights = torch.where(mask, torch.randn((N, self.Ny, self.Nx)), weights)
                continue
            
            break
            
        xs = torch.rand((N, self.C, self.Nx)) * 2.0 - 1.0
        ys = torch.einsum('ijk,ick->icj', weights, xs)

        return xs, ys, weights


if __name__ == '__main__':
    data = MULData(10000, 12, 20, 10, epsW=13.84)
    start = time.perf_counter()
    xs, ys, weights = data.sample_dr(10000)
    end = time.perf_counter()
    print(end - start)
    print(xs.size(), ys.size(), weights.size())
        
