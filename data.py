import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader

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
        
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs, self.ys, self.weights
    
    def sample_df(self, N):
        """
        Get N samples from the forget set
        """
        tIdx = torch.randint(0, self.Nf, (N,1))
        orig_weights = self.weightsF[tIdx] 

        noise = torch.randn_like(orig_weights)
        while True:
            mask = F.norm(noise) <= 1.0
            if torch.any(mask):
                noise = torch.where(mask, torch.randn_like(orig_weights), noise)
                continue

            break

        weights = orig_weights + self.epsW * noise
        xs = torch.rand((N, self.C, self.Nx)) * 2.0 - 1.0
        ys = torch.einsum('ijk,ick->icj', self.weights, xs)
        return xs, ys, weights
    
    def sample_rf(self, N):
        """
        Get N samples from the retain set
        """
        weights = torch.randn((N, self.Ny, self.Nx))
        while True:
            mask = torch.zeros()
            for i in range(self.Nf):
                mask = torch.logical_or(F.norm(weights-self.weightsF[i], dim=(-2,-1), keepdim=True) <= self.epsW, mask)
            
            if torch.any(mask):
                weights = torch.where(mask, torch.randn((N, self.Ny, self.Nx)), weights)
                continue
            
            break
            
        xs = torch.rand((N, self.C, self.Nx)) * 2.0 - 1.0
        ys = torch.einsum('ijk,ick->icj', weights, xs)

        return xs, ys, weights
        
