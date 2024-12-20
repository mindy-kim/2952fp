import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import math
from scipy.special import factorial
from tqdm import tqdm

def gamma(x):
    return math.exp(torch.lgamma())

def empricalEps(forgetW, Nx, Ny, thresh=0.025, num_trials=1000):
    candEps = [0.1 + 0.1 * i for i in range(50)]
    closestDist = float('inf')
    chosenEps = None
    for epsW in tqdm(candEps, desc='selecting epsW'):
        tot = 0
        for i in range(num_trials):
            wgts = torch.randn((1000, Ny, Nx))
            mask = F.norm(wgts-forgetW, dim=(-2,-1), keepdim=True) < epsW
            tot += torch.mean(mask.to(dtype=torch.float)).item()
        tot /= num_trials
        if tot > 2 * thresh:
            break
        if abs(tot - thresh) < closestDist:
            closestDist = abs(tot - thresh)
            chosenEps = epsW
    
    print(f"Selected {chosenEps}")
    return chosenEps

class FullData(Dataset):
    def __init__(self, *, C, Nx, Ny, N=None):
        super().__init__()
        C = C + 1 # for query token
        self.N, self.C = N, C
        self.Nx, self.Ny = Nx, Ny

        # full dataset
        if N:
            self.weights = torch.randn((N, Ny, Nx))
            self.xs = torch.rand((N, C, Nx)) * 2.0 - 1.0
            self.ys = torch.einsum('ijk,ick->icj', self.weights, self.xs)

        # initialize forget tasks

    def __len__(self):
        return self.N if self.N else int(1e8)
    
    def __getitem__(self, idx):
        out_dict = {}
        if self.N:
            out_dict['xs'], out_dict['ys'], out_dict['weights'] = self.xs[idx], self.ys[idx], self.weights[idx]
        else:
            weights = torch.randn((self.Ny, self.Nx))
            xs = torch.rand((self.C, self.Nx)) * 2.0 - 1.0
            ys = torch.einsum('jk,ck->cj', weights, xs)
            out_dict['xs'], out_dict['ys'], out_dict['weights'] = xs, ys, weights
            
        return out_dict

class MULData(Dataset):
    def __init__(self, *, C, Nx, Ny, N=None, Nf=1, forgetThresh=0.025):
        super().__init__()
        C = C + 1 # for query token
        self.N, self.Nf, self.C = N, Nf, C
        self.Nx, self.Ny = Nx, Ny
        self.epsW = torch.zeros(Nf)
        self.forgetThresh = forgetThresh / Nf

        # full dataset
        if N:
            self.weights = torch.randn((N, Ny, Nx))
            self.xs = torch.rand((N, C, Nx)) * 2.0 - 1.0
            self.ys = torch.einsum('ijk,ick->icj', self.weights, self.xs)

        # initialize forget tasks
        self.weightsF = torch.randn((Nf, Ny, Nx))
        for i in range(Nf):
            self.epsW[i] = empricalEps(self.weightsF[i], Nx, Ny, thresh=self.forgetThresh)
        dim = Nx * Ny
        self.EN = 1 / (2 ** (1 / 2) * factorial((dim + 1) / 2) / factorial(dim / 2)) # normalizing factor for generation

    def __len__(self):
        return self.N if self.N else int(1e8)
    
    def __getitem__(self, idx):
        out_dict = {}
        if self.N:
            out_dict['xs'], out_dict['ys'], out_dict['weights'] = self.xs[idx], self.ys[idx], self.weights[idx]
        else:
            xs, ys, weights = self.sample_dr(1) # retain 
            out_dict['xs'], out_dict['ys'], out_dict['weights'] = xs[0], ys[0], weights[0]

            xsF, ysF, weightsF = self.sample_df(1) # forget
            out_dict['xsF'], out_dict['ysF'], out_dict['weightsF'] = xsF[0], ysF[0], weightsF[0]
            
        return out_dict
    
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
        
        weights = orig_weights + self.epsW[tIdx].view(-1,1,1) * noise
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
                mask = torch.logical_or(F.norm(weights-self.weightsF[i], dim=(-2,-1), keepdim=True) < self.epsW[i], mask)
            
            if torch.any(mask):
                weights = torch.where(mask, torch.randn((N, self.Ny, self.Nx)), weights)
                continue
            
            break
            
        xs = torch.rand((N, self.C, self.Nx)) * 2.0 - 1.0
        ys = torch.einsum('ijk,ick->icj', weights, xs)
        return xs, ys, weights


if __name__ == '__main__':
    ''' BASELINE DATA '''
    # start = time.perf_counter()
    # xs, ys, weights = data.sample_df(10000)
    # end = time.perf_counter()
    # print(end - start)
    # print(xs.size(), ys.size(), weights.size())

    ''' UNLEARNING DATA '''
    NUM_TRIALS = 1000
    tot = 0
    for i in range(NUM_TRIALS):
        data = MULData(N=10000, Nx=10, Ny=1, C=10, epsW=2.5)
        wgts = data.weights
        forgetW = data.weightsF[0]
        mask = F.norm(wgts-forgetW, dim=(-2,-1), keepdim=True) < data.epsW
        tot += torch.mean(mask.to(dtype=torch.float)).item()
    print(tot / NUM_TRIALS)

        
