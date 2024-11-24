import torch
import torch.nn as nn
from typing import Callable, Union

from models.utils import LinearCustom

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_key=None, d_val=None):
        super(MultiHeadAttention, self).__init__()
        if not d_key:
            d_key = d_model
        if not d_val:
            d_val = d_model

        self.n_head = n_head
        self.w_q = LinearCustom(d_model, d_model * n_head)
        self.w_k = LinearCustom(d_model, d_model * n_head)
        self.w_v = LinearCustom(d_model, d_model * n_head)
        self.w_p = LinearCustom(d_model * n_head, d_model)

    def forward(self, q, k, v, is_concat=False):
        Bq, Tq, Dq = q.size()
        Bv, Tv, Dv = v.size()
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q = q.view(Bq, Tq, self.n_head, Dq)
        k = k.view(Bv, Tv, self.n_head, Dv)
        v = v.view(Bv, Tv, self.n_head, Dv)
        
        attn_weights = torch.einsum('...thd,...Thd->...htT', q, k)
        attn_out = torch.einsum('...htT,...Thd->...thd', attn_weights, v)
        attn_out = attn_out.reshape([Bq, Tq, -1])

        out = self.w_p(attn_out)

        return out

    def attention(self, q, k, v):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        k_t = k.transpose(2, 3)
        score = q @ k_t
        weighted = score @ v

        return weighted

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