import torch
import torch.nn as nn
from torch.nn.functional import softmax
from sandstone.models.pools.abstract_pool import AbstractPool
from sandstone.models.pools.factory import RegisterPool
import numpy as np
import pdb


@RegisterPool('Simple_AttentionPool')
class Simple_AttentionPool(AbstractPool):
    def __init__(self, args, num_chan):
        super(Simple_AttentionPool, self).__init__(args, num_chan)

        self.attention_fc = nn.Linear(num_chan, 1)
        self.softmax = nn.Softmax(dim=-1)

    def replaces_fc(self):
        return False

    def forward(self, x, batch=None):
        # X dim: B, L, N
        x = x.permute(0,2,1) #B, N, L
        attention_scores = self.attention_fc(x.transpose(1,2)) #B, L, 1
        attention_scores = self.softmax( attention_scores.transpose(1,2)) #B, 1, L
        x = x * attention_scores #B, N, L
        x = torch.sum(x, dim=-1) #B, N
        return x
