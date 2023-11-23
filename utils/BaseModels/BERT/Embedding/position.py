import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=1024):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(self.pe[0, :x.size(1)])
        return self.pe[:, :x.size(1)]


class PositionalEmbeddingLearnable(nn.Module):
    def __init__(self, input_lenth, embedding_dim=1024):
        super().__init__()
        self.input_lenth = input_lenth
        self.embedding_dim = embedding_dim
        self.position_embedding = nn.Parameter(torch.randn(1, self.input_lenth, self.embedding_dim))

    def forward(self, x):
        # print(self.pe[0, :x.size(1)])
        return self.position_embedding
        # return nn.Parameter(torch.randn(1, self.input_lenth, self.embedding_dim)).cuda()
