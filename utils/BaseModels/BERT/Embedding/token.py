import torch.nn as nn
import torch
from torch.autograd import Variable
from utils.Methods.methods import SEBlock, NoSEBlock


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class Tokenalize(nn.Module):
    def __init__(self, vocab_size, input_dim, embed_dim=512):
        super(Tokenalize, self).__init__()
        self.embedding = nn.Linear(vocab_size * input_dim, vocab_size * embed_dim)
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim

    def forward(self, x):
        x_flatten = x.view(x.size(0), -1)
        output = self.embedding(x_flatten)
        output = output.view(x.size(0), self.vocab_size, self.embed_dim)
        return output


class Tokenalize_Mutifeature(nn.Module):
    def __init__(self, vocab_size, input_dim, feature_dim, embed_dim=512, use_se=False):
        super(Tokenalize_Mutifeature, self).__init__()
        self.embedding = nn.Linear(vocab_size * input_dim * feature_dim, vocab_size * embed_dim)
        self.SE = SEBlock(feature_dim, 3) if use_se else NoSEBlock(feature_dim, 3)
        self.vocab_size = vocab_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim

    def forward(self, x):
        x = self.SE(x)
        x_flatten = x.view(x.size(0), -1)
        output = self.embedding(x_flatten)
        output = output.view(x.size(0), self.vocab_size, self.embed_dim)
        return output


if __name__ == "__main__":
    test = torch.ones([1, 200, 1]).long().cuda()
    test = Variable(test)
    token = TokenEmbedding(200, 24, ).cuda()
    test = token(test)
    print(test.shape)
    print(test)
