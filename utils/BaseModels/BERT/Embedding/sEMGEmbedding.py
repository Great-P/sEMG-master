import torch.nn as nn
from utils.BaseModels.BERT.Embedding.position import PositionalEmbeddingLearnable
from utils.BaseModels.BERT.Embedding.token import Tokenalize_Mutifeature


class sEMGEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, feature_size, input_dims=12, dropout=0.1,use_se=False):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.token = Tokenalize_Mutifeature(vocab_size, input_dims, feature_size, embed_size,use_se=use_se)
        self.position = PositionalEmbeddingLearnable(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, ):
        # print(sequence.shape)
        # print(self.token(sequence).shape)
        x = self.token(sequence) + self.position(sequence)
        # print(x.shape)
        return self.dropout(x)
        # return sequence
