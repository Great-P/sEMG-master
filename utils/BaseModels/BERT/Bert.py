import torch.nn as nn
import torch

from .Transformer import TransformerBlock
from .Embedding.sEMGEmbedding import sEMGEmbedding


# import torchsnooper
# @torchsnooper.snoop()
class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, input_dim=12, hidden=768, feature_dim=1, n_layers=12, attn_heads=12, dropout=0.1,
                 use_se=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = sEMGEmbedding(vocab_size=vocab_size, embed_size=hidden, feature_size=feature_dim, use_se=False,
                                       input_dims=input_dim)
        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        # print(x.shape, (x > 0))
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1, 1).permute(0, 3, 1, 2)
        # print(mask.shape)
        mask = None
        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)
        # print(x.shape)
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
