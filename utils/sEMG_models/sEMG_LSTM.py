import torch.nn as nn
import torch


class sEMG_LSTM(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=24, n_layers=32, input_channels=12, feature_dim=1, attn_heads=12,
                 dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.name = "LSTM"
        # self.bert = BERT(vocab_size, hidden, n_layers, attn_heads, dropout)
        # self.token = Tokenalize_Mutifeature(vocab_size, input_channels, feature_dim, embed_dim=input_channels)
        self.fc = nn.Linear(vocab_size * hidden * 1, 1 * 10)
        self.lstm = nn.LSTM(input_size=12, hidden_size=hidden, num_layers=n_layers, batch_first=True)
        self.vocab_size = vocab_size
        self.hidden = hidden

    def forward(self, x, hidden=None):
        x = x.view(x.size(0), x.size(1), x.size(2))
        hidden_1 = hidden
        x1, hidden_1 = self.lstm(x, hidden_1)
        # x1 = self.bert(x1)
        x1_flatten = x1.contiguous().view(x1.size(0), -1)
        output = self.fc(x1_flatten)
        # output = output.reshape([x1.size(0), self.vocab_size, 10])
        output = output.view(x1.size(0), 1, 10)
        return output, hidden_1


if __name__ == "__main__":
    sim_batch = torch.rand([1, 200, 12]).float().cuda()
    model = sEMG_LSTM(200, hidden=12).cuda()
    print(sim_batch.shape)
    output = model(sim_batch)
    print(output.shape)
