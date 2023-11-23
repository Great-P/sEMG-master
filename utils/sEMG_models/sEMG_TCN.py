from utils.BaseModels.TCN.TCN import TemporalConvNet
from torch import nn


class sEMG_TCN(nn.Module):
    def __init__(self, num_inputs=12, num_channels=[32, 64, 64, 32, 10], kernel_size=3, dropout=0.2):
        super(sEMG_TCN, self).__init__()
        self.name = "TCN"
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(200 * 10, 10)

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2))
        x = x.permute(0, 2, 1)

        output = self.tcn(x)
        # output = output.permute(0, 2, 1)
        # output = output[:, 99:100, :]

        output = output.view(x.size(0), -1)
        output = self.fc(output)
        output = output.reshape([x.size(0), 1, 10])
        return output,x
