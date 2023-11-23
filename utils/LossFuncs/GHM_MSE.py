from utils.Methods.GHMLoss import GHM_Loss
import torch
from torch import nn


class GHM_MSELoss(GHM_Loss):
    def __init__(self, bins, alpha):
        super(GHM_MSELoss, self).__init__(bins, alpha)
        self.sigmod = nn.Sigmoid()

    def _custom_loss(self, x, target, weight):
        d = torch.pow((x - target), 2)
        N = x.size(0)
        weight = weight.cuda()
        d = d.cuda()
        return (d * weight).sum() / N

    def _custom_loss_grad(self, x, target):
        # d = torch.abs(x - target)
        d = x - target
        # weight = (d-torch.min(d))/(torch.max(d)-torch.min(d))
        N = (x.size(0) * x.size(1))
        weight = self.sigmod(2*d/N)
        return weight
