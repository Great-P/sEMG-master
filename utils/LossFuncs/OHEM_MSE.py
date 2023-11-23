from utils.Methods.OHEMLoss import OHEM_Loss
import torch


class OHEM_MSE(OHEM_Loss):
    def __init__(self, batchsize):
        super(OHEM_MSE, self).__init__(batchsize)

    def loss_function(self, target, predict):
        d = torch.pow((predict - target), 2)
        d = d.cuda()
        return d
