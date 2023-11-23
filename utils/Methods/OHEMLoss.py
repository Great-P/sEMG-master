import torch
import torch.nn.functional as F
import torch.nn as nn

class OHEM_Loss(nn.Module):
    def __init__(self, batchsize,):
        super(OHEM_Loss, self).__init__()
        self._batchsize = batchsize
        """batch_size (int): number of sampled rois for bbox head training"""
    def loss_function(self,target,predict):
        raise NotImplementedError

    def forward(self,target,predict):
        # ohem_cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='none', ignore_index=-1)
        ohem_loc_loss = self.loss_function(target, predict).sum(dim=1)
        # 这里先暂存下正常的分类loss和回归loss
        # print(ohem_cls_loss.shape, ohem_loc_loss.shape)
        loss = ohem_loc_loss
        # 然后对分类和回归loss求和

        sorted_ohem_loss, idx = torch.sort(loss, descending=True)
        # 再对loss进行降序排列

        keep_num = min(sorted_ohem_loss.size()[0], self._batchsize)
        # 得到需要保留的loss数量

        if keep_num < sorted_ohem_loss.size()[0]:
            # 这句的作用是如果保留数目小于现有loss总数，则进行筛选保留，否则全部保留

            keep_idx_cuda = idx[:keep_num]        # 保留到需要keep的数目
            # ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
            ohem_loc_loss = ohem_loc_loss[keep_idx_cuda]        # 分类和回归保留相同的数目

        # cls_loss = ohem_cls_loss.sum() / keep_num
        loc_loss = ohem_loc_loss.sum() / keep_num    # 然后分别对分类和回归loss求均值
        return loc_loss