import copy
import random

import torch
import torch.nn as nn
import numpy as np
from torch import optim
import torch.nn.functional as F

from utils.TCN.TCN import TemporalConvNet, TemporalBlock


def KLloss(miu, sig):
    return torch.mean((sig.exp() + sig - 1 + torch.pow(miu, 2)))


class CBAM(nn.Module):
    """ Convolutional Block Attention Module
        c1: 输入通道数
        r: 全连接层隐藏层通道缩放比
        k: 空间注意力模块卷积核大小"""

    def __init__(self, c1, r=16, k=7):
        super(CBAM, self).__init__()
        c_ = int(c1 // r)
        self.mlp = nn.Sequential(
            nn.Conv1d(c1, c_, kernel_size=(1,)),
            nn.Conv1d(c_, c1, kernel_size=(1,))
        )
        assert k & 1, '卷积核尺寸需为奇数'
        self.conv = nn.Conv1d(2, 1, kernel_size=(k,), padding=k // 2)

    def forward(self, x):
        # Channel Attention
        ca = torch.cat([
            F.adaptive_avg_pool1d(x, 1),
            F.adaptive_max_pool1d(x, 1)
        ], dim=2)
        ca = torch.sigmoid(self.mlp(ca).sum(dim=2, keepdims=True))
        x1 = ca * x
        # Spatial Attention
        sa = torch.sigmoid(self.conv(torch.cat([
            x.mean(dim=1, keepdims=True),
            x.max(dim=1, keepdims=True)[0]
        ], dim=1)))
        x2 = sa * x1
        return x2


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class TCNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, SE_redcution=16):
        super(TCNEncoder, self).__init__()
        layers = nn.ModuleList([])
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.extend([TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size - 1) * dilation_size, dropout=dropout)])

        self.network = layers
        # self.se_block = SEBlock(sum(num_channels), SE_redcution) if SE_redcution else nn.Identity()
        self.se_block = CBAM(sum(num_channels), SE_redcution) if SE_redcution else nn.Identity()

    def forward(self, x):
        """
        x:(b, 12, 200)
        """
        each_layer_outputs = []
        x_last = x
        for each_layer in self.network:
            x_last = each_layer(x_last)
            each_layer_outputs.append(x_last)
        output = torch.cat(each_layer_outputs, dim=1)
        output = self.se_block(output)
        return output


class Adapter(nn.Module):
    def __init__(self, hidden):
        super(Adapter, self).__init__()
        self.hidden = hidden
        self.down = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
        )
        self.nonlinearity = nn.Sequential(
            # nn.Linear(self.hidden // 2, self.hidden // 2),
            nn.GELU(),
        )
        self.up = nn.Sequential(
            nn.Linear(self.hidden // 2, self.hidden)
        )

    def forward(self, x):
        x1 = self.down(x)
        x1 = self.nonlinearity(x1)
        x1 = self.up(x1)
        return x + x1


class RegressionEncoder(nn.Module):
    def __init__(self, vocab_size, hidden, classnum, jointchannels=10, emg_channels=12):
        super().__init__()
        self.classnum = classnum
        self.jointchannels = jointchannels
        self.hidden = hidden
        self.emg_channels = emg_channels

        self.feature_extractor = None
        self.capsules = nn.ModuleList([Adapter(hidden) for _ in range(classnum)])
        self.sm = nn.Softmax()

    def forward(self, x):
        b, t, c = x.size()
        # z_c = self.feature_extractor(x)[0]
        z_c = self.feature_extractor(x)
        z_capsules = []
        z_c = z_c.permute(0, 2, 1)
        for each_capsule in self.capsules:
            z_capsules.append(each_capsule(z_c).permute(0, 2, 1).unsqueeze(1))
        z_capsules = torch.cat(z_capsules, dim=1)
        return z_capsules

    def check(self):
        if self.feature_extractor is not None:
            print(f"[!]Extractor is Not None:{self.feature_extractor}")
            return
        else:
            self.load_extractor("TCN")

    def load_extractor(self, extractor="LSTM"):
        if not isinstance(extractor, (str, nn.Module)):
            raise "[!]Wrong Extractor type!"
        if not isinstance(extractor, str):
            self.feature_extractor = extractor
            return
        if extractor.upper() == "LSTM":
            self.feature_extractor = nn.LSTM(input_size=self.emg_channels, hidden_size=self.hidden, num_layers=4,
                                             batch_first=True)
        if extractor.upper() == "TCN":
            self.feature_extractor = TemporalConvNet(self.emg_channels, [self.hidden] * 4, 3, 0.2)
            # assert self.hidden % 4 == 0, "[!!]PLZ redesign TCN hidden layers"
            # self.feature_extractor = TCNEncoder(self.emg_channels, [self.hidden // 4] * 4, 3, 0.2)

    def extend_capsules(self, cls_num):
        self.classnum += cls_num
        temp_capsules = nn.ModuleList([Adapter(self.hidden) for _ in range(cls_num)])
        self.capsules.extend(temp_capsules)


class Regressor(nn.Module):
    def __init__(self, vocab_size, hidden, jointchannels):
        super(Regressor, self).__init__()
        self.adapter = Adapter(hidden)
        self.prediction_head = nn.Linear(vocab_size * hidden, jointchannels)

    def forward(self, x):
        b, c, t = x.size()
        x = self.adapter(x)
        x = x.contiguous().view(b, -1)
        y_pred = self.prediction_head(x)
        y_pred = y_pred.unsqueeze(1)
        return y_pred


class ClassificationEncoder(nn.Module):
    def __init__(self, vocab_size, hidden, classnum, emg_channels=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.classnum = classnum
        self.emg_channels = emg_channels
        self.feature_extractor = None

    def forward(self, x):
        # z_s = self.feature_extractor(x)[0]
        z_s = self.feature_extractor(x)
        return z_s

    def check(self):
        if self.feature_extractor is not None:
            print(f"[!]Extractor is Not None:{self.feature_extractor}")
            return
        else:
            self.load_extractor("TCN")

    def load_extractor(self, extractor="LSTM"):
        if not isinstance(extractor, (str, nn.Module)):
            raise "[!]Wrong Extractor type!"
        if not isinstance(extractor, str):
            self.feature_extractor = extractor
            return
        if extractor.upper() == "LSTM":
            self.feature_extractor = nn.LSTM(input_size=self.emg_channels, hidden_size=self.hidden, num_layers=4,
                                             batch_first=True)
        if extractor.upper() == "TCN":
            self.feature_extractor = TemporalConvNet(self.emg_channels, [self.hidden] * 4, 3, 0.2)
            # assert self.hidden % 4 == 0, "[!!]PLZ redesign TCN hidden layers"
            # self.feature_extractor = TCNEncoder(self.emg_channels, [self.hidden // 4] * 4, 3, 0.2)


class SharedClassificationHead(nn.Module):
    def __init__(self, vocabsize, hidden, classnum):
        super().__init__()
        self.vocabsize = vocabsize
        self.hidden = hidden
        self.classnum = classnum
        self.adapter = Adapter(hidden)
        self.net = nn.Linear(vocabsize * hidden, classnum)
        self.sm = nn.Softmax(dim=1)

    def forward(self, z):
        b, t, h = z.size()
        z = self.adapter(z.permute(0, 2, 1))
        z = z.contiguous().view(b, -1)
        softlabel = self.net(z)
        softlabel = softlabel.view(b, -1)
        softlabel = self.sm(softlabel)
        return softlabel

    def extend_cls(self, cls_nums, ):
        self.prev_weight = self.net.weight.data
        self.classnum += cls_nums
        self.net = nn.Linear(self.vocabsize * self.hidden, self.classnum)

    def expert_gate_init(self, idx, init_expert):
        old_cls_num = self.prev_weight.size()[0]
        self.net.weight.data[:old_cls_num] = self.prev_weight
        self.net.weight.data[old_cls_num + idx] = self.prev_weight[init_expert]


class ReconstructionDecoder(nn.Module):
    def __init__(self, vocab_size, hidden, emg_channels=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.emg_channels = emg_channels
        self.reconstruction_executor = None

    def forward(self, z):
        # x_produce = self.reconstruction_executor(z)[0]
        x_produce = self.reconstruction_executor(z)
        return x_produce

    def check(self):
        if self.reconstruction_executor is not None:
            print(f"[!]Extractor is Not None:{self.reconstruction_executor}")
            return
        else:
            self.load_extractor("TCN")

    def load_extractor(self, executor="LSTM"):
        if not isinstance(executor, (str, nn.Module)):
            raise "[!]Wrong Extractor type!"
        if not isinstance(executor, str):
            self.reconstruction_executor = executor
            return
        if executor.upper() == "LSTM":
            self.reconstruction_executor = nn.LSTM(input_size=self.hidden, hidden_size=self.emg_channels, num_layers=4,
                                                   batch_first=True)
        if executor.upper() == "TCN":
            self.reconstruction_executor = TemporalConvNet(self.hidden, [self.emg_channels] * 4, 3, 0.2)


class MemBuffer:
    def __init__(self, init_subjects=10, buf_nums=10):
        """memory_unit = (EMGdata->Tensor:(1,t,c), score->Tensor(b,1)))"""
        self.buf_nums = buf_nums
        self.memory = list()
        for _ in range(init_subjects):
            if torch.cuda.is_available():
                self.memory.append([(None, torch.Tensor([float("inf")]).cuda())] * self.buf_nums)
            else:
                self.memory.append([(None, torch.Tensor([float("inf")]))] * self.buf_nums)

    # def check_and_update(self, data):
    #     """
    #     data:(emg, y_s_label, score)
    #     """
    #     # assert 0 <= sub_idx < len(self.memory), "[!!]Subject Index out of range!"
    #     emg, label, score = data
    #     b = emg.size()[0]
    #     for i in range(b):
    #         # print(label[i])
    #         sub_idx = torch.max(label[i], dim=0)[1]
    #         cur_buffer = self.memory[sub_idx]
    #         # print(cur_buffer[0][1],score)
    #         if cur_buffer[0][1] > score[i]:
    #             cur_buffer.pop(0)
    #             cur_buffer.append((emg[i], score[i]))
    #             cur_buffer.sort(key=lambda x: x[1], reverse=True)
    #         self.memory[sub_idx] = cur_buffer
    def check_and_update(self, data):
        """
        data:(emg, y_s_label, score)
        """
        # assert 0 <= sub_idx < len(self.memory), "[!!]Subject Index out of range!"
        random.seed(0)
        np.random.seed(seed=0)
        torch.manual_seed(0)
        emg, label, score = data
        b = emg.size()[0]
        for i in range(b):
            # print(label[i])
            sub_idx = torch.max(label[i], dim=0)[1]
            cur_buffer = self.memory[sub_idx]
            # print(cur_buffer[0][1],score)
            if torch.isinf(cur_buffer[-1][1]) or random.choice([0, 1]) == 1:
                cur_buffer.pop(0)
                cur_buffer.append((emg[i], score[i]))
                # cur_buffer.sort(key=lambda x: x[1], reverse=True)
            self.memory[sub_idx] = cur_buffer

    def visit(self, sub_idx=None):
        if sub_idx is None:
            return self.memory
        assert 0 <= sub_idx < len(self.memory), "[!!]Subject Index out of range!"
        return self.memory[sub_idx]

    def extend_subjects(self, subject_nums, ):
        for _ in range(subject_nums):
            if torch.cuda.is_available():
                self.memory.append([(None, torch.Tensor([float("inf")]).cuda())] * self.buf_nums)
            else:
                self.memory.append([(None, torch.Tensor([float("inf")]))] * self.buf_nums)

    def data_confusion(self, emg_data, y_s_label, replay_ratio=0.5):
        """Confuse Input with Memory Data"""
        b = emg_data.size()[0]

        new_data_idx_pool = [i for i in range(b)]
        sub_idx_pool = [i for i in range(self.__len__()[0])]
        block_idx_pool = []
        for _ in range(self.__len__()[0]):
            block_idx_pool.append([i for i in range(self.buf_nums)])

        confusion_emg = torch.Tensor([])
        confusion_label = torch.Tensor([])
        if torch.cuda.is_available():
            confusion_emg = confusion_emg.cuda()
            confusion_label = confusion_label.cuda()
        posibility = np.array([1 - replay_ratio, replay_ratio])
        flag = np.random.choice([0, 1], p=posibility.ravel())
        for cur_idx in range(b):
            if flag == 0:
                idx = random.choice(new_data_idx_pool)
                confusion_emg = torch.cat([confusion_emg, emg_data[idx].unsqueeze(0)], dim=0)
                # print(emg_data[idx].size())
                confusion_label = torch.cat([confusion_label, y_s_label[idx].unsqueeze(0)], dim=0)
                new_data_idx_pool.remove(idx)
                flag = np.random.choice([0, 1], p=posibility.ravel())
            else:
                sub_idx = random.choice(sub_idx_pool)
                block_idx = random.choice(block_idx_pool[sub_idx])
                # print(self.memory[sub_idx][block_idx][0].size())
                confusion_emg = torch.cat([confusion_emg, self.memory[sub_idx][block_idx][0].unsqueeze(0)], dim=0)

                temp_label = torch.zeros([1, y_s_label.size()[1]]).scatter_(1, torch.tensor([sub_idx],
                                                                                            dtype=torch.int64).unsqueeze(
                    0),
                                                                            1).cuda()
                # temp_label = torch.zeros([1, y_s_label.size()[1]]).scatter_(1,
                #                                                             torch.tensor([sub_idx], dtype=torch.int64).unsqueeze(0),
                #                                                             1)
                confusion_label = torch.cat([confusion_label, temp_label], dim=0)
                block_idx_pool[sub_idx].remove(block_idx)
                flag = np.random.choice([0, 1], p=posibility.ravel())
                if len(block_idx_pool[sub_idx]) == 0:
                    sub_idx_pool.remove(sub_idx)
                if len(sub_idx_pool) == 0:
                    flag = 0
        return confusion_emg, confusion_label

    def __len__(self):
        return len(self.memory), self.buf_nums


class LifelongCrossSubjectNetwork(nn.Module):
    def __init__(self, vocab_size, hidden, class_num, joint_channels=10, emg_channels=12, subject_nums=10,
                 mem_capa_per_subject=200):
        super().__init__()
        self.name = "LCSN"
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.classnum = class_num
        self.joint_channels = joint_channels
        self.gen_encoder = RegressionEncoder(vocab_size, hidden, class_num, joint_channels)
        self.chara_encoder = ClassificationEncoder(vocab_size, hidden, class_num)
        self.regressor = Regressor(vocab_size, hidden, joint_channels)
        self.shared_cls_head = SharedClassificationHead(vocab_size, hidden, class_num)
        self.rec_cls = ReconstructionDecoder(vocab_size, hidden, emg_channels)
        self.rec_reg = ReconstructionDecoder(vocab_size, hidden, emg_channels)
        # self.gen_BN = nn.LayerNorm([hidden,vocab_size])
        # self.chara_BN = nn.LayerNorm([hidden,vocab_size])
        self.memory = MemBuffer(subject_nums, mem_capa_per_subject)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.squeeze(3)
        z_r_capsules = self.gen_encoder(x)
        z_c = self.chara_encoder(x)
        # z_s = self.chara_BN(z_s)

        y_c = self.shared_cls_head(z_c)
        # print(z_c_capsules.size(), y_s.size())
        # y_s = (y_s - torch.min(y_s, dim=1)[0].unsqueeze(1)) / (
        #         torch.max(y_s, dim=1)[0].unsqueeze(1) - torch.min(y_s, dim=1)[0].unsqueeze(1))
        z_r = torch.mul(z_r_capsules, y_c.unsqueeze(2).unsqueeze(3))
        z_r = torch.sum(z_r, dim=1).squeeze(1)
        y_pred = self.regressor(z_r.permute(0, 2, 1))

        x_hat_c = self.rec_cls(z_r)
        x_hat_r = self.rec_reg(z_c)

        return y_pred, y_c, x_hat_c, x_hat_r

    def inference(self, x):
        if len(x.size()) == 4:
            x = x.squeeze(3)
        z_c_capsules = self.gen_encoder(x)
        z_s = self.chara_encoder(x)
        # z_s = self.chara_BN(z_s)

        y_s = self.shared_cls_head(z_s)
        # print(z_c_capsules.size(), y_s.size())
        z_c = torch.mul(z_c_capsules, y_s.unsqueeze(2).unsqueeze(3))
        z_c = torch.sum(z_c, dim=1).squeeze(1)
        # z_c = self.gen_BN(z_c)
        # print(z_c.size())
        y_pred = self.regressor(z_c.permute(0, 2, 1))
        return y_pred

    def check(self):
        self.gen_encoder.check()
        self.chara_encoder.check()
        self.rec_cls.check()
        self.rec_reg.check()

    def reconstruct_cls_head(self, new_class_num):
        """
        For Transfer
        """
        self.shared_cls_head = SharedClassificationHead(self.vocab_size, self.hidden, new_class_num)


if __name__ == "__main__":
    """
    Test
    """
    GDN = LifelongCrossSubjectNetwork(200, 128, 10, 10, 12, 1, 8)
    GDN.check()
    # GDN.memory.extend_subjects(1)
    # GDN.check()
    GDN = GDN.cuda()
    dummy_input = torch.randn([8, 200, 12]).cuda()
    dummy_input = dummy_input.permute(0, 2, 1)
    dumpy_label = torch.zeros(8, 1).scatter_(1, torch.zeros([8, 1], dtype=torch.int64), 1).cuda()
    # print(dumpy_label)
    output = GDN(dummy_input)
    y_pred, y_c, y_s, x_hat, z_s = output
    dummy_input = dummy_input.permute(0, 2, 1)
    GDN.memory.check_and_update((dummy_input, dumpy_label, torch.randn([8, 1], dtype=torch.float64).cuda()))
    # GDN.memory.check_and_update((dummy_input, dumpy_label, torch.randn([8, 1], dtype=torch.float64).cuda()))
    # print("1:",GDN.memory.visit(sub_idx=0)[0][0].size())
    new_input_emg = torch.randn([8, 200, 12]).cuda()
    new_input_label = torch.zeros(8, 2).scatter_(1, torch.ones([8, 1], dtype=torch.int64), 1).cuda()
    emg, label = GDN.memory.data_confusion(new_input_emg, new_input_label, 0.5)
    print(emg.size(), label.size())
    print(label)
