import h5py
import torch.utils.data as data
import torch
import os
import pandas
import scipy.signal as signal
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


## 0.5ms采集一次数据(为一行)

class NinaPro(data.Dataset):
    def __init__(self, EMGtrain_dir, GloveTrain_dir, window_size=100, subframe=200, normalization="minmax", mu=2 ** 20,
                 dummy_label=1, class_num=1, dummy_tsk=0, tsk_num=1):
        # self.isTrain = isTrain
        self.EMG_dir = EMGtrain_dir
        self.glove_dir = GloveTrain_dir
        self.window_size = window_size
        self.subframe = subframe
        self.normalization = normalization
        self.mu = mu
        self.dummy_label = dummy_label
        self.class_num = class_num
        self.dummy_tsk = dummy_tsk
        self.tsk_num = tsk_num

        self.y_c = torch.ones([self.class_num]) / self.class_num
        self.y_s = torch.zeros([self.class_num])
        self.y_s[dummy_label] = 1
        self.t_c = torch.ones([self.tsk_num]) / self.tsk_num
        self.t_s = torch.zeros([self.tsk_num])
        self.t_s[dummy_tsk] = 1
        # self.y_c = self.y_c.long()
        # self.y_s = self.y_s.long()
        self._EMGload()

    def _EMGload(self):
        # print(self.EMG_dir + ' is loading')
        with h5py.File(self.EMG_dir, "r") as f:
            self.EMGtrain_data = f.get("featureset")[:]
        # print(self.glove_dir + ' is loading')
        with h5py.File(self.glove_dir, "r") as f:
            self.Glovetrain_data = f.get("featureset")[:]

        self.EMGtrain_data = torch.from_numpy(self.EMGtrain_data).float()
        self.Glovetrain_data = torch.from_numpy(self.Glovetrain_data).float()

        H, W, C = self.EMGtrain_data.shape
        # self.EMGtrain_data = self.EMGtrain_data.reshape([H, W])
        print("[*]Cur normalization type is: ", end='')
        if self.normalization.lower() == "minmax":
            print("Min-max")
            for index_w in range(W):
                for index_c in range(C):
                    self.EMGtrain_data[:, index_w, index_c] = (
                            (self.EMGtrain_data[:, index_w, index_c] - torch.min(
                                self.EMGtrain_data[:, index_w, index_c])) / (
                                    torch.max(self.EMGtrain_data[:, index_w, index_c]) - torch.min(
                                self.EMGtrain_data[:, index_w, index_c])))
        elif self.normalization.lower() == "miu":
            print(f"Mu-normalization with miu={self.mu}")
            self.EMGtrain_data = self.Mu_Normalization(self.EMGtrain_data, Mu=self.mu)
        else:
            raise Exception("[x]Have a wrong normalization type!")
        # print(self.EMGtrain_data.shape)
        # print(self.Glovetrain_data.shape)

    def Mu_Normalization(self, data, Mu=256):
        # print(data.shape)
        result = np.sign(data) * np.log((1 + Mu * np.abs(data))) / np.log((1 + Mu))
        return result

    def __getitem__(self, index):
        glovedata = self.Glovetrain_data[index * self.subframe:index * self.subframe + self.subframe, :]
        indices = torch.LongTensor([1, 2, 4, 5, 7, 8, 11, 12, 15, 16])
        glovefulldata = torch.index_select(glovedata, 1, indices)
        glovedata = torch.index_select(glovedata[self.subframe // 2 - 1:self.subframe // 2, :], 1, indices)
        # glovedata = torch.index_select(glovedata, 1, indices)
        return self.EMGtrain_data[index * self.subframe:index * self.subframe + self.subframe,
               :], glovedata, self.y_c, self.y_s, self.t_c, self.t_s

    def __len__(self):
        return (self.EMGtrain_data.shape[0] // self.subframe) - 1

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train'
        fmt_str += '    Split: {}\n'.format(tmp)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    train_emg_dir = "../Data/featureset/S2_E2_A1_rms_train.h5"
    train_glove_dir = "../Data/featureset/S2_E2_A1_glove_train.h5"
    # train_emg_dir = "../Data/S1emgtrain_rms.csv"
    # train_glove_dir = "../Data/S1glovetrain_rms.csv"
    dataset = NinaPro(train_emg_dir, train_glove_dir, subframe=200, normalization="miu", mu=2 ** 20, dummy_label=1,
                      class_num=10)
    DLoader = DataLoader(dataset=dataset, num_workers=12, drop_last=True, batch_size=1, shuffle=False)
    print(len(DLoader))
    for (i, data) in enumerate(DLoader):
        # print(i, data[0].shape, data[1].shape)
        # print(data[0][0])
        # print(data[1][0])
        print(data[2].size())
        print(data[3].size())
        break
