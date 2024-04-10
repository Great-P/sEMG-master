import ast
import sys

import scipy

sys.path.append("..")

import numpy

import DataProcess.NinaPro as ninapro
import matplotlib.pyplot as plt
import datetime

import argparse
import re
import os, glob, datetime, time
import shutil
from utils.methods import str2bool, js_div

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataset import ConcatDataset

from skimage import metrics
import sklearn.metrics as skmetrics
from utils.methods import pearson_CC, avg_smoothing_np, draw_graph
from LossFuncs.GHM_MSE import GHM_MSELoss
from LossFuncs.OHEM_MSE import OHEM_MSE

from sEMG_models.sEMG_Bert import sEMG_BERT
from sEMG_models.sEMG_LSTM import sEMG_lstm
from sEMG_models.sEMG_TCN import sEMG_TCN
from sEMG_models.LE_ConvMN import LE_convMN
from utils.GenMiningNet.GenMiningNet import GeneralityMiningNetwork
from utils.GenMiningNet.GenDeepMiningNet import GeneralityDeepMiningNetwork, MemBuffer
from utils.GenMiningNet.LifelongCrossSubjectNetwork import LifelongCrossSubjectNetwork, MemBuffer
# from utils.GenMiningNet.TaskIncrementalGMN import TaskIncrementalGMN, MemBuffer

from tensorboardX import SummaryWriter
from utils.torchsummary import summary

# Params
parser = argparse.ArgumentParser(description='PyTorch sEMG-Bert')
# dataset
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--subframe', default=200, type=int)
parser.add_argument('--epoch', default=200, type=int)
# parser.add_argument('--epoch', default=1000, type=int)
parser.add_argument('--lr', default=1e-1, type=float)
parser.add_argument('--milestones', default=[200], type=list)
# parser.add_argument('--milestones', default=[300, 600, 900], type=list)
parser.add_argument('--device_ids', type=list, default=[0])
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--normalization', type=str, default="miu")
parser.add_argument('--miu', type=int, default=2 ** 20)
parser.add_argument('--smooth', type=str2bool, default=False)
parser.add_argument('--use_se', type=str2bool, default=False)

parser.add_argument('--subject', type=str, default=[f'S9', 'S21', 'S26', 'S20', 'S35', 'S5', 'S18', 'S7', 'S32', 'S31'])
# parser.add_argument('--subject', type=str, default=[f"S0"])
# [f"S{i+1}" for i in range(40)]
parser.add_argument('--subject_name', type=str, default="E1")

args = parser.parse_args()

cuda = torch.cuda.is_available()
assert cuda

# subject = args.subject.strip()
subject = args.subject
subject_name = args.subject_name.strip()
if not isinstance(subject, list):
    subject = [subject]
if "S0" in subject:
    subject.remove("S0")
    subject = ["S1", "S2", "S3", "S13", "S25", "S11", "S14", "S18", "S19", "S22"] + subject


def cls_new_loss(y_pred, y_label):
    return torch.sum(-y_label * y_pred.log())


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(seed=0)
    torch.manual_seed(0)

    # model selection
    print('===> Building model')
    # model = sEMG_lstm(vocab_size=args.subframe, hidden=args.hidden, n_layers=args.num_layers)
    # model = sEMG_BERT(vocab_size=args.subframe, hidden=args.hidden, feature_dim=1, n_layers=args.num_layers,
    #                   attn_heads=8, use_se=args.use_se)
    # model = sEMG_TCN(12, [128, 128, 128, 128, 10], 3, 0.2)
    # model = LE_convMN((12, 200), 1, 1, [64, 32, 10], (3, 3), 3, 20, batch_first=True, bias=True,
    #                   return_all_layers=False, withCBAM=False)
    # model = torch.load(os.path.join(args.save_dir, 'model_latest.pth'))
    # model = GeneralityMiningNetwork(vocab_size=args.subframe, hidden=args.hidden, class_num=len(subject),
    #                                 joint_channels=10, emg_channels=12)
    # model = GeneralityDeepMiningNetwork(vocab_size=args.subframe, hidden=args.hidden, class_num=len(subject),
    #                                     joint_channels=10, emg_channels=12)
    model = LifelongCrossSubjectNetwork(vocab_size=args.subframe, hidden=args.hidden, class_num=len(subject),
                                        joint_channels=10, emg_channels=12)
    # model = TaskIncrementalGMN(vocab_size=args.subframe, hidden=args.hidden, class_num=len(subject),
    #                            joint_channels=10, emg_channels=12)

    model.check()
    initial_epoch = 0
    cur_model_name = model.name
    if args.smooth:
        cur_model_name = "s" + cur_model_name
    # criterion = sum_squared_error()

    # save_dir = args.save_dir
    save_dir = f"../models/{args.normalization}/{subject_name}/{cur_model_name}"

    model = torch.load(os.path.join(save_dir, 'model_best.pth'))
    # model.memory.extend_subjects(len(subject))
    # task_num = model.task_num
    model.memory = MemBuffer(len(subject), 100)
    # model.memory.new_task_init(len(subject))
    # print(model.memory.visit()[11])
    # exit(0)
    # dataset
    train_dataset_list = []
    test_dataset_list = []
    for i, each_subject in enumerate(subject):
        emgtrain_dir = f"../Data/featureset/{each_subject}_E2_A1_rms_train_ff.h5"
        glovetrain_dir = f"../Data/featureset/{each_subject}_E2_A1_glove_train_ff.h5"
        emgtest_dir = f"../Data/featureset/{each_subject}_E2_A1_rms_test.h5"
        glovetest_dir = f"../Data/featureset/{each_subject}_E2_A1_glove_test.h5"
        train_dataset = ninapro.NinaPro(emgtrain_dir, glovetrain_dir, window_size=200, subframe=args.subframe,
                                        normalization=args.normalization, mu=args.miu, dummy_label=i,
                                        class_num=len(subject), )
        # dummy_tsk=task_num - 1, tsk_num=task_num)
        test_dataset = ninapro.NinaPro(emgtest_dir, glovetest_dir, window_size=200, subframe=args.subframe,
                                       normalization=args.normalization, mu=args.miu, dummy_label=i,
                                       class_num=len(subject), )
        # dummy_tsk=task_num - 1, tsk_num=task_num)
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
    # concat_train_dataset = ConcatDataset(train_dataset_list)
    # concat_test_dataset = ConcatDataset(test_dataset_list)
    train_dataset_list.extend(test_dataset_list)
    concat_dataset = ConcatDataset(train_dataset_list)
    TrainLoader = DataLoader(dataset=concat_dataset, num_workers=12, drop_last=True, batch_size=args.batch_size,
                             shuffle=True, )  # sampler=train_sampler, pin_memory=True)
    # DLoader_eval = DataLoader(dataset=concat_test_dataset, num_workers=1, drop_last=True, batch_size=args.batch_size,
    #                           shuffle=True, )  # sampler=test_sampler,pin_memory=True)
    criterion = nn.CrossEntropyLoss(reduce=False)
    if cuda:
        print("[*]Training on GPU......")
        model = model.cuda()
        criterion = criterion.cuda()
    # add log
    log_file = os.path.join(save_dir, 'train_result.txt')
    with open(log_file, 'w') as f:
        f.write('----Begin logging----\n')
        f.write("Training begin:" + str(datetime.datetime.now()) + "\n")
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('================ Training loss ================\n')

    # trainning
    best_epoch = {'epoch': 0, 'NRMSE': 10, 'CC': 0, 'R2': -10}
    loss_count = []
    elpased_time_list = []
    hidden = None
    model.eval()
    # scheduler = MultiStepLR(model.ae_optim, milestones=args.milestones, gamma=0.5)  # learning rates
    for epoch in range(initial_epoch, 1):
        # print('[*]Current Learning rate ={:.6f}'.format(scheduler.get_last_lr()[0]))

        epoch_loss = 0
        start_time = time.time()
        # training phase

        for n_count, batch_tr in enumerate(TrainLoader):
            batch_emg = batch_tr[0].squeeze(3).permute(0, 2, 1).cuda()
            # batch_emg = batch_tr[0].squeeze(3).cuda()
            # print(batch_emg.shape)
            batch_glove = batch_tr[1].cuda()
            y_c_label = batch_tr[2].cuda()
            y_s_label = batch_tr[3].cuda()
            t_c_label = batch_tr[4].cuda()
            t_s_label = batch_tr[5].cuda()

            # y_pred, y_c, y_s, x_hat, z_s, z_c = model(batch_emg)
            _, y_s, _, _, = model(batch_emg)
            score = criterion(y_s, torch.max(y_s_label, dim=1)[1])
            # y_pred, y_c, y_s, x_hat, z_s, t_s, t_c = model(batch_emg)
            # score = criterion(t_s, torch.max(t_s_label, dim=1)[1])
            # print(torch.max(y_s_label,dim=1)[1],score, score.size())
            # exit()
            model.memory.check_and_update((batch_emg.permute(0, 2, 1), y_s_label, score))
            y_s.detach_()
            score.detach_()

        elapsed_time = time.time() - start_time
        elpased_time_list.append(elapsed_time)
        # evaluation phase

        torch.save(model, os.path.join(save_dir, 'model_best.pth'))
        # scheduler.step()  # step to the learning rate in this epoch

    elpased_time_list = numpy.array(elpased_time_list)
    time_message = f"[*]Average time cost: {numpy.sum(elpased_time_list) / args.epoch}"
    print(time_message)
    with open(log_file, "a") as f:
        f.write("\n")
        f.write(time_message)

    torch.cuda.empty_cache()

    # plt.figure(f'PyTorch_{cur_model_name}_Loss')
    # plt.plot(loss_count, label='Loss')
    # plt.legend()
    # plt.show()
