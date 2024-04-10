import sys

sys.path.append("..")

import numpy

import argparse
import os, datetime, time
import shutil
from utils.Methods.methods import str2bool

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

from utils.Methods.methods import pearson_CC, avg_smoothing_np, draw_graph
from utils.DataProcess import NinaPro
from utils.sEMG_models.LE_ConvMN import LE_ConvMN
from utils.sEMG_models.LifelongCrossSubjectNetwork import LifelongCrossSubjectNetwork, MemBuffer

from tensorboardX import SummaryWriter
from utils.Methods.torchsummary import summary
from utils.sEMG_models.sEMG_BERT import sEMG_BERT
from utils.sEMG_models.sEMG_LSTM import sEMG_LSTM
from utils.sEMG_models.sEMG_TCN import sEMG_TCN

# Params
parser = argparse.ArgumentParser(description='PyTorch sEMG-Bert')
# dataset
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--subframe', default=200, type=int)
parser.add_argument('--epoch', default=2, type=int)
# parser.add_argument('--epoch', default=1000, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--milestones', default=[200], type=list)
# parser.add_argument('--milestones', default=[300, 600, 900], type=list)
parser.add_argument('--device_ids', type=list, default=[0])
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--normalization', type=str, default="miu")
parser.add_argument('--miu', type=int, default=2 ** 20)
parser.add_argument('--smooth', type=str2bool, default=False)
parser.add_argument('--use_se', type=str2bool, default=False)

parser.add_argument('--subject', type=str, default=['S0'])
# [f"S{i+1}" for i in range(40)]
parser.add_argument('--subject_name', type=str, default="S0")
parser.add_argument('--model', type=str, default="LE_ConvMN")

args = parser.parse_args()

cuda = torch.cuda.is_available()
assert cuda

# subject = args.subject.strip()
subject = args.subject
subject_name = args.subject_name.strip()
model_name = args.model.strip()
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
    if model_name == "BERT":
        model = sEMG_BERT(vocab_size=args.subframe, hidden=args.hidden, feature_dim=1, n_layers=args.num_layers,
                          attn_heads=8, use_se=args.use_se)
    elif model_name == "LSTM":
        model = sEMG_LSTM(vocab_size=args.subframe, hidden=args.hidden, n_layers=args.num_layers)

    elif model_name == "TCN":
        model = sEMG_TCN(12, [128, 128, 128, 128, 10], 3, 0.2)
    elif model_name == "LE_ConvMN":
        model = LE_ConvMN((12, 200), 1, 1, [64, 32, 10], (3, 3), 3, 20, batch_first=True, bias=True,
                          return_all_layers=False, withCBAM=False)
    elif model_name == "LCSN":
        model = LifelongCrossSubjectNetwork(vocab_size=args.subframe, hidden=args.hidden, class_num=len(subject),
                                            joint_channels=10, emg_channels=12, subject_nums=len(subject),
                                            mem_capa_per_subject=100)
        model.check()
    else:
        raise Exception("[x] Wrong model name!")


    def data_wrapper_permute(data):
        return data.permute(0, 2, 1)


    def data_wrapper_leconv(data):
        return data.unsqueeze(3)


    def data_wrapper_identity(data):
        return data


    if model_name in ["LCSN"]:
        data_wrapper = data_wrapper_permute
    elif model_name in ["LE_ConvMN"]:
        data_wrapper = data_wrapper_leconv
    else:
        data_wrapper = data_wrapper_identity

    initial_epoch = 0
    reg_loss = nn.MSELoss()
    cls_loss = nn.CrossEntropyLoss()
    cur_model_name = model.name
    if args.smooth:
        cur_model_name = "s" + cur_model_name

    tensorboard_record_path = f"../runs/{args.normalization}/{subject_name}/{cur_model_name}"

    save_dir = f"../models/{args.normalization}/{subject_name}/{cur_model_name}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(tensorboard_record_path):
        print("[*]Cleaning previous cache of tensorboard...")
        shutil.rmtree(tensorboard_record_path)
    record_writer = SummaryWriter(tensorboard_record_path)
    dummy_input = data_wrapper(torch.randn([1, 200, 12])).cuda()
    print(f"[*]Current model is {cur_model_name}")
    if cuda:
        print("[*]Training on GPU......")
        model = model.cuda()
        record_writer.add_graph(model, dummy_input, )
        device_ids = args.device_ids
        reg_loss = reg_loss.cuda()
        # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)  # learning rates
    ## Todo
    # record_writer.add_text("Model Scale", summary(model, (12, args.subframe, 1), args.batch_size))
    # record_writer.add_text("Model Scale", summary(model, (args.subframe, 12, 1), args.batch_size)) # TCN
    # dataset
    train_dataset_list = []
    test_dataset_list = []
    for i, each_subject in enumerate(subject):
        emgtrain_dir = f"../Data/featureset/{each_subject}_E2_A1_rms_train.h5"
        glovetrain_dir = f"../Data/featureset/{each_subject}_E2_A1_glove_train.h5"
        emgtest_dir = f"../Data/featureset/{each_subject}_E2_A1_rms_test.h5"
        glovetest_dir = f"../Data/featureset/{each_subject}_E2_A1_glove_test.h5"
        train_dataset = NinaPro.NinaPro(emgtrain_dir, glovetrain_dir, window_size=200, subframe=args.subframe,
                                        normalization=args.normalization, mu=args.miu, dummy_label=i,
                                        class_num=len(subject))
        test_dataset = NinaPro.NinaPro(emgtest_dir, glovetest_dir, window_size=200, subframe=args.subframe,
                                       normalization=args.normalization, mu=args.miu, dummy_label=i,
                                       class_num=len(subject))
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
    concat_train_dataset = ConcatDataset(train_dataset_list)
    concat_test_dataset = ConcatDataset(test_dataset_list)
    TrainLoader = DataLoader(dataset=concat_train_dataset, num_workers=12, drop_last=True, batch_size=args.batch_size,
                             shuffle=True, )  # sampler=train_sampler, pin_memory=True)
    DLoader_eval = DataLoader(dataset=concat_test_dataset, num_workers=12, drop_last=True, batch_size=args.batch_size,
                              shuffle=False, )  # sampler=test_sampler,pin_memory=True)

    # add log
    log_file = os.path.join(save_dir, 'train_result.txt')
    with open(log_file, 'w') as f:
        f.write('----Begin logging----\n')
        f.write("Training begin:" + str(datetime.datetime.now()) + "\n")
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('================ Training loss ================\n')

    # training
    best_epoch = {'epoch': 0, 'NRMSE': 10, 'CC': 0, 'R2': -10}
    loss_count = []
    elpased_time_list = []
    hidden = None
    for epoch in range(initial_epoch, args.epoch):
        print('[*]Current Learning rate ={:.6f}'.format(scheduler.get_last_lr()[0]))

        epoch_loss = 0
        start_time = time.time()
        # training phase
        model.train()
        for n_count, batch_tr in enumerate(TrainLoader):
            batch_emg = data_wrapper(batch_tr[0].squeeze(3)).cuda()
            batch_glove = batch_tr[1].cuda()
            # y_c_label = batch_tr[2].cuda()
            y_s_label = batch_tr[3].cuda()

            y_pred = model(batch_emg)[0]  # BERT/TCN
            # output, hidden = model(batch_emg, hidden) # RNN
            # for l in range(len(hidden)):  # LE-Conv
            #     for p in range(len(hidden[l])):
            #         hidden[l][p].detach_()

            # hidden[0].detach_() # LSTM
            # hidden[1].detach_()

            ## adjust your loss func
            loss = reg_loss(y_pred, batch_glove)

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if n_count % (len(TrainLoader) // 10) == 0:
                message = '[{}] {} / {} loss = {}'.format(epoch + 1, n_count, len(TrainLoader),
                                                          loss.item() / args.batch_size)
                loss_count.append(loss.item() / args.batch_size)
                print(message)
                with open(log_file, 'a') as f:
                    f.write(message)
                    f.write('\n')
        elapsed_time = time.time() - start_time
        elpased_time_list.append(elapsed_time)
        # evaluation phase
        model.eval()
        NRMSEs = []
        CCs = []
        total_output_test = torch.Tensor([]).cuda()
        total_target_test = torch.Tensor([]).cuda()
        with torch.no_grad():
            i = 0
            hidden_1 = hidden
            for _, batch_eval in enumerate(DLoader_eval):
                i += 1
                glove_true = batch_eval[1].cuda()
                glove_true = glove_true.view(glove_true.shape[1] * args.batch_size,
                                             glove_true.shape[2])  # .cpu().numpy()  # .astype(np.float32)

                glove_pred = model(data_wrapper(batch_eval[0].squeeze(3)).cuda(), )[0]  # BERT/TCN
                # glove_pred, hidden_1 = model(data_wrapper(batch_eval[0].squeeze(3)).cuda(), hidden_1)  # RNN with hidden

                glove_pred = glove_pred.view(glove_pred.shape[1] * args.batch_size,
                                             glove_pred.shape[2])  # .cpu().numpy()  # .astype(np.float32)

                total_output_test = torch.cat([total_output_test, glove_pred])
                total_target_test = torch.cat([total_target_test, glove_true])
        total_output_test = total_output_test.detach().cpu().numpy()
        if args.smooth:
            total_output_test = avg_smoothing_np(5, total_output_test)
        total_target_test = total_target_test.detach().cpu().numpy()
        NRMSE = metrics.normalized_root_mse(total_target_test, total_output_test, normalization="min-max")
        CC_pearson = pearson_CC(total_target_test, total_output_test)
        r2 = skmetrics.r2_score(total_target_test.T, total_output_test.T, multioutput="variance_weighted")
        # add log
        record_writer.add_scalar('Loss', epoch_loss, global_step=epoch + 1)
        record_writer.add_scalar('CC', CC_pearson, global_step=epoch + 1)
        record_writer.add_scalar('NRMSE', NRMSE, global_step=epoch + 1)
        record_writer.add_scalar('R2', r2, global_step=epoch + 1)
        if NRMSE < best_epoch['NRMSE']:
            # if CC_pearson > best_epoch['CC']:
            torch.save(model, os.path.join(save_dir, 'model_best.pth'))
            best_epoch['NRMSE'] = NRMSE
            best_epoch['epoch'] = epoch + 1
            best_epoch['CC'] = CC_pearson
            best_epoch['R2'] = r2
            record_writer.add_text("Best epoch", str(epoch + 1))
            record_writer.add_text("Best CC now", str(best_epoch['CC']), )
            record_writer.add_text("Best NRMSE now", str(best_epoch['NRMSE']), )
            record_writer.add_text("Best R2 now", str(best_epoch['R2']), )
            fig = draw_graph(total_output_test, total_target_test)
            record_writer.add_figure("test_results", fig, global_step=epoch + 1)

        message1 = 'epoch = {:03d}, [time] = {:.2f}s, [NRMSE of {}-frames]:{:.3f}, [CC of {}-frames]:{:.3f},[R2 of {}-frames]:{:.3f},[loss] = {:.7f}.'.format(
            epoch + 1,
            elapsed_time,
            i, NRMSE,
            i, CC_pearson,
            i, r2,
            epoch_loss)
        message2 = 'Best @ {:03d}, with NRMSE {:.3f}, CC {:.3f}, R2 {:.3f}. \n'.format(best_epoch['epoch'],
                                                                                       best_epoch['NRMSE'],
                                                                                       best_epoch["CC"],
                                                                                       best_epoch['R2'])
        print(message1)
        print(message2)
        with open(log_file, 'a') as f:
            f.write(message1)
            f.write("\n")
            f.write(message2)
        torch.save(model, os.path.join(save_dir, 'model_latest.pth'))
        scheduler.step()  # step to the learning rate in this epoch

    elpased_time_list = numpy.array(elpased_time_list)
    time_message = f"[*]Average time cost: {numpy.sum(elpased_time_list) / args.epoch}"
    record_writer.add_text("Average time cost", str(numpy.sum(elpased_time_list) / args.epoch), )
    print(time_message)
    with open(log_file, "a") as f:
        f.write("\n")
        f.write(time_message)

    torch.cuda.empty_cache()
    record_writer.close()

    ## Store the data for CSLN(LCSN)
    model = torch.load(os.path.join(save_dir, 'model_best.pth'))
    model.memory = MemBuffer(len(subject), 100)
    score_criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    for n_count, batch_tr in enumerate(TrainLoader):
        batch_emg = batch_tr[0].squeeze(3).permute(0, 2, 1).cuda()
        batch_glove = batch_tr[1].cuda()
        y_c_label = batch_tr[2].cuda()
        y_s_label = batch_tr[3].cuda()
        # y_pred, y_c, y_s, x_hat, z_s, z_c = model(batch_emg)
        y_pred, y_s, _, _ = model(batch_emg)
        score = score_criterion(y_s, torch.max(y_s_label, dim=1)[1])
        model.memory.check_and_update((batch_emg.permute(0, 2, 1), y_s_label, score))
    for n_count, batch_tr in enumerate(DLoader_eval):
        batch_emg = batch_tr[0].squeeze(3).permute(0, 2, 1).cuda()
        batch_glove = batch_tr[1].cuda()
        y_c_label = batch_tr[2].cuda()
        y_s_label = batch_tr[3].cuda()
        # y_pred, y_c, y_s, x_hat, z_s, z_c = model(batch_emg)
        y_pred, y_s, _, _ = model(batch_emg)
        score = score_criterion(y_s, torch.max(y_s_label, dim=1)[1])
        model.memory.check_and_update((batch_emg.permute(0, 2, 1), y_s_label, score))
    torch.save(model, os.path.join(save_dir, 'model_best.pth'))
    torch.cuda.empty_cache()
    record_writer.close()
