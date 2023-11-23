"""
Transfer via Lifelong learning strategy

"""

import sys

from utils.DataProcess import NinaPro

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
import torch.nn.functional as F

from skimage import metrics
import sklearn.metrics as skmetrics
from utils.Methods.methods import pearson_CC, avg_smoothing_np, draw_graph

from utils.sEMG_models.LifelongCrossSubjectNetwork import LifelongCrossSubjectNetwork

from tensorboardX import SummaryWriter
from utils.Methods.torchsummary import summary
import copy

# Params
parser = argparse.ArgumentParser(description='PyTorch sEMG-Bert')
# dataset
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--subframe', default=200, type=int)
parser.add_argument('--epoch', default=200, type=int)
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

parser.add_argument('--new_subject', type=str, default=[f'S30', 'S36', 'S3'])
# [f"S{i+1}" for i in range(40)]
parser.add_argument('--source_name', type=str, default="E1-E2-E3-E4")
parser.add_argument('--target_name', type=str, default="E5")

args = parser.parse_args()

cuda = torch.cuda.is_available()
assert cuda

# subject = args.subject.strip()
subject = args.new_subject
source_name = args.source_name.strip()
target_name = args.target_name.strip()
if not isinstance(subject, list):
    subject = [subject]
if "S0" in subject:
    subject.remove("S0")
    subject = ["S1", "S2", "S3", "S13", "S25", "S11", "S14", "S18", "S19", "S22"] + subject

new_cls_num = len(subject)


def cls_new_loss(y_pred, y_label):
    outputs_S = F.softmax(y_pred, dim=1)
    outputs_T = F.softmax(y_label, dim=1)
    loss2 = outputs_T.mul(-1 * torch.log(outputs_S))
    loss2 = loss2.sum(1)
    loss2 = loss2.mean()
    return loss2


def cls_old_loss(y_pred, y_label, T):
    outputs_S = F.softmax(y_pred / T, dim=1)
    outputs_T = F.softmax(y_label / T, dim=1)
    # Cross entropy between output of the old task and output of the old model
    loss2 = outputs_T.mul(-1 * torch.log(outputs_S))
    loss2 = loss2.sum(1)
    loss2 = loss2.mean() * T * T
    return loss2


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
    model = LifelongCrossSubjectNetwork(vocab_size=args.subframe, hidden=args.hidden, class_num=len(subject),
                                        joint_channels=10, emg_channels=12)
    # model.check()

    initial_epoch = 0
    reg_loss = nn.MSELoss()
    cls_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    cur_model_name = model.name
    if args.smooth:
        cur_model_name = "s" + cur_model_name
    # criterion = sum_squared_error()
    tensorboard_record_path = f"../runs/{args.normalization}/{source_name}-{target_name}/{cur_model_name}"
    # save_dir = args.save_dir
    source_dir = f"../models/{args.normalization}/{source_name}/{cur_model_name}/model_best.pth"
    save_dir = f"../models/{args.normalization}/{source_name}-{target_name}/{cur_model_name}"
    model = torch.load(source_dir)
    old_cls_num = model.shared_cls_head.classnum

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(tensorboard_record_path):
        print("[*]Cleaning previous cache of tensorboard...")
        shutil.rmtree(tensorboard_record_path)
    record_writer = SummaryWriter(tensorboard_record_path)
    dummy_input = torch.randn([1, 200, 12]).permute(0, 2, 1).cuda()
    print(f"[*]Current model is {cur_model_name}")

    # model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # dataset
    train_dataset_list = []
    test_dataset_list = []
    for i, each_subject in enumerate(subject):
        emgtrain_dir = f"../Data/featureset/{each_subject}_E2_A1_rms_train.h5"
        glovetrain_dir = f"../Data/featureset/{each_subject}_E2_A1_glove_train.h5"
        emgtest_dir = f"../Data/featureset/{each_subject}_E2_A1_rms_test.h5"
        glovetest_dir = f"../Data/featureset/{each_subject}_E2_A1_glove_test.h5"
        train_dataset = NinaPro.NinaPro(emgtrain_dir, glovetrain_dir, window_size=200, subframe=args.subframe,
                                        normalization=args.normalization, mu=args.miu, dummy_label=old_cls_num + i,
                                        class_num=old_cls_num + new_cls_num)
        test_dataset = NinaPro.NinaPro(emgtest_dir, glovetest_dir, window_size=200, subframe=args.subframe,
                                       normalization=args.normalization, mu=args.miu, dummy_label=old_cls_num + i,
                                       class_num=old_cls_num + new_cls_num)
        train_dataset_list.append(train_dataset)
        test_dataset_list.append(test_dataset)
    concat_train_dataset = ConcatDataset(train_dataset_list)
    concat_test_dataset = ConcatDataset(test_dataset_list)
    TrainLoader = DataLoader(dataset=concat_train_dataset, num_workers=12, drop_last=True, batch_size=args.batch_size,
                             shuffle=True, )  # sampler=train_sampler, pin_memory=True)
    DLoader_eval = DataLoader(dataset=concat_test_dataset, num_workers=12, drop_last=True, batch_size=args.batch_size,
                              shuffle=False, )
    ELoader = DataLoader(dataset=concat_test_dataset, num_workers=12, drop_last=True, batch_size=1,
                              shuffle=False, ) # sampler=test_sampler,pin_memory=True)
    """=====transfer reconstruction====="""
    expert_model = nn.Sequential(
        copy.deepcopy(model.chara_encoder),
        copy.deepcopy(model.shared_cls_head),
    )
    model.gen_encoder.extend_capsules(new_cls_num)
    cls_model = nn.Sequential(
        model.chara_encoder,
        model.shared_cls_head
    )
    model.shared_cls_head.extend_cls(new_cls_num, )

    """Expert Gate Initialization"""
    for i in range(len(subject)):
        for _, data_tr in enumerate(ELoader):
            if torch.max(data_tr[3], dim=1) == old_cls_num + i:
                expert_sel = expert_model(data_tr[0][0].squeeze(3).permute(0, 2, 1).cuda())
                init_capsule_no = torch.max(expert_sel, dim=1)[1]
                # weight_init = model.gen_encoder.capsules[init_capsule_no].weight.data
                model.gen_encoder.capsules[old_cls_num + i] = copy.deepcopy(model.gen_encoder.capsules[init_capsule_no])
                # model.shared_cls_head.expert_gate_init(i, init_capsule_no)
                break
    del ELoader
    for name, parameter in expert_model.named_parameters():
        parameter.requires_grad = False
    # for name, parameter in teacher_mem.named_parameters():
    #     parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model.chara_encoder.named_parameters():
        parameter.requires_grad = True
    for name, parameter in model.shared_cls_head.named_parameters():
        parameter.requires_grad = True
    for name, parameter in model.rec_cls.named_parameters():
        parameter.requires_grad = True
    for name, parameter in model.rec_reg.named_parameters():
        parameter.requires_grad = True
    # for name, parameter in model.mem.named_parameters():
    #     parameter.requires_grad = True
    for name, parameter in model.gen_encoder.capsules[-new_cls_num:].named_parameters():
        parameter.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=cur_lr,weight_decay=1e-10)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)  # learning rates
    if cuda:
        print("[*]Training on GPU......")
        model = model.cuda()
        # record_writer.add_graph(model, dummy_input, )
        device_ids = args.device_ids
        reg_loss = reg_loss.cuda()
        cls_loss = cls_loss.cuda()
        kl_loss = kl_loss.cuda()
    record_writer.add_text("Model Scale", summary(model, (12, args.subframe, 1), args.batch_size))
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
    replay_ratio = old_cls_num / (old_cls_num + new_cls_num)
    # posibility = np.array([0.7, 0.3])

    for epoch in range(initial_epoch, args.epoch):
        print('[*]Current Learning rate ={:.6f}'.format(scheduler.get_last_lr()[0]))

        epoch_loss = 0
        start_time = time.time()
        # training phase
        model.train()

        for n_count, batch_tr in enumerate(TrainLoader):
            batch_emg = batch_tr[0].squeeze(3).permute(0, 2, 1).cuda()
            # batch_emg = batch_tr[0].squeeze(3).cuda()
            # print(batch_emg.shape)
            batch_glove = batch_tr[1].cuda()

            y_c_label = batch_tr[2].cuda()
            y_s_label = batch_tr[3].cuda()

            emg_shuffle, shuffle_label = model.memory.data_confusion(batch_emg.permute(0, 2, 1), y_s_label,
                                                                     replay_ratio)
            emg_shuffle = emg_shuffle.permute(0, 2, 1)

            # y_s_shuffle = cls_model(emg_shuffle)

            _, y_c_shuffle, x_hat_r, x_hat_c = model(emg_shuffle)

            y_pred = model.inference(batch_emg)
            # y_pred, y_c, y_s, _, z_s, z_c = model(batch_emg)

            # loss = 0.01 * reg_loss(y_s, y_s_label) + 0.99 * reg_loss(y_s_remember, y_s_old)
            loss = (reg_loss(y_pred, batch_glove) + reg_loss(x_hat_r, emg_shuffle) + reg_loss(x_hat_c,
                                                                                            emg_shuffle)) + 1e4 * (
                       cls_loss(y_c_shuffle, torch.max(shuffle_label, dim=1)[1]))
            # + kl_loss(F.log_softmax(z_c,dim=1), F.softmax(z_c_shuffle))
            # if epoch < 20:
            #     loss = cls_loss(y_s, torch.max(y_s_label, dim=1)[1]) + cls_new_loss(y_c, y_c_label)
            # else:
            #     loss = reg_loss(y_pred, batch_glove) + reg_loss(x_hat, batch_emg) \
            #            + (cls_loss(y_s, torch.max(y_s_label, dim=1)[1]) + cls_new_loss(y_c, y_c_label))

            epoch_loss += loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
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
                glove_true = glove_true.view(glove_true.shape[1]*args.batch_size,
                                             glove_true.shape[2])  # .cpu().numpy()  # .astype(np.float32)

                # glove_pred, hidden_1 = model(batch_eval[0].cuda(), hidden_1)  # LSTM
                glove_pred = model.inference(batch_eval[0].squeeze(3).permute(0, 2, 1).cuda(), )
                # glove_pred,_ = model(batch_eval[0].cuda(), )

                glove_pred = glove_pred.view(glove_pred.shape[1]*args.batch_size,
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
        # torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'model_latest.pth'))
        scheduler.step()  # step to the learning rate in this epoch

    elpased_time_list = numpy.array(elpased_time_list)
    time_message = f"[*]Average time cost: {numpy.sum(elpased_time_list) / args.epoch}"
    record_writer.add_text("Average time cost", str(numpy.sum(elpased_time_list) / args.epoch), )
    print(time_message)
    with open(log_file, "a") as f:
        f.write("\n")
        f.write(time_message)
    """收集样本"""
    model = torch.load(os.path.join(save_dir, 'model_best.pth'))
    model.memory.extend_subjects(new_cls_num)
    score_criterion = nn.CrossEntropyLoss(reduce=False).cuda()
    for n_count, batch_tr in enumerate(TrainLoader):
        batch_emg = batch_tr[0].squeeze(3).permute(0, 2, 1).cuda()
        batch_glove = batch_tr[1].cuda()
        y_c_label = batch_tr[2].cuda()
        y_s_label = batch_tr[3].cuda()
        # y_pred, y_c, y_s, x_hat, z_s, z_c = model(batch_emg)
        y_pred, y_s, _,_ = model(batch_emg)
        score = score_criterion(y_s, torch.max(y_s_label, dim=1)[1])
        model.memory.check_and_update((batch_emg.permute(0, 2, 1), y_s_label, score))
    for n_count, batch_tr in enumerate(DLoader_eval):
        batch_emg = batch_tr[0].squeeze(3).permute(0, 2, 1).cuda()
        batch_glove = batch_tr[1].cuda()
        y_c_label = batch_tr[2].cuda()
        y_s_label = batch_tr[3].cuda()
        # y_pred, y_c, y_s, x_hat, z_s, z_c = model(batch_emg)
        y_pred, y_s, _,_ = model(batch_emg)
        score = score_criterion(y_s, torch.max(y_s_label, dim=1)[1])
        model.memory.check_and_update((batch_emg.permute(0, 2, 1), y_s_label, score))
    torch.save(model, os.path.join(save_dir, 'model_best.pth'))
    torch.cuda.empty_cache()
    record_writer.close()
    # plt.figure(f'PyTorch_{cur_model_name}_Loss')
    # plt.plot(loss_count, label='Loss')
    # plt.legend()
    # plt.show()
