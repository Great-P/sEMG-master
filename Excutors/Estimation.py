import sys

import numpy as np

from utils.DataProcess import NinaPro

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils.Methods.methods import pearson_CC, draw_graph_2c
import time
from skimage import metrics
import sklearn.metrics as skmetrics
from utils.Methods.methods import avg_smoothing_np, get_smooth_curve

sys.path.append("..")

normalization = "miu"

test_name = "A1-A2-A3-A4-A5"  # trained model name
model_name = "LCSN"  # trained model type

test_subjects = ['S8']  # which subjects to test

if "S0" in test_subjects:
    test_subjects.remove("S0")
    test_subjects = ["S1", "S2", "S3", "S13", "S25", "S11", "S14", "S18", "S19", "S22"] + test_subjects

try:
    model = torch.load(f'../models/{normalization}/{test_name}/{model_name}/model_best.pth')

except Exception as e:
    raise e


def estimation(test_subject):
    print("=" * 49 + test_subject + "=" * 49)
    emgtest_dir = f"../Data/featureset/{test_subject}_E2_A1_rms_test.h5"
    glovetest_dir = f"../Data/featureset/{test_subject}_E2_A1_glove_test.h5"

    data_read_test = NinaPro.NinaPro(emgtest_dir, glovetest_dir, window_size=200, subframe=200,
                                     normalization=normalization, mu=2 ** 20, dummy_label=0, class_num=1, )
    # dummy_tsk=model.task_num - 1, tsk_num=model.task_num)
    loader_test = DataLoader(dataset=data_read_test, batch_size=1, shuffle=False, drop_last=True)
    output_predict = torch.Tensor([])
    output_target = torch.Tensor([])
    x_produce = torch.Tensor([])
    x_true = torch.Tensor([])
    model.eval()
    hidden = None
    print(len(loader_test))
    for step, batch_tr in tqdm(enumerate(loader_test), total=len(loader_test)):
        start_time = time.time()
        # x_true = torch.cat([x_true, batch_tr[0].permute(0,2,1).squeeze().detach().cpu()])
        x_true = torch.cat([x_true, batch_tr[0].squeeze().detach().cpu()])
        data = batch_tr[0].squeeze(3).permute(0, 2, 1).cuda()
        target = batch_tr[1].cuda()

        # output_test = model.gen_encoder(data)[0]  # BERT-based/TCN
        # output_test, y_c, y_s, x_hat, = model(data)[0:4]  # BERT-based/TCN
        # output_test, y_c, x_hat_c, x_hat_r, = model(data)[0:4]  # BERT-based/TCN
        # output_test = model.inference(data)  # LCSN

        output_test = output_test.view(output_test.shape[1],
                                       output_test.shape[2]).detach().cpu()
        target = target.view(target.shape[1],
                             target.shape[2]).detach().cpu()

        output_predict = torch.cat([output_predict, output_test])
        output_target = torch.cat([output_target, target])

    output_target = output_target.numpy()
    output_predict = output_predict.numpy()

    if model_name[0] == "s":
        output_predict = avg_smoothing_np(5, output_predict)
    nrmses = list()
    ccs = list()
    r2s = list()
    for i in range(10):
        NRMSE = metrics.normalized_root_mse(output_predict[:, i], output_target[:, i], normalization="min-max")
        CC = pearson_CC(output_predict[:, i], output_target[:, i])
        r2 = skmetrics.r2_score(output_predict[:, i], output_target[:, i], multioutput="variance_weighted")
        nrmses.append(NRMSE)
        ccs.append(CC)
        r2s.append(r2)
    std_nrmse = np.std(nrmses, ddof=1)
    std_cc = np.std(ccs, ddof=1)
    std_r2 = np.std(r2s, ddof=1)
    CC = pearson_CC(output_predict, output_target)
    NRMSE = metrics.normalized_root_mse(output_target, output_predict, normalization="min-max")
    R2 = skmetrics.r2_score(output_target, output_predict, multioutput="variance_weighted")
    # rec = pearson_CC(x_true, x_produce)
    rec = -1
    smooth = 0
    for i in range(10):
        smooth += get_smooth_curve(output_predict[:, i])[0]
    smooth /= 10
    # R2 = skmetrics.r2_score(output_target, output_predict,)

    print(f"[*]CC:{CC},NRMSE:{NRMSE},R2:{R2}, Smooth:{smooth}, Recovery:{rec}")
    print(f"[*]CCstd:{std_cc}, NRMSEstd:{std_nrmse}, R2std:{std_r2}")
    print("-" * 100 + "\n")

    # if test_subject == "S1":
    fig = draw_graph_2c(output_predict, output_target)
    # fig = draw_graph(x_produce, x_true,12)
    plt.savefig("../1.pdf")
    plt.show()
    return CC, NRMSE, std_cc, std_nrmse


if __name__ == "__main__":
    cclist = []
    mselist = []
    stdc = []
    stdn = []
    for subject in test_subjects:
        cc, mse, stdcc, stdnrmse = estimation(subject)
        cclist.append(cc)
        mselist.append(mse)
        stdc.append(stdcc)
        stdn.append(stdnrmse)

    print("=" * 49 + "==" + "=" * 49)

    print(
        f"[*]TaskCC:{sum(cclist) / len(cclist)},TaskNRMSE:{sum(mselist) / len(mselist)}, TaskstdC:{sum(stdc) / len(stdc)}, TaskstdC:{sum(stdn) / len(stdn)}")
