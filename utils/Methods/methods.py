import argparse
import math

import numpy as np
import sklearn.metrics
import torch
from torch import nn
import matplotlib.pyplot as plt

import torch.nn.functional as F

"""=========================Feature Extraction============================="""


def rms(data):
    return np.sqrt((np.sum(data ** 2)) / data.shape[0])

def USTD(data):
    """Unbiased Standard Deviation"""
    return np.std(data, ddof=1)


"""=======================Criteria========================="""


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    # log_mean_output = ((p_output + q_output )/2).log()
    log_mean_output = ((p_output + q_output) / 2)
    return (KLDivLoss(p_output, log_mean_output) + KLDivLoss(q_output, log_mean_output)) / 2


def pearson_CC(x, y):
    """
    :param x: A tensor
    :param y: A tensor
    :return: Pearson CC of X & Y
    """
    x = np.array(x)
    y = np.array(y)
    assert x.shape == y.shape
    stdx = x.std()
    stdy = y.std()
    covxy = np.mean((x - x.mean()) * (y - y.mean()))
    return covxy / (stdx * stdy)


def curvature(x, y):
    import numpy.linalg as LA
    """
    input  : the coordinate of the three point
    output : the curvature and norm direction
    """

    t_a = LA.norm([x[1] - x[0], y[1] - y[0]])
    t_b = LA.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([
        [1, -t_a, t_a ** 2],
        [1, 0, 0],
        [1, t_b, t_b ** 2]
    ])

    a = np.matmul(LA.inv(M), x)
    b = np.matmul(LA.inv(M), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** (1.5)
    return kappa, [b[1], -a[1]] / np.sqrt(a[1] ** 2. + b[1] ** 2.)


def get_smooth_curve(curve):
    ka = []
    no = []
    pos = []
    for idx, theta in enumerate(curve[1:-2]):
        x = list(range(idx, idx + 3))
        y = curve[idx: idx + 3]
        # print(x,y)
        kappa, norm = curvature(x, y)
        ka.append(np.abs(kappa))
        no.append(norm)
        pos.append((x[1], y[1]))

    return np.average(ka), no, pos


"""================Normalization========================="""


def Mu_Normalization(data, Mu=256):
    # print(data.shape)
    result = np.sign(data) * np.log((1 + Mu * np.abs(data))) / np.log((1 + Mu))
    return result


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


"""==================Common Module======================="""


class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class NoSEBlock(nn.Module):
    def __init__(self, input_dim, reduction):
        super().__init__()

    def forward(self, x):
        return x


"""============Post-process============="""


def avg_smoothing(window_size, s):
    """
    For Pytorch
    :param window_size: processing window size (int)
    :param s: serial data to handle(numpy array)
    :return: smoothened data(numpy array)
    """
    for j in range(s.shape[0]):
        for i in range(s.shape[1]):
            s[j, i] = torch.mean(s[j:j + window_size, i])
    return s


def avg_smoothing_np(window_size, s):
    """
    For numpy
    :param window_size: processing window size (int)
    :param s: serial data to handle(numpy array)
    :return: smoothened data(numpy array)
    """
    for j in range(s.shape[0]):
        for i in range(s.shape[1]):
            s[j, i] = np.mean(s[j:j + window_size, i])
    return s


"""=========================Misc=============================="""


def draw_graph(output, target, channel_num=10):
    fig = plt.figure(figsize=(20, channel_num))
    for i in range(channel_num):
        plt.subplot(math.ceil(channel_num / 2), 2, i + 1)
        # plt.plot(total_output_test[:, i].detach().cpu().numpy(), label="predict", color="b")
        # plt.plot(total_target_test[:, i].detach().cpu().numpy(), label="truth", color="r")
        plt.plot(output[:, i], label="predict", color="b")
        plt.plot(target[:, i], label="truth", color="r")
        # print(total_output_test[:, i].detach().cpu().numpy().shape)
    return fig

def draw_graph_2c(output, target):
    fig = plt.figure(figsize=(10, 4))

    plt.subplot(2, 1, 1)  # 第一个子图在上方
    plt.plot(output[:, 3], label="predict", color="b")
    plt.plot(target[:, 3], label="truth", color="r")

    plt.subplot(2, 1, 2)  # 第二个子图在下方
    plt.plot(output[:, 5], label="predict", color="b")
    plt.plot(target[:, 5], label="truth", color="r")
    fig.tight_layout()
    return fig

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif str(v).strip().lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. Your input:{v} and input type:{type(v)}')


if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([6, 7, 8, 9, 100])
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    f = m * x + c
    print(f)
    print(pearson_CC(x, y))
    print(pearson_CC(x, y) ** 2)
    print(sklearn.metrics.r2_score(x, y))
    print(sklearn.metrics.r2_score(y, f))
