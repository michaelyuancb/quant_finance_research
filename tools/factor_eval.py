from torch import nn
import pandas as pd
import numpy as np
import torch


class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return self.loss(x, y)


class RMSELoss(nn.Module):

    def __init__(self):
        super(RMSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        return torch.sqrt(self.loss(x, y))


class ICLoss(nn.Module):

    def __init__(self):
        super(ICLoss, self).__init__()
        pass

    def forward(self, x, y):
        x, y = x.reshape(-1), y.reshape(-1)
        x_mean, y_mean = x.mean(), y.mean()
        x_std, y_std = x.std(), y.std()
        x_z, y_z = (x - x_mean) / x_std, (y - y_mean) / y_std
        pr = torch.sum(x_z * y_z) / (x.shape[0] - 1)
        return -pr  # -pearson


class CCCLoss(nn.Module):

    def __init__(self):
        super(CCCLoss, self).__init__()
        pass

    def forward(self, x, y):
        x, y = x.reshape(-1), y.reshape(-1)
        sxy = torch.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
        ccc = 2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
        return -ccc  # -ccc


def evaluate_build(pred, label, time_id):
    df_tmp = pd.DataFrame(np.stack((time_id, pred, label), axis=1))
    return df_tmp


def evaluate_IC(pred, label, time_id):
    df = evaluate_build(pred, label, time_id)
    ic = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        icv = dft.corr(method='pearson').iloc[0, 1]
        ic.append(icv)
    return np.mean(np.array(ic)), ic


def evaluate_RankIC(pred, label, time_id):
    df = evaluate_build(pred, label, time_id)
    ric = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        ricv = dft.corr(method='spearman').iloc[0, 1]
        ric.append(ricv)
    return np.mean(np.array(ric)), ric


def evaluate_IR(pred, label, time_id):
    _, ic = evaluate_IC(pred, label, time_id)
    ir = np.array(ic)
    return np.mean(ir) / np.std(ir)


def evaluate_RankIR(pred, label, time_id):
    _, ric = evaluate_RankIC(pred, label, time_id)
    rir = np.array(ric)
    return np.mean(ric) / np.std(ric)


def evaluate_classTop_acc(pred, label, time_id, class_num=5):
    df = evaluate_build(pred, label, time_id)
    top_acc = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        cnt = dft.shape[0]
        if cnt < 1000:
            continue
        pv = np.argsort(dft.values[:, 0]).argsort() < cnt // class_num
        lv = np.argsort(dft.values[:, 1]).argsort() < cnt // class_num
        acc = np.sum(pv * lv) / (cnt // class_num)
        top_acc.append(acc)
    return np.mean(np.array(top_acc)), top_acc


def evaluate_classBottom_acc(pred, label, time_id, class_num=5):
    df = evaluate_build(pred, label, time_id)
    top_acc = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        cnt = dft.shape[0]
        if cnt < 1000:
            continue
        pv = np.argsort(dft.values[:, 0]).argsort() > cnt - cnt // class_num - 1
        lv = np.argsort(dft.values[:, 1]).argsort() > cnt - cnt // class_num - 1
        acc = np.sum(pv * lv) / (cnt // class_num)
        top_acc.append(acc)
    return np.mean(np.array(top_acc)), top_acc


def evaluate_factor(pred, label, time_id, model_name='default'):
    # pred & label & time_id: 1D numpy.
    ic = evaluate_IC(pred, label, time_id)
    ir = evaluate_IR(pred, label, time_id)
    ric = evaluate_RankIC(pred, label, time_id)
    rir = evaluate_RankIR(pred, label, time_id)
    t5 = evaluate_classTop_acc(pred, label, time_id, class_num=5)
    b5 = evaluate_classBottom_acc(pred, label, time_id, class_num=5)
    t10 = evaluate_classTop_acc(pred, label, time_id, class_num=10)
    b10 = evaluate_classBottom_acc(pred, label, time_id, class_num=10)
    t20 = evaluate_classTop_acc(pred, label, time_id, class_num=20)
    b20 = evaluate_classBottom_acc(pred, label, time_id, class_num=20)
    data_list = [ic[0], ric[0], ir, rir, t5[0], b5[0], t10[0], b10[0], t20[0], b20[0]]
    data_list = [np.round(d, 3) for d in data_list]
    df = pd.DataFrame([[model_name] + data_list],
                      columns=['Model', 'IC', 'RankIC', 'IR', 'RankIR',
                               '5CTopAcc', '5CBotAcc',
                               '10CTopAcc', '10CBotAcc',
                               '20CTopAcc', '20CBotAcc'])
    return df
