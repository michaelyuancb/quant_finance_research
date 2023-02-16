from torch import nn
import pandas as pd
import numpy as np
import torch


def evaluate_build(pred, label):
    df_tmp = pd.DataFrame(np.stack((pred, label), axis=1))
    return df_tmp


def evaluate_mse(pred, label):
    return np.mean((pred - label) ** 2)


def evaluate_rmse(pred, label):
    return np.sqrt(evaluate_mse(pred, label))


def evaluate_IC(pred, label):
    df = evaluate_build(pred, label)
    icv = df.corr(method='pearson').iloc[0, 1]
    return icv


def evaluate_RankIC(pred, label):
    df = evaluate_build(pred, label)
    icv = df.corr(method='spearman').iloc[0, 1]
    return icv


def evaluate_CCC(pred, label):
    sxy = np.sum((pred - pred.mean()) * (label - label.mean())) / pred.shape[0]
    ccc = 2 * sxy / (pred.var() + label.var() + (pred.mean() - label.mean()) ** 2)
    return ccc


def evaluate_build_time(pred, label, time_id):
    if type(time_id) is list:
        time_id = np.array(time_id)
    df_tmp = pd.DataFrame(np.stack((time_id, pred, label), axis=1))
    return df_tmp


def evaluate_mse_time(pred, label, time_id):
    df = evaluate_build_time(pred, label, time_id)
    mse = []
    for t in list(set(time_id)):
        dft = df[df[0] == t]
        imse = evaluate_mse(dft.iloc[:, 1].values.reshape(-1), dft.iloc[:, 2].values.reshape(-1))
        mse.append(imse)
    return np.mean(np.array(mse)), mse


def evaluate_rmse_time(pred, label, time_id):
    df = evaluate_build_time(pred, label, time_id)
    rmse = []
    for t in list(set(time_id)):
        dft = df[df[0] == t]
        irmse = evaluate_rmse(dft.iloc[:, 1].values.reshape(-1), dft.iloc[:, 2].values.reshape(-1))
        rmse.append(irmse)
    return np.mean(np.array(rmse)), rmse


def evaluate_IC_time(pred, label, time_id):
    df = evaluate_build_time(pred, label, time_id)
    ic = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        icv = dft.corr(method='pearson').iloc[0, 1]
        ic.append(icv)
    return np.mean(np.array(ic)), ic


def evaluate_RankIC_time(pred, label, time_id):
    df = evaluate_build_time(pred, label, time_id)
    ric = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        ricv = dft.corr(method='spearman').iloc[0, 1]
        ric.append(ricv)
    return np.mean(np.array(ric)), ric


def evaluate_CCC_time(pred, label, time_id):
    df = evaluate_build_time(pred, label, time_id)
    rccc = []
    for t in list(set(time_id)):
        dft = df[df[0] == t]
        ccc = evaluate_CCC(dft.iloc[:, 1].values.reshape(-1), dft.iloc[:, 2].values.reshape(-1))
        rccc.append(ccc)
    return np.mean(np.array(rccc)), rccc


def evaluate_IR_time(pred, label, time_id):
    _, ic = evaluate_IC_time(pred, label, time_id)
    ir = np.array(ic)
    if np.std(ir) == 0.0:
        return np.inf
    else:
        return np.mean(ir) / np.std(ir)


def evaluate_RankIR_time(pred, label, time_id):
    _, rir = evaluate_RankIC_time(pred, label, time_id)
    rir = np.array(rir)
    if np.std(rir) == 0.0:
        return np.inf
    else:
        return np.mean(rir) / np.std(rir)


def evaluate_classTop_acc_time(pred, label, time_id, class_num=5):
    df = evaluate_build_time(pred, label, time_id)
    top_acc = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        cnt = dft.shape[0]
        if cnt < class_num:
            top_acc.append(np.nan)
            continue
        pv = np.argsort(dft.values[:, 0]).argsort() > cnt - cnt // class_num - 1
        lv = np.argsort(dft.values[:, 1]).argsort() > cnt - cnt // class_num - 1
        acc = np.sum(pv * lv) / (cnt // class_num)
        top_acc.append(acc)
    return np.nanmean(np.array(top_acc)), top_acc


def evaluate_classBottom_acc_time(pred, label, time_id, class_num=5):
    df = evaluate_build_time(pred, label, time_id)
    top_acc = []
    for t in list(set(time_id)):
        dft = df[df[0] == t].iloc[:, 1:]
        cnt = dft.shape[0]
        if cnt < class_num:
            top_acc.append(np.nan)
            continue
        pv = np.argsort(dft.values[:, 0]).argsort() < cnt // class_num
        lv = np.argsort(dft.values[:, 1]).argsort() < cnt // class_num
        acc = np.sum(pv * lv) / (cnt // class_num)
        top_acc.append(acc)
    return np.nanmean(np.array(top_acc)), top_acc


def evaluate_factor_time_classic_1(pred, label, time_id, model_name='Default'):
    # pred & label & time_id: 1D numpy.
    ic = evaluate_IC_time(pred, label, time_id)
    ir = evaluate_IR_time(pred, label, time_id)
    ric = evaluate_RankIC_time(pred, label, time_id)
    rir = evaluate_RankIR_time(pred, label, time_id)
    t5 = evaluate_classTop_acc_time(pred, label, time_id, class_num=5)
    b5 = evaluate_classBottom_acc_time(pred, label, time_id, class_num=5)
    t10 = evaluate_classTop_acc_time(pred, label, time_id, class_num=10)
    b10 = evaluate_classBottom_acc_time(pred, label, time_id, class_num=10)
    data_list = [ic[0], ric[0], ir, rir, t5[0], b5[0], t10[0], b10[0]]
    data_list = [np.round(d, 3) for d in data_list]
    df = pd.DataFrame([[model_name] + data_list],
                      columns=['Model', 'IC', 'RankIC', 'IR', 'RankIR',
                               '5CTopAcc', '5CBotAcc',
                               '10CTopAcc', '10CBotAcc'])
    return df
