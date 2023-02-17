from torch import nn
import pandas as pd
import numpy as np
import torch


eps = 1e-20


def evaluate_build(pred, label, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    df_tmp = pd.DataFrame(np.stack((pred, label), axis=1))
    return df_tmp


def evaluate_MSE(pred, label, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    return np.mean((pred - label) ** 2)


def evaluate_RMSE(pred, label, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    return np.sqrt(evaluate_MSE(pred, label))


def evaluate_IC(pred, label, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    df = evaluate_build(pred, label)
    icv = df.corr(method='pearson').iloc[0, 1]
    return icv


def evaluate_RankIC(pred, label, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    df = evaluate_build(pred, label)
    icv = df.corr(method='spearman').iloc[0, 1]
    return icv


def evaluate_CCC(pred, label, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    sxy = np.sum((pred - pred.mean()) * (label - label.mean())) / pred.shape[0]
    ccc = 2 * sxy / (pred.var() + label.var() + (pred.mean() - label.mean()) ** 2)
    return ccc


def evaluate_classTop_acc(pred, label, class_num=5, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    cnt = pred.shape[0]
    if cnt < class_num:
        return np.nan
    pv = pred.argsort().argsort() > cnt - cnt // class_num - 1
    lv = label.argsort().argsort() > cnt - cnt // class_num - 1
    acc = np.sum(pv * lv) / (cnt // class_num)
    return acc


def evaluate_classBottom_acc(pred, label, class_num=5, need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    cnt = pred.shape[0]
    if cnt < class_num:
        return np.nan
    pv = pred.argsort().argsort() > cnt - cnt // class_num - 1
    lv = label.argsort().argsort() > cnt - cnt // class_num - 1
    acc = np.sum(pv * lv) / (cnt // class_num)
    return acc


def evaluate_factor_classic_1(pred, label, model_name='Default', need_reshape=True):
    if need_reshape:
        pred, label = pred.reshape(-1), label.reshape(-1)
    # pred & label & time_id: 1D numpy.
    ic = evaluate_IC(pred, label)
    ric = evaluate_RankIC(pred, label)
    t5 = evaluate_classTop_acc(pred, label, class_num=5, need_reshape=False)
    b5 = evaluate_classBottom_acc(pred, label, class_num=5, need_reshape=False)
    t10 = evaluate_classTop_acc(pred, label, class_num=10, need_reshape=False)
    b10 = evaluate_classBottom_acc(pred, label, class_num=10, need_reshape=False)
    data_list = [ic, ric, t5, b5, t10, b10]
    data_list = [np.round(d, 3) for d in data_list]
    df = pd.DataFrame([[model_name] + data_list],
                      columns=['Model', 'IC', 'RankIC',
                               '5CTopAcc', '5CBotAcc',
                               '10CTopAcc', '10CBotAcc'])
    return df


def evaluate_build_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df_tmp = pd.DataFrame(np.stack((pred, label), axis=1))
    df_tmp = pd.concat([pd.Series(time_id), df_tmp], axis=1)
    df_tmp.columns = [0, 1, 2]
    return df_tmp


def evaluate_MSE_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_time(pred, label, time_id, need_reshape=False)
    mse = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        imse = evaluate_MSE(dft.iloc[:, 1].values.reshape(-1), dft.iloc[:, 2].values.reshape(-1))
        mse.append(imse)
    return np.mean(np.array(mse)), mse, time_id_list


def evaluate_RMSE_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_time(pred, label, time_id, need_reshape=False)
    rmse = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        irmse = evaluate_RMSE(dft.iloc[:, 1].values.reshape(-1), dft.iloc[:, 2].values.reshape(-1))
        rmse.append(irmse)
    return np.mean(np.array(rmse)), rmse, time_id_list


def evaluate_IC_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_time(pred, label, time_id, need_reshape=False)
    ic = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]].iloc[:, 1:]
        # dft = df.iloc[[0,1,4], 1:]
        icv = evaluate_IC(dft.iloc[:, 0].values, dft.iloc[:, 1].values, need_reshape=False)
        ic.append(icv)
    return np.mean(np.array(ic)), ic, time_id_list


def evaluate_RankIC_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_time(pred, label, time_id, need_reshape=False)
    ric = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]].iloc[:, 1:]
        ricv = dft.corr(method='spearman').iloc[0, 1]
        ric.append(ricv)
    return np.mean(np.array(ric)), ric, time_id_list


def evaluate_CCC_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_time(pred, label, time_id, need_reshape=False)
    rccc = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        ccc = evaluate_CCC(dft.iloc[:, 1].values.reshape(-1), dft.iloc[:, 2].values.reshape(-1))
        rccc.append(ccc)
    return np.mean(np.array(rccc)), rccc, time_id_list


def evaluate_IR_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    _, ic, time_id_list = evaluate_IC_time(pred, label, time_id, need_reshape=False)
    ir = np.array(ic)
    if np.std(ir) == 0.0:
        return np.inf, time_id_list
    else:
        return np.mean(ir) / np.std(ir), time_id_list


def evaluate_RankIR_time(pred, label, time_id, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    _, rir, time_id_list = evaluate_RankIC_time(pred, label, time_id, need_reshape=False)
    rir = np.array(rir)
    if np.std(rir) == 0.0:
        return np.inf, time_id_list
    else:
        return np.mean(rir) / np.std(rir), time_id_list


def evaluate_classTop_acc_time(pred, label, time_id, class_num=5, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_time(pred, label, time_id, need_reshape=False)
    top_acc = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]].iloc[:, 1:]
        cnt = dft.shape[0]
        if cnt < class_num:
            top_acc.append(np.nan)
            continue
        pv = np.argsort(dft.values[:, 0]).argsort() > cnt - cnt // class_num - 1
        lv = np.argsort(dft.values[:, 1]).argsort() > cnt - cnt // class_num - 1
        acc = np.sum(pv * lv) / (cnt // class_num)
        top_acc.append(acc)
    return np.nanmean(np.array(top_acc)), top_acc, time_id_list


def evaluate_classBottom_acc_time(pred, label, time_id, class_num=5, need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_time(pred, label, time_id, need_reshape=False)
    top_acc = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]].iloc[:, 1:]
        cnt = dft.shape[0]
        if cnt < class_num:
            top_acc.append(np.nan)
            continue
        pv = np.argsort(dft.values[:, 0]).argsort() < cnt // class_num
        lv = np.argsort(dft.values[:, 1]).argsort() < cnt // class_num
        acc = np.sum(pv * lv) / (cnt // class_num)
        top_acc.append(acc)
    return np.nanmean(np.array(top_acc)), top_acc, time_id_list


def evaluate_factor_time_classic_1(pred, label, time_id, model_name='Default', need_reshape=True):
    if need_reshape:
        pred, label, time_id = pred.reshape(-1), label.reshape(-1), time_id.reshape(-1)
    # pred & label & time_id: 1D numpy.
    ic = evaluate_IC_time(pred, label, time_id, need_reshape=False)
    ir = evaluate_IR_time(pred, label, time_id, need_reshape=False)
    ric = evaluate_RankIC_time(pred, label, time_id, need_reshape=False)
    rir = evaluate_RankIR_time(pred, label, time_id, need_reshape=False)
    t20 = evaluate_classTop_acc_time(pred, label, time_id, class_num=20, need_reshape=False)
    b20 = evaluate_classBottom_acc_time(pred, label, time_id, class_num=20, need_reshape=False)
    t10 = evaluate_classTop_acc_time(pred, label, time_id, class_num=10, need_reshape=False)
    b10 = evaluate_classBottom_acc_time(pred, label, time_id, class_num=10, need_reshape=False)
    data_list = [ic[0], ric[0], ir[0], rir[0], t10[0], b10[0], t20[0], b20[0]]
    data_list = [np.round(d, 3) for d in data_list]
    df = pd.DataFrame([[model_name] + data_list],
                      columns=['Model', 'IC', 'RankIC', 'IR', 'RankIR',
                               '10CTopAcc', '10CBotAcc', '20CTopAcc', '20CBotAcc'])
    return df


def evaluate_MSE_StaticWeight(pred, label, weight, need_reshape=True):
    if need_reshape:
        pred, label, weight = pred.reshape(-1), label.reshape(-1), weight.reshape(-1)
    return np.mean(weight * (pred - label) ** 2)


def evaluate_RMSE_StaticWeight(pred, label, weight, need_reshape=True):
    if need_reshape:
        pred, label, weight = pred.reshape(-1), label.reshape(-1), weight.reshape(-1)
    return np.sqrt(evaluate_MSE_StaticWeight(pred, label, weight))


def evaluate_IC_StaticWeight(pred, label, weight, need_reshape=True):
    if need_reshape:
        pred, label, weight = pred.reshape(-1), label.reshape(-1), weight.reshape(-1)
    x, y = pred.reshape(-1), label.reshape(-1)
    weight = weight.reshape(-1)
    x_mean, y_mean = (x * weight).mean(), (y * weight).mean()
    cov = np.sum(weight * x * y) - x_mean * y_mean
    x_var = np.sum(weight * (x ** 2)) - x_mean ** 2
    y_var = np.sum(weight * (y ** 2)) - y_mean ** 2
    if x_var < 0:
        x_var = x_var + eps
    if y_var < 0:
        y_var = y_var + eps
    x_std = np.sqrt(x_var)  
    y_std = np.sqrt(y_var)
    wic = cov / (x_std * y_std)
    return wic


def evaluate_RankIC_StaticWeight(pred, label, weight, need_reshape=True):
    if need_reshape:
        pred, label, weight = pred.reshape(-1), label.reshape(-1), weight.reshape(-1)
    pred = pred.argsort().argsort()
    label = label.argsort().argsort()
    ric = evaluate_IC_StaticWeight(pred, label, weight, need_reshape=False)
    return ric


def evaluate_CCC_StaticWeight(pred, label, weight, need_reshape=True):
    if need_reshape:
        pred, label, weight = pred.reshape(-1), label.reshape(-1), weight.reshape(-1)
    x, y = pred.reshape(-1), label.reshape(-1)
    weight = weight.reshape(-1)
    x_mean, y_mean = (x * weight).mean(), (y * weight).mean()
    cov = np.sum(weight * x * y) - x_mean * y_mean
    x_var = np.sum(weight * (x ** 2)) - x_mean ** 2
    y_var = np.sum(weight * (y ** 2)) - y_mean ** 2
    ccc = 2 * cov / (x_var + y_var + (x_mean - y_mean) ** 2)
    return ccc  # -ccc


def evaluate_factor_StaticWeight_classic_1(pred, label, weight, model_name='Default', need_reshape=True):
    if need_reshape:
        pred, label, weight = pred.reshape(-1), label.reshape(-1), weight.reshape(-1)
    # pred & label & time_id: 1D numpy.
    ic = evaluate_IC_StaticWeight(pred, label, weight, need_reshape=False)
    ric = evaluate_IC_StaticWeight(pred, label, weight, need_reshape=False)
    data_list = [ic, ric]
    data_list = [np.round(d, 3) for d in data_list]
    df = pd.DataFrame([[model_name] + data_list],
                      columns=['Model', 'swIC', 'swRankIC'])
    return df


def evaluate_build_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    df_tmp = pd.DataFrame(np.stack((pred, label, weight), axis=1))
    df_tmp = pd.concat([pd.Series(time_id), df_tmp], axis=1)
    df_tmp.columns = [0, 1, 2, 3]
    return df_tmp


def evaluate_MSE_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    mse = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        imse = evaluate_MSE_StaticWeight(dft.iloc[:, 1].values.reshape(-1),
                                         dft.iloc[:, 2].values.reshape(-1),
                                         dft.iloc[:, 3].values.reshape(-1))
        mse.append(imse)
    return np.mean(np.array(mse)), mse, time_id_list


def evaluate_RMSE_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    rmse = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        irmse = evaluate_RMSE_StaticWeight(dft.iloc[:, 1].values.reshape(-1),
                                           dft.iloc[:, 2].values.reshape(-1),
                                           dft.iloc[:, 3].values.reshape(-1))
        rmse.append(irmse)
    return np.mean(np.array(rmse)), rmse, time_id_list


def evaluate_IC_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    ic = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        icv = evaluate_IC_StaticWeight(dft.iloc[:, 1].values.reshape(-1),
                                       dft.iloc[:, 2].values.reshape(-1),
                                       dft.iloc[:, 3].values.reshape(-1))
        ic.append(icv)
    return np.mean(np.array(ic)), ic, time_id_list


def evaluate_RankIC_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    ric = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        ricv = evaluate_RankIC_StaticWeight(dft.iloc[:, 1].values.reshape(-1),
                                            dft.iloc[:, 2].values.reshape(-1),
                                            dft.iloc[:, 3].values.reshape(-1))
        ric.append(ricv)
    return np.mean(np.array(ric)), ric, time_id_list


def evaluate_CCC_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    df = evaluate_build_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    rccc = []
    time_id_list = np.sort(np.unique(time_id))
    n_time_idx = len(time_id_list)
    for i in range(n_time_idx):
        dft = df[df[0] == time_id_list[i]]
        ccc = evaluate_CCC_StaticWeight(dft.iloc[:, 1].values.reshape(-1),
                                        dft.iloc[:, 2].values.reshape(-1),
                                        dft.iloc[:, 3].values.reshape(-1))
        rccc.append(ccc)
    return np.mean(np.array(rccc)), rccc, time_id_list


def evaluate_IR_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    _, ic, time_id_list = evaluate_IC_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    ir = np.array(ic)
    if np.std(ir) == 0.0:
        return np.inf, time_id_list
    else:
        return np.mean(ir) / np.std(ir), time_id_list


def evaluate_RankIR_StaticWeight_time(pred, label, weight, time_id, need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    _, rir, time_id_list = evaluate_RankIC_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    rir = np.array(rir)
    if np.std(rir) == 0.0:
        return np.inf, time_id_list
    else:
        return np.mean(rir) / np.std(rir), time_id_list


def evaluate_factor_StaticWeight_time_classic_1(pred, label, weight, time_id, model_name='Default', need_reshape=True):
    if need_reshape:
        pred, label, weight, time_id = pred.reshape(-1), label.reshape(-1), weight.reshape(-1), time_id.reshape(-1)
    # pred & label & time_id: 1D numpy.
    ic = evaluate_IC_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    ir = evaluate_IR_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    ric = evaluate_RankIC_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    rir = evaluate_RankIR_StaticWeight_time(pred, label, weight, time_id, need_reshape=False)
    data_list = [ic[0], ric[0], ir[0], rir[0]]
    data_list = [np.round(d, 3) for d in data_list]
    df = pd.DataFrame([[model_name] + data_list],
                      columns=['Model', 'swIC', 'swRankIC', 'swIR', 'swRankIR'])
    return df
