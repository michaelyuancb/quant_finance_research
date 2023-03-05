import copy
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import statsmodels.api as sm
from wind_get_data import get_wind_data, get_toy_data
from wind_module import DoubleMSELoss, DoubleRMSELoss, DoubleICLoss, evaluate

from quant_finance_research.utils import load_pickle, save_pickle, set_seed
from quant_finance_research.prediction.tscv import QuantTimeSplit_GroupTime, QuantTimeSplit_PreprocessSplit, \
    QuantTimeSplit_SeqPast, QuantTimeSplit_RollingPredict, QuantTimeSplit_NormalTimeCut, QuantTimeSplit_PurgeGroupTime
from quant_finance_research.fe.fe_utils import PiplinePreprocess, update_df_column_package, add_preprocess_pipeline, \
    find_column_idx
from quant_finance_research.fe.fe_orth import orth_panel_local_residual
from quant_finance_research.fe.fe_portfolio import build_pca_cov_portfolio, build_ica_cov_portfolio
from quant_finance_research.fe.fe_feat import delete_Feature

from quant_finance_research.prediction.gbdt_framework import GBDTCVEnsemble
from quant_finance_research.fe.fe_val import panel_time_zscore_normalize_preprocess
from lightgbm import LGBMRegressor

from quant_finance_research.prediction.nn.ft_transformer import FT_Transformer
from quant_finance_research.prediction.nn.base_dnn import Base_DNN
from quant_finance_research.prediction.nn_framework import NeuralNetworkCVEnsemble
from quant_finance_research.fe.fe_feat import add_GlobalAbsIcRank_LocalTimeMeanFactor


# =======================================  New Label ============================================

def get_loss_label(init_df, init_df_column, loss_type):
    df_new, df_column_new = init_df, init_df_column
    print(f"USE LOSS_TYPE={loss_type}")
    if loss_type == 'corr_weight':
        init_df_column['y'] = dy_col
        df_new, _ = panel_time_zscore_normalize_preprocess(init_df, init_df_column, init_df_column['y'],
                                                           time_col='time_id')
        df_new['new_y'] = df_new[df_new.columns[init_df_column['y']]].apply(lambda x: x[0] + x[1], axis=1)
        df_column_new = init_df_column
        df_column_new['y'] = [df_new.shape[1] - 1]
    return df_new, df_column_new


# ============================================  GBDT ============================================

def get_gbdt(model_name):
    if 'lgbm' in model_name:
        gbdt = LGBMRegressor(boosting_type='gbdt', objective='regression', n_estimators=300,
                             random_state=0, learning_rate=0.1, max_depth=7)
    else:
        raise ValueError(f"Unknown GBDT {model_name}")
    return gbdt


def solve_predict_gbdt_avg(insample, outsample, df_column, model_name, **kwargs):
    print(f"SOLVE MODEL={model_name}")
    use_fspliter = copy.deepcopy(fspliter)
    lgbm_list = [get_gbdt(model_name) for i in range(cv_k)]
    lgbm_en_1d = GBDTCVEnsemble(lgbm_list)
    df_column['y'] = dy_col[0]
    lgbm_en_1d.train(insample, df_column, use_fspliter, early_stopping_rounds=gbdt_early_stopping_rounds, verbose=500,
                     **kwargs['pro_pack'])
    pred_1d, pred_cv_list_1d = lgbm_en_1d.predict(outsample, df_column, spliter=use_fspliter)
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())

    use_fspliter = copy.deepcopy(fspliter)
    lgbm_list = [get_gbdt(model_name) for i in range(cv_k)]
    lgbm_en_10d = GBDTCVEnsemble(lgbm_list)
    df_column['y'] = dy_col[1]
    lgbm_en_10d.train(insample, df_column, use_fspliter, early_stopping_rounds=gbdt_early_stopping_rounds, verbose=100,
                      **kwargs['pro_pack'])
    pred_10d, pred_cv_list_10d = lgbm_en_10d.predict(outsample, df_column, spliter=use_fspliter)
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())

    pred = 0.5 * (pred_1d + pred_10d)
    save_pickle(path_name + 'model.pkl', (lgbm_en_1d, lgbm_en_10d))
    return pred.reshape(-1)


def solve_predict_gbdt_single(insample, outsample, df_column, model_name, **kwargs):
    print(f"SOLVE MODEL={model_name}")
    use_fspliter = copy.deepcopy(fspliter)
    lgbm_list = [get_gbdt(model_name) for i in range(cv_k)]
    lgbm_en = GBDTCVEnsemble(lgbm_list)
    print("META-COLUMN:")
    print(df_column)
    lgbm_en.train(insample, df_column, use_fspliter, early_stopping_rounds=gbdt_early_stopping_rounds, verbose=100,
                  **kwargs['pro_pack'])
    pred, pred_cv_list_1d = lgbm_en.predict(outsample, df_column, spliter=use_fspliter)
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())
    save_pickle(path_name + 'model.pkl', lgbm_en)
    return pred.reshape(-1)


# ======================================  Regression ============================================

def solve_predict_lreg(insample, outsample, df_column, model_name, **kwargs):
    print(f"SOLVE MODEL={model_name}")
    X = insample.iloc[:, df_column['x']].values
    X_out = outsample.iloc[:, df_column['x']].values
    if 'lreg_avg' in model_name:
        y_1d = insample.iloc[:, dy_col[0]].values.reshape(-1)
        y_10d = insample.iloc[:, dy_col[1]].values.reshape(-1)
        pred = 0.5 * (sm.WLS(y_1d, X).fit().predict(X_out) + sm.WLS(y_10d, X).fit().predict(X_out))
    elif 'lreg_corr_weight' in model_name:
        y_corr_weight = insample.iloc[:, df_column['y']].values.reshape(-1)
        pred = sm.WLS(y_corr_weight, X).fit().predict(X_out)
    else:
        raise ValueError(f"Unknown Regression Type = {model_name} in train_lreg()")
    return pred


# ====================================== Neural Network ============================================


def get_nn_optim_list(model_name, **kwargs):
    dnn_list, optimizer_list = [], []
    if 'ftt' in model_name:
        for i in range(cv_k):
            dnn = FT_Transformer(input_dim=len(kwargs['df_input_column']['x']), token_dim=16,
                                 index_num=[i for i in range(len(kwargs['df_input_column']['x']))], index_cat=[],
                                 total_n_cat=0,
                                 n_layers=1, n_heads=2, output_dim=1, activation=nn.GELU(), kv_compression_rate=0.25,
                                 kv_compression_sharing='layerwise')
            optim = torch.optim.Adam(dnn.parameters(), lr=1e-4)
            dnn_list.append(dnn)
            optimizer_list.append(optim)
    elif 'base_dnn' in model_name:
        for i in range(cv_k):
            dnn = Base_DNN(input_dim=len(kwargs['df_input_column']['x']),
                           hidden_dim=int(len(kwargs['df_input_column']['x']) * 1.2),
                           output_dim=1,
                           dropout_rate=0.2)
            optim = torch.optim.Adam(dnn.parameters(), lr=1e-4)
            dnn_list.append(dnn)
            optimizer_list.append(optim)
    return dnn_list, optimizer_list


def solve_predict_nn(insample, outsample, df_column, model_name, **kwargs):
    print(f"SOLVE MODEL={model_name}")
    use_fspliter = copy.deepcopy(fspliter)
    dnn_list, optimizer_list = get_nn_optim_list(model_name, **kwargs)
    dnn_en = NeuralNetworkCVEnsemble(dnn_list, optimizer_list, seed=0, device='cuda')
    df_column['y'] = dy_col
    dnn_en.train(insample, df_column, use_fspliter, nn_loss_function, use_loss_column=False,
                 early_stop=15, epoch_print=1, num_workers=num_workers, batch_size=batch_size, max_epoch=max_epoch,
                 **kwargs['pro_pack'])
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())
    dnn_en.save_best_param_file(path_name + 'model.pkl')
    pred, predict_list = dnn_en.predict(outsample, df_column, spliter=use_fspliter, batch_size=batch_size)
    return pred.reshape(-1)


# ====================================== Solver ===============================================

def solve_predict(train_df, test_df, df_column, model_name, **kwargs):
    if 'lgbm_avg' in model_name:
        return solve_predict_gbdt_avg(train_df, test_df, df_column, model_name, **kwargs)
    elif 'lgbm_corr_weight' in model_name:
        return solve_predict_gbdt_single(train_df, test_df, df_column, model_name, **kwargs)
    elif 'lreg' in model_name:
        return solve_predict_lreg(train_df, test_df, df_column, model_name, **kwargs)
    else:
        return solve_predict_nn(train_df, test_df, df_column, model_name, **kwargs)


# ====================================== BootStrap ===============================================

def fix_horizon_bootstrap(train_df, horizon, time_col='time_id'):
    time_idx = train_df[time_col].values
    time_idx = np.sort(np.unique(time_idx))[-horizon]
    train_df = train_df[train_df[time_col] > time_idx]
    return train_df


# ====================================== CV & Preprocess =====================================

def get_tscv(insample, pro_seq, cv_k, df_column_org, barra_idx, wind_factor_idx, loss_type):
    # preprocess setting.
    number_GAIR = 10
    pca_cov_n_components = 80

    # fspliter = QuantTimeSplit_PurgeGroupTime(k=cv_k, gap=1)
    fspliter = QuantTimeSplit_GroupTime(k=cv_k)

    tmp_df = insample.copy().iloc[:100, :]
    df_column = copy.deepcopy(df_column_org)
    df_column_tmp = copy.deepcopy(df_column)

    preprocess_func = PiplinePreprocess()
    pro_pack = dict()
    for pro in pro_seq:
        if pro == 'barraPanelReg':
            tmp_df, df_column_tmp, pro_pack, pfb = add_preprocess_pipeline(preprocess_func, 'val',
                                                                           orth_panel_local_residual, tmp_df,
                                                                           df_column_tmp, barra_idx, pro_pack,
                                                                           orth_column_base=wind_factor_idx)
            print("Init barraPanelReg")
        elif pro == "gair":
            tmp_df, df_column_tmp, pro_pack, pfb = add_preprocess_pipeline(preprocess_func, 'feat',
                                                                           add_GlobalAbsIcRank_LocalTimeMeanFactor,
                                                                           tmp_df, df_column_tmp, df_column['x'],
                                                                           pro_pack,
                                                                           number_GAIR=number_GAIR)
            print("Init gair")
        elif pro == 'pcaCovWind':
            barra_name = tmp_df.columns[barra_idx]
            tmp_df, df_column_tmp, pro_pack, pfb = add_preprocess_pipeline(preprocess_func, 'feat',
                                                                           build_pca_cov_portfolio, tmp_df,
                                                                           df_column_tmp, wind_factor_idx, pro_pack,
                                                                           pca_cov_n_components=pca_cov_n_components,
                                                                           pca_cov_portfolio_inplace=True)
            wind_factor_idx = pfb['new_column_idx']
            barra_idx = find_column_idx(barra_name, tmp_df.columns)
            print(f"Init PCA & Delete Org; num_barra={len(barra_idx)}, num_wind={len(wind_factor_idx)}, "
                  f"num_factor={len(df_column_tmp['x'])}")
    print(f"pro_df.shape={tmp_df.shape}")

    if preprocess_func.func_num() > 0:
        fspliter = QuantTimeSplit_PreprocessSplit(fspliter, df_column=df_column,
                                                  preprocess_column=[], preprocess_func=preprocess_func)
    process_pack = {"preprocess_func": preprocess_func, "cv_k": cv_k, "fspliter": fspliter,
                    "df_input_column": df_column_tmp,
                    "pro_pack": pro_pack}
    return fspliter, process_pack


# ====================================== User =================================================

if __name__ == "__main__":
    # input_setting
    file = 'barra'
    out_sample_start = datetime.strptime('2018-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')
    # insample, outsample, df_column, dy_col, barra_idx, wind_factor_idx = get_toy_data(
    #     out_sample_start=out_sample_start)
    insample, outsample, df_column, dy_col, barra_idx, wind_factor_idx = get_wind_data(
        out_sample_start=out_sample_start, file=file)

    # insample = insample.iloc[:60000, :]
    # outsample = outsample.iloc[:10000, :]

    print(f"Original shape: {insample.shape}")
    model_name = 'lgbm_corr_weight'  # base_dnn, ftt, lgbm_avg,  lgbm_corr_weight, lreg_avg, lreg_corr_weight

    if model_name in ['lgbm_corr_weight', 'lreg_corr_weight']:
        loss_type = 'corr_weight'
    elif model_name in ['lgbm_avg', 'lreg_avg']:
        loss_type = 'avg'
    else:
        lose_type = 'nn_loss'
    insample, _ = get_loss_label(insample, df_column, loss_type)
    outsample, df_column = get_loss_label(outsample, df_column, loss_type)

    # nn_setting
    nn_early_stopping_rounds = 10
    max_epoch = 200
    batch_size = 32768
    num_workers = 0
    nn_loss_function = DoubleRMSELoss()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # gbdt_setting
    gbdt_early_stopping_rounds = 30

    config_param = {"nn_early_stopping_rounds": nn_early_stopping_rounds,
                    "max_epoch": max_epoch,
                    "nn_loss_function": nn_loss_function, "gbdt_early_stopping_rounds": gbdt_early_stopping_rounds,
                    "loss_type": loss_type, "batch_size": batch_size, "num_workers": num_workers}

    # cv & preprocess
    pro_seq = ['pcaCovWind', 'barraPanelReg']  # ['pcaCovWind', 'barraPanelReg']
    pro_name = ''
    for i, pro in enumerate(pro_seq):
        if i > 0:
            pro_name = pro_name + '_' + pro
        else:
            pro_name = pro
    model_name = model_name + '_' + pro_name
    cv_k = 8
    fspliter, process_pack = get_tscv(insample, pro_seq, cv_k, df_column, barra_idx, wind_factor_idx, loss_type)
    config_param.update(process_pack)

    # basic setting
    set_seed(0)
    base_file = 'result/' + file + '/'
    if not os.path.exists(base_file):
        os.mkdir(base_file)
    if not os.path.exists(base_file + model_name + '/'):
        os.mkdir(base_file + model_name + '/')
    path_name = base_file + model_name + '/'
    config_param['path_name'] = path_name

    # =============================  Rolling Training & Prediction ===========================
    # roll_iter = 2
    roll_iter = 20 if 'lgbm' in model_name else 40 if 'lreg' in model_name else 5  # 20 for gbdt, 5 for nn, 40 for lreg.
    roll_spliter = QuantTimeSplit_RollingPredict(k=roll_iter)
    print(insample.shape)
    print(outsample.shape)
    roll_spliter.split(insample, outsample, time_col='time_id')
    pred_list = []

    for fold_idx in range(roll_spliter.get_k()):
        print(f"============================ Start Training Stage-{fold_idx} ==============================")
        outsample_train_df, test_df = roll_spliter.get_folder(insample, outsample, fold_idx)
        presample = pd.concat([insample, outsample_train_df], axis=0)
        presample = fix_horizon_bootstrap(presample, horizon=int(2 * 250))
        pred = solve_predict(presample, test_df, df_column, model_name, **config_param)
        pred_list.append(pred)

    # ======================================  Evaluation ======================================
    pred = np.concatenate([prd.reshape(-1) for prd in pred_list])
    evaluate(outsample, df_column, dy_col, model_name, path_name, pred)
    print("Finish.")
