import copy
from datetime import datetime
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
import torch
import statsmodels.api as sm
from wind_get_data import get_toy_data, get_wind_data
from wind_module import DoubleMSELoss, DoubleICLoss, evaluate

from quant_finance_research.utils import load_pickle, save_pickle, set_seed
from quant_finance_research.prediction.tscv import QuantTimeSplit_GroupTime, QuantTimeSplit_PreprocessSplit, \
    QuantTimeSplit_SeqPast, QuantTimeSplit_RollingPredict

from quant_finance_research.prediction.gbdt_framework import GBDTCVEnsemble
from quant_finance_research.fe.fe_val import panel_time_zscore_normalize_preprocess
from lightgbm import LGBMRegressor

from quant_finance_research.prediction.nn.ft_transformer import FT_Transformer
from quant_finance_research.prediction.nn_framework import NeuralNetworkCVEnsemble
from quant_finance_research.fe.fe_feat import add_GlobalAbsIcRank_LocalTimeMeanFactor


# =======================================  New Label ============================================

def get_loss_label(init_df, init_df_column):
    df_new, df_column_new = None, None
    if gbdt_loss_method == 'corr_weight':
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
        gbdt = LGBMRegressor(boosting_type='gbdt', objective='regression', n_estimators=2000,
                             random_state=0, learning_rate=0.01, max_depth=7)
    else:
        raise ValueError(f"Unknown GBDT {model_name}")
    return gbdt


def solve_predict_gbdt_avg(insample, outsample, df_column, model_name, **kwargs):
    use_fspliter = copy.deepcopy(fspliter)
    lgbm_list = [get_gbdt(model_name) for i in range(cv_k)]
    lgbm_en_1d = GBDTCVEnsemble(lgbm_list)
    df_column['y'] = dy_col[0]
    lgbm_en_1d.train(insample, df_column, use_fspliter, early_stopping_rounds=gbdt_early_stopping_rounds, verbose=500)
    pred_1d, pred_cv_list_1d = lgbm_en_1d.predict(outsample, df_column, spliter=use_fspliter)
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())

    use_fspliter = copy.deepcopy(fspliter)
    lgbm_list = [get_gbdt(model_name) for i in range(cv_k)]
    lgbm_en_10d = GBDTCVEnsemble(lgbm_list)
    df_column['y'] = dy_col[1]
    lgbm_en_10d.train(insample, df_column, use_fspliter, early_stopping_rounds=gbdt_early_stopping_rounds, verbose=500)
    pred_10d, pred_cv_list_10d = lgbm_en_10d.predict(outsample, df_column, spliter=use_fspliter)
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())

    pred = 0.5 * (pred_1d + pred_10d)
    save_pickle(path_name + 'model.pkl', (lgbm_en_1d, lgbm_en_10d))
    return pred.reshape(-1)


def solve_predict_gbdt_single(insample, outsample, df_column, model_name, **kwargs):
    use_fspliter = copy.deepcopy(fspliter)
    lgbm_list = [get_gbdt(model_name) for i in range(cv_k)]
    lgbm_en = GBDTCVEnsemble(lgbm_list)
    pro_insample, pro_column = get_loss_label(insample, df_column)
    lgbm_en.train(pro_insample, pro_column, use_fspliter, early_stopping_rounds=gbdt_early_stopping_rounds, verbose=500)
    pred, pred_cv_list_1d = lgbm_en.predict(outsample, df_column, spliter=use_fspliter)
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())
    save_pickle(path_name + 'model.pkl', lgbm_en)
    return pred.reshape(-1)


# ======================================  Regression ============================================

def solve_predict_lreg(insample, outsample, df_column, model_name, **kwargs):
    X = insample.iloc[:, df_column['x']].values
    X_out = outsample.iloc[:, df_column['x']].values
    if model_name == 'lreg_avg':
        y_1d = insample.iloc[:, dy_col[0]].values.reshape(-1)
        y_10d = insample.iloc[:, dy_col[1]].values.reshape(-1)
        pred = 0.5 * (sm.WLS(y_1d, X).fit().predict(X_out) + sm.WLS(y_10d, X).fit().predict(X_out))
    elif model_name == 'lreg_corr_weight':
        pro_insample, pro_column = get_loss_label(insample, df_column)
        y_corr_weight = pro_insample.iloc[:, pro_column['y']].values.reshape(-1)
        pred = sm.WLS(y_corr_weight, X).fit().predict(X_out)
    else:
        raise ValueError(f"Unknown Regression Type = {model_name} in train_lreg()")
    return pred


# ====================================== Neural Network ============================================

def get_nn_optim_list(model_name, **kwargs):
    dnn_list, optimizer_list = [], []
    df_tmp_column = copy.deepcopy(df_column)
    if 'gair' in model_name:
        df_tmp_column['y'] = dy_col
        df_tmp, df_tmp_column = add_GlobalAbsIcRank_LocalTimeMeanFactor(insample.iloc[:10, :], df_tmp_column,
                                                                        df_column['x'], **kwargs)
        df_tmp_column = df_tmp_column['df_column_new']
    if 'ftt' in model_name:
        for i in range(cv_k):
            dnn = FT_Transformer(input_dim=len(df_tmp_column['x']), token_dim=16,
                                 index_num=[i for i in range(len(df_tmp_column['x']))], index_cat=[], total_n_cat=0,
                                 n_layers=2, n_heads=4, output_dim=1,
                                 )
            optim = torch.optim.Adam(dnn.parameters(), lr=5e-5)
            dnn_list.append(dnn)
            optimizer_list.append(optim)
    return dnn_list, optimizer_list


def solve_predict_nn(insample, outsample, df_column, model_name, **kwargs):
    use_fspliter = copy.deepcopy(fspliter)
    dnn_list, optimizer_list = get_nn_optim_list(model_name)
    dnn_en = NeuralNetworkCVEnsemble(dnn_list, optimizer_list, seed=0, device='cuda')
    df_column['y'] = dy_col
    dnn_en.train(insample, df_column, fspliter, nn_loss_function, use_loss_column=False,
                 early_stop=15, epoch_print=1, num_workers=num_workers, batch_size=batch_size, max_epoch=max_epoch,
                 number_GAIR=number_GAIR)
    if hasattr(use_fspliter, "get_preprocess_package"):
        save_pickle(path_name + 'pro_pack.pkl', use_fspliter.get_preprocess_package())
    dnn_en.save_best_param_file(path_name + 'model.pkl')
    pred, predict_list = dnn_en.predict(outsample, df_column, spliter=fspliter, batch_size=batch_size)
    return pred.reshape(-1)


# ====================================== Solver ===============================================

def solve_predict(train_df, test_df, df_column, model_name, **kwargs):
    if model_name in ['lgbm_avg']:
        return solve_predict_gbdt_avg(train_df, test_df, df_column, model_name, **kwargs)
    elif model_name in ['lgbm_corr_weight']:
        return solve_predict_gbdt_single(train_df, test_df, df_column, model_name, **kwargs)
    elif model_name in ['lreg_avg', 'lreg_corr_weight']:
        return solve_predict_lreg(train_df, test_df, df_column, model_name, **kwargs)
    elif model_name in ['ftt', 'ftt_gair']:
        return solve_predict_nn(train_df, test_df, df_column, model_name, **kwargs)


# ====================================== BootStrap ===============================================

def fix_horizon_bootstrap(train_df, horizon, time_col='time_id'):
    time_idx = train_df[time_col].values
    time_idx = np.sort(np.unique(time_idx))[-horizon]
    train_df = train_df[train_df[time_col] > time_idx]
    return train_df


# ====================================== User =================================================

if __name__ == "__main__":
    model_name = 'ftt_gair'  # ftt, ftt_gair, lgbm_avg, lgbm_corr_weight, lreg_avg, lreg_corr_weight

    # tscv setting
    cv_k = 5
    fspliter = QuantTimeSplit_GroupTime(k=cv_k)

    # nn_setting
    nn_early_stopping_rounds = 50
    max_epoch = 200
    use_GAIR = False
    number_GAIR = 25
    batch_size = 4096
    num_workers = 1
    nn_loss_function = DoubleMSELoss()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # gbdt_setting
    gbdt_early_stopping_rounds = 25
    gbdt_loss_method = 'corr_weight'

    config_param = {"cv_k": cv_k, "fspliter": fspliter, "nn_early_stopping_rounds": nn_early_stopping_rounds,
                    "max_epoch": max_epoch, "number_GAIR": number_GAIR, "use_GAIR": use_GAIR,
                    "nn_loss_function": nn_loss_function, "gbdt_early_stopping_rounds": gbdt_early_stopping_rounds,
                    "gbdt_loss_method": gbdt_loss_method, "batch_size": batch_size, "num_workers": num_workers}

    # input_setting
    out_sample_start = datetime.strptime('2018-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')
    # insample, outsample, df_column, dy_col, barra_idx, wind_factor_idx = get_toy_data(out_sample_start=out_sample_start)
    insample, outsample, df_column, dy_col, barra_idx, wind_factor_idx = get_wind_data(out_sample_start=out_sample_start)

    # basic setting
    set_seed(0)
    if not os.path.exists('result/' + model_name + '/'):
        os.mkdir('result/' + model_name + '/')
    path_name = 'result/' + model_name + '/'
    config_param['path_name'] = path_name

    # =============================  Rolling Training & Prediction ===========================
    roll_iter = 40
    roll_spliter = QuantTimeSplit_RollingPredict(k=roll_iter)
    roll_spliter.split(insample, outsample, time_col='time_id')
    pred_list = []

    for fold_idx in range(roll_spliter.get_k()):
        print("============================ Start Training Stage-{fold_idx} ==============================")
        outsample_train_df, test_df = roll_spliter.get_folder(insample, outsample, fold_idx)
        presample = pd.concat([insample, outsample_train_df], axis=0)

        presample = fix_horizon_bootstrap(presample, horizon=int(250*2))

        pred = solve_predict(presample, test_df, df_column, model_name, **config_param)
        pred_list.append(pred)

    # ======================================  Evaluation ======================================
    pred = np.concatenate([prd.reshape(-1) for prd in pred_list])
    evaluate(outsample, df_column, dy_col, model_name, path_name, pred)
    print("Finish.")
