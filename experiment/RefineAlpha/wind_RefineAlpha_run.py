import copy
import os

import torch
import time
from torch import nn
import statsmodels.api as sm
from wind_get_data import get_wind_data, get_toy_data
from wind_module import DoubleMSELoss, DoubleRMSELoss, DoubleICLoss, evaluate

from quant_finance_research.utils import load_pickle, save_pickle, set_seed
from quant_finance_research.prediction.sample import *
from quant_finance_research.prediction.tscv import *

from wind_stable_tabnn import *

from wind_preprocess import get_tool, get_loss_label

if __name__ == "__main__":
    # input_setting
    file = 'barra'
    out_sample_start = datetime.strptime('2018-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')
    # insample, outsample, df_column, dy_col, barra_idx, wind_factor_idx = get_toy_data(
    #     out_sample_start=out_sample_start)
    insample, outsample, df_column, dy_col, barra_idx, wind_factor_idx = get_wind_data(
        out_sample_start=out_sample_start, file=file)
    print(f"Original shape: {insample.shape}")

    loss_type = 'corr_weight'
    insample, _ = get_loss_label(insample, df_column, loss_type, dy_col)
    outsample, df_column = get_loss_label(outsample, df_column, loss_type, dy_col)
    # must apply for the same shape of insample & outsample

    # roll & cv & preprocess
    roll_iter = 5
    fixed_horizon = 2
    sampler = Sample_Horizon(purge=10, horizon=int(fixed_horizon * 250))

    pro_seq = ['windCrossReg']  # ['windCrossReg', 'barraCrossReg']
    cv_type = 'normal'
    split_ratio = 0.1
    cv_k = 5 if cv_type != 'normal' else 1
    purge = 10
    fspliter, df_column, pro_pack, insample, outsample = \
        get_tool(insample, outsample, pro_seq, cv_k, df_column, barra_idx, wind_factor_idx, cv_type, split_ratio, purge)
    print(df_column)

    # stable_tabnet setting
    learning_rate = 1e-4
    casual_ratio = 0.8
    n_cluster = 3
    lambda_var = 0.2
    lambda_mask = 0.5
    loss_func = nn.MSELoss()
    n_epochs = 100
    early_stop_epochs = 10
    verbose = 1
    batch_size = 131072
    cluster_max_iter = 300
    device = 'cuda'

    model_name = 'stablenn_' + 'r' + str(roll_iter) + '_h' + str(fixed_horizon) + '_' + cv_type + '_lr4_hard'
    for seq in pro_seq:
        model_name = model_name + '_' + seq

    # basic setting
    set_seed(0)
    base_file = 'result/'
    if not os.path.exists(base_file):
        os.mkdir(base_file)
    if not os.path.exists(base_file + model_name + '/'):
        os.mkdir(base_file + model_name + '/')
    path_name = base_file + model_name + '/'

    # =============================  Rolling Training & Prediction ===========================
    roll_spliter = QuantTimeSplit_RollingPredict(k=roll_iter)
    roll_spliter.split(insample, outsample, time_col='time_id')
    pred_list = []

    for fold_idx in range(roll_spliter.get_k()):
        print(f"============================ Start Training Stage-{fold_idx} ==============================")
        presample, test_df = roll_spliter.get_folder(insample, outsample, fold_idx)
        if sampler is not None:
            presample = sampler.sample(presample)

        casual_masker = CasualMask(input_dim=len(df_column['x']), casual_ratio=casual_ratio,
                                   num_hidden=2,
                                   activate=nn.ReLU()).to(device)
        assert len(df_column['y']) == 1
        inv_predictor = BaseMLP(input_dim=len(df_column['x']), output_dim=len(df_column['y'])).to(device)

        optimizer = torch.optim.Adam(itertools.chain(casual_masker.parameters(), inv_predictor.parameters()),
                                     lr=learning_rate)

        fspliter.clear()
        fspliter.split(presample, time_col="time_id")
        train_df, val_df = fspliter.get_folder(presample, 0)

        stable_tabnet_solve(train_df.iloc[:, df_column['x']].values, train_df.iloc[:, df_column['y']].values,
                            val_df.iloc[:, df_column['x']].values, val_df.iloc[:, df_column['y']].values,
                            casual_masker=casual_masker, n_clusters=n_cluster, inv_predictor=inv_predictor,
                            lambda_var=lambda_var, lambda_mask=lambda_mask, optimizer=optimizer,
                            loss_func=loss_func,
                            n_epochs=n_epochs, early_stop_epochs=early_stop_epochs,
                            batch_size=batch_size,
                            verbose=verbose, cluster_max_iter=cluster_max_iter, device=device)

        pred = stable_tabnet_predict(test_df.iloc[:, df_column['x']].values, casual_masker=casual_masker,
                                     inv_predictor=inv_predictor,
                                     batch_size=batch_size, device='cuda')
        pred_list.append(pred)

    # ======================================  Evaluation ======================================
    pred = np.concatenate([prd.reshape(-1) for prd in pred_list])
    evaluate(outsample, df_column, dy_col, model_name, path_name, pred)
    print("Finish.")
