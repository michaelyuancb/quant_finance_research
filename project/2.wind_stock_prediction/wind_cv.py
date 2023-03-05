import os
from datetime import datetime
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import torch
from wind_get_data import get_wind_data

from quant_finance_research.prediction.gbdt_framework import GBDTGridCVBase
from quant_finance_research.prediction.nn_framework import NeuralNetworkGridCVBase
from quant_finance_research.prediction.tscv import QuantTimeSplit_GroupTime
from quant_finance_research.prediction.nn.ft_transformer import FT_Transformer
from wind_module import DoubleLoss

out_sample_start = datetime.strptime('2018-12-31 23:00:00', '%Y-%m-%d %H:%M:%S')
insample, outsample, df_column, dy_col = get_wind_data(out_sample_start=out_sample_start)


class GBDTGridCV_LightGBM(GBDTGridCVBase):

    def __init__(self, k, param_dict):
        super(GBDTGridCV_LightGBM, self).__init__(k, param_dict)

    def get_gbdt(self, param):
        lgbm = LGBMRegressor(
            boosting_type='gbdt',
            objective='regression',
            n_estimators=1000,
            random_state=0,
            subsample=param['subsample'],
            learning_rate=param['learning_rate'],
            max_depth=param['max_depth'],
        )
        return lgbm


class GBDTGridCV_CatBoost(GBDTGridCVBase):

    def __init__(self, k, param_dict):
        super(GBDTGridCV_CatBoost, self).__init__(k, param_dict)

    def get_gbdt(self, param):
        catb = CatBoostRegressor(
            loss_function='RMSE',
            iterations=1000,
            random_seed=0,
            learning_rate=param['learning_rate'],
            depth=param['max_depth'],
            l2_leaf_reg=param['l2_leaf_reg']
        )
        return catb


class NeuralNetworkGridCV_FTT(NeuralNetworkGridCVBase):

    def __init__(self, k, param_dict, device='cuda'):
        super(NeuralNetworkGridCV_FTT, self).__init__(k, param_dict, device)

    def get_model_list(self, param):
        dnn_list = []
        optimizer_list = []
        for i in range(self.k):
            dnn = FT_Transformer(input_dim=len(df_column['x']), token_dim=param['token_dim'],
                                 index_num=[i for i in range(len(df_column['x']))], index_cat=[], total_n_cat=0,
                                 n_layers=param['n_layer'], n_heads=4, output_dim=2,
                                 )
            optim = torch.optim.Adam(dnn.parameters(), lr=param['learning_rate'])
            dnn_list.append(dnn)
            optimizer_list.append(optim)
        assert self.k == len(dnn_list)
        return dnn_list, optimizer_list


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# model_name = 'ftt'
# if not os.path.exists('result/' + model_name + '/'):
#     os.mkdir('result/' + model_name + '/')
# param_dict_fft = {'learning_rate': [5e-4, 5e-5], 'token_dim': [16], 'n_layer': [2]}
# grid_cv = NeuralNetworkGridCV_FTT(k=5, param_dict=param_dict_fft, device='cuda')
# cv_spliter = QuantTimeSplit_GroupTime(k=5)
# df_column['y'] = dy_col
# result_grid_cv = grid_cv.cv(insample, df_column, cv_spliter, loss_func=DoubleLoss(), early_stop=10,
#                             batch_size=4096, num_workers=1, epoch_print=1, max_epoch=50)
# result_grid_cv.to_csv('result/' + model_name + '/cv_result_dic.csv')
# del grid_cv


model_name = 'lgbm'
if not os.path.exists('result/' + model_name + '/'):
    os.mkdir('result/' + model_name + '/')
param_dict_lgbm = {'learning_rate': [0.1, 0.01], 'max_depth': [7, 15, 25], 'subsample': [0.8, 1.0]}
grid_cv = GBDTGridCV_LightGBM(k=5, param_dict=param_dict_lgbm)
cv_spliter = QuantTimeSplit_GroupTime(k=5)
df_column['y'] = [dy_col[0]]
result_grid_cv = grid_cv.cv(insample, df_column, cv_spliter, early_stopping_rounds=50, verbose=50)
result_grid_cv.to_csv('result/' + model_name + '/cv_result_1d.csv')
del grid_cv
grid_cv = GBDTGridCV_LightGBM(k=5, param_dict=param_dict_lgbm)
df_column['y'] = [dy_col[1]]
result_grid_cv = grid_cv.cv(insample, df_column, cv_spliter, early_stopping_rounds=50, verbose=50, need_split=False)
result_grid_cv.to_csv('result/' + model_name + '/cv_result_10d.csv')
del grid_cv


model_name = 'catb'
if not os.path.exists('result/' + model_name + '/'):
    os.mkdir('result/' + model_name + '/')
param_dict_lgbm = {'learning_rate': [0.1, 0.01], 'max_depth': [7, 15, 25], 'l2_leaf_reg': [1.0, 3.0]}
grid_cv = GBDTGridCV_CatBoost(k=5, param_dict=param_dict_lgbm)
cv_spliter = QuantTimeSplit_GroupTime(k=5)
df_column['y'] = [dy_col[0]]
result_grid_cv = grid_cv.cv(insample, df_column, cv_spliter, early_stopping_rounds=50, verbose=50)
result_grid_cv.to_csv('result/' + model_name + '/cv_result_1d.csv')
del grid_cv
grid_cv = GBDTGridCV_CatBoost(k=5, param_dict=param_dict_lgbm)
df_column['y'] = [dy_col[1]]
result_grid_cv = grid_cv.cv(insample, df_column, cv_spliter, early_stopping_rounds=50, verbose=50, need_split=False)
result_grid_cv.to_csv('result/' + model_name + '/cv_result_10d.csv')
del grid_cv

