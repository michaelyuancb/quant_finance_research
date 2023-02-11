import copy
import itertools

from tqdm import tqdm
import numpy as np
import abc
from lightgbm import LGBMRegressor

from quant_finance_research.tools.util import *
from quant_finance_research.tools.factor_eval import evaluate_rmse


class LightGBMEnsembleBase(abc.ABC):

    def __init__(self, k, lgbm):
        self.k = k
        self.lgbm_list = []
        for i in range(k):
            self.lgbm_list.append(copy.deepcopy(lgbm))

    def get_k(self):
        return self.k

    def predict(self, xtest):
        pred_list = []
        for i in range(self.k):
            pred = self.lgbm_list[i].predict(xtest)
            pred_list.append(pred)
        pred_list = np.array(pred_list)
        pred = np.mean(pred_list, axis=0)
        return pred, pred_list  # np.float, np.ndarray

    def get_ensemble_model(self):
        return self.lgbm_list

    def set_ensemble_model(self, lgbm_list):
        self.lgbm_list = lgbm_list

    def save_ensemble_model_file(self, filename):
        save_pickle(filename, self.get_ensemble_model())

    def load_ensemble_model_file(self, filename):
        en_model = load_pickle(filename)
        self.set_ensemble_model(en_model)


class LightGBMCVEnsemble(LightGBMEnsembleBase):

    def __init__(self, k, lgbm):
        super(LightGBMCVEnsemble, self).__init__(k, lgbm)

    def train(self, data_df, spliter, x_column, y_column,
              inv_col='investment_id',
              time_col='time_id',
              need_split=True,
              early_stopping_rounds=20,
              verbose=0):
        assert self.k == spliter.get_k()
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        for i in tqdm(range(spliter.get_k())):
            xtr, ytr, xvl, yvl = spliter.get_folder(data_df, x_column, y_column, i)
            self.lgbm_list[i].fit(xtr, ytr, eval_set=[(xvl, yvl)], early_stopping_rounds=early_stopping_rounds,
                                  verbose=verbose)


class LightGBMAvgBaggingEnsemble(LightGBMEnsembleBase):

    def __init__(self, k, lgbm, seed_list):
        super(LightGBMAvgBaggingEnsemble, self).__init__(k, lgbm)
        self.seed_list = seed_list
        for i in range(k):
            self.lgbm_list[i].random_state = seed_list[i]

    def train(self, xtrain, ytrain, xtest, ytest,
              early_stopping_rounds=20,
              verbose=0):
        for i in tqdm(range(self.k)):
            self.lgbm_list[i].fit(xtrain, ytrain, eval_set=[(xtest, ytest)], verbose=verbose,
                                  early_stopping_rounds=early_stopping_rounds)


class LightGBMCV:

    def __init__(self, k, lgbm):
        self.k = k
        self.lgbm = lgbm
        self.lgbm_cv_list = []

    def cv(self, data_df, spliter, x_column, y_column,
           evaluate_func=evaluate_rmse,
           inv_col='investment_id',
           time_col='time_id',
           need_split=True,
           early_stopping_rounds=20,
           verbose=0):
        assert spliter.get_k() == self.k
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        folder_err_list = []
        for i in range(spliter.get_k()):
            print(f"Start Traning Model{i}")
            lgbm_base = copy.deepcopy(self.lgbm)
            xtr, ytr, xvl, yvl = spliter.get_folder(data_df, x_column, y_column, i)
            lgbm_base.fit(xtr, ytr, eval_set=[(xvl, yvl)], verbose=verbose,
                          early_stopping_rounds=early_stopping_rounds)
            pred_vl = lgbm_base.predict(xvl)
            rmse = evaluate_func(pred_vl.reshape(-1), yvl.reshape(-1))
            folder_err_list.append(rmse)
            self.lgbm_cv_list.append(lgbm_base)
        folder_err = np.array(folder_err_list)
        return np.mean(folder_err), folder_err

    def get_cv_model(self):
        return self.lgbm_cv_list

    def save_cv_model_file(self, filename):
        save_pickle(filename, self.get_cv_model())

    def load_param_file(self, filename):
        lgbm_cv = load_pickle(filename)
        self.lgbm_cv_list = lgbm_cv


class LightGBMGridCVBase:

    def __init__(self, k, param_dict):
        self.k = k
        self.param_dict = param_dict
        self.cv_model_list = []

    def cv(self, data_df, spliter, x_column, y_column,
           evaluate_func=evaluate_rmse,
           inv_col='investment_id',
           time_col='time_id',
           need_split=True,
           early_stopping_rounds=20,
           verbose=0):
        assert spliter.get_k() == self.k
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        keys, values = zip(*self.param_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        cv_result = []
        for param in tqdm(permutations_dicts):
            lgbm_base = self.get_lgbm(param)
            cv_lgbm = LightGBMCV(self.k, lgbm_base)
            err_mean, err_list = cv_lgbm.cv(data_df, spliter, x_column, y_column, evaluate_func,
                                            inv_col=inv_col,
                                            time_col=time_col,
                                            need_split=False,
                                            early_stopping_rounds=early_stopping_rounds,
                                            verbose=verbose)
            cv_result.append((err_mean, err_list))
            self.cv_model_list.append(cv_lgbm)
        return cv_result, permutations_dicts

    def get_cv_model(self, idx):
        return self.cv_model_list[idx]

    def get_cv_model_list(self, idx):
        return self.cv_model_list

    def save_cv_model_list_file(self, filename):
        save_pickle(filename, self.get_cv_model_list())

    def get_lgbm(self, param):
        """
        User should write their own function to get dnn_list & optimizer_list from param
        with "from lightgbm import * (LightGBM sklearn API)"
        """
        lgbm = None
        return lgbm


class LightGBMGridCVBase_Example(LightGBMGridCVBase):

    def __init__(self, k, param_dict):
        super(LightGBMGridCVBase_Example, self).__init__(k, param_dict)

    def get_lgbm(self, param):
        lgbm = LGBMRegressor(
            boosting_type='gbdt',
            objective='regression',
            silent=True,
            learning_rate=param['learning_rate'],
            max_depth=param['max_depth']
        )
        return lgbm
