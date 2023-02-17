import copy
import itertools

from tqdm import tqdm
import numpy as np
import abc
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation

from quant_finance_research.utils import *
from quant_finance_research.eval.factor_eval import evaluate_RMSE
from quant_finance_research.fe.fe_utils import update_df_column_package


class LightGBMWrapper:
    """
        Wrap the lightGBM framework for convenient usage.
    """

    def __init__(self, lgbm, seed=0):
        self.lgbm = lgbm
        self.lgbm.random_state = seed

    def train(self, train_df, val_df, df_column,
              inv_col='investment_id', time_col='time_id',
              early_stopping_rounds=100,
              verbose=0,
              **kwargs):
        xtrain, ytrain, xval, yval = get_numpy_from_df_train_val(train_df, val_df, df_column)
        ytrain = ytrain.reshape(-1)
        yval = yval.reshape(-1)
        callbacks = [log_evaluation(period=verbose), early_stopping(stopping_rounds=early_stopping_rounds)]
        self.lgbm.fit(xtrain, ytrain, eval_set=[(xval, yval)], callbacks=callbacks)

    def predict(self, test_df, df_column, inv_col='investment_id', time_col='time_id',):
        xtest = test_df.iloc[:, df_column['x']].values
        pred = self.lgbm.predict(xtest)
        return pred

    def get_model(self):
        return self.lgbm

    def load_model(self, filename):
        self.lgbm = load_pickle(filename)

    def save_model(self, filename):
        save_pickle(filename, self.lgbm)


class LightGBMEnsembleBase(abc.ABC):

    def __init__(self, lgbm_list):
        self.k = len(lgbm_list)
        self.lgbm_list = [LightGBMWrapper(lgbm_r) for lgbm_r in lgbm_list]

    def get_k(self):
        return self.k

    def predict(self, test_df, df_column, inv_col='investment_id', time_col='time_id'):
        pred_list = []
        for i in range(self.k):
            pred = self.lgbm_list[i].predict(test_df, df_column, inv_col=inv_col, time_col=time_col)
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

    def __init__(self, lgbm_list):
        super(LightGBMCVEnsemble, self).__init__(lgbm_list)

    def train(self, data_df, df_column, spliter,
              inv_col='investment_id', time_col='time_id',
              need_split=True,
              early_stopping_rounds=20,
              verbose=0,
              **kwargs):
        assert self.k == spliter.get_k()
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        for i in tqdm(range(spliter.get_k())):
            train_df, val_df = spliter.get_folder(data_df, i, **kwargs)
            if hasattr(spliter, "get_folder_preprocess_package"):
                pro_package = spliter.get_folder_preprocess_package(i)
                df_column = update_df_column_package(df_column, pro_package)
            self.lgbm_list[i].train(train_df, val_df, df_column,
                                    time_col=time_col, inv_col=inv_col,
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose=verbose, **kwargs)

    def predict(self, test_df, df_column, inv_col='investment_id', time_col='time_id',
                spliter=None):
        pred_list = []
        for i in range(self.k):
            if spliter is not None:
                xtest_df, _ = spliter.get_folder_preprocess(test_df, i)
                if hasattr(spliter, "get_folder_preprocess_package"):
                    pro_package = spliter.get_folder_preprocess_package(i)
                    df_column = update_df_column_package(df_column, pro_package)
            else:
                xtest_df = test_df
            pred = self.lgbm_list[i].predict(xtest_df, df_column)
            pred_list.append(pred)
        pred_list = np.array(pred_list)
        pred = np.mean(pred_list, axis=0)
        return pred, pred_list  # np.float, np.ndarray


class LightGBMAvgBaggingEnsemble(LightGBMEnsembleBase):

    def __init__(self, lgbm_list, seed_list):
        super(LightGBMAvgBaggingEnsemble, self).__init__(lgbm_list)
        self.seed_list = seed_list
        for i in range(self.k):
            self.lgbm_list[i].lgbm.seed = seed_list[i]

    def train(self, train_df, val_df, df_column,
              early_stopping_rounds=20,
              verbose=0,
              **kwargs):
        for i in tqdm(range(self.k)):
            self.lgbm_list[i].train(train_df, val_df, df_column, verbose=verbose,
                                    early_stopping_rounds=early_stopping_rounds)


class LightGBMCV:

    def __init__(self, k, lgbm):
        self.k = k
        self.lgbm = LightGBMWrapper(lgbm)
        self.lgbm_cv_list = []

    def cv(self, data_df, df_column, spliter, evaluate_func=evaluate_RMSE,
           inv_col='investment_id',
           time_col='time_id',
           need_split=True,
           early_stopping_rounds=20,
           verbose=0,
           **kwargs):
        assert spliter.get_k() == self.k
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        folder_err_list = []
        for i in range(spliter.get_k()):
            print(f"Start Traning Model{i}")
            lgbm_base = copy.deepcopy(self.lgbm)
            train_df, val_df = spliter.get_folder(data_df, i, **kwargs)
            if hasattr(spliter, "get_folder_preprocess_package"):
                pro_package = spliter.get_folder_preprocess_package(i)
                df_column = update_df_column_package(df_column, pro_package)

            lgbm_base.train(train_df, val_df, df_column, verbose=verbose, time_col=time_col, inv_col=inv_col,
                            early_stopping_rounds=early_stopping_rounds)
            pred_vl = lgbm_base.predict(val_df, df_column)
            verr = evaluate_func(pred_vl.reshape(-1), val_df.iloc[:, df_column['y']].values.reshape(-1))
            folder_err_list.append(verr)
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

    def cv(self, data_df, df_column, spliter, evaluate_func=evaluate_RMSE,
           inv_col='investment_id', time_col='time_id',
           need_split=True,
           early_stopping_rounds=20,
           verbose=0,
           **kwargs):
        assert spliter.get_k() == self.k
        if need_split:
            spliter.split(data_df, inv_col=inv_col, time_col=time_col)
        keys, values = zip(*self.param_dict.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        cv_result = []
        for param in tqdm(permutations_dicts):
            lgbm_base = self.get_lgbm(param)
            cv_lgbm = LightGBMCV(self.k, lgbm_base)
            err_mean, err_list = cv_lgbm.cv(data_df, df_column, spliter, evaluate_func,
                                            inv_col=inv_col, time_col=time_col,
                                            need_split=False,
                                            early_stopping_rounds=early_stopping_rounds,
                                            verbose=verbose,
                                            **kwargs)
            cv_result.append((err_mean, err_list))
            self.cv_model_list.append(cv_lgbm)
        return generate_cv_result_df(cv_result, permutations_dicts)

    def get_cv_model(self, idx):
        return self.cv_model_list[idx]

    def get_cv_model_list(self):
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
            n_estimators=100,
            learning_rate=param['learning_rate'],
            max_depth=param['max_depth']
        )
        return lgbm
