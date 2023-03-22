import copy
import itertools

from tqdm import tqdm
import numpy as np
import abc
from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation

from lightgbm import LGBMModel
from xgboost import XGBModel
from catboost import CatBoost
from quant_finance_research.utils import *
from quant_finance_research.prediction.prediction_utils import *
from quant_finance_research.eval.factor_eval import evaluate_RMSE
from quant_finance_research.fe.fe_utils import update_df_column_package


class GBDTWrapper:
    """
        Wrap the lightGBM framework for convenient usage.
    """

    def __init__(self, gbdt, seed=0):
        self.gbdt = gbdt
        self.gbdt.random_state = seed
        self.n_feature = None

    def train(self, train_df, val_df, df_column,
              inv_col='investment_id', time_col='time_id',
              early_stopping_rounds=100,
              verbose=0,
              **kwargs):
        xtrain, ytrain, xval, yval = get_numpy_from_df_train_val(train_df, val_df, df_column)
        if (xtrain.shape[1] != xval.shape[1]) or (ytrain.shape[1] != 1 and ytrain.ndim == 2) or \
                (yval.shape[1] != 1 and yval.ndim == 2) or (xtrain.shape[0] != ytrain.shape[0]) or \
                (xval.shape[0] != yval.shape[0]):
            raise ValueError(f"The shape of data is illegal ; xtrain.shape={xtrain.shape}, ytrain.shape={ytrain.shape},"
                             f"xval.shape={xval.shape}, yval.shape={yval.shape}")
        self.n_feature = xtrain.shape[1:]
        ytrain = ytrain.reshape(-1)
        yval = yval.reshape(-1)
        if isinstance(self.gbdt, LGBMModel):
            callbacks = [log_evaluation(period=verbose), early_stopping(stopping_rounds=early_stopping_rounds)]
            self.gbdt.fit(xtrain, ytrain, eval_set=[(xval, yval)], callbacks=callbacks)
        elif isinstance(self.gbdt, CatBoost):
            self.gbdt.set_params(allow_writing_files=False)
            self.gbdt.fit(xtrain, ytrain, eval_set=[(xval, yval)], verbose=verbose,
                          early_stopping_rounds=early_stopping_rounds)
        elif isinstance(self.gbdt, XGBModel):
            self.gbdt.set_params(early_stopping_rounds=early_stopping_rounds)
            self.gbdt.fit(xtrain, ytrain, eval_set=[(xval, yval)],
                          verbose=verbose, sample_weight=None)
        else:
            raise AttributeError(f"Unsupported GDBT Type: {type(self.gbdt)} in Training Step.")

    def predict(self, test_df, df_column, inv_col='investment_id', time_col='time_id', ):
        xtest = test_df.iloc[:, df_column['x']].values
        if (self.n_feature is not None) and (xtest.shape[1:] != self.n_feature):
            raise ValueError(f"The Train Feature Dim={self.n_feature}, But the Test Feature Dim={xtest.shape[1:]}. "
                             f"They should be the same.")
        pred = self.gbdt.predict(xtest)
        return pred

    def get_model(self):
        return self.gbdt

    def load_model(self, filename):
        self.gbdt = load_pickle(filename)

    def save_model(self, filename):
        save_pickle(filename, self.gbdt)


class GBDTEnsembleBase(abc.ABC):

    def __init__(self, gbdt_list):
        self.k = len(gbdt_list)
        self.gbdt_list = [GBDTWrapper(gbdt_r) for gbdt_r in gbdt_list]

    def get_k(self):
        return self.k

    def predict(self, test_df, df_column, inv_col='investment_id', time_col='time_id'):
        pred_list = []
        for i in range(self.k):
            pred = self.gbdt_list[i].predict(test_df, df_column, inv_col=inv_col, time_col=time_col)
            pred_list.append(pred)
        pred_list = np.array(pred_list)
        pred = np.mean(pred_list, axis=0)
        return pred, pred_list  # np.float, np.ndarray

    def get_ensemble_model(self):
        return self.gbdt_list

    def set_ensemble_model(self, gbdt_list):
        self.gbdt_list = gbdt_list

    def save_ensemble_model_file(self, filename):
        save_pickle(filename, self.get_ensemble_model())

    def load_ensemble_model_file(self, filename):
        en_model = load_pickle(filename)
        self.set_ensemble_model(en_model)


class GBDTCVEnsemble(GBDTEnsembleBase):

    def __init__(self, gbdt_list):
        super(GBDTCVEnsemble, self).__init__(gbdt_list)

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
            self.gbdt_list[i].train(train_df, val_df, df_column,
                                    time_col=time_col, inv_col=inv_col,
                                    early_stopping_rounds=early_stopping_rounds,
                                    verbose=verbose, **kwargs)

    def predict(self, test_df, df_column, inv_col='investment_id', time_col='time_id',
                spliter=None):
        pred_list = []
        for i in range(self.k):
            if spliter is not None:
                if hasattr(spliter, "get_folder_preprocess"):
                    xtest_df, _ = spliter.get_folder_preprocess(test_df, i)
                else:
                    xtest_df = test_df
                if hasattr(spliter, "get_folder_preprocess_package"):
                    pro_package = spliter.get_folder_preprocess_package(i)
                    df_column = update_df_column_package(df_column, pro_package)
            else:
                xtest_df = test_df
            pred = self.gbdt_list[i].predict(xtest_df, df_column)
            pred_list.append(pred)
        pred_list = np.array(pred_list)
        pred = np.mean(pred_list, axis=0)
        return pred, pred_list  # np.float, np.ndarray


class GBDTAvgBaggingEnsemble(GBDTEnsembleBase):

    def __init__(self, gbdt_list, seed_list):
        super(GBDTAvgBaggingEnsemble, self).__init__(gbdt_list)
        self.seed_list = seed_list
        for i in range(self.k):
            self.gbdt_list[i].gbdt.seed = seed_list[i]

    def train(self, train_df, val_df, df_column,
              early_stopping_rounds=20,
              verbose=0,
              **kwargs):
        for i in tqdm(range(self.k)):
            self.gbdt_list[i].train(train_df, val_df, df_column, verbose=verbose,
                                    early_stopping_rounds=early_stopping_rounds)


class GBDTCV:

    def __init__(self, k, gbdt):
        self.k = k
        self.gbdt = GBDTWrapper(gbdt)
        self.gbdt_cv_list = []

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
            gbdt_base = copy.deepcopy(self.gbdt)
            train_df, val_df = spliter.get_folder(data_df, i, **kwargs)
            if hasattr(spliter, "get_folder_preprocess_package"):
                pro_package = spliter.get_folder_preprocess_package(i)
                df_column = update_df_column_package(df_column, pro_package)

            gbdt_base.train(train_df, val_df, df_column, verbose=verbose, time_col=time_col, inv_col=inv_col,
                            early_stopping_rounds=early_stopping_rounds)
            pred_vl = gbdt_base.predict(val_df, df_column)
            verr = evaluate_func(pred_vl.reshape(-1), val_df.iloc[:, df_column['y']].values.reshape(-1))
            folder_err_list.append(verr)
            self.gbdt_cv_list.append(gbdt_base)
        folder_err = np.array(folder_err_list)
        return np.mean(folder_err), folder_err

    def get_cv_model(self):
        return self.gbdt_cv_list

    def save_cv_model_file(self, filename):
        save_pickle(filename, self.get_cv_model())

    def load_param_file(self, filename):
        gbdt_cv = load_pickle(filename)
        self.gbdt_cv_list = gbdt_cv


class GBDTGridCVBase:

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
            gbdt_base = self.get_gbdt(param)
            cv_gbdt = GBDTCV(self.k, gbdt_base)
            err_mean, err_list = cv_gbdt.cv(data_df, df_column, spliter, evaluate_func,
                                            inv_col=inv_col, time_col=time_col,
                                            need_split=False,
                                            early_stopping_rounds=early_stopping_rounds,
                                            verbose=verbose,
                                            **kwargs)
            cv_result.append((err_mean, err_list))
            self.cv_model_list.append(cv_gbdt)
        return generate_cv_result_df(cv_result, permutations_dicts)

    def get_cv_model(self, idx):
        return self.cv_model_list[idx]

    def get_cv_model_list(self):
        return self.cv_model_list

    def save_cv_model_list_file(self, filename):
        save_pickle(filename, self.get_cv_model_list())

    def get_gbdt(self, param):
        """
        User should write their own function to get dnn_list & optimizer_list from param
        with "from lightgbm import * (LightGBM sklearn API or other gbdt library)"
        """
        gbdt = None
        return gbdt


class GBDTGridCVBase_Example(GBDTGridCVBase):

    def __init__(self, k, param_dict):
        super(GBDTGridCVBase_Example, self).__init__(k, param_dict)

    def get_gbdt(self, param):
        gbdt = LGBMRegressor(
            boosting_type='gbdt',
            objective='regression',
            n_estimators=100,
            learning_rate=param['learning_rate'],
            max_depth=param['max_depth']
        )
        return gbdt
