import numpy as np

from quant_finance_research.utils import *
from quant_finance_research.prediction.gbdt_framework import *
from quant_finance_research.prediction.tscv import *
from quant_finance_research.fe.fe_val import *
from quant_finance_research.fe.fe_feat import *
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


def get_debug_lgbm():
    lgbm_debug = LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
    )
    return lgbm_debug


def get_debug_lgbm_list(k):
    lgbm_list = []
    for i in range(k):
        lgbm_list.append(get_debug_lgbm())
    return lgbm_list


class DebugGBDTWrapper:

    def __init__(self):
        lgbm = get_debug_lgbm()
        self.wrapper = GBDTWrapper(lgbm, seed=0)

    def debug_train(self):
        df, df_column = get_example_df()
        train_df, val_df = df, df
        self.wrapper.train(train_df, val_df, df_column,
                           early_stopping_rounds=10,
                           verbose=2)

    def debug_predict(self):
        df, df_column = get_example_df()
        pred = self.wrapper.predict(df, df_column)
        print(pred)

    def debug_all(self):
        self.debug_train()
        self.debug_predict()


class DebugGBDTCVEnsemble:

    def __init__(self):
        k = 4
        lgbm_list = get_debug_lgbm_list(k)
        df, df_column = get_example_df()
        self.en = GBDTCVEnsemble(lgbm_list)
        fsplit_base = QuantTimeSplit_GroupTime(k=k)
        pro_func = classic_x_preprocess_package_1
        self.fsplit = QuantTimeSplit_PreprocessSplit(fsplit_base, df_column, preprocess_func=pro_func,
                                                     preprocess_column=df_column['x'])

    def debug_train(self):
        df, df_column = get_example_df()
        self.en.train(df, df_column, self.fsplit,
                      early_stopping_rounds=10,
                      verbose=2,
                      mad_factor=0.5)
        print(self.en.get_ensemble_model())
        print(self.fsplit.get_preprocess_package())

    def debug_predict(self):
        df, df_column = get_example_df()
        pred, pred_l = self.en.predict(df, df_column, spliter=self.fsplit)
        print(pred)
        print(pred_l)
        print(type(pred_l))

    def debug_all(self):
        self.debug_train()
        self.debug_predict()


class DebugGBDTAvgBaggingEnsemble:

    def __init__(self):
        k = 4
        lgbm_list = get_debug_lgbm_list(k)
        self.en = GBDTAvgBaggingEnsemble(lgbm_list, seed_list=[0, 1, 2, 3])

    def debug_train(self):
        df, df_column = get_example_df()
        train_df, val_df = df, df
        self.en.train(train_df, val_df, df_column,
                      early_stopping_rounds=10,
                      verbose=2)
        print(self.en.get_ensemble_model())

    def debug_predict(self):
        df, df_column = get_example_df()
        pred, pred_l = self.en.predict(df, df_column)
        print(pred)
        print(pred_l)
        print(type(pred_l))

    def debug_all(self):
        self.debug_train()
        self.debug_predict()


class DebugGBDTCV:

    def __init__(self):
        k = 4
        lgbm = get_debug_lgbm()
        self.cv = GBDTCV(k, lgbm)
        fsplit_base = QuantTimeSplit_GroupTime(k=k)
        df, df_column = get_example_large_df()
        pro_func = add_GlobalAbsIcRank_LocalTimeMeanFactor
        self.fsplit = QuantTimeSplit_PreprocessSplit(fsplit_base, df_column, preprocess_func=pro_func,
                                                     preprocess_column=df_column['x'])

    def debug_cv(self):
        df, df_column = get_example_large_df()
        train_df, val_df = df, df
        self.cv.cv(df, df_column, self.fsplit,
                   early_stopping_rounds=10,
                   verbose=2,
                   number_GAIR=20)
        print(self.cv.get_cv_model())
        print(self.fsplit.get_preprocess_package())


class GBDTGridCV_Debug(GBDTGridCVBase):
    def __init__(self, k, param_dict):
        super(GBDTGridCV_Debug, self).__init__(k, param_dict)

    def get_gbdt(self, param):

        # gbdt = XGBRegressor(
        #     booster='gbtree',
        #     n_estimators=100,
        #     max_depth=5,
        #     learning_rate=0.1,
        #     reg_lambda=0.005,
        # )

        # gbdt = LGBMRegressor(
        #     boosting_type='gbdt',
        #     objective='regression',
        #     learning_rate=param['learning_rate'],
        #     max_depth=param['max_depth']
        # )

        gbdt = CatBoostRegressor(
            loss_function='RMSE',
            iterations=100,
            learning_rate=0.03,
            random_seed=233,
            depth=6,
            thread_count=-1,
            nan_mode='Min',
            l2_leaf_reg=3.0,
        )

        return gbdt


class DebugGBDTGridCV:

    def __init__(self):
        k = 3
        df, df_column = get_example_df()
        self.param_dict = {"learning_rate": [0.1, 0.01], "max_depth": [3, 5]}
        fsplit_base = QuantTimeSplit_GroupTime(k=k)
        pro_func = classic_x_preprocess_package_1
        self.grid_cv = GBDTGridCV_Debug(k, self.param_dict)
        self.fsplit = QuantTimeSplit_PreprocessSplit(fsplit_base, df_column, preprocess_func=pro_func,
                                                     preprocess_column=df_column['x'])

    def debug_get_model_list(self):
        lgbm = self.grid_cv.get_gbdt(param={"learning_rate": 0.1, "max_depth": 3})
        print(lgbm)

    def debug_cv(self):
        df, df_column = get_example_df()
        dff = self.grid_cv.cv(df, df_column, self.fsplit,
                              early_stopping_rounds=10,
                              verbose=2,
                              mad_factor=0.5)
        print(dff)


if __name__ == "__main__":
    # DebugGBDTWrapper().debug_all()
    DebugGBDTCVEnsemble().debug_all()
    # DebugGBDTAvgBaggingEnsemble().debug_all()
    # DebugGBDTCV().debug_cv()
    # DebugGBDTGridCV().debug_cv()
