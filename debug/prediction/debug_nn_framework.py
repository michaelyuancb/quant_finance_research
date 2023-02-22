import numpy as np
import torch.nn

from quant_finance_research.utils import *
from quant_finance_research.prediction.nn_framework import *
from quant_finance_research.prediction.nn.base_dnn import Base_DNN
from quant_finance_research.prediction.tscv import *
from quant_finance_research.fe.fe_val import *
from quant_finance_research.fe.fe_feat import *
from quant_finance_research.prediction.nn_module.nn_loss import *


def get_debug_dnn(input_dim=None, output_dim=None):
    df, df_column = get_example_df()
    input_dim = len(df_column['x']) if input_dim is None else input_dim
    output_dim = len(df_column['y']) if output_dim is None else output_dim
    dnn = torch.nn.Linear(input_dim, output_dim)
    optimizer = torch.optim.Adam(dnn.parameters(), lr=1e-2)
    return dnn, optimizer


def get_debug_dnn_list(k=5, input_dim=None, output_dim=None):
    dnn_list = []
    optimizer_list = []
    for i in range(k):
        dnn, optimizer = get_debug_dnn(input_dim, output_dim)
        dnn_list.append(dnn)
        optimizer_list.append(optimizer)
    return dnn_list, optimizer_list


class DebugNeuralNetworkWrapper:

    def __init__(self):
        dnn, optimizer = get_debug_dnn()
        self.wrapper = NeuralNetworkWrapper(dnn, optimizer, device='cuda')

    def debug_train(self):
        df, df_column = get_example_df()
        train_df, val_df = df.iloc[:6], df.iloc[-6:]
        loss_func = StaticWeightRMSELoss()
        loss_val = self.wrapper.train(train_df, val_df, df_column, loss_func, use_loss_column=True,
                                      seed=0,
                                      early_stop=10,
                                      epoch_print=1,
                                      max_epoch=20,
                                      batch_size=6)
        print(loss_val)
        print(self.wrapper.get_best_param())

    def debug_predict(self):
        df, df_column = get_example_df()
        pred = self.wrapper.predict(df, df_column, batch_size=6)
        print(pred)
        self.wrapper.set_device(device='cuda')
        pred1 = self.wrapper.predict(df, df_column, batch_size=6)
        self.wrapper.set_device(device='cpu')
        pred2 = self.wrapper.predict(df, df_column, batch_size=6)
        print("===============")
        pp = np.concatenate([pred1, pred2], axis=1)
        print(pp)

    def debug_all(self):
        self.debug_train()
        self.debug_predict()


class DebugNeuralNetworkAvgBaggingEnsemble:

    def __init__(self):
        dnn_list, optimizer_list = get_debug_dnn_list(k=3)
        self.en = NeuralNetworkAvgBaggingEnsemble(dnn_list, optimizer_list, seed_list=[0, 1, 2], device='cuda')

    def debug_train(self):
        df, df_column = get_example_df()
        train_df, val_df = df.iloc[:6], df.iloc[-6:]
        loss_func = nn.MSELoss()
        loss_val = self.en.train(train_df, val_df, df_column, loss_func,
                                 early_stop=10,
                                 epoch_print=1,
                                 max_epoch=20,
                                 batch_size=6)
        print(loss_val)
        print(self.en.get_best_param())

    def debug_predict(self):
        df, df_column = get_example_df()
        pred, pred_l = self.en.predict(df, df_column, batch_size=6)
        print(pred)
        print(pred_l)
        print(type(pred_l))
        self.en.set_device(device='cuda')
        pred1, _ = self.en.predict(df, df_column, batch_size=6)
        self.en.set_device(device='cpu')
        pred2, _ = self.en.predict(df, df_column, batch_size=6)
        print("===============")
        pp = np.concatenate([pred1, pred2], axis=1)
        print(pp)

    def debug_all(self):
        self.debug_train()
        self.debug_predict()


class DebugNeuralNetworkCVEnsemble:

    def __init__(self):
        k = 4
        dnn_list, optimizer_list = get_debug_dnn_list(k=k)
        df, df_column = get_example_df()
        self.en = NeuralNetworkCVEnsemble(dnn_list, optimizer_list, device='cuda')
        fsplit_base = QuantTimeSplit_GroupTime(k=k)
        pro_func = classic_x_preprocess_package_1
        self.fsplit = QuantTimeSplit_PreprocessSplit(fsplit_base, df_column=df_column, preprocess_func=pro_func,
                                                     preprocess_column=df_column['x'])

    def debug_train(self):
        df, df_column = get_example_df()
        train_df, val_df = df.iloc[:6], df.iloc[-6:]
        loss_func = nn.MSELoss()
        loss_val = self.en.train(df, df_column, self.fsplit, loss_func,
                                 early_stop=10,
                                 epoch_print=1,
                                 max_epoch=20,
                                 batch_size=6,
                                 mad_factor=0.5)
        print(loss_val)
        print(self.en.get_best_param())
        print(self.fsplit.get_preprocess_package())

    def debug_predict(self):
        df, df_column = get_example_df()
        pred, pred_l = self.en.predict(df, df_column, spliter=self.fsplit, batch_size=6)
        print(pred)
        print(pred_l)
        print(type(pred_l))
        self.en.set_device(device='cuda')
        pred1, _ = self.en.predict(df, df_column, batch_size=6)
        self.en.set_device(device='cpu')
        pred2, _ = self.en.predict(df, df_column, batch_size=6)
        print("===============")
        pp = np.concatenate([pred1, pred2], axis=1)
        print(pp)

    def debug_all(self):
        self.debug_train()
        self.debug_predict()


class DebugNeuralNetworkCV:

    def __init__(self):
        k = 4
        dnn_list, optimizer_list = get_debug_dnn_list(k=k, input_dim=3, output_dim=1)
        df, df_column = get_example_df()
        self.en = NeuralNetworkCV(dnn_list, optimizer_list, device='cuda')
        fsplit_base = QuantTimeSplit_GroupTime(k=k)
        pro_func = add_GlobalAbsIcRank_LocalTimeMeanFactor
        self.fsplit = QuantTimeSplit_PreprocessSplit(fsplit_base, df_column=df_column, preprocess_func=pro_func,
                                                     preprocess_column=df_column['x'])

    def debug_cv(self):
        df, df_column = get_example_df()
        train_df, val_df = df.iloc[:6], df.iloc[-6:]
        loss_func = StaticWeightCCCLoss()
        loss_val = self.en.cv(df, df_column, self.fsplit, loss_func, use_loss_column=True,
                              early_stop=10,
                              epoch_print=1,
                              max_epoch=20,
                              batch_size=6,
                              mad_factor=0.5,
                              number_GAIR=1, time_col='time_id')
        print(loss_val)
        print(self.en.get_best_param())
        print(self.fsplit.get_preprocess_package())


class NeuralNetworkGridCV_Debug(NeuralNetworkGridCVBase):
    """
    An Example to show how to use NeuralNetworkGridCV
    """

    def __init__(self, k, param_dict, device='cuda'):
        super(NeuralNetworkGridCV_Debug, self).__init__(k, param_dict, device)

    def get_model_list(self, param):
        """
        User should write their own function to get dnn_list & optimizer_list from param
        """
        dnn_list = []
        optimizer_list = []
        param = {'learning_rate': 0.1}
        for i in range(self.k):
            dnn = Base_DNN(input_dim=2, hidden_dim=1, dropout_rate=0)
            optim = torch.optim.Adam(dnn.parameters(), lr=param['learning_rate'])
            dnn_list.append(dnn)
            optimizer_list.append(optim)
        assert self.k == len(dnn_list)
        return dnn_list, optimizer_list


class DebugNeuralNetworkGridCV:

    def __init__(self):
        k = 3
        df, df_column = get_example_df()
        self.param_dict = {"learning_rate": [0.1, 0.01, 0.001]}
        self.grid_cv = NeuralNetworkGridCV_Debug(k, self.param_dict)
        fsplit_base = QuantTimeSplit_GroupTime(k=k)
        pro_func = classic_x_preprocess_package_1
        self.fsplit = QuantTimeSplit_PreprocessSplit(fsplit_base, df_column, preprocess_func=pro_func,
                                                     preprocess_column=df_column['x'])

    def debug_get_model_list(self):
        dnn_list, optimizer_list = self.grid_cv.get_model_list(param={"learning_rate": 0.1})
        print(dnn_list)
        print(optimizer_list)

    def debug_cv(self):
        df, df_column = get_example_df()
        loss_func = StaticWeightICLoss()
        df = self.grid_cv.cv(df, df_column, self.fsplit, loss_func, use_loss_column=True,
                             early_stop=10,
                             epoch_print=1,
                             max_epoch=20,
                             batch_size=6,
                             mad_factor=0.5)
        print(df)


if __name__ == "__main__":
    # DebugNeuralNetworkWrapper().debug_all()
    # DebugNeuralNetworkAvgBaggingEnsemble().debug_all()
    # DebugNeuralNetworkCVEnsemble().debug_all()
    # DebugNeuralNetworkCV().debug_cv()
    DebugNeuralNetworkGridCV().debug_cv()

