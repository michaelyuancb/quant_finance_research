import torch
import os
import matplotlib.pyplot as plt

from quant_finance_research.prediction.nn_module.nn_loss import ICLoss, MSELoss
from torch import nn

from quant_finance_research.utils import load_pickle, save_pickle
from quant_finance_research.eval.factor_eval import evaluate_IC_StaticWeight_time, evaluate_IC_time
from quant_finance_research.plot.plot_utils import plot_seq_setting
from quant_finance_research.eval.eval_utils import seq_cumsum


class DoubleICLoss(nn.Module):
    def __init__(self):
        super(DoubleICLoss, self).__init__()
        self.ic_loss = ICLoss()

    def forward(self, pred, label):
        dic_loss = self.ic_loss(pred[:, 0], label[:, 0]) + self.ic_loss(pred[:, 0], label[:, 1]) * 2.0
        return dic_loss


class DoubleMSELoss(nn.Module):
    def __init__(self):
        super(DoubleMSELoss, self).__init__()
        self.mse_loss = MSELoss()

    def forward(self, pred, label):
        dic_loss = self.mse_loss(pred[:, 0], label[:, 0]) + self.mse_loss(pred[:, 0], label[:, 1])
        return dic_loss


class DoubleRMSELoss(nn.Module):
    def __init__(self):
        super(DoubleRMSELoss, self).__init__()
        self.rmse_loss = MSELoss()

    def forward(self, pred, label):
        dic_loss = self.rmse_loss(pred[:, 0], label[:, 0]) + self.rmse_loss(pred[:, 0], label[:, 1])
        return dic_loss


def evaluate(outsample, df_column, dy_col, model_name, path_name, pred):
    save_pickle(path_name + 'pred.pkl', pred)

    label_1d = outsample.iloc[:, dy_col[0]].values.reshape(-1)
    label_10d = outsample.iloc[:, dy_col[1]].values.reshape(-1)
    weight = outsample.iloc[:, df_column['loss']].values.reshape(-1)
    time_id = outsample.time_id.values.reshape(-1)

    print(pred.shape)
    print(label_1d.shape)
    print(label_10d.shape)
    print(weight.shape)
    print(time_id.shape)

    nic_mean_1d, nic_list_1d, time_id_list = evaluate_IC_time(pred, label_1d, time_id)
    nic_mean_10d, nic_list_10d, time_id_list = evaluate_IC_time(pred, label_10d, time_id)
    wic_mean_1d, wic_list_1d, time_id_list = evaluate_IC_StaticWeight_time(pred, label_1d, weight, time_id)
    wic_mean_10d, wic_list_10d, time_id_list = evaluate_IC_StaticWeight_time(pred, label_10d, weight, time_id)
    n_time = len(time_id_list)

    print("========================= MultiTask Result ===========================")
    print(f"Model={model_name}")
    print(f"MEAN(IC_1d)={nic_mean_1d}")
    print(f"MEAN(IC_10d)={nic_mean_10d}")
    print(f"MEAN(WIC_1d)={wic_mean_1d}")
    print(f"MEAN(WIC_10d)={wic_mean_10d}")
    print("")
    print("========================= ================ ===========================")
    save_pickle(path_name + 'eval.pkl', (nic_mean_1d, nic_list_1d, nic_mean_10d, nic_list_10d, wic_mean_1d, wic_list_1d,
                                         wic_mean_10d, wic_list_10d, time_id_list))
    if not os.path.exists(path_name + 'pic/'):
        os.mkdir(path_name + 'pic/')

    def get_fig(title):
        plot_seq_setting(figsize=(12, 7.5), subplot=111,
                         xlabel='date', ylabel='Value', label_font_size=None,
                         ax_rotation=-10, ax_base=n_time / 10, ax_font_size=12, ay_font_size=12,
                         title=title, title_font_size=15)

    print("plotting...")

    get_fig(model_name + '_1d_IC')
    plt.plot(time_id_list, nic_list_1d, label="IC_1d", linewidth=2.5)
    plt.plot(time_id_list, wic_list_1d, label="WeightIC_1d", linewidth=2.5)
    plt.legend()
    plt.savefig(path_name + 'pic/' + model_name + "_ic_1d.png", dpi=200)

    get_fig(model_name + '_10d_IC')
    plt.plot(time_id_list, nic_list_10d, label="IC_10d", linewidth=2.5)
    plt.plot(time_id_list, wic_list_10d, label="WeightIC_10d", linewidth=2.5)
    plt.legend()
    plt.savefig(path_name + 'pic/' + model_name + "_ic_10d.png", dpi=200)

    get_fig(model_name + '_1d_AccumIC')
    plt.plot(time_id_list, seq_cumsum(wic_list_1d), label="WeightIC_1d", linewidth=2.5)
    plt.plot(time_id_list, seq_cumsum(nic_list_1d), label="IC_1d", linewidth=2.5)
    plt.legend()
    plt.savefig(path_name + 'pic/' + model_name + "_ic_accum_1d.png", dpi=200)

    get_fig(model_name + '_10d_AccumIC')
    plt.plot(time_id_list, seq_cumsum(wic_list_10d), label="WeightIC_10d", linewidth=2.5)
    plt.plot(time_id_list, seq_cumsum(nic_list_10d), label="IC_10d", linewidth=2.5)
    plt.legend()
    plt.savefig(path_name + 'pic/' + model_name + "_ic_accum_10d.png", dpi=200)


if __name__ == "__main__":
    pred = torch.randn(64, 2)
    label = torch.randn(64, 2)
    db = DoubleICLoss()
    loss = db(pred, label)
    print(loss)
