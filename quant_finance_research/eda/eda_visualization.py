import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from quant_finance_research.plot.plot_utils import *


# ========================================== Asset-Target ==================================================

def eda_target_asset_count_hist(df, df_column, inv_col='investment_id', bins=60, target_col=None, fig_size=(7, 3),
                                save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    obs_by_asset = df.groupby(inv_col)[target_col].count()
    plot_seq_setting(figsize=fig_size, xlabel='Asset observations',
                     title=f"Target[{target_col}]-Count by Asset Observations")
    obs_by_asset.plot.hist(bins=bins)
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_time_asset_exist_scatter(df, df_column, inv_col='investment_id', time_col='time_id', fig_size=(20, 20),
                                 s=0.5, save_filename=None, save_dpi=150):
    df[[inv_col, time_col]].plot.scatter(time_col, inv_col, figsize=fig_size, s=s)
    plt.title("Asset vs Time Exist-Map")
    plt.xlabel("time column")
    plt.ylabel("asset column")
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_target_asset_mean_hist(df, df_column, inv_col='investment_id', bins=60, target_col=None, fig_size=(7, 3),
                               save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    obs_by_asset = df.groupby(inv_col)[target_col].mean()
    plot_seq_setting(figsize=fig_size, xlabel='Asset observations',
                     title=f"Target[{target_col}]-Mean by Asset Observations")
    obs_by_asset.plot.hist(bins=bins)
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_target_asset_std_hist(df, df_column, inv_col='investment_id', bins=60, target_col=None, fig_size=(7, 3),
                              save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    obs_by_asset = df.groupby(inv_col)[target_col].std()
    plot_seq_setting(figsize=fig_size, xlabel='observations',
                     title=f"Target[{target_col}]-Std by Asset Observations")
    obs_by_asset.plot.hist(bins=bins)
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_target_asset_count_mean_join(df, df_column, inv_col='investment_id', target_col=None, height=5,
                                     save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    obs_by_asset = df.groupby(inv_col)[target_col].count()
    mean_target = df.groupby([inv_col])[target_col].mean()
    ax = sns.jointplot(x=obs_by_asset, y=mean_target, kind="reg",
                       height=height, joint_kws={'line_kws': {'color': 'red'}})
    ax.ax_joint.set_xlabel('observations')
    ax.ax_joint.set_ylabel('mean target')
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_target_asset_count_std_join(df, df_column, inv_col='investment_id', target_col=None, height=5,
                                    save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    obs_by_asset = df.groupby(inv_col)[target_col].count()
    mean_target = df.groupby([inv_col])[target_col].std()
    ax = sns.jointplot(x=obs_by_asset, y=mean_target, kind="reg",
                       height=height, joint_kws={'line_kws': {'color': 'red'}})
    ax.ax_joint.set_xlabel('observations')
    ax.ax_joint.set_ylabel('std target')
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


# ========================================== Time Series ==================================================

def eda_asset_time_count_plot(df, df_column, inv_col='investment_id', time_col='time_id', fig_size=(7, 3),
                              save_filename=None, save_dpi=150):
    plot_seq_setting(figsize=fig_size, xlabel='Time Series', ylabel='Asset Count',
                     title=f"Asset-Count with Time Series")
    df.groupby(time_col)[inv_col].nunique().plot()
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_target_time_mean_plot(df, df_column, time_col='time_id', target_col=None, fig_size=(7, 3),
                              save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    plot_seq_setting(figsize=fig_size, xlabel='Time Series', ylabel='Target Mean',
                     title=f"Target[{target_col}]-Mean with Time Series")
    mean_target = df.groupby([time_col])[target_col].mean()
    mean_target.plot()
    plt.axhline(y=np.mean(mean_target), color='r', linestyle='--', label="mean")
    plt.legend()
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_target_time_std_plot(df, df_column, time_col='time_id', target_col=None, fig_size=(7, 3),
                             save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    plot_seq_setting(figsize=fig_size, xlabel='Time Series', ylabel='Target Mean',
                     title=f"Target[{target_col}]-Std with Time Series")
    std_target = df.groupby([time_col])[target_col].std()
    std_target.plot()
    plt.axhline(y=np.mean(std_target), color='r', linestyle='--', label="mean")
    plt.legend()
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_target_time_mean_std_plot(df, df_column, time_col='time_id', target_col=None, fig_size=(12, 7), s=1.0,
                                  save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    time2target_mean = df.groupby([time_col])[target_col].mean()
    time2target_std = df.groupby([time_col])[target_col].std()
    plot_seq_setting(figsize=fig_size, xlabel='Time Series', ylabel='Target Mean',
                     title=f"Target[{target_col}]-MeanStd with Time Series")

    plt.fill_between(
        time2target_mean.index,
        time2target_mean - time2target_std,
        time2target_mean + time2target_std,
        alpha=0.1,
        color="b",
    )
    plt.plot(time2target_mean.index, time2target_mean, "o-", color="b", markersize=s, label="Training score")
    plt.axhline(y=np.mean(time2target_mean), color='r', linestyle='--', label="mean")
    plt.legend()
    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)


def eda_time_basic_summary_plot(df, df_column, time_col='time_id', inv_col='investment_id', target_col=None,
                                fig_size=(15, 18), save_filename=None, save_dpi=150):
    target_col = target_col if target_col is not None else df.columns[df_column['y'][0]]
    plt.figure(figsize=fig_size)

    plt.subplot(3, 1, 1)
    df.groupby(time_col)[inv_col].nunique().plot()
    plt.title(f"Asset-Count with Time Series")
    plt.xlabel(None)

    plt.subplot(3, 1, 2)
    mean_target = df.groupby(time_col)[target_col].mean()
    mean_target.plot()
    plt.title(f"Target[{target_col}]-Mean with Time Series")
    plt.axhline(y=mean_target.mean(), color='r', linestyle='--', label="mean")
    plt.legend(loc='lower left')
    plt.xlabel(None)

    plt.subplot(3, 1, 3)
    std_target = df.groupby(time_col)[target_col].std()
    std_target.plot()
    plt.title(f"Target[{target_col}]-Std with Time Series")
    plt.axhline(y=std_target.mean(), color='r', linestyle='--', label="mean")
    plt.legend(loc='lower left')

    if save_filename:
        plt.savefig(save_filename, dpi=save_dpi)
