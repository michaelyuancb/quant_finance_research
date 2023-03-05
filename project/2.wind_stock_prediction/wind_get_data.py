import copy

import pandas as pd
import numpy as np
import sklearn
import pickle
import datetime
import datatable as dtb
from datetime import datetime
import matplotlib.pyplot as plt

from quant_finance_research.utils import load_pickle, save_pickle
from quant_finance_research.fe.fe_val import panel_time_sumone_normalize_preprocess
from quant_finance_research.utils import get_example_df


def get_toy_data(out_sample_start):
    del out_sample_start
    df, df_column = get_example_df()
    df['target_2'] = copy.deepcopy(df['target'])
    df['wind_extra'] = copy.deepcopy(df['factor_1'].apply(lambda x: x*x+0.5+np.sin(x)-np.exp(x)))
    df_column['x'] = df_column['x'] + [df.shape[1]-1]

    return df, df.copy(), df_column, [df_column['y'][0], df.shape[1] - 2], [2], [3, df.shape[1]-1]


def set_df_continue_index(df):
    df.index = range(len(df))


def get_wind_data(out_sample_start, file='no_barra'):
    if file == 'no_barra':
        data = load_pickle('../../../datasets/wind_factor/wind_factor_insample.pkl')
    elif file == 'barra':
        data = load_pickle('../../../datasets/wind_factor/wind_and_barra_factor_insample.pkl')
    else:
        raise ValueError(f"No such file type: {file}")
    print(data.shape)
    # data = data.iloc[:10000, :]

    data_columns = data.columns.tolist()

    data = data.reset_index()
    del data['MicroSecondSinceEpoch'], data['positive']
    data = data.rename(columns={'time': 'time_id', 'YID': 'investment_id'})
    del data['id']

    x_column = [i for i in range(5, data.shape[1])]
    loss_column = [2]
    y_column = [3, 4]
    df_column = {'x': x_column, 'y': y_column, 'loss': loss_column}
    data = data.dropna(axis=0, how='any')

    if file == 'barra':
        barra_idx = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]   # see wind_init_analysis.ipynb for more details.
        wind_factor_idx = [i for i in df_column['x'] if i not in barra_idx]
    else:
        barra_idx = []
        wind_factor_idx = df_column['x']

    insample = data[data['time_id'] < out_sample_start]
    outsample = data[data['time_id'] >= out_sample_start]
    set_df_continue_index(insample)
    set_df_continue_index(outsample)

    insample, _ = panel_time_sumone_normalize_preprocess(insample, df_column, df_column['loss'], time_col='time_id')
    outsample, _ = panel_time_sumone_normalize_preprocess(outsample, df_column, df_column['loss'], time_col='time_id')

    return insample, outsample, df_column, [3, 4], barra_idx, wind_factor_idx


if __name__ == "__main__":
    insample, outsmple, df_column, dy_col, barra_idx, wind_idx = get_toy_data(1)
    print(insample)
    print(df_column)
    print(dy_col)
    print(insample.columns)
