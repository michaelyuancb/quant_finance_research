import random

import pandas as pd
import numpy as np

from quant_finance_research.fe.fe_feat import *
from quant_finance_research.utils import *


def debug_delete_Feature():
    df, df_column = get_example_df()
    dfval = df.copy()
    column = [2]
    print(df)
    df2, pro = delete_Feature(df, df_column, column)
    assert(df.values == dfval.values).all()
    print(df2)
    print(pro)
    df3, _ = delete_Feature(df, df_column, column, **pro)
    assert (df3.values == df2.values).all()
    print("success.")


def debug_delete_NanRatio_Feature():
    df, df_column = get_example_df()
    dfval = df.copy()
    for j in range(8):
        df.iloc[j, 2] = np.nan
    column = df_column['x']
    print(df)
    df2, pro = delete_NanRatio_Feature(df, df_column, column, del_nan_ratio=0.3)
    print(pro)
    df3, _ = delete_NanRatio_Feature(df, df_column, column, **pro)
    assert (df3.values == df2.values).all()
    print("success.")


def debug_find_GlobalAbsIcRank():

    df, df_column = get_example_df()
    print(df)
    df2, pro = find_GlobalAbsIcRank(df, df_column, df_column['x'], number_GAIR=1)
    print(df2)
    print(pro)
    df3, pro2 = find_GlobalAbsIcRank(df, df_column, df_column['x'], **pro)
    assert (pro == pro2)
    print("success.")


def debug_add_LocalTimeMeanFactor():

    df, df_column = get_example_df()
    print(df)
    column = [2]
    df2, pro = add_LocalTimeMeanFactor(df, df_column, column, time_col="time_id")
    print(df2)
    print(pro)
    df3, _ = add_LocalTimeMeanFactor(df, df_column, column, **pro)
    assert (df3.values == df2.values).all()
    print("success.")


def debug_add_GlobalAbsIcRank_LocalTimeMeanFactor():

    df, df_column = get_example_df()
    print(df)
    df2, pro = add_GlobalAbsIcRank_LocalTimeMeanFactor(df, df_column, df_column['x'], number_GAIR=1, time_col="time_id")
    print(df2)
    print(pro)
    df3, _ = add_GlobalAbsIcRank_LocalTimeMeanFactor(df, df_column, df_column['x'], **pro)
    assert (df3.values == df2.values).all()
    print("success.")


def debug_build_portfolio():
    df, df_column = get_example_df()
    print(df)
    df_val = copy.deepcopy(df)
    df, pro_pack = build_portfolio(df, df_column, df_column['x'],
                                   build_portfolio_weight=np.array([[0.4, 0.6], [0.1, 0.9], [0.05, 0.2]]),
                                   name_prefix='test_portfolio_',
                                   inplace=True)
    print(df)
    print(pro_pack)
    df_val, _ = build_portfolio(df_val, df_column, df_column['x'], **pro_pack)
    assert (df.values == df_val.values).all()
    print("success.")


if __name__ == "__main__":
    # debug_delete_Feature()
    debug_delete_NanRatio_Feature()
    # debug_find_GlobalAbsIcRank()
    # debug_add_LocalTimeMeanFactor()
    # debug_add_GlobalAbsIcRank_LocalTimeMeanFactor()
    # debug_build_portfolio()