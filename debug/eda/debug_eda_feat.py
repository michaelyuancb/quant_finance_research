import random

import pandas as pd
import numpy as np

from quant_finance_research.eda.eda_feat import *
from quant_finance_research.utils import *


def debug_eda_data_type():
    df, df_column = get_example_cat_df()
    print(df)
    print(type(df.iloc[0, -1]))
    idx_abs_dict, idx_rel_dict, dfx = eda_data_type(df, df_column)
    print(idx_abs_dict)
    print(idx_rel_dict)


def debug_eda_nan_analysis():
    df, df_column = get_example_df()
    for j in range(8):
        df.iloc[j, 2] = np.nan
    print(df)
    res = eda_nan_analysis(df, df_column, df_column['x'] + df_column['y'])
    print(res)
    print("success.")


def debug_eda_factor_ic_analysis():
    df, df_column = get_example_df()
    df_res = eda_factor_ic_analysis(df, df_column)
    print(df_res)
    print("success.")


if __name__ == "__main__":
    # debug_eda_data_type()
    # debug_eda_nan_analysis()
    debug_eda_factor_ic_analysis()
