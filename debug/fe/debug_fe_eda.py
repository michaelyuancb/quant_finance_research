import random

import pandas as pd
import numpy as np

from quant_finance_research.fe.fe_eda import *
from quant_finance_research.utils import *


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
    debug_eda_nan_analysis()
    # debug_eda_factor_ic_analysis()
