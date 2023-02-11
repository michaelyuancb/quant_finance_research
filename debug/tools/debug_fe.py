import random

import pandas as pd

from quant_finance_research.tools.fe import *


def debug_add_GlobalAbsIcRank_LocalTimeMeanFactor():
    data = np.random.randn(10, 6)
    data[:, 0] = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4, 6])
    data[:, 1] = np.array([0, 1, 0, 2, 1, 0, 2, 0, 1, 2])
    df = pd.DataFrame(data, columns=['time', 'investment', 'f_1', 'f_2', 'f_3', 'y'])
    df_new, best_col, best_col_name = add_GlobalAbsIcRank_LocalTimeMeanFactor(df, x_column=[2, 3, 4], y_column=5,
                                                                              number=2,
                                                                              time_col='time')
    print(df)
    print(df_new)
    print(best_col)
    print(best_col_name)


if __name__ == "__main__":
    debug_add_GlobalAbsIcRank_LocalTimeMeanFactor()
