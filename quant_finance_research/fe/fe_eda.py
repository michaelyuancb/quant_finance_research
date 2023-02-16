import pandas as pd
import numpy as np


def eda_nan_analysis(df, df_column, column, **kwargs):
    col_null_ratio = df.iloc[:, column].isnull().sum(axis=0).values / df.shape[0] * 100.0
    row_null_ratio = (df.iloc[:, column].isnull().sum(axis=1)).values / len(column) * 100.0
    max_cn = np.max(col_null_ratio)
    min_cn = np.min(col_null_ratio)
    mean_cn = np.mean(col_null_ratio)
    median_cn = np.median(col_null_ratio)
    max_rn = np.max(row_null_ratio)
    min_rn = np.min(row_null_ratio)
    mean_rn = np.mean(row_null_ratio)
    median_rn = np.median(row_null_ratio)
    print(f"Column_Nan_Ratio: max={np.round(max_cn,2)}%, min={np.round(min_cn,2)}%, mean={np.round(mean_cn,2)}%, median={np.round(median_cn,2)}%")
    print(f"Row_Nan_Ratio: max={np.round(max_rn,2)}%, min={np.round(min_rn,2)}%, mean={np.round(mean_rn,2)}%, median={np.round(median_rn,2)}%")
    return col_null_ratio / 100.0, row_null_ratio / 100.0


def eda_factor_ic_analysis(data_df, df_column, **kwargs):
    x_column, y_column, _ = df_column.values()
    x_name = data_df.columns[x_column].tolist()
    y_name = data_df.columns[y_column].tolist()

    corr = data_df[x_name + y_name].corr()[y_name].drop(labels=y_name).reset_index()
    corr[y_name] = abs(corr[y_name])
    corr = pd.concat([corr, pd.DataFrame(x_column)], axis=1)
    corr.sort_values(y_name, ascending=False, inplace=True)
    print("Finish Generate Factor-IC-Rank DataFrame.")
    corr = corr.rename(columns={0: 'x_column_idx'})
    return corr
