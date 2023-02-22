import pandas as pd
import numpy as np


def eda_data_type(df, df_column, print_result=True):
    """
        Analysis the data-type of column_x, column_y, column_loss.
        The data type is split to three group, real/object/int, with real to be the numerical feature, object to be the
        category feature, and int is leave for user to decide. dict_absolute_index and dict_relative_index is provided
        for later process.

        Input: df, df_column
        Output: dict_absolute_index, dict_relative_index, df_result.

        dict_absolute_index: a dict with keys: 'x_type'/'y_type'/'loss_type', type in ['real', 'object', 'int'].
                             the value of each key is a list with absolute index in DataFrame.

        dict_relative_index: a dict with keys: 'x_type'/'y_type'/'loss_type', type in ['real', 'object', 'int'].
                             the value of each key is a list with relative index in x_column/y_column/loss_column.
                             One usage of this dict is to determine the category column in model.embedding.
        df_result: the result DataFrame with data_type as column and column_type as index.

    """

    def _analysis_data_type(dft, idx_list, prefix=""):
        tps = dft.dtypes
        real_list, relative_real_list = [], []
        object_list, relative_object_list = [], []
        int_list, relative_int_list = [], []
        for i in range(len(idx_list)):
            tp = str(tps[i])
            if 'float' in tp or 'double' in tp:
                real_list.append(idx_list[i])
                relative_real_list.append(i)
            elif 'int' in tp:
                int_list.append(idx_list[i])
                relative_int_list.append(i)
            else:
                object_list.append(idx_list[i])
                relative_object_list.append(i)
        dict_abs = {prefix+'real': real_list, prefix+'object': object_list, prefix+'int': int_list}
        dict_rel = {prefix+'real': relative_real_list, prefix+'object': relative_object_list,
                    prefix+'int': relative_int_list}
        return dict_abs, dict_rel

    dict_abs, dict_rel = dict(), dict()
    dabs, drel = _analysis_data_type(df.iloc[:, df_column['x']], df_column['x'], prefix='x_')
    dict_abs.update(dabs)
    dict_rel.update(drel)
    dabs, drel = _analysis_data_type(df.iloc[:, df_column['y']], df_column['y'], prefix='y_')
    dict_abs.update(dabs)
    dict_rel.update(drel)
    dabs, drel = _analysis_data_type(df.iloc[:, df_column['loss']], df_column['loss'], prefix='loss_')
    dict_abs.update(dabs)
    dict_rel.update(drel)
    dfx = pd.DataFrame([[len(dict_abs['x_real']), len(dict_abs['x_object']), len(dict_abs['x_int'])],
                        [len(dict_abs['y_real']), len(dict_abs['y_object']), len(dict_abs['y_int'])],
                        [len(dict_abs['loss_real']), len(dict_abs['loss_object']), len(dict_abs['loss_int'])]],
                       columns=['num_real', 'num_object', 'num_int'], index=['x', 'y', 'loss'])
    if print_result:
        print("=========================== eda_data_type ===========================")
        print(dfx)
    return dict_abs, dict_rel, dfx


def eda_nan_analysis(df, df_column, column, print_result=True):
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
    if print_result:
        print("=========================== eda_nan_analysis ===========================")
        print(f"Column_Nan_Ratio: max={np.round(max_cn,2)}%, min={np.round(min_cn,2)}%, mean={np.round(mean_cn,2)}%, median={np.round(median_cn,2)}%")
        print(f"Row_Nan_Ratio: max={np.round(max_rn,2)}%, min={np.round(min_rn,2)}%, mean={np.round(mean_rn,2)}%, median={np.round(median_rn,2)}%")
    return col_null_ratio / 100.0, row_null_ratio / 100.0


def eda_factor_ic_analysis(data_df, df_column, print_result=True):
    x_column, y_column, _ = df_column.values()
    x_name = data_df.columns[x_column].tolist()
    y_name = data_df.columns[y_column].tolist()

    corr = data_df[x_name + y_name].corr()[y_name].drop(labels=y_name).reset_index()
    corr[y_name] = abs(corr[y_name])
    corr = pd.concat([corr, pd.DataFrame(x_column)], axis=1)
    corr.sort_values(y_name, ascending=False, inplace=True)
    corr = corr.rename(columns={0: 'x_column_idx'})
    if print_result:
        print("=========================== eda_nan_analysis ===========================")
        print(corr)
    return corr
