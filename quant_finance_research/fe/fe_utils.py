import pandas as pd
import numpy as np


def find_column_idx(column_name, df_columns):
    """
        Find column_idx in O(n) Time Complexity.
        Notice that the column_name appearance order should be the same with df_columns.
        ::input column_name: list of column's name, each item is a str.
        ::input df_columns: the Column of DataFrame, df.columns.
    """
    df_idx = 0
    column_idx = []
    for j in range(len(column_name)):
        while df_columns[df_idx] != column_name[j]:
            df_idx = df_idx + 1
        column_idx.append(df_idx)
    return column_idx


def update_df_column_package(df_column, pro_package):
    if "df_column_new" in pro_package.keys():
        df_column = pro_package['df_column_new']
    return df_column


class PiplinePreprocess:

    def __init__(self):
        self.preprocess_function_list = []

    def __call__(self, data_df, df_column, column, **kwargs):
        pro_package = dict()

        for func_tuple in self.preprocess_function_list:
            func_type, func, df_column_input, column_input = func_tuple
            if func_type == 'feat':
                data_df, package = func(data_df, df_column_input, column_input, **kwargs)
                df_column = update_df_column_package(df_column, package)
            elif func_type == 'val':
                data_df, package = func(data_df, df_column_input, column_input, **kwargs)
            else:
                raise ValueError(f"Unsupported Preprocess Function Type: {func_type} for {func}.")
            pro_package.update(package)

        return data_df, pro_package

    def func_num(self):
        return len(self.preprocess_function_list)

    def add_feat_preprocess(self, func, df_column, column):
        self.preprocess_function_list.append(('feat', func, df_column, column))

    def add_val_preprocess(self, func, df_column, column):
        self.preprocess_function_list.append(('val', func, df_column, column))


def add_preprocess_pipeline(pipeline, add_type, preprocess_func, df, df_column, column, pro_pack=None, **kwargs):
    """
        Add the preprocess_func to pipeline (PiplinePreprocess), and then monk the process for later usage.
        ::input pipeline: the PiplinePreprocess Class
        ::input add_type: str, 'val' of 'feat', the add_type of the preprocess_func
        ::input preprocess_func: the preprocess function with standard format
        ::input df, df_column, column
        ::input pro_pack: a dict to collect all the input parameter for preprocess_function_pipeline.
                          If None, it means that this is the first func to be added to the pipeline, then the output
                          pro_pack could be used for later collection.
                          If not None, it means to inherit the parameter collected before.

        :output tmp_df: the dataframe after the preprocess by preprocess_func
        :output df_column: the new df_column
        :output pro_pack: the input parameter collection dict. (has been added the parameter of this func.)
        output prop_return: the return pro_pack by the preprocess_func
    """
    assert isinstance(pipeline, PiplinePreprocess)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_column, dict)
    assert isinstance(column, list)
    assert add_type in ['feat', 'val']
    pro_pack = dict() if pro_pack is None else pro_pack
    if add_type in ['val']:
        pipeline.add_val_preprocess(preprocess_func, df_column, column)
    elif add_type in ['feat']:
        pipeline.add_feat_preprocess(preprocess_func, df_column, column)
    tmp_df, prop_return = preprocess_func(df, df_column, column, **kwargs)
    pro_pack.update(kwargs)
    df_column = update_df_column_package(df_column, prop_return)
    return tmp_df, df_column, pro_pack, prop_return