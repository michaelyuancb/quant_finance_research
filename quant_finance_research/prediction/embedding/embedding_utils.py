import numpy as np
import pandas as pd


def category2int_global(df_org, df_column, cat_column):
    """
        Transfer the Category Feature to Int Globally.
        "Global" means that there will not be any same value(int) in difference columns.
        The return of this function could be used to satisfy the demand of Embedding Class Construction.

        Input: df, df_column, cat_column, where cat_column is a list which is the index to be processed.
        Return: df_new, total_n_cat, where total_n_cat is the number of int used in transfer.

    """
    df = df_org.copy()
    total_n_cat = 0
    for col in cat_column:
        cls = df.iloc[:, col].unique()
        dict_trasfer = dict()
        for c in cls:
            dict_trasfer[c] = total_n_cat
            total_n_cat = total_n_cat + 1
        df.iloc[:, col] = df.iloc[:, col].apply(lambda x: dict_trasfer[x])
    return df, total_n_cat
