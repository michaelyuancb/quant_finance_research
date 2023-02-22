import abc

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

from quant_finance_research.utils import get_example_df
from quant_finance_research.prediction.tscv import *


class DebugQuantTimeSplit_SeqHorizon:

    def debug_get_folder(self):
        df, df_column = get_example_df()
        fsplit = QuantTimeSplit_SeqHorizon(k=3)
        fsplit.split(df)
        train_df, test_df = fsplit.get_folder(df, folder_idx=2)
        print(train_df)
        print(test_df)
        print(train_df.values.shape)
        print(test_df.values.shape)


class DebugQuantTimeSplit_SeqPast:

    def debug_get_folder(self):
        df, df_column = get_example_df()
        fsplit = QuantTimeSplit_SeqPast(k=3)
        fsplit.split(df)
        train_df, test_df = fsplit.get_folder(df, folder_idx=2)
        print(train_df)
        print(test_df)
        print(train_df.values.shape)
        print(test_df.values.shape)


class DebugQuantTimeSplit_GroupTime:

    def debug_get_folder(self):
        df, df_column = get_example_df()
        fsplit = QuantTimeSplit_GroupTime(k=4)
        fsplit.split(df)
        train_df, test_df = fsplit.get_folder(df, folder_idx=3)
        print(train_df)
        print(test_df)
        print(train_df.values.shape)
        print(test_df.values.shape)


class DebugQuantTimeSplit_PurgeSeqHorizon:

    def debug_get_folder(self):
        df, df_column = get_example_df()
        fsplit = QuantTimeSplit_PurgeSeqHorizon(k=2, gap=1)
        fsplit.split(df)
        train_df, test_df = fsplit.get_folder(df, folder_idx=1)
        print(train_df)
        print(test_df)
        print(train_df.values.shape)
        print(test_df.values.shape)


class DebugQuantTimeSplit_PurgeSeqPast:

    def debug_get_folder(self):
        df, df_column = get_example_df()
        fsplit = QuantTimeSplit_PurgeSeqPast(k=2, gap=1)
        fsplit.split(df)
        train_df, test_df = fsplit.get_folder(df, folder_idx=1)
        print(train_df)
        print(test_df)
        print(train_df.values.shape)
        print(test_df.values.shape)


class DebugQuantTimeSplit_PurgeGroupTime:

    def debug_get_folder(self):
        df, df_column = get_example_df()
        fsplit = QuantTimeSplit_PurgeGroupTime(k=4, gap=1)
        fsplit.split(df)
        train_df, test_df = fsplit.get_folder(df, folder_idx=1)
        print(train_df)
        print(test_df)
        print(train_df.values.shape)
        print(test_df.values.shape)


class DebugQuantTimeSplit_PreprocessSplit:

    def __init__(self):
        self.base_spliter = QuantTimeSplit_SeqPast(k=3)
        from quant_finance_research.fe.fe_val import classic_x_preprocess_package_1
        df, df_column = get_example_df()
        self.spliter = QuantTimeSplit_PreprocessSplit(fsplit=self.base_spliter, df_column=df_column,
                                                      preprocess_func=classic_x_preprocess_package_1,
                                                      preprocess_column=df_column['x'])

    def debug_get_folder(self):
        df, df_column = get_example_df()
        print(f"k={self.spliter.get_k()}")
        self.spliter.split(df)
        for idx in tqdm(range(self.spliter.get_k())):
            self.spliter.get_folder(df, idx, mad_factor=0.5)
        train_df, test_df = self.spliter.get_folder(df, folder_idx=2)
        print(train_df)
        print(test_df)
        print(train_df.values.shape)
        print(test_df.values.shape)
        print("==========preprocess info================")
        print(self.spliter.get_preprocess_package())
        dff = self.spliter.get_folder_preprocess(df, folder_idx=0)
        print("=============External Preprocess===============")
        print(dff)


if __name__ == "__main__":
    # DebugQuantTimeSplit_SeqHorizon().debug_get_folder()
    # DebugQuantTimeSplit_SeqPast().debug_get_folder()
    # DebugQuantTimeSplit_GroupTime().debug_get_folder()
    # DebugQuantTimeSplit_PurgeSeqHorizon().debug_get_folder()
    # DebugQuantTimeSplit_PurgeSeqPast().debug_get_folder()
    # DebugQuantTimeSplit_PurgeGroupTime().debug_get_folder()
    DebugQuantTimeSplit_PreprocessSplit().debug_get_folder()
