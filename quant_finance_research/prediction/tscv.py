import abc

import pandas as pd
import numpy as np
import sys
from datetime import datetime

from sklearn.model_selection import GroupKFold


class QuantSplit_Base(abc.ABC):
    """
        The input DataFrame:
        1. one column to be the investment_id
        2. one column to be the time_id
        3. some column to be the input / factor
        4. some column to be the prediction target
    """

    def __init__(self, k):
        """
        :param k: the split number of DataFrame
        """
        self.k = k
        self.seq_time = []
        self.split_idx = []

    def get_k(self):
        return self.k

    def clear(self):
        self.split_idx = []

    def reset(self, k):
        self.split_idx = []
        self.k = k

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        """
        :param df: the DataFrame to split
        :param k:
        :param inv_col:
        :param time_col:

        No return, but solve the split idx of DataFrame.
        """
        pass

    def get_folder(self, df, folder_idx, **kwargs):
        """
        :param df: the target DataFrame
        :param folder_idx: the idx of folder, 0 <= folder_idx < k
        :return: train_df, test_df: DataFrame with split.
        """
        if len(self.split_idx) == 0:
            raise NotImplementedError("Please split the DataFrame through .split attribute first.")
        if self.k <= folder_idx:
            raise ValueError(f"The folder_idx={folder_idx} is out of range for k={self.k}.")
        train_index = self.split_idx[folder_idx][0]
        test_index = self.split_idx[folder_idx][1]
        train_df = df.loc[train_index]
        test_df = df.loc[test_index]
        return train_df, test_df

    def get_folder_idx(self, folder_idx, **kwargs):
        if self.k <= folder_idx:
            raise ValueError(f"The folder_idx={folder_idx} is out of range for k={self.k}.")
        return self.split_idx[folder_idx][0], self.split_idx[folder_idx][1]

    def _split_time_seq(self, df, kk, inv_col='investment_id', time_col='time_id', **kwargs):
        time_idx = df[time_col].values
        time_idx = np.sort(np.unique(time_idx))
        n_time_idx = len(time_idx)
        seq_time = [n_time_idx // kk] * kk
        if seq_time[0] == 0:
            raise ValueError(f"The DataFrame is to small for {self.k}-seq time split folder.")
        rest = n_time_idx - (n_time_idx // kk) * kk
        for j in range(rest):
            seq_time[kk - j - 1] = seq_time[kk - j - 1] + 1
        seq_time = [0] + seq_time
        for j in range(kk):
            seq_time[j + 1] = seq_time[j + 1] + seq_time[j]
        self.seq_time = [time_idx[seq_time[i]] for i in range(0, len(seq_time) - 1)] + ['+inf_time']
        # Notice that the last item (seq_time[kk], the (kk+1)th) is a string, which can not be compared with others.

    def get_seq_time(self):
        return self.seq_time


class QuantTimeSplit_NormalTimeCut(QuantSplit_Base):
    """
        Split Time Series with Sequence Validation (Same Horizon):
        val_ratio = [0.5, 0.25, 0.25]
        [A B C D E F G H I | J K L] ===> (A B C D E F G H I), (J K L)
    """

    def __init__(self, val_ratio, purge=0):
        """
        :param val_ratio: the ratio of validation set.
        """
        k = 1
        super(QuantTimeSplit_NormalTimeCut, self).__init__(k)
        self.val_ratio = val_ratio
        self.purge = purge

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        self.clear()
        time_idx = df[time_col].values
        time_idx = np.sort(np.unique(time_idx))
        n_time_idx = len(time_idx)
        n_train = int(n_time_idx * (1.0 - self.val_ratio))
        if n_train <= self.purge:
            raise ValueError(f"n_train={n_train}, purge={self.purge} ; it should be n_train > purge")
        n_test = n_time_idx - n_train
        print(f"Total TimeStampNum={n_time_idx}, Train TimeStampNum={n_train}, Val TimeStampNum={n_test}")
        self.seq_time = [time_idx[0], time_idx[n_train]] + ['+inf_time']
        idx_train = df[df[time_col] < time_idx[n_train - self.purge]].index
        idx_val = df[df[time_col] >= self.seq_time[1]].index
        self.split_idx.append((idx_train, idx_val))


class QuantTimeSplit_SeqHorizon(QuantSplit_Base):
    """
        Split Time Series with Sequence Validation (Same Horizon):
        k = 3
        [A B C | D E F | G H I | J K L] ==> ([A B C] [D E F]) ,
                                            ([D E F] [G H I]) ,
                                            ([G H I] [J K L])
    """

    def __init__(self, k):
        """
        :param k: the number of folder
        """
        super(QuantTimeSplit_SeqHorizon, self).__init__(k)

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        self.clear()
        kk = self.k + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            if i != self.k - 1:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1]) & (df[time_col] < self.seq_time[i + 2])].index
            else:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1])].index
            self.split_idx.append((idx_train, idx_val))


class QuantTimeSplit_SeqPast(QuantSplit_Base):
    """
        Split Time Series with Sequence Validation (All Past Data):
        k = 3
        [A B C | D E F | G H I | J K L] ==> ([A B C] [D E F]) ,
                                            ([A B C D E F] [G H I]) ,
                                            ([A B C D E F G H I] [J K L])
    """

    def __init__(self, k):
        """
        :param k: the number of folder
        """
        super(QuantTimeSplit_SeqPast, self).__init__(k)

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        self.clear()
        kk = self.k + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[0]) & (df[time_col] < self.seq_time[i + 1])].index
            if i != self.k - 1:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1]) & (df[time_col] < self.seq_time[i + 2])].index
            else:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1])].index
            self.split_idx.append((idx_train, idx_val))


class QuantTimeSplit_GroupTime(QuantSplit_Base):
    """
        Split Time Series with Group Split on Time Dimension:
        k = 4
        [A B C | D E F | G H I | J K L] ==> ([A B C G H I J K L] [D E F]) ,
                                            ([A B C D E F J K L] [G H I]) ,
                                            ([A B C D E F G H I] [J K L]) ,
                                            ([D E F G H I J K L] [A B C]),
    """

    def __init__(self, k):
        """
        :param k: the number of folder
        """
        assert k > 1
        super(QuantTimeSplit_GroupTime, self).__init__(k)
        self.seq_time = []
        self.group_index = None

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        self.clear()
        kk = self.k
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(0, self.k):
            if i != self.k - 1:
                idx_val = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            else:
                idx_val = df[(df[time_col] >= self.seq_time[i])].index
            df_idx_train = None
            if i - 1 >= 0:
                df_idx_train = df[time_col] < self.seq_time[i]
            if i + 1 < self.k:
                df_tmp = df[time_col] >= self.seq_time[i + 1]
                df_idx_train = df_tmp if df_idx_train is None else df_idx_train | df_tmp
            idx_train = df[df_idx_train].index
            self.split_idx.append((idx_train, idx_val))

    def get_group_index(self):
        return self.group_index


class QuantTimeSplit_PurgeSeqHorizon(QuantSplit_Base):
    """
        Split Time Series with Sequence Validation (Same Horizon), while gap is between train & validation:
        k = 2, gap = 1
        [A B C | D E F | G H I | J K L] ==> ([A B C] [G H I]) , ([D E F] [J K L])
    """

    def __init__(self, k, gap):
        """
        :param k: the number of folder
        :param gap: the number of gap block, which means the sequence will be divided into k+1+gap block.
        """
        super(QuantTimeSplit_PurgeSeqHorizon, self).__init__(k)
        self.gap = gap
        if gap < 1:
            raise ValueError(f"The gap should be larger than 0, but now get {gap}.")

    def get_gap(self):
        return self.gap

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        self.clear()
        kk = self.k + self.gap + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            if i != self.k - 1:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1 + self.gap]) & (
                        df[time_col] < self.seq_time[i + 2 + self.gap])].index
            else:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1 + self.gap])].index
            self.split_idx.append((idx_train, idx_val))


class QuantTimeSplit_PurgeSeqPast(QuantSplit_Base):
    """
        Split Time Series with Sequence Validation (All Past Data), while gap is between train & validation:
        k = 2, gap = 1
        [A B C | D E F | G H I | J K L] ==> ([A B C] [G H I]) ,
                                            ([A B C D E F] [J K L])
    """

    def __init__(self, k, gap):
        """
        :param k: the number of folder
        :param gap: the number of gap block, which means the sequence will be divided into k+1+gap block.
        """
        super(QuantTimeSplit_PurgeSeqPast, self).__init__(k)
        self.gap = gap
        if gap < 1:
            raise ValueError(f"The gap should be larger than 0, but now get {gap}.")

    def get_gap(self):
        return self.gap

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        self.clear()
        kk = self.k + self.gap + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[0]) & (df[time_col] < self.seq_time[i + 1])].index
            if i != self.k - 1:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1 + self.gap]) & (
                        df[time_col] < self.seq_time[i + 2 + self.gap])].index
            else:
                idx_val = df[(df[time_col] >= self.seq_time[i + 1 + self.gap])].index
            self.split_idx.append((idx_train, idx_val))


class QuantTimeSplit_PurgeGroupTime(QuantSplit_Base):
    """
        Split Time Series with Group Split on Time Dimension , while gap is between train & validation:
        k = 2, gap = 1
        [A B C | D E F | G H I | J K L] ==> ([G H I J K L] [A B C]) ,
                                            ([J K L] [D E F]) ,
                                            ([A B C] [G H I]) ,
                                            ([A B C D E F] [J K L]) ,
    """

    def __init__(self, k, gap):
        """
        :param k: the number of folder
        :param gap: the number of gap block, the sequence will be divided into k block.
                                             and the train & validation interval is at least gap block.
        """
        super(QuantTimeSplit_PurgeGroupTime, self).__init__(k)
        self.gap = gap
        if 2 * self.gap + 1 >= self.k:
            # at least one block for training folder.
            raise ValueError("The gap is to big for split, need 2*gap+1 < k.")
        if gap < 1:
            raise ValueError(f"The gap should be larger than 0, but now get {gap}.")

    def get_gap(self):
        return self.gap

    def split(self, df, inv_col='investment_id', time_col='time_id', **kwargs):
        self.clear()
        kk = self.k
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(0, self.k):
            if i != self.k - 1:
                idx_val = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            else:
                idx_val = df[(df[time_col] >= self.seq_time[i])].index
            df_idx_train = None
            if i - self.gap - 1 >= 0:
                df_idx_train = (df[time_col] >= self.seq_time[0]) & (df[time_col] < self.seq_time[i - self.gap])
            if i + self.gap + 1 < self.k:
                df_tmp = (df[time_col] >= self.seq_time[i + self.gap + 1])
                df_idx_train = df_tmp if df_idx_train is None else df_idx_train | df_tmp
            idx_train = df[df_idx_train].index
            self.split_idx.append((idx_train, idx_val))


class QuantTimeSplit_PreprocessSplit:

    def __init__(self, fsplit, df_column, preprocess_func, preprocess_column):
        """
        :param fsplit: The base QuantTimeSplit Class
        :param df_column: the (x,y,l) dict for DataFrame to be processed.
        :param preprocess_func: a preprocess function with param(train_df, test_df, x_column, y_column, **kwargs).
                                see example for more things.
               The function should be a standard preprocess form with:
               Input: func(df, self.preprocess_column, **{other_parameters})
               Return: (df, pro_package).
               You can use the function in fe_val directly, or build your own pipeline for more complex preprocess.
        :param preprocess_column: the column to be preprocessed, list
        """
        self.fsplit = fsplit
        self.preprocess_func = preprocess_func
        self.preprocess_package = dict()
        self.preprocess_column = preprocess_column
        self.df_column = df_column

    def get_k(self):
        return self.fsplit.get_k()

    def clear(self):
        self.fsplit.clear()
        self.preprocess_package = dict()

    def reset(self, k):
        self.fsplit.reset(k)
        self.preprocess_package = dict()

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        self.clear()
        self.fsplit.split(df, inv_col=inv_col, time_col=time_col)

    def get_folder(self, df, folder_idx, verbose=False, **kwargs):
        """
        Process Some Transformation before get the value of train_df & val_df
        :param df:  the target DataFrame
        :param folder_idx: the idx of folder, 0 <= folder_idx < k
        :return:
        """
        train_df, val_df = self.fsplit.get_folder(df, folder_idx)
        package_name = 'pro_package_folder_' + str(folder_idx)
        if package_name not in self.get_preprocess_package().keys():
            if verbose:
                print(f"Generate PreProcessor For Folder-{folder_idx}.")
            train_df, pro_package = self.preprocess_func(train_df, self.df_column, self.preprocess_column, **kwargs)
            self.preprocess_package[package_name] = pro_package
        else:
            pro_package = self.get_folder_preprocess_package(folder_idx)
            train_df, _ = self.preprocess_func(train_df, self.df_column, self.preprocess_column,
                                               **pro_package)
        # print(pro_package)
        val_df, _ = self.preprocess_func(val_df, self.df_column, self.preprocess_column,
                                         **pro_package)
        return train_df, val_df

    def get_folder_preprocess_package(self, folder_idx):
        """
            Get the pro_package for Fold-idx.  (PreprocessPackage-idx)
            :param folder_idx:
            :return:
        """
        if 'pro_package_folder_' + str(folder_idx) not in self.preprocess_package.keys():
            raise ValueError(f"The Pro_Package_Folder_{folder_idx} has not been generated.")
        return self.preprocess_package['pro_package_folder_' + str(folder_idx)]

    def get_folder_preprocess(self, df, folder_idx):
        """
            Preprocess the DataFrame with the PreprocessPackage-idx
        """
        if 'pro_package_folder_' + str(folder_idx) not in self.preprocess_package.keys():
            raise ValueError(f"The Pro_Package_Folder_{folder_idx} has not been generated.")
        df, package = self.preprocess_func(df, self.df_column, self.preprocess_column,
                                           **self.get_folder_preprocess_package(folder_idx))
        return df, package

    def get_preprocess_column(self):
        return self.preprocess_column

    def get_preprocess_func(self):
        return self.preprocess_func

    def get_preprocess_package(self):
        """
           Get the whole PreprocessPackage-idx
        """
        return self.preprocess_package


class QuantTimeSplit_RollingPredict:
    """
        A Spliter for Rolling Prediction. Given the insample_df and outsample_df, cut the predict process
        to k step, in each step TrainingSet={insample_df, outsample_df_train}, TestSet={outsample_df_test}.
        The Split of outsample_df using the QuantTimeSplit_SeqPast(k-1). The first folder for rolling prediction
        is TrainingSet={insample_df}, TestSet={outsample_df_test[1/k]}
    """
    def __init__(self, k):
        if k > 1:
            self.spliter_test = QuantTimeSplit_SeqPast(k=k-1)
        self.k = k

    def get_k(self):
        return self.k

    def clear(self):
        self.spliter_test.clear()

    def reset(self, k):
        self.spliter_test.reset(k)

    def split(self, insample, outsample, inv_col='investment_id', time_col='time_id'):
        if self.k > 1:
            self.clear()
            self.spliter_test.split(outsample, inv_col=inv_col, time_col=time_col)
        else:
            pass

    def get_folder(self, insample, outsample, folder_idx):
        if self.k == 1:
            return insample, outsample
        if folder_idx == 0:
            outsample_pred, _ = self.spliter_test.get_folder(outsample, folder_idx=0)
            return insample, outsample_pred
        else:
            train_df, outsample_pred = self.spliter_test.get_folder(outsample, folder_idx=folder_idx-1)
            train_df = pd.concat([insample, train_df], axis=0)
            return train_df, outsample_pred


"""
    Introduction of QuantTimeSplit For DataFrame.

    The Input DataFrame:
                investment_id  time_id factor[0:m1] target[0:m2]
        idx_0                     |
        idx_1                     |
        ...                       |
        idx_n                     |
                            validation split

    # fsplit = QuantTimeSplit_SeqHorizon(k=3)   # sequence with fixed horizon(folder), k+1 split-block
    # fsplit = QuantTimeSplit_SeqPast(k=3)      # sequence with past-folder, k+1 split-block
    # fsplit = QuantTimeSplit_GroupTime(k=4)    # group split on time dimension with k folder, k split-block
    # fsplit = QuantTimeSplit_PurgeSeqHorizon(k=2, gap=1)   # sequence with fixed horizon and gap, k+1+gap split-block
    # fsplit = QuantTimeSplit_PurgeSeqPast(k=2, gap=1)   # sequence with past-folder and gap, k+1+gap split-block
    # fsplit = QuantTimeSplit_PurgeGroupTime(k=4, gap=1)   # group split on time with k folder and gap, k split-block

    API:
    fsplit.split(df, inv_col, time_col)  # split the data (get the index) with time columns
    fsplit.get_folder_idx(folder_idx)  # get the dataframe-idx of one folder, return (index_train, index_val)
    train_df, test_df = fsplit.get_folder(df, x_column, y_column, fidx)  # get the train & validation data, DataFrame

    fsplit.get_seq_time()  # get the split sequence on time-stamps-cluster.
    fsplit.get_k()
    fsplit.get_gap()
    
    For More Complex Usage, please see debug-py file.
"""

if __name__ == "__main__":
    pass
