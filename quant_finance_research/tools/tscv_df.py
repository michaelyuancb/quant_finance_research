import abc

import pandas as pd
import numpy as np

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

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        """
        :param df: the DataFrame to split
        :param k:
        :param inv_col:
        :param time_col:

        No return, but solve the split idx of DataFrame.
        """
        pass

    def get_folder(self, df, x_column, y_column, folder_idx):
        """
        :param df: the target DataFrame
        :param x_column: the column idx of X, list
        :param y_column: the column idx of y, list
        :param folder_idx: the idx of folder, 0 <= folder_idx < k
        :return: numpy, xtrain, ytrain, xval, yval
        """
        if len(self.split_idx) == 0:
            raise NotImplementedError("Please split the DataFrame through .split attribute first.")
        if self.k <= folder_idx:
            raise ValueError(f"The folder_idx={folder_idx} is out of range for k={self.k}.")
        train_index = self.split_idx[folder_idx][0]
        test_index = self.split_idx[folder_idx][1]
        xtrain = df.iloc[train_index, x_column].values
        ytrain = df.iloc[train_index, y_column].values
        xtest = df.iloc[test_index, x_column].values
        ytest = df.iloc[test_index, y_column].values
        return xtrain, ytrain, xtest, ytest

    def get_folder_idx(self, folder_idx):
        if self.k <= folder_idx:
            raise ValueError(f"The folder_idx={folder_idx} is out of range for k={self.k}.")
        return self.split_idx[folder_idx][0], self.split_idx[folder_idx][1]

    def _split_time_seq(self, df, kk, inv_col='investment_id', time_col='time_id'):
        time_idx = df[time_col].values
        time_idx = np.unique(time_idx).tolist()
        time_idx = list(set(time_idx))
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
        self.seq_time = seq_time

    def get_seq_time(self):
        return self.seq_time


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

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        self.clear()
        kk = self.k + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            idx_val = df[(df[time_col] >= self.seq_time[i + 1]) & (df[time_col] < self.seq_time[i + 2])].index
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

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        self.clear()
        kk = self.k + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[0]) & (df[time_col] < self.seq_time[i + 1])].index
            idx_val = df[(df[time_col] >= self.seq_time[i + 1]) & (df[time_col] < self.seq_time[i + 2])].index
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
        super(QuantTimeSplit_GroupTime, self).__init__(k)
        self.seq_time = []
        self.group_index = None

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        self.clear()
        kk = self.k
        self._split_time_seq(df, kk, inv_col, time_col)
        group_index_list = []
        for i in range(self.k):
            idx_v = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            group_index_list.append(idx_v)
        group_index = np.zeros(df.shape[0])
        for i in range(self.k):
            group_index[group_index_list[i]] = i
        self.group_index = group_index
        kf = GroupKFold(n_splits=self.k)
        self.split_idx = [sp for sp in kf.split(np.zeros((df.shape[0], 1)), groups=self.group_index)]

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

    def get_gap(self):
        return self.gap

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        self.clear()
        kk = self.k + self.gap + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            idx_val = df[(df[time_col] >= self.seq_time[i + 1 + self.gap]) & (
                    df[time_col] < self.seq_time[i + 2 + self.gap])].index
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

    def get_gap(self):
        return self.gap

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        self.clear()
        kk = self.k + self.gap + 1
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(self.k):
            idx_train = df[(df[time_col] >= self.seq_time[0]) & (df[time_col] < self.seq_time[i + 1])].index
            idx_val = df[(df[time_col] >= self.seq_time[i + 1 + self.gap]) & (
                    df[time_col] < self.seq_time[i + 2 + self.gap])].index
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

    def get_gap(self):
        return self.gap

    def split(self, df, inv_col='investment_id', time_col='time_id'):
        self.clear()
        kk = self.k
        self._split_time_seq(df, kk, inv_col, time_col)
        for i in range(0, self.k):
            idx_val = df[(df[time_col] >= self.seq_time[i]) & (df[time_col] < self.seq_time[i + 1])].index
            df_idx_train = None
            if i - self.gap - 1 >= 0:
                df_idx_train = (df[time_col] >= self.seq_time[0]) & (df[time_col] < self.seq_time[i - self.gap])
            if i + self.gap + 1 < self.k:
                df_tmp = (df[time_col] >= self.seq_time[i + self.gap + 1]) & (df[time_col] < self.seq_time[self.k])
                df_idx_train = df_tmp if df_idx_train is None else df_idx_train | df_tmp
            idx_train = df[df_idx_train].index
            self.split_idx.append((idx_train, idx_val))


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
    xtr, ytr, xvl, yvl = fsplit.get_folder(df, x_column, y_column, fidx)  # get the train & validation data, numpy

    fsplit.get_seq_time()  # get the split sequence on time-stamps-cluster.
    fsplit.get_k()
    fsplit.get_gap()
"""

if __name__ == "__main__":
    v = np.array([[0, 0, 1, 2], [0, 1, 3, 2], [2, 0, 4, 5], [1, 2, 3, 3], [1, 0, 0, 5], [2, 1, 0, 3],
                  [2, 3, 1, 1], [0, 3, 2, 1], [1, 3, 4, 0], [2, 3, 4, 1], [0, 4, 0, 1], [1, 4, 4, 1]])
    v[:, 0] = v[:, 1]
    df_t = pd.DataFrame(np.array(v),
                        columns=['time_id', 'investment_id', 'factor_0', 'factor_1'])
    print(df_t)
    print("------")
    # fsplit = QuantTimeSplit_SeqHorizon(k=3)
    # fsplit = QuantTimeSplit_SeqPast(k=3)
    # fsplit = QuantTimeSplit_GroupTime(k=4)
    # fsplit = QuantTimeSplit_PurgeSeqHorizon(k=2, gap=1)
    # fsplit = QuantTimeSplit_PurgeSeqPast(k=2, gap=1)
    fsplit = QuantTimeSplit_PurgeGroupTime(k=4, gap=1)
    # fsplit = QuantTimeGroupSplit(k=3)
    fsplit.split(df_t)
    fidx = 3
    print(fsplit.get_folder_idx(fidx))
    print(fsplit.get_seq_time())
    xtr, ytr, xvl, yvl = fsplit.get_folder(df_t, [2, 3], [3], fidx)
    print(xtr.shape)
    print(ytr.shape)
    print(xvl.shape)
    print(yvl.shape)
