import copy
import pickle
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def set_df_continue_index(df):
    df.index = range(len(df))


def datetime2int(date):
    st = datetime.strftime(date, "%Y%m%d%H%M%S")
    return int(st)


def list_datetime2int(date_list):
    dint = [datetime2int(date) for date in date_list]
    return dint


def get_tmp_str():
    return "_@@QwQ##!"


def seq_data_transform(seq):
    if type(seq) is list:
        seq = np.array(seq)
    seq = seq.reshape(-1)
    return seq


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def save_pickle(pickle_file, data):
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(data, pfile)


def reduce_mem_usage_df(df, column):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns[column]:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def transfer_numpy_cpu(x):
    tp = str(type(x))
    if 'numpy.ndarray' in tp:
        return x
    elif 'torch.Tensor' in tp or 'torch.tensor' in tp:
        if 'cuda' in str(x.device):
            return x.detach().cpu().numpy()
        elif 'cpu' in str(x.device):
            return x.numpy()
    else:
        raise TypeError(f"Unknown type for {tp}")


def copy_model_list(model, k):
    model_list = []
    for i in range(k):
        model_list.append(copy.deepcopy(model))
    return model_list


def _get_example_vdt_base():
    v = np.array([[1, 0, 1, 2, 1], [2, 0, 3, 2, 3], [1, 1, 4, 5, 4], [3, 0, 3, 3, 3], [1, 2, 0, 5, 0],
                  [2, 1, 0, 3, 0],
                  [4, 2, 1, 1, 1], [4, 0, 2, 1, 2], [4, 3, 4, 0, 4], [4, 1, 4, 1, 4.1], [5, 1, 0, 1, 0],
                  [5, 3, 4, 1, 4]])
    v[:, 0] = v[:, 1]
    dt = [
        datetime.strptime("2022-11-1", "%Y-%m-%d"), datetime.strptime("2022-11-2", "%Y-%m-%d"),
        datetime.strptime("2022-11-1", "%Y-%m-%d"), datetime.strptime("2022-11-4", "%Y-%m-%d"),
        datetime.strptime("2022-11-1", "%Y-%m-%d"), datetime.strptime("2022-11-2", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"), datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"), datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-10", "%Y-%m-%d"), datetime.strptime("2022-11-10", "%Y-%m-%d")
    ]
    return v, dt


def get_example_df():
    v, dt = _get_example_vdt_base()
    np.random.seed(0)
    weight = np.array([0.1, 0.05, 0.05, 0.1, 0.2, 0.02, 0.08, 0.15, 0.05, 0.1, 0.04, 0.06]).reshape(-1, 1)
    dt = np.array(dt)
    df_t = pd.DataFrame(np.concatenate([v, weight], axis=1),
                        columns=['time_id', 'investment_id', 'factor_0', 'factor_1', 'target', 'weight'])
    df_t['investment_id'] = df_t['investment_id'].astype(np.int32)
    df_t.iloc[:, 0] = pd.Series(dt)
    x_column = [2, 3]
    y_column = [4]
    loss_column = [5]
    df_column = {"x": x_column, "y": y_column, "loss": loss_column}
    return df_t, df_column


def get_example_large_df():
    v, dt = _get_example_vdt_base()
    n = len(dt)
    extra_x = np.random.randn(n, 201)
    v = np.concatenate([v, extra_x], axis=1)
    np.random.seed(0)
    weight = np.array([0.1, 0.05, 0.05, 0.1, 0.2, 0.02, 0.08, 0.15, 0.05, 0.1, 0.04, 0.06]).reshape(-1, 1)
    cls = ['time_id', 'investment_id'] + ['factor_' + str(i) for i in range(200)] + ['target'] + ['weight']
    dt = np.array(dt)
    df_t = pd.DataFrame(np.concatenate([np.array(v), weight], axis=1),
                        columns=cls)
    df_t['investment_id'] = df_t['investment_id'].astype(np.int32)
    df_t.iloc[:, 0] = pd.Series(dt)
    x_column = [i for i in range(2, 202)]
    y_column = [202]
    loss_column = [203]
    df_column = {"x": x_column, "y": y_column, "loss": loss_column}
    return df_t, df_column


def get_example_cat_df():
    df, df_column = get_example_df()
    class_1 = np.array(["A", "B", "A", "C", "A", "B", "B", "D", "C", "C", "B", "C"])
    class_2 = np.array([1, 2, 1, 3, 1, 2, 2, 4, 3, 3, 2, 3])
    cls = df_column['x'] + [df.shape[1], df.shape[1]+1]
    df_column['x'] = cls
    dfx = np.array([class_1, class_2]).T
    dfx[:, 1] = dfx[:, 1].astype(np.int32)
    cls_pd = pd.DataFrame(dfx, columns=['cls_factor_1', 'cls_factor_2'])
    df = pd.concat([df, cls_pd], axis=1)
    df.iloc[:, -1] = df.iloc[:, -1].astype(np.int32)
    return df, df_column


def get_example_cat_matrix():
    def get_cat(n, ofs=0):
        lst = []
        for i in range(n):
            t = np.random.randint(0, 3)
            if t == 0:
                lst.append(ofs)
            elif t == 1:
                lst.append(ofs+1)
            else:
                lst.append(ofs+2)
        return np.array(lst).reshape(-1, 1)

    X = np.concatenate([np.random.randn(50, 3), get_cat(50, ofs=0),
                        np.random.randn(50, 1), get_cat(50, ofs=3), get_cat(50, ofs=6)], axis=1)
    y = np.random.randn(50, 1)

    return X, y, {'index_num': [0, 1, 2, 4], 'index_cat': [3, 5, 6]}

