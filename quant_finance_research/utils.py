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


def get_example_df():
    v = np.array([[1, 0, 1, 2, 1], [2, 0, 3, 2, 3], [1, 1, 4, 5, 4], [3, 0, 3, 3, 3], [1, 2, 0, 5, 0],
                  [2, 1, 0, 3, 0],
                  [4, 2, 1, 1, 1], [4, 0, 2, 1, 2], [4, 3, 4, 0, 4], [4, 1, 4, 1, 4.1], [5, 1, 0, 1, 0],
                  [5, 3, 4, 1, 4]])
    v[:, 0] = v[:, 1]
    dt = [
        datetime.strptime("2022-11-1", "%Y-%m-%d"),
        datetime.strptime("2022-11-2", "%Y-%m-%d"),
        datetime.strptime("2022-11-1", "%Y-%m-%d"),
        datetime.strptime("2022-11-4", "%Y-%m-%d"),
        datetime.strptime("2022-11-1", "%Y-%m-%d"),
        datetime.strptime("2022-11-2", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-10", "%Y-%m-%d"),
        datetime.strptime("2022-11-10", "%Y-%m-%d")
    ]
    np.random.seed(0)
    weight = np.random.rand(12, 1)
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
    v = np.array([[1, 0, 1, 2, 1], [2, 0, 3, 2, 3], [1, 1, 4, 5, 4], [3, 0, 3, 3, 3], [1, 2, 0, 5, 0],
                  [2, 1, 0, 3, 0],
                  [4, 2, 1, 1, 1], [4, 0, 2, 1, 2], [4, 3, 4, 0, 4], [4, 1, 4, 1, 100], [5, 1, 0, 1, 0],
                  [5, 3, 4, 1, 3]])
    v = v[:, :2]
    dt = [
        datetime.strptime("2022-11-1", "%Y-%m-%d"),
        datetime.strptime("2022-11-2", "%Y-%m-%d"),
        datetime.strptime("2022-11-1", "%Y-%m-%d"),
        datetime.strptime("2022-11-4", "%Y-%m-%d"),
        datetime.strptime("2022-11-1", "%Y-%m-%d"),
        datetime.strptime("2022-11-2", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-9", "%Y-%m-%d"),
        datetime.strptime("2022-11-10", "%Y-%m-%d"),
        datetime.strptime("2022-11-10", "%Y-%m-%d")
    ]
    n = len(dt)
    extra_x = np.random.randn(n, 201)
    v = np.concatenate([v, extra_x], axis=1)
    np.random.seed(0)
    weight = np.random.rand(12, 1)
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


def generate_cv_result_df(cv_result, param_combination):
    # cv_result:  list of (np.float, np.array(k, ))
    column = list(param_combination[0].keys())
    param_np = []
    for dct in param_combination:
        param_np.append(list(dct.values()))
    param_np = np.array(param_np)
    column = column + ['mean'] + ['std']
    result_np = []
    for mn, rlist in cv_result:
        lst = [mn, np.std(rlist)] + np.round(rlist, 5).tolist()
        result_np.append(lst)
    column = column + ['cv_' + str(i) for i in range(rlist.shape[0])]
    result_np = np.concatenate([param_np, result_np], axis=1)
    cv_result_df = pd.DataFrame(result_np, columns=column)
    return cv_result_df


def get_numpy_from_df_train_val(train_df, val_df, df_column):
    x_column = df_column['x']
    y_column = df_column['y']
    xtrain = train_df.iloc[:, x_column].values
    ytrain = train_df.iloc[:, y_column].values
    xval = val_df.iloc[:, x_column].values
    yval = val_df.iloc[:, y_column].values
    return xtrain, ytrain, xval, yval


def copy_model_list(model, k):
    model_list = []
    for i in range(k):
        model_list.append(copy.deepcopy(model))
    return model_list


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    z = torch.tensor([7, 8, 9]).to('cuda')
    print(transfer_numpy_cpu(x))
    print(transfer_numpy_cpu(y))
    print(transfer_numpy_cpu(z))
