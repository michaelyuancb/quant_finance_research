import pickle
import random
import pandas as pd
import numpy as np

import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def reduce_mem_usage_df(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
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


if __name__ == "__main__":
    x = np.array([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    z = torch.tensor([7, 8, 9]).to('cuda')
    print(transfer_numpy_cpu(x))
    print(transfer_numpy_cpu(y))
    print(transfer_numpy_cpu(z))
