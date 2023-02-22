import numpy as np
import pandas as pd


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