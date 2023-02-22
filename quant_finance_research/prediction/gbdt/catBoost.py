import copy

import numpy as np

from catboost import CatBoostRegressor
from catboost import CatBoostClassifier, CatBoost

"""
    For more thing about LightGBM, see the reference below.
    [1] Anna Veronika Dorogush, Andrey Gulin, Gleb Gusev, Nikita Kazeev, Liudmila Ostroumova Prokhorenkova, 
        Aleksandr Vorobev “Fighting biases with dynamic boosting”. arXiv:1706.09516, 2017
    [2] Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin,
        “CatBoost: gradient boosting with categorical features support”. Workshop on ML Systems at NIPS 2017
    [3] https://github.com/catboost/catboost
    [4] https://catboost.ai/
    [5] https://catboost.ai/en/docs/concepts/python-quickstart
    [6] https://catboost.ai/en/docs/concepts/python-reference_catboostregressor
    [7] https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier
"""


def get_CatboostRegressor(cat_column=None,
                          iterations=1000,
                          learning_rate=0.03,
                          depth=6,
                          random_seed=0,
                          thread_count=-1,
                          loss_function='RMSE',
                          nan_mode='Min',
                          l2_leaf_reg=3.0,
                          **kwargs):
    my_catb = CatBoostRegressor(
        cat_features=cat_column,  # Key parameter, the indices of categorical columns, Default=None,
        loss_function=loss_function,  # Supported by Catboost: RMSE, Logloss, MAE, CrossEntropy, Quantile,
        # LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE,Poisson.
        iterations=iterations,  # Key parameter, boosting num, Default=1000.
        learning_rate=learning_rate,  # Key parameter, Default=0.03,
        random_seed=random_seed,  # Key parameter, Default=None,
        depth=depth,  # Key parameter, range in [1, 16], Default=6，
        thread_count=thread_count,  # Key parameter, Default=-1.
        nan_mode=nan_mode,  # Supported Mode: Forbidden, Min, Ma. Default='Min'.
        l2_leaf_reg=l2_leaf_reg,  # L2 regularization term on weights, Default=3.0.
        allow_writing_files=False,
        **kwargs
    )
    return my_catb


def get_cat(n):
    lst = []
    for i in range(n):
        t = np.random.randint(0, 3)
        if t == 0:
            lst.append('A')
        elif t == 1:
            lst.append('B')
        else:
            lst.append('C')
    return np.array(lst)


def example_catBoost_regressor():
    cat_column = [5]
    my_catb = CatBoostRegressor(
        cat_features=cat_column,  # Key parameter, the indices of categorical columns, Default=None,
        loss_function='RMSE',  # Supported by Catboost: RMSE, Logloss, MAE, CrossEntropy, Quantile,
        # LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE,Poisson.
        iterations=1000,  # Key parameter, boosting num, Default=1000.
        learning_rate=0.03,  # Key parameter, Default=0.03,
        random_seed=233,  # Key parameter, Default=None,
        depth=6,  # Key parameter, range in [1, 16], Default=6，
        thread_count=-1,  # Key parameter, Default=-1.
        nan_mode='Min',  # Supported Mode: Forbidden, Min, Ma. Default='Min'.
        l2_leaf_reg=3.0,  # L2 regularization term on weights, Default=3.0.
        allow_writing_files=False
    )
    my_catb_copy = copy.deepcopy(my_catb)

    X = np.concatenate([np.random.randn(50, 5), get_cat(50).reshape(-1, 1)], axis=1)
    y = np.random.randn(50, 1)
    Xval = np.concatenate([np.random.randn(20, 5), get_cat(20).reshape(-1, 1)], axis=1)
    yval = np.random.randn(20, 1)

    print(X.shape)
    print(y.shape)
    print(X.ndim)
    print(X[:, cat_column[0]])
    print(type(X[0, cat_column[0]]))

    # ===================================== Training =====================================
    judge = isinstance(my_catb, CatBoost)
    print(f"Judge=={judge}")

    # ===================================== Training =====================================
    verbose = 10  # Key input, the epoch interval of print.
    early_stopping_rounds = 20  # Key input, the early_stop.
    my_catb.fit(X, y,
                eval_set=[(Xval, yval)],  # Key input, the validation set.
                verbose=verbose,
                early_stopping_rounds=early_stopping_rounds
                )

    # ===================================== Prediction =====================================
    Xtest = np.concatenate([np.random.randn(50, 5), get_cat(50).reshape(-1, 1)], axis=1)
    pred = my_catb.predict(Xtest)
    print(pred.shape)

    # ===================================== Saving =====================================
    param = my_catb.get_params()
    print(param)

    # ===================================== Loading =====================================
    my_catb_copy.set_params(**param)

    print("success.")


if __name__ == "__main__":
    # example_catBoost_regressor()
    catb = get_CatboostRegressor(cat_column=None, iterations=100, learning_rate=0.1, depth=5)
    print(catb)
