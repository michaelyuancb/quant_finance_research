import numpy as np

from xgboost import XGBModel
from xgboost import XGBRegressor
from xgboost import XGBClassifier

"""
    For more thing about LightGBM, see the reference below.
    [1] Chen T, Guestrin C. Xgboost: A scalable tree boosting system[C]
    //Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. 2016: 785-794.
    [2] https://github.com/microsoft/LightGBM
    [3] https://xgboost.readthedocs.io/en/latest/index.html
    [4] https://xgboost.readthedocs.io/en/latest/python/python_api.html
"""

# objective:
# Objective candidate: survival:aft
# Objective candidate: binary:hinge
# Objective candidate: multi:softmax
# Objective candidate: multi:softprob
# Objective candidate: rank:pairwise
# Objective candidate: rank:ndcg
# Objective candidate: rank:map
# Objective candidate: survival:cox
# Objective candidate: reg:gamma
# Objective candidate: reg:tweedie
# Objective candidate: reg:squarederror
# Objective candidate: reg:squaredlogerror
# Objective candidate: reg:logistic
# Objective candidate: binary:logistic
# Objective candidate: binary:logitraw
# Objective candidate: reg:linear
# Objective candidate: reg:absoluteerror
# Objective candidate: reg:pseudohubererror
# Objective candidate: count:poisson


def get_XGBRegressor(
        booster='gbtree',
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=233,
        n_jobs=-1,
        max_leaves=0,
        subsample=1.0,
        reg_alpha=0.0,
        reg_lambda=0.005,
        **kwargs):
    my_xgbt = XGBRegressor(
        booster=booster,        # 'gbtree', 'gblinear', 'dart'. No need to change for most of the time.
        objective=objective,  # 'regression' or a custom objective callbacks. It is noteworthy that if
        # objective is not provided, the xgb will choose the loss function adaptively according to the data.
        n_estimators=n_estimators,  # Key parameter, Default=100.
        max_depth=max_depth,  # Key parameter, Default=-1
        learning_rate=learning_rate,  # Key parameter, Default=0.1.
        random_state=random_state,  # Key parameter, Default=None.
        n_jobs=n_jobs,  # Key parameter, number of parallel threads, Default=-1.
        max_leaves=max_leaves,    # default=0, which means no limit.
        subsample=subsample,  # Subsample ratio of the training instance, Default=1.0.
        reg_alpha=reg_alpha,  # L1 regularization term on weights, Default=0.0
        reg_lambda=reg_lambda,  # L2 regularization term on weights, Default=0.0
        **kwargs
    )
    return my_xgbt


def example_xgBoost_regressor():
    my_xgbt = XGBRegressor(
        booster='gbtree',        # 'gbtree', 'gblinear', 'dart'. No need to change for most of the time.
        objective='reg:squarederror',  # 'regression' or a custom objective callbacks. It is noteworthy that if
        # objective is not provided, the xgb will choose the loss function adaptively according to the data.
        n_estimators=100,  # Key parameter, Default=100.
        max_depth=5,  # Key parameter, Default=-1
        learning_rate=0.1,  # Key parameter, Default=0.1.
        random_state=233,  # Key parameter, Default=None.
        n_jobs=-1,  # Key parameter, number of parallel threads, Default=-1.
        max_leaves=0,    # default=0, which means no limit.
        subsample=1.0,  # Subsample ratio of the training instance, Default=1.0.
        reg_alpha=0.0,  # L1 regularization term on weights, Default=0.0
        reg_lambda=0.005,  # L2 regularization term on weights, Default=0.0
    )
    X = np.random.randn(50, 5)
    y = np.random.randn(50, 1)
    Xval = np.random.randn(20, 5)
    yval = np.random.randn(20, 1)

    print(X.shape)
    print(y.shape)
    print(X.ndim)
    # ===================================== Training =====================================
    judge = isinstance(my_xgbt, XGBModel)
    print(f"Judge=={judge}")

    # ===================================== Training =====================================
    verbose = 10                   # Key input, the epoch interval of print.
    early_stopping_rounds = 20     # Key input, the early_stop.
    my_xgbt.set_params(early_stopping_rounds=early_stopping_rounds,
                       eval_metric=None)
    my_xgbt.fit(X, y,
                eval_set=[(Xval, yval)],  # Key input, the validation set.
                verbose=verbose,
                sample_weight=None       # The sample weight.
                )

    # ===================================== Prediction =====================================
    Xtest = np.random.randn(30, 5)
    pred = my_xgbt.predict(Xtest)
    print(pred.shape)

    # ===================================== Saving =====================================
    param = my_xgbt.get_params()
    print(param)

    # ===================================== Loading =====================================
    my_xgbt.set_params(**param)

    print("success.")


if __name__ == "__main__":
    example_xgBoost_regressor()