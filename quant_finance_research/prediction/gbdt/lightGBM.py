import numpy as np

from lightgbm import LGBMModel
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from lightgbm import early_stopping, log_evaluation

"""
    For more thing about LightGBM, see the reference below.
    [1] Ke G, Meng Q, Finley T, et al. Lightgbm: A highly efficient gradient boosting decision tree[J]. 
        Advances in neural information processing systems, 2017, 30.
    [2] https://github.com/microsoft/LightGBM
    [3] https://lightgbm.readthedocs.io/en/v3.3.2/
    [4] https://lightgbm.readthedocs.io/en/v3.3.2/Python-API.html
    [5] https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/lightgbm.LGBMRegressor.html#lightgbm.LGBMRegressor
    [6] https://lightgbm.readthedocs.io/en/v3.3.2/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier
"""


def get_LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=233,
        n_jobs=-1,
        num_leaves=40,
        subsample=1.0,
        reg_alpha=0.0,
        reg_lambda=0.005,
        **kwargs):
    my_lgbm = LGBMRegressor(
        boosting_type=boosting_type,  # 'gdbt', 'dart', 'goss'. No need to change for most of the time.
        objective=objective,  # 'regression' or a custom objective callbacks.
        learning_rate=learning_rate,  # Key parameter, Default=0.1.
        max_depth=max_depth,  # Key parameter, Default=-1
        n_estimators=n_estimators,  # Key parameter, Default=100.
        random_state=random_state,  # Key parameter, Default=None.
        n_jobs=n_jobs,  # Key parameter, number of parallel threads, Default=-1.
        num_leaves=num_leaves,  # default=31.
        subsample=subsample,  # Subsample ratio of the training instance, Default=1.0.
        reg_alpha=reg_alpha,  # L1 regularization term on weights, Default=0.0
        reg_lambda=reg_lambda,  # L2 regularization term on weights, Default=0.0
        **kwargs
    )
    return my_lgbm


def example_lightGBM_regressor():
    my_lgbm = LGBMRegressor(
        boosting_type='gbdt',  # 'gdbt', 'dart', 'goss'. No need to change for most of the time.
        objective='regression',  # 'regression' or a custom objective callbacks.
        learning_rate=0.1,  # Key parameter, Default=0.1.
        max_depth=5,  # Key parameter, Default=-1
        n_estimators=100,  # Key parameter, Default=100.
        random_state=233,  # Key parameter, Default=None.
        n_jobs=-1,  # Key parameter, number of parallel threads, Default=-1.
        num_leaves=40,  # default=31.
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
    judge = isinstance(my_lgbm, LGBMModel)
    print(f"Judge=={judge}")

    # ===================================== Training =====================================
    verbose = 10                   # Key input, the epoch interval of print.
    early_stopping_rounds = 20     # Key input, the early_stop.
    callbacks = [log_evaluation(period=verbose), early_stopping(stopping_rounds=early_stopping_rounds)]
    my_lgbm.fit(X, y,
                eval_set=[(Xval, yval)],  # Key input, the validation set.
                callbacks=callbacks,
                eval_metric='l2',       # Evaluate metric, 'l2' custom eval_metric callbacks.
                eval_sample_weight=None,  # The sample weight.
                )

    # ===================================== Prediction =====================================
    Xtest = np.random.randn(30, 5)
    pred = my_lgbm.predict(Xtest)
    print(pred.shape)

    # ===================================== Saving =====================================
    param = my_lgbm.get_params()
    print(param)

    # ===================================== Loading =====================================
    my_lgbm.set_params(**param)

    print("success.")


if __name__ == "__main__":
    example_lightGBM_regressor()