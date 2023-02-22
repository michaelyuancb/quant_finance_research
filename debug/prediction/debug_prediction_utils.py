from quant_finance_research.prediction.prediction_utils import *
from quant_finance_research.utils import *


def debug_generate_cv_result_df():
    cv_result = [np.array([0.8, 0.5, 0.3]), np.array([0.8, 0.6, 0.25]), np.array([0.7, 0.5, 0.4])]
    for i in range(len(cv_result)):
        cv_result[i] = (np.mean(cv_result[i]), cv_result[i])
    param_combination = [{"lr": 0.1, "pn": 3}, {"lr": 0.1, "pn": 4}, {"lr": 0.01, "pn": 3}]
    cv_df = generate_cv_result_df(cv_result, param_combination)
    print(cv_df)


def debug_get_numpy_from_df_train_val():
    df, df_column = get_example_df()
    xtrain, ytrain, xval, yval = get_numpy_from_df_train_val(df, df, df_column)
    print(xtrain.shape)
    print(ytrain.shape)
    print(xval.shape)
    print(yval.shape)


if __name__ == "__main__":
    debug_generate_cv_result_df()
    debug_get_numpy_from_df_train_val()