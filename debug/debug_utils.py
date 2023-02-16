from quant_finance_research.utils import *


def debug_set_seed():
    seed = 0
    set_seed(seed)
    print("succeed.")


def debug_get_example_df():
    df, df_column = get_example_df()
    print(df)
    print(df_column)


def def_get_example_large_df():
    df, df_column = get_example_large_df()
    print(df)
    print(df_column)


def debug_reduce_mem_usage_df():
    df, df_column = get_example_df()
    df2 = reduce_mem_usage_df(df, df_column['x'] + df_column['y'])
    print(df2)


def debug_transfer_numpy_cpu():
    x = np.array([1, 2, 3])
    y = torch.tensor([4, 5, 6])
    z = torch.tensor([7, 8, 9]).to('cuda')
    print(transfer_numpy_cpu(x))
    print(transfer_numpy_cpu(y))
    print(transfer_numpy_cpu(z))
    print(type(transfer_numpy_cpu(x)))
    print(type(transfer_numpy_cpu(y)))
    print(type(transfer_numpy_cpu(z)))


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
    # debug_set_seed()
    # debug_get_example_df()
    # def_get_example_large_df()
    # debug_reduce_mem_usage_df()
    # debug_transfer_numpy_cpu()
    # debug_get_numpy_from_df_train_val()