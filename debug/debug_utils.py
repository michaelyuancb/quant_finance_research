from quant_finance_research.utils import *


def debug_set_seed():
    seed = 0
    set_seed(seed)
    print("succeed.")


def debug_set_df_continue_index():
    df, df_column = get_example_df()
    df.index = [6, 5, 4, 3, 2, 1] + [0, 0, 0, 0, 0, 0]
    print(df)
    set_df_continue_index(df)
    print(df)


def debug_get_example_df():
    df, df_column = get_example_df()
    print(df)
    print(df_column)
    print(df.iloc[:, df_column['loss']].values.sum())


def debug_get_example_large_df():
    df, df_column = get_example_large_df()
    print(df)
    print(df_column)
    print(df.iloc[:, df_column['loss']].values.sum())


def debug_get_example_cat_df():
    df, df_column = get_example_cat_df()
    print(df)
    print(df_column)
    print(df.iloc[:, df_column['loss']].values.sum())


def debug_get_example_cat_matrix():
    x, y, idx = get_example_cat_matrix()
    print(x)
    print(x[:, idx['index_cat']])
    print(x.shape)
    print(y.shape)
    print(idx)


def debug_datetime2int():
    dtt = datetime.strptime("2022-02-17 11:11:05", "%Y-%m-%d %H:%M:%S")
    print(dtt)
    print(datetime2int(dtt))


def debug_list_datetime2int():
    dtt = datetime.strptime("2022-02-17 11:11:05", "%Y-%m-%d %H:%M:%S")
    dtt2 = datetime.strptime("2022-02-17 11:12:08", "%Y-%m-%d %H:%M:%S")
    dt_list = [dtt, dtt2]
    print(dt_list)
    print(list_datetime2int(dt_list))


def debug_seq_data_transform():
    seq = [1, 3, 2, 5, 3, 2]
    print(seq_data_transform(seq))
    print(type(seq_data_transform(seq)))
    seq = np.array(seq)
    print(seq_data_transform(seq))
    print(type(seq_data_transform(seq)))
    seq = seq.reshape(2, 3)
    print(seq_data_transform(seq))
    print(type(seq_data_transform(seq)))


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


if __name__ == "__main__":
    # debug_set_seed()
    # debug_set_df_continue_index()
    # debug_datetime2int()
    # debug_list_datetime2int()
    # debug_seq_data_transform()
    # debug_reduce_mem_usage_df()
    # debug_transfer_numpy_cpu()
    # debug_get_example_df()
    # debug_get_example_large_df()
    # debug_get_example_cat_df()
    debug_get_example_cat_matrix()