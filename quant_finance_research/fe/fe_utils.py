
def find_column_idx(column_name, df_columns):
    # Find column_idx in O(n) Time Complexity.
    # Notice that the column_name appearance order should be the same with df_columns.
    df_idx = 0
    column_idx = []
    for j in range(len(column_name)):
        while df_columns[df_idx] != column_name[j]:
            df_idx = df_idx + 1
        column_idx.append(df_idx)
    return column_idx


def update_df_column_package(df_column, pro_package):
    if "df_column_new" in pro_package.keys():
        df_column = pro_package['df_column_new']
    return df_column


class PiplinePreprocess:

    def __init__(self):
        self.preprocess_function_list = []

    def __call__(self, data_df, df_column, column, **kwargs):
        pro_package = dict()
        column_name = data_df.columns[column]

        for func_tuple in self.preprocess_function_list:
            func_type, func = func_tuple
            if func_type == 'feat':
                data_df, package = func(data_df, df_column, column, **kwargs)
                df_column = update_df_column_package(df_column, package)
                column_name = [cname for cname in column_name if cname in data_df.columns]
                column = find_column_idx(column_name, data_df.columns)
            elif func_type == 'val':
                data_df, package = func(data_df, df_column, column, **kwargs)
            else:
                raise ValueError(f"Unsupported Preprocess Function Type: {func_type} for {func}.")
            pro_package.update(package)

        return data_df, pro_package

    def add_feat_preprocess(self, func):
        self.preprocess_function_list.append(('feat', func))

    def add_val_preprocess(self, func):
        self.preprocess_function_list.append(('val', func))
