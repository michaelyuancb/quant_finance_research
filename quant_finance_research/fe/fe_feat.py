from quant_finance_research.eval.factor_eval import *
from quant_finance_research.fe.fe_utils import *


def delete_Feature(df_org, df_column, column, **kwargs):
    """
        Delete the Feature of some column.
    """
    df = df_org.copy()
    col_name = df.columns
    del_name = col_name[column]
    x_column, y_column, loss_column = df_column.values()
    save_x_idx = [i for i in x_column if i not in column]
    save_y_idx = [i for i in y_column if i not in column]
    save_x_name = col_name[save_x_idx]
    save_y_name = col_name[save_y_idx]
    save_l_idx = [i for i in loss_column if i not in column]
    save_l_name = col_name[save_l_idx]
    new_df = df.drop(del_name, axis=1)

    x_column = find_column_idx(save_x_name, new_df.columns)
    y_column = find_column_idx(save_y_name, new_df.columns)
    loss_column = find_column_idx(save_l_name, new_df.columns)
    df_column_new = {"x": x_column, "y": y_column, "loss": loss_column}
    return new_df, {"df_column_new": df_column_new}


# Delete NaN
def delete_NanRatio_Feature(df_org, df_column, column, del_nan_ratio, del_idx=None, **kwargs):
    """
        Delete the Feature with NaN-Ratio > threshold.
    """
    df = df_org.copy()
    if del_idx is None and del_nan_ratio is None:
        raise ValueError("At least one of del_x_idx and del_nan_ratio should not be None.")
    elif del_idx is not None:
        pass
    else:
        col_null_ratio = df.iloc[:, column].isnull().sum(axis=0).values / df.shape[0]
        del_idx = np.where(col_null_ratio > del_nan_ratio)[0].tolist()
        del_idx = [column[i] for i in del_idx]
    new_df, pro_package = delete_Feature(df, df_column, del_idx)
    pro_package.update({"del_idx": del_idx, "del_nan_ratio": del_nan_ratio})

    return new_df, pro_package


def find_GlobalAbsIcRank(df_org, df_column, column, number_GAIR, column_GAIR=None, **kwargs):
    """
        Find the feature ranked 1~number on the abc(IC) of the column,
        The IC is calculated with df_column['y'],
        when len(df_column['y'])>0, the rank is according to the average performance.
    """
    data_df = df_org.copy()
    x_column, y_column, loss_column = df_column.values()
    if column_GAIR is None:
        column_name = data_df.columns[column].tolist()
        y_name = data_df.columns[y_column].tolist()
        corr = data_df[column_name + y_name].corr()[y_name].drop(labels=y_name).reset_index()
        corr[y_name] = abs(corr[y_name])
        corr = pd.concat([corr, pd.DataFrame(x_column, index=corr.index)], axis=1)
        corr['GlobalAbsIcScore'] = corr[y_name].apply(lambda x: x.mean(), axis=1)
        corr.sort_values(by='GlobalAbsIcScore', ascending=False, inplace=True)
        if number_GAIR > corr.shape[0]:
            raise ValueError(
                f"The required mean-factor number={number_GAIR} is larger than the factor number {corr.shape[0]}.")
        column_GAIR = corr.iloc[:number_GAIR, -2].to_list()
    return data_df, {"df_column_new": df_column, "column_GAIR": column_GAIR, "number_GAIR": number_GAIR}


def add_LocalTimeMeanFactor(df_org, df_column, column, time_col='time_id', **kwargs):
    """
        Generate the mean-factor of each timestamps for column. (mean of all investment for each time-stamps)
    """
    data_df = df_org.copy()
    x_column, y_column, loss_column = df_column.values()
    width_1 = data_df.shape[1]
    best_corr = data_df.columns[column].tolist()
    df_mean = data_df[[time_col] + best_corr].groupby(time_col).mean()
    mean_name = ['LocalTimeMean_' + x for x in best_corr]
    mean_name_dict = {best_corr[i]: mean_name[i] for i in range(len(mean_name))}
    df_mean = df_mean.rename(columns=mean_name_dict)
    data_df = pd.merge(data_df, df_mean, on=time_col, how='inner')
    width_2 = data_df.shape[1]
    new_column_idx = [i for i in range(width_1, width_2)]
    x_column = x_column + new_column_idx
    df_column_new = {"x": x_column, "y": y_column, "loss": loss_column}
    del df_mean
    return data_df, {"df_column_new": df_column_new, "time_col": time_col, "new_column_idx": new_column_idx}


def add_GlobalAbsIcRank_LocalTimeMeanFactor(df_org, df_column, column, number_GAIR,
                                            time_col='time_id',
                                            column_GAIR=None, **kwargs
                                            ):
    """
        For the feature ranked 1~number on the abc(IC) of the column.
        The IC is calculated with df_column['y'],
        when len(df_column['y'])>0, the rank is according to the average performance.
        Generate the mean-factor of each timestamps for those feature. (mean of all investment for each time-stamps)
    """
    data_df = df_org.copy()
    _, pro_package = find_GlobalAbsIcRank(data_df, df_column, column, number_GAIR, column_GAIR=column_GAIR)
    column_GAIR = pro_package['column_GAIR']
    data_df, pro_package_2 = add_LocalTimeMeanFactor(data_df, df_column, column_GAIR, time_col=time_col)
    pro_package.update(pro_package_2)
    return data_df, pro_package


def build_portfolio(df_org, df_column, column, build_portfolio_weight, name_prefix="default_portfolio_",
                    inplace=False, **kwargs):
    """
    Input
        ::input df_org.iloc[:, column].values: the [N, M] array, where N is the number of bar,
                                                                       M is the number of features.
        ::input portfolio_weight: the [NC, M] array, where NC is the number of portfolio.
        ::input name_prefix: the name of the portfolio, end with '_'
        ::input inplace: if True, delete the original column, and update the meta  column list. Default=False.
    Output
        ::output df: a new dataframe with new portfolio in the last NC column.
        ::output df_column_new: match the df.
    """

    def _inplace_portfolio(df, df_column, column, new_column_idx):
        new_column_name = df.columns[new_column_idx]
        df, pro = delete_Feature(df, df_column, column)
        new_column_idx = find_column_idx(new_column_name, df.columns)
        return df, pro['df_column_new'], new_column_idx

    df = df_org.copy()
    if not name_prefix.endswith('_'):
        raise ValueError(f"The name_prefix should end with '_', but now name_prefix={name_prefix}")
    assert len(column) == build_portfolio_weight.shape[1]
    data_val = np.dot(df.iloc[:, column].values, build_portfolio_weight.T).reshape(-1, build_portfolio_weight.shape[0])
    n_component = data_val.shape[1]
    portfolio_name = [name_prefix + str(i) for i in range(n_component)]
    portfolio_df = pd.DataFrame(data_val, columns=portfolio_name, index=df.index)
    n_org = df.shape[1]
    df = pd.concat([df, portfolio_df], axis=1)
    new_column_idx = [n_org+i for i in range(data_val.shape[1])]
    df_column_new = {"x": df_column['x'] + new_column_idx, "y": df_column['y'],
                     "loss": df_column['loss']}
    if inplace:
        df, df_column_new, new_column_idx = _inplace_portfolio(df, df_column_new, column, new_column_idx)
    return df, {"df_column_new": df_column_new, "build_portfolio_weight": build_portfolio_weight,
                "name_prefix": name_prefix, "new_column_idx": new_column_idx, "inplace": inplace}