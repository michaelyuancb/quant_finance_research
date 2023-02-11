
from quant_finance_research.tools.factor_eval import *


def find_GlobalAbsIcRank(data_df, x_column, y_column, number):
    """
        Find the feature ranked 1~number on the abc(IC) of the whole DataFrame,
    """
    x_name = data_df.columns[x_column].tolist()
    y_name = data_df.columns[y_column]

    corr = data_df[x_name + [y_name]].corr()[y_name].drop(labels=y_name).reset_index()
    corr[y_name] = abs(corr[y_name])
    corr = pd.concat([corr, pd.DataFrame(x_column)], axis=1)
    corr.sort_values(y_name, ascending=False, inplace=True)
    if number > corr.shape[0]:
        raise ValueError(f"The require mean-factor number={number} is larger than the factor number {corr.shape[0]}.")
    best_col = corr.iloc[:number, 2].to_list()
    best_col_name = corr.iloc[:number, 0].to_list()
    return best_col, best_col_name  # idx & name


def add_LocalTimeMeanFactor(data_df, fe_column, time_col='time_id',
                            dtype=np.float32):
    """
        Generate the mean-factor of each timestamps for fe_column. (mean of all investment for each time-stamps)
    """
    best_corr = data_df.columns[fe_column].tolist()
    df_mean = data_df[[time_col] + best_corr].groupby(time_col).mean().astype(dtype)
    mean_name = ['LocalTimeMean_' + x for x in best_corr]
    mean_name_dict = {best_corr[i]: mean_name[i] for i in range(len(mean_name))}
    df_mean = df_mean.rename(columns=mean_name_dict)
    data_df = pd.merge(data_df, df_mean, on=time_col, how='inner')
    del df_mean
    return data_df


def add_GlobalAbsIcRank_LocalTimeMeanFactor(data_df, x_column, y_column, number,
                                            time_col='time_id',
                                            dtype=np.float32  # for memory saving.
                                            ):
    """
        For the feature ranked 1~number on the abc(IC) of the whole DataFrame,
        Generate the mean-factor of each timestamps for those feature. (mean of all investment for each time-stamps)
    """
    best_col, best_col_name = find_GlobalAbsIcRank(data_df, x_column, y_column, number)
    data_df = add_LocalTimeMeanFactor(data_df, best_col, time_col=time_col, dtype=dtype)
    return data_df, best_col, best_col_name
