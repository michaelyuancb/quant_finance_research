from quant_finance_research.utils import get_example_df
from quant_finance_research.fe.fe_val import *


def debug_mad_preprocess():
    df, df_column = get_example_df()
    df_val = df.copy()
    df, package = mad_preprocess(df, df_column, df_column['x'], mad_factor=0.5)  # df, {"mad_bound": (x_upper, x_lower)}
    print(package)
    print(df)
    df_val, _ = mad_preprocess(df_val, df_column, df_column['x'], **package)
    assert (df_val.values == df.values).all()
    print("success")


def debug_zscore_preprocess():
    df, df_column = get_example_df()
    df_val = df.copy()
    df, package = zscore_preprocess(df, df_column, df_column['x'])  # df, {"zscore_trans_scaler": zscore_trans_scaler}
    print(package)
    print(df)
    df_val, _ = zscore_preprocess(df_val, df_column, df_column['x'], **package)
    assert (df_val.values == df.values).all()
    print("success")


def debug_zscore_inverse_preprocess():
    df, df_column = get_example_df()
    df_org = df.copy()
    df, package = zscore_preprocess(df, df_column, df_column['x'])  # df, {"zscore_rev_scaler": zscore_rev_scaler}
    dft = df.copy()
    scaler = package['zscore_trans_scaler']
    df, package = zscore_inverse_preprocess(df, df_column, df_column['x'], zscore_rev_scaler=scaler)
    assert (df.values == df_org.values).all()
    df2, _ = zscore_inverse_preprocess(dft, df_column, df_column['x'], **package)
    assert (df.values == df2.values).all()
    print("success")


def debug_fillna_fixval_preprocess():
    df, df_column = get_example_df()
    df.iloc[0, 2] = np.nan
    df.iloc[1, 3] = np.nan
    df_val = df.copy()
    df, package = fillna_fixval_preprocess(df, df_column, df_column['x'], fillna_val=1000)  # df, {"fillna_val": fillna_val}
    print(package)
    print(df)
    df_val, _ = fillna_fixval_preprocess(df_val, df_column, df_column['x'], **package)
    assert (df_val.values == df.values).all()
    print("success")


def debug_classic_x_preprocess_package_1():
    df, df_column = get_example_df()
    df.iloc[0, 2] = np.nan
    df.iloc[1, 3] = np.nan
    df_val = df.copy()
    df, package = classic_x_preprocess_package_1(df, df_column, df_column['x'], fillna_val=1000, mad_factor=0.5)
    print(package)
    print(df)
    df_val, _ = classic_x_preprocess_package_1(df_val, df_column, df_column['x'], **package)
    assert (df_val.values == df.values).all()
    print("success")


if __name__ == "__main__":
    debug_mad_preprocess()
    debug_zscore_preprocess()
    debug_zscore_inverse_preprocess()
    debug_fillna_fixval_preprocess()
    debug_classic_x_preprocess_package_1()
