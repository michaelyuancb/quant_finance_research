import copy

from quant_finance_research.fe.fe_portfolio import *
from quant_finance_research.fe.fe_utils import *
from quant_finance_research.utils import *


def debug_build_pca_portfolio():
    df, df_column = get_example_df()
    df_val = copy.deepcopy(df)
    print(df)
    df, pro = build_pca_portfolio(df, df_column, df_column['x'], pca_n_components=1,
                                  pca_portfolio_inplace=True)
    print(df)
    print(pro)
    df_val, _ = build_pca_portfolio(df_val, df_column, df_column['x'], **pro)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_build_pca_cov_portfolio():
    df, df_column = get_example_df()
    df_val = copy.deepcopy(df)
    print(df)
    df, pro = build_pca_cov_portfolio(df, df_column, df_column['x'], pca_cov_n_components='mle',
                                      pca_cov_portfolio_inplace=True)
    print(df)
    print(pro)
    df_val, _ = build_pca_cov_portfolio(df_val, df_column, df_column['x'], **pro)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_build_ica_portfolio():
    df, df_column = get_example_df()
    df_val = copy.deepcopy(df)
    print(df)
    df, pro = build_ica_portfolio(df, df_column, df_column['x'], ica_n_components=1, ica_portfolio_inplace=True)
    print(df)
    print(pro)
    df_val, _ = build_ica_portfolio(df_val, df_column, df_column['x'], **pro)
    assert (df_val.values == df.values).all()
    print("success.")


def debug_build_ica_cov_portfolio():
    df, df_column = get_example_df()
    df_val = copy.deepcopy(df)
    print(df)
    df, pro = build_ica_cov_portfolio(df, df_column, df_column['x'], ica_cov_n_components=1,
                                      ica_cov_portfolio_inplace=True)
    print(df)
    print(pro)
    df_val, _ = build_ica_cov_portfolio(df_val, df_column, df_column['x'], **pro)
    assert (df_val.values == df.values).all()
    print("success.")


if __name__ == "__main__":
    debug_build_pca_portfolio()
    # debug_build_pca_cov_portfolio()
    # debug_build_ica_portfolio()
    # debug_build_ica_cov_portfolio()