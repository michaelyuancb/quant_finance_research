import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FastICA

from quant_finance_research.fe.fe_feat import build_portfolio, delete_Feature
from quant_finance_research.fe.fe_utils import find_column_idx, update_df_column_package


def build_pca_portfolio(df_org, df_column, column, pca_n_components, pca_portfolio_weight=None,
                        pca_portfolio_inplace=False,
                        **kwargs):
    df = df_org.copy()
    data_val = df.iloc[:, column]
    if pca_portfolio_weight is None:
        pca_clf = PCA(n_components=pca_n_components)
        pca_clf.fit(data_val)
        pca_portfolio_weight = pca_clf.components_  # (n_comp, n_feat)
    df, pro_pack = build_portfolio(df, df_column, column, pca_portfolio_weight, "pca_portfolio_",
                                   inplace=pca_portfolio_inplace)
    return df, {"df_column_new": pro_pack['df_column_new'], "pca_portfolio_weight": pca_portfolio_weight,
                "pca_n_components": pca_n_components, "pca_portfolio_inplace": pca_portfolio_inplace,
                "new_column_idx": pro_pack['new_column_idx']}


def build_pca_cov_portfolio(df_org, df_column, column, pca_cov_n_components, pca_cov_portfolio_weight=None,
                            pca_cov_portfolio_inplace=False, **kwargs):
    df = df_org.copy()
    data_val = df.iloc[:, column]
    data_cov = data_val.cov()
    if pca_cov_portfolio_weight is None:
        pca_clf = PCA(n_components=pca_cov_n_components)
        pca_clf.fit(data_cov)
        pca_cov_portfolio_weight = pca_clf.components_  # (n_comp, n_feat)
    df, pro_pack = build_portfolio(df, df_column, column, pca_cov_portfolio_weight, "pca_cov_portfolio_",
                                   inplace=pca_cov_portfolio_inplace)
    return df, {"df_column_new": pro_pack['df_column_new'], "pca_cov_portfolio_weight": pca_cov_portfolio_weight,
                "pca_cov_n_components": pca_cov_n_components, "pca_cov_portfolio_inplace": pca_cov_portfolio_inplace,
                "new_column_idx": pro_pack['new_column_idx']}


def build_ica_portfolio(df_org, df_column, column, ica_n_components, ica_portfolio_weight=None,
                        ica_portfolio_inplace=False, **kwargs):
    df = df_org.copy()
    data_val = df.iloc[:, column]
    if ica_portfolio_weight is None:
        ica_clf = FastICA(n_components=ica_n_components)
        ica_clf.fit(data_val)
        ica_portfolio_weight = ica_clf.components_  # (n_comp, n_feat)
    df, pro_pack = build_portfolio(df, df_column, column, ica_portfolio_weight, "ica_portfolio_",
                                   inplace=ica_portfolio_inplace)
    return df, {"df_column_new": pro_pack['df_column_new'], "ica_portfolio_weight": ica_portfolio_weight,
                "ica_n_components": ica_n_components, "ica_portfolio_inplace": ica_portfolio_inplace,
                "new_column_idx": pro_pack['new_column_idx']}


def build_ica_cov_portfolio(df_org, df_column, column, ica_cov_n_components, ica_cov_portfolio_weight=None,
                            ica_cov_portfolio_inplace=False, **kwargs):
    df = df_org.copy()
    data_val = df.iloc[:, column]
    if ica_cov_portfolio_weight is None:
        ica_clf = FastICA(n_components=ica_cov_n_components)
        ica_clf.fit(data_val)
        ica_cov_portfolio_weight = ica_clf.components_  # (n_comp, n_feat)
    df, pro_pack = build_portfolio(df, df_column, column, ica_cov_portfolio_weight, "ica_cov_portfolio_",
                                   inplace=ica_cov_portfolio_inplace)
    return df, {"df_column_new": pro_pack['df_column_new'], "ica_cov_portfolio_weight": ica_cov_portfolio_weight,
                "ica_cov_n_components": ica_cov_n_components, "ica_cov_portfolio_inplace": ica_cov_portfolio_inplace,
                "new_column_idx": pro_pack['new_column_idx']}

