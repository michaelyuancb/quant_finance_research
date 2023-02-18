from sklearn.preprocessing import StandardScaler
from quant_finance_research.utils import get_tmp_str
import pandas as pd
import numpy as np

"""
The standard preprocess Function:
    Input: func(df_org, column, other parameters, **kwargs)
    Output: df, package.
            where pacakge is a dict with **kwargs-form for other parameter for replicate the preprocess. 
            (for later val_process & test_process)
            
    The first usage: data, pro_package = func(df_org, column, other parameters)
    The later usage: data, _ = func(df, column, **pro_package)
"""


# MAD-X
def mad_preprocess(df_org, df_column, column, mad_bound=None, mad_factor=3, **kwargs):
    """
        Process the MAD (Mean Absolute Deviation) Tail Shrink for feature.
        ::param mad_factor: the tail range should be mad_factor * mad
        ::param mad_bound: the tuple with (x_lower, x_upper), where both of them are 1D-numpy with shape (len(column),)
        Notice that one of mad_factor and mad_bound should not be None.
        If both mad_factor and mad_bound is not None, then mad_bound is adopted for usage.
    """
    df = df_org.copy()
    if mad_bound is None and mad_factor is None:
        raise ValueError("At least one of mad_factor and mad_bound should not be None.")
    elif mad_bound is not None:
        assert type(mad_bound) is tuple
        x_lower, x_upper = mad_bound
    else:
        x_median = df.iloc[:, column].median(axis=0)
        x_mad = 1.483 * (df.iloc[:, column] - x_median).abs().median(axis=0)
        x_upper = x_mad + x_median * mad_factor
        x_lower = x_mad - x_median * mad_factor
        x_upper = x_upper.values
        x_lower = x_lower.values

    for idx, x in enumerate(column):
        upper = x_upper[idx]
        df.iloc[:, x] = df.iloc[:, x].apply(lambda v: upper if v > upper else v)
        lower = x_lower[idx]
        df.iloc[:, x] = df.iloc[:, x].apply(lambda v: lower if v < lower else v)

    return df, {"mad_bound": (x_lower, x_upper), "mad_factor": mad_factor}


# ZScore
def zscore_normalize_preprocess(df_org, df_column, column, zscore_trans_scaler=None, **kwargs):
    """
        Process the ZScore for feature with sklearn StandardNorm scaler.
    """
    df = df_org.copy()
    if zscore_trans_scaler is None:
        zscore_trans_scaler = StandardScaler()
        zscore_trans_scaler.fit(df.iloc[:, column])
    df.iloc[:, column] = zscore_trans_scaler.transform(df.iloc[:, column])
    return df, {"zscore_trans_scaler": zscore_trans_scaler}


# Inverse Zscore
def zscore_normalize_inverse_preprocess(df_org, df_column, column, zscore_rev_scaler, **kwargs):
    """
        Process the Inverse-ZScore for feature with sklearn StandardNorm scaler.
    """
    df = df_org.copy()
    df.iloc[:, column] = zscore_rev_scaler.inverse_transform(df.iloc[:, column])
    return df, {"zscore_rev_scaler": zscore_rev_scaler}


def sumone_normalize_preprocess(df_org, df_column, column, sumone_denominator=None, **kwargs):
    """
        Redistribute the value of column to make the sum of each column equal to one.
    """
    df = df_org.copy()
    if sumone_denominator is None:
        sumone_denominator = df.iloc[:, column].sum()
    df.iloc[:, column] = df.iloc[:, column] / sumone_denominator.apply(lambda x: 1.0 if x == 0 else x)
    return df, {"sumone_denominator": sumone_denominator}


def panel_time_sumone_normalize_preprocess(df_org, df_column, column, time_col="time_id", **kwargs):
    df = df_org.copy()
    dfg = df[[time_col]+df.columns[column].tolist()].groupby(time_col).sum()
    org_name = dfg.columns.tolist()
    trf_name = {rname: rname+get_tmp_str() for rname in org_name}
    dfg = dfg.rename(columns=trf_name)
    df = pd.merge(df, dfg, how='left', left_on=time_col, right_on=time_col)
    del dfg
    for org_name, tname in trf_name.items():
        df[org_name] = df[org_name] / df[tname].apply(lambda x: 1.0 if x == 0 else x)
    for tname in trf_name.values():
        del df[tname]
    return df, {"time_col": time_col}


# Fillna-FixValue
def fillna_fixval_preprocess(df_org, df_column, column, fillna_val=0, **kwargs):
    """
        Fill NaN for feature with Fixed Value
    """
    df = df_org.copy()
    df.iloc[:, column] = df.iloc[:, column].fillna(fillna_val)
    return df, {"fillna_val": fillna_val}


def classic_x_preprocess_package_1(df_org, df_column, column, mad_factor, fillna_val=0,
                                   mad_bound=None,
                                   zscore_trans_scaler=None,
                                   **kwargs):
    """
        One Classic Preprocess Combination for Features.
        MAD Tail Shrink ---> ZScore ---> Fillna with 0.
        ::param mad_factor: the tail range should be mad_factor * mad
        ::param mad_bound: the tuple with (x_lower, x_upper), where both of them are 1D-numpy with shape (len(column),)
        Notice that one of mad_factor and bound should not be None. Other situation is illegal.
    """
    df = df_org.copy()
    df, package = mad_preprocess(df, df_column, column, mad_bound=mad_bound, mad_factor=mad_factor)
    df, package_2 = zscore_normalize_preprocess(df, df_column, column, zscore_trans_scaler=zscore_trans_scaler)
    package.update(package_2)
    df, package_3 = fillna_fixval_preprocess(df, df_column, column, fillna_val=fillna_val)
    package.update(package_3)
    return df, package   # {"mad_bound", "zscore_trans_scaler", "fillna_val"}
