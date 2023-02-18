import pandas as pd
import numpy as np

from quant_finance_research.utils import seq_data_transform


def seq_cumsum(seq):
    return np.cumsum(seq_data_transform(seq))


def seq_cumprod(seq):
    return np.cumprod(seq_data_transform(seq))


def seq_nancumsum(seq):
    return np.nancumsum(seq_data_transform(seq))


def seq_nancumprod(seq):
    return np.nancumprod(seq_data_transform(seq))