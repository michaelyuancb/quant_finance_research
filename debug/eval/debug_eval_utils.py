
from quant_finance_research.eval.eval_utils import *


def get_short_seq():
    seq = [1, 3, 2, 1, 4, 5]
    return seq


def debug_seq_cumsum():
    print(seq_cumsum(get_short_seq()))


def debug_seq_cumprod():
    print(seq_cumprod(get_short_seq()))


if __name__ == "__main__":
    debug_seq_cumsum()
    debug_seq_cumprod()