import numpy as np

from quant_finance_research.plot.plot_utils import *


def get_seq(n=100):
    x = np.random.randn(n)
    x = np.cumsum(x)
    y = ['some_date_time_' + str(i) for i in range(x.shape[0])]
    return y, x


def debug_plot_seq_setting():
    x, y = get_seq(n=100)
    plot_seq_setting(figsize=(6, 4), xlabel='time', ylabel='value', label_font_size=10,
                     ax_rotation=20, ax_base=20, ax_font_size=5,
                     ay_font_size=5,
                     title='Test Title', title_font_size=15
                     )
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    debug_plot_seq_setting()
