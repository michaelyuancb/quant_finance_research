import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_seq_setting(figsize=(10, 6), subplot=111,
                     xlabel='time', ylabel='value', label_font_size=None,
                     ax_rotation=0, ax_base=None, ax_font_size=None,
                     ay_rotation=0, ay_base=None, ay_font_size=None,
                     title="Default Title", title_font_size=None):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(subplot)
    plt.xlabel(xlabel, fontsize=label_font_size)
    plt.ylabel(ylabel, fontsize=label_font_size)
    plt.xticks(rotation=ax_rotation, fontsize=ax_font_size)
    plt.yticks(rotation=ay_rotation, fontsize=ay_font_size)
    if ax_base is not None:
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(base=ax_base))
    if ay_base is not None:
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(base=ay_base))
    plt.title(title, fontsize=title_font_size)
