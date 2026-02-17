"""Small plotting helpers used in notebooks."""

import matplotlib.pyplot as plt


def quick_plot(x, y, title=None):
    plt.figure()
    plt.plot(x, y)
    if title:
        plt.title(title)
    plt.show()
