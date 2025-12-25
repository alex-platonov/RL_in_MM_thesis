import numpy as np
import matplotlib.pyplot as plt


def plot_LOB(lob_data, k=3):
    """
    Plot a limit order book (LOB).
    
    Parameters
    ----------
    lob_data : np.array
        Numpy array containing the LOB data.
    k : int
        Number of outer price levels excluded from the plot. A value of k = 3 is typically suitable when the spread is small
        (≤ 2–3). If the spread is larger, k can be reduced. The minimum value is 0.


    Returns
    -------
    None
    """

    # Adjusting k. k < 0 fails, and k >= 5 makes the plot non-representable
    if k < 0 or k >= 5:
        k = 0

    ask = lob_data[0, 0]
    spread = -lob_data[1, 0]
    buy_volumes = lob_data[1, spread:]  # non-zero buys
    sell_volumes = lob_data[0, spread:]  # non-zero sells
    lob_volumes = np.concatenate(
        (np.flip(buy_volumes), np.zeros((spread - 1,)), sell_volumes)
    )
    lob_volumes = lob_volumes[k:-k]
    prices = np.array(
        range(ask - len(buy_volumes) - spread + 1, ask + len(sell_volumes))
    )
    prices = prices[k:-k]

    # Coloring the LOB
    color = np.repeat("b", len(lob_volumes))
    color[np.where(lob_volumes > 0)] = "r"

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xticks(ticks=prices)
    ax.grid(True, linewidth=1, linestyle="dashed", color="grey")
    ax.set_axisbelow(True)
    ax.bar(x=prices, height=lob_volumes, color=color)
    ax.set_title("LOB")

    fig.tight_layout()


if __name__ == "__main__":
    """
    Used for plotting test.
    
    """

    # number of price levels (on each side)
    d = 10

    # other data
    spread = 2  # should be in absolute terms
    ask = 100

    # lob data
    x0 = np.zeros((2, d + 1), dtype=int)
    x0[0, 0] = ask
    x0[1, 0] = -spread
    x0[0, spread] = 2
    x0[1, spread] = -2
    x0[0, spread + 1] = 3
    x0[1, spread + 1] = -3
    x0[0, spread + 2] = 5
    x0[1, spread + 2] = -5
    x0[0, spread + 3] = 3
    x0[1, spread + 3] = -3
    x0[0, spread + 4] = 2
    x0[1, spread + 4] = -2
    x0[0, spread + 5 :] = 1
    x0[1, spread + 5 :] = -1

    plot_LOB(x0)
