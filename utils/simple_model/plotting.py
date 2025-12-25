import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from environments.simple_model.simple_model_mm import SimpleEnv
from collections import defaultdict
import pickle


def plot_optimal_depth(D, bid=True, discrete=True):
    """
    Plot the optimal depths implied by D.
    
    Parameters
    ----------
    D : np.array
        Array of shape (2Q+1, T+1) containing the optimal depths for all inventory levels q across all time steps.
    bid : bool
        Flag indicating whether the data corresponds to bid (True) or ask (False).
    discrete : bool
        Flag indicating whether the depths are rounded to discrete tick levels.
    Returns
    -------
    None
    """

    LO_type = "bid" if bid else "ask"

    n_levels = D.shape[0]

    if discrete:
        D = D[:, 0 : (D.shape[1] - 1)]

    plt.figure()
    for level in range(n_levels):
        plt.plot(
            D[level],
            ("-o" if discrete else "-"),
            label="q = " + str(int(level + (1 - n_levels) / 2)),
        )

    plt.title(
        "Optimal ("
        + ("discrete" if discrete else "continuous")
        + ") "
        + LO_type
        + " depths as a function of t"
    )
    plt.ylabel(LO_type + " depth")
    plt.xlabel("time (t)")
    plt.ylim([-0.001, 0.021])
    plt.yticks(np.arange(3) * 0.010)
    if discrete and D.shape[1] < 10:
        plt.xticks(np.arange(0, D.shape[1]))

    plt.legend()
    plt.show()


def generate_optimal_depth(T=30, Q=3, dp=0.01, phi=1e-5, bid=True, discrete=True):
    """
    Generate the optimal depths for either the bid or the ask side.
    
    Parameters
    ----------
    T : int
        Episode length.
    Q : int
        Maximum absolute inventory that can be held.
    dp : float
        Tick size.
    phi : float
        Running inventory penalty parameter.
    bid : bool
        Flag indicating whether bid-side (True) or ask-side (False) depths are generated.
    discrete : bool
        Flag indicating whether depths are rounded to discrete tick levels.
    
    Returns
    -------
    data : np.array
        Array of shape (2Q+1, T+1) containing optimal depths for all inventory levels q across all time steps.
    """

    env = SimpleEnv(T, Q=Q, dp=dp, phi=phi)

    data = []

    q_s = np.arange(start=-env.Q, stop=env.Q + 1)

    for q in q_s:
        data_q = []
        for t in range(T + 1):
            env.t = t
            env.Q_t = q
            if discrete:
                depth = env.transform_action(env.discrete_analytically_optimal())[
                    1 - bid
                ] * (1 - 2 * bid)
            else:
                depth = env.calc_analytically_optimal()[1 - bid]

            data_q.append(depth)

        data.append(data_q)

    data = np.array(data)

    return data


def heatmap_Q(Q_tab, file_path=None):
    """
    Generate a heatmap from Q_tab.
    
    Parameters
    ----------
    Q_tab : dict
        Dictionary containing values for all state–action pairs.
    file_path : str
        Path to where the output files are saved.

    Returns
    -------
    None
    """

    optimal_bid = dict()
    optimal_ask = dict()

    plt.figure()
    for state in list(Q_tab.keys()):
        optimal_action = np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
        optimal_bid[state] = optimal_action[0]
        optimal_ask[state] = optimal_action[1]

    for state in list(Q_tab.keys()):
        if state[0] == 3:
            optimal_bid.pop(state, None)
        if state[0] == -3:
            optimal_ask.pop(state, None)

    ser = pd.Series(
        list(optimal_bid.values()), index=pd.MultiIndex.from_tuples(optimal_bid.keys())
    )
    df = ser.unstack().fillna(0)
    fig = sns.heatmap(df, vmin=0, vmax=3)
    fig.set_title("Optimal bid depth")
    fig.set_xlabel("time (t)")
    fig.set_ylabel("inventory (q)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_bid_heat")
        plt.close()

    plt.figure()
    ser = pd.Series(
        list(optimal_ask.values()), index=pd.MultiIndex.from_tuples(optimal_ask.keys())
    )
    df = ser.unstack().fillna(0)
    fig = sns.heatmap(df, vmin=0, vmax=3)
    fig.set_title("Optimal ask depth")
    fig.set_xlabel("time (t)")
    fig.set_ylabel("inventory (q)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_ask_heat")
        plt.close()


def heatmap_Q_std(Q_std, file_path=None):
    """
    Plot a heatmap of the standard deviation of the Q-value associated with the optimal action.
    
    Parameters
    ----------
    Q_std : defaultdict
        Defaultdict with states as keys and standard deviations as values.
    file_path : str
        Path to where the output files are saved.

    Returns
    -------
    None
    """

    plt.figure()

    ser = pd.Series(list(Q_std.values()), index=pd.MultiIndex.from_tuples(Q_std.keys()))
    df = ser.unstack().fillna(0)
    fig = sns.heatmap(df)
    fig.set_title("Standard deviation of optimal actions")
    fig.set_xlabel("time (t)")
    fig.set_ylabel("inventory (q)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "heatmap_of_std")
        plt.close()


def heatmap_Q_n_errors(Q_mean, Q_tables, n_unique=True, file_path=None):
    """
    Plot a heatmap of differences in optimal actions across runs, showing either the number of unique actions or the number
    of actions that disagree with the mean-optimal action.
    
    Parameters
    ----------
    Q_mean : defaultdict
        Defaultdict with states as keys and mean Q-values as values.
    Q_tables : list
        List of defaultdicts with states as keys and Q-values as values.
    n_unique : bool
        Flag indicating whether the number of unique actions is plotted. If False, disagreements with the mean-optimal
        action are plotted instead.
    file_path : str
        Path to where the output files are saved.

    Returns
    -------
    None
    """

    Q_n_errors = Q_mean

    if n_unique:
        # ----- CALCULATE THE NUMBER OF UNIQUE ACTIONS -----
        title = "Number of unique of optimal actions"
        vmin = 1
        for state in list(Q_mean.keys()):
            opt_action_array = []

            for Q_tab in Q_tables:
                opt_action = np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
                opt_action_array.append(opt_action)

            n_unique_opt_actions = len(set(opt_action_array))

            Q_n_errors[state] = n_unique_opt_actions

    else:
        # ----- CALCULATE THE NUMBER ERROS COMPARED TO MEAN OPTIMAL -----
        title = "Number of actions not agreeing with mean optimal action"
        vmin = 0
        for state in list(Q_mean.keys()):
            num_errors = 0

            for Q_tab in Q_tables:
                error = np.unravel_index(
                    Q_tab[state].argmax(), Q_tab[state].shape
                ) != np.unravel_index(Q_mean[state].argmax(), Q_mean[state].shape)
                num_errors += error

            Q_n_errors[state] = num_errors

    plt.figure()

    ser = pd.Series(
        list(Q_n_errors.values()), index=pd.MultiIndex.from_tuples(Q_n_errors.keys())
    )
    df = ser.unstack().fillna(0)
    fig = sns.heatmap(df, vmin=vmin, vmax=len(Q_tables))
    fig.set_title(title)
    fig.set_xlabel("time (t)")
    fig.set_ylabel("inventory (q)")

    if file_path == None:
        plt.show()

    else:
        if n_unique:
            plt.savefig(file_path + "n_unique_opt_actions")
            plt.close()
        else:
            plt.savefig(file_path + "n_errors_compared_to_mean")
            plt.close()


def remove_last_t(Q_tab, T=5):
    """
    Remove all entries at time t = T from a defaultdict.
    
    Parameters
    ----------
    Q_tab : defaultdict
        Defaultdict with states as keys and Q-values as values.
    T : int
        Time index for which values are removed.
    
    Returns
    -------
    Q_tab : defaultdict
        Defaultdict with states as keys and Q-values as values.
    """

    for state in list(Q_tab.keys()):
        if state[1] == T:
            Q_tab.pop(state, None)

    return Q_tab


def Q_table_to_array(Q_tab, env):
    """
    Compute the optimal depth for each state and store the results in two arrays, one for bids and one for asks.
    
    Parameters
    ----------
    Q_tab : defaultdict
        Defaultdict with states as keys and Q-values as values.
    env : object
        Environment used to simulate the market.
    
    Returns
    -------
    array_bid : np.array
        Array containing optimal bid depths.
    array_ask : np.array
        Array containing optimal ask depths.
    """

    optimal_bid = dict()
    optimal_ask = dict()

    for state in list(Q_tab.keys()):
        optimal_action = np.array(
            np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
        )
        [optimal_bid[state], optimal_ask[state]] = (
            optimal_action + env.min_dp
        ) * env.dp

    for state in list(Q_tab.keys()):
        if state[0] == 3:
            optimal_bid[state] = np.inf
        if state[0] == -3:
            optimal_ask[state] = np.inf

    # ===== BID =====

    ser = pd.Series(
        list(optimal_bid.values()), index=pd.MultiIndex.from_tuples(optimal_bid.keys())
    )
    df = ser.unstack()

    df = df.to_numpy()

    array_bid = df[0 : (df.shape[0] - 1), 0 : (df.shape[1] - 1)]

    # ===== ASK =====
    ser = pd.Series(
        list(optimal_ask.values()), index=pd.MultiIndex.from_tuples(optimal_ask.keys())
    )
    df = ser.unstack()

    df = df.to_numpy()

    array_ask = df[1 : (df.shape[0]), 0 : (df.shape[1] - 1)]

    return array_bid, array_ask


def show_Q(Q_tab, env, file_path=None):
    """
    Plot the optimal depths implied by Q_tab.
    
    Parameters
    ----------
    Q_tab : dict
        Dictionary containing values for all state–action pairs.
    env : object
        Environment instance used during training.
    file_path : str
        Path to where the output files are saved.

    Returns
    -------
    None
    """

    optimal_bid = dict()
    optimal_ask = dict()

    for state in list(Q_tab.keys()):
        optimal_action = np.array(
            np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
        )
        [optimal_bid[state], optimal_ask[state]] = (
            optimal_action + env.min_dp
        ) * env.dp

    for state in list(Q_tab.keys()):
        if state[0] == 3:
            optimal_bid[state] = np.inf
        if state[0] == -3:
            optimal_ask[state] = np.inf

    ser = pd.Series(
        list(optimal_bid.values()), index=pd.MultiIndex.from_tuples(optimal_bid.keys())
    )
    df = ser.unstack()
    df = df.T
    df.columns = "q=" + df.columns.map(str)
    df.plot.line(title="Optimal bid depth", style=".-")
    plt.legend(loc="upper right")
    plt.xlabel("time (t)")
    plt.ylabel("depth")
    plt.xticks(np.arange(df.shape[0]))
    plt.yticks(np.arange(3) * 0.010)

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_bid_strategy")
        plt.close()

    ser = pd.Series(
        list(optimal_ask.values()), index=pd.MultiIndex.from_tuples(optimal_ask.keys())
    )
    df = ser.unstack()
    df = df.T
    df.columns = "q=" + df.columns.map(str)
    df.plot.line(title="Optimal ask depth", style=".-")
    plt.legend(loc="upper right")
    plt.xlabel("time (t)")
    plt.ylabel("depth")
    plt.xticks(np.arange(df.shape[0]))
    plt.yticks(np.arange(3) * 0.010)

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_ask_strategy")
        plt.close()
