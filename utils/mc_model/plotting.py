import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from environments.mc_model.mc_environment import MonteCarloEnv
from collections import defaultdict
import pickle


def heatmap_Q(Q_tab, file_path=None, skip_T=False):
    """
    Generate a heatmap from Q_tab.
    
    Parameters
    ----------
    Q_tab : dict
        Dictionary containing values for all state–action pairs.
    file_path : str
        Path to where the output files are saved.
    skip_T : bool
        Flag indicating whether the final time step is excluded.

    Returns
    -------
    None
    """

    optimal_bid = dict()
    optimal_ask = dict()
    optimal_MO = dict()

    plt.figure()
    for state in list(Q_tab.keys()):
        optimal_action = np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
        optimal_bid[state] = optimal_action[0] + 1
        optimal_ask[state] = optimal_action[1] + 1
        optimal_MO[state] = optimal_action[2]

    ser = pd.Series(
        list(optimal_bid.values()), index=pd.MultiIndex.from_tuples(optimal_bid.keys())
    )
    df = ser.unstack().fillna(0)
    if skip_T:
        df = df.iloc[:, :-1]
    fig = sns.heatmap(df)
    fig.set_title("Optimal bid depth")
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

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
    if skip_T:
        df = df.iloc[:, :-1]
    fig = sns.heatmap(df)
    fig.set_title("Optimal ask depth")
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_ask_heat")
        plt.close()

    plt.figure()
    ser = pd.Series(
        list(optimal_MO.values()), index=pd.MultiIndex.from_tuples(optimal_MO.keys())
    )
    df = ser.unstack().fillna(0)
    if skip_T:
        df = df.iloc[:, :-1]
    fig = sns.heatmap(df)
    fig.set_title("Market order")
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_mo_heat")
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
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

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
        title = "Number of unique of optimal actions."
        vmin = 1
        for state in list(Q_mean.keys()):
            opt_action_array = []

            for Q_tab in Q_tables:
                opt_action = np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
                opt_action_array.append(opt_action)

            n_unique_opt_actions = len(set(opt_action_array))

            Q_n_errors[state] = n_unique_opt_actions

    else:
        # ----- CALCULATE THE NUMBER ERRORS COMPARED TO MEAN OPTIMAL -----
        title = "Number of actions not agreeing with mean optimal action."
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
    fig.set_xlabel("t (grouped)")
    fig.set_ylabel("inventory (grouped)")

    if file_path == None:
        plt.show()

    else:
        if n_unique:
            plt.savefig(file_path + "n_unique_opt_actions")
            plt.close()
        else:
            plt.savefig(file_path + "n_errors_compared_to_mean")
            plt.close()


def show_Q(Q_tab, file_path=None):
    """
    Plot the optimal depths implied by Q_tab.
    
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

    for state in list(Q_tab.keys()):
        optimal_action = np.array(
            np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
        )
        [optimal_bid[state], optimal_ask[state]] = optimal_action[0:2] + 1

    ser = pd.Series(
        list(optimal_bid.values()), index=pd.MultiIndex.from_tuples(optimal_bid.keys())
    )
    df = ser.unstack()
    df = df.T
    df.columns = "q=" + df.columns.map(str)
    df.plot.line(title="Optimal bid depth", style=".-")
    plt.legend(loc="upper right")
    plt.xlabel("t (grouped)")
    plt.ylabel("depth")
    plt.xticks(np.arange(1, df.shape[0] + 1))

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
    plt.xlabel("t (grouped)")
    plt.ylabel("depth")
    plt.xticks(np.arange(1, df.shape[0] + 1))

    if file_path == None:
        plt.show()
    else:
        plt.savefig(file_path + "opt_ask_strategy")
        plt.close()


def load_Q(filename, default=True):
    """
    Load a Q table from a pkl file.    
    
Parameter    s
--------    --
filename :     str
    File n    ame.
default :     bool
    Flag indicating whether a defaultdict or a standard dict is ret    u    rned.
    
Returns    
-------    
Q : dict
    A defaultdict or dict containing all Q tables. The keys are actions and the values are the correspondin    g Q tables.    
args : dict
    Mode    l parame    ters.
n : int
    Number of episodes for which Q-le    arning was run.    
rewards : list
    Rewards saved during training.
    """

    # Load the file
    file = open("Q_tables/" + filename + ".pkl", "rb")
    Q_raw, args, n, rewards = pickle.load(file)

    # If defaultdict isn't needed, just return a dict.
    if not default:
        return Q_raw

    # Find d
    dim = Q_raw[(0, 0)].shape[0]

    # Transform to a default_dict.
    Q_loaded = defaultdict(lambda: np.zeros((dim, dim)))
    Q_loaded.update(Q_raw)

    return Q_loaded, args, n, rewards
