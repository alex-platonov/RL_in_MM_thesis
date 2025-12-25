from simple_model_mm_q_learning import (
    load_Q,
    show_Q,
    Q_learning_multiple,
    fetch_table_name,
    heatmap_Q,
)
from environments.simple_model.simple_model_mm import SimpleEnv
from utils.simple_model.plotting import (
    Q_table_to_array,
    generate_optimal_depth,
    heatmap_Q_n_errors,
    heatmap_Q_std,
    heatmap_Q_n_errors,
    remove_last_t,
)
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import defaultdict


def evaluate_Q_matrix(
    matrix_path, n, folder_mode=False, folder_name=None, Q_tab=None, args=None
):
    """
    Compute episode rewards over n simulated runs for a given Q-table.

    Parameters
    ----------
    matrix_path : str
        File path to the Q-table to be evaluated.
    n : int
        Number of episodes to simulate.
    folder_mode : bool
        Flag indicating whether results are loaded from and/or saved to files.
    folder_name : str
        Directory used for loading/saving files.
    Q_tab : object
        Q-table to evaluate; if not provided, it is loaded from matrix_path.
    args : dict
        Dictionary of arguments used to initialise the environment.

    Returns
    -------
    rewards : list
        List of simulated episode rewards.
    opt_action : tuple
        Optimal action at state (0,0).
    Q_star : float
        State-value at (0,0).
    """

    if Q_tab == None:
        Q_tab, args, _, _, _, _ = load_Q(
            matrix_path, folder_mode=folder_mode, folder_name=folder_name
        )

    env = SimpleEnv(**args, printing=False, debug=False, analytical=False)

    rewards = list()

    for _ in range(int(n)):
        env.reset()
        disc_reward = 0

        while env.t < env.T:
            state = env.state()
            action = np.array(
                np.unravel_index(Q_tab[state].argmax(), Q_tab[state].shape)
            )

            _, action_reward = env.step(
                np.array(action)
            )  # Get the new state and the reward

            disc_reward += action_reward  # * (gamma ** env.t)  # Discounting with gamma

        rewards.append(disc_reward)

    opt_action = np.unravel_index(Q_tab[(0, 0)].argmax(), Q_tab[(0, 0)].shape)
    Q_star = Q_tab[(0, 0)][opt_action]

    return rewards, opt_action, Q_star


def evaluate_analytical_strategy(
    args_environment, args_generating, n=1000, discrete=True
):
    """
    Simulate n episodes and return the reward distribution obtained under the analytically optimal strategy.

    Parameters
    ----------
    args_environment : dict
        Parameters used to initialise the environment.
    args_generating : dict
        Parameters used to generate the strategy.
    n : int
        Number of episodes to simulate.
    discrete : bool
        Flag indicating whether the analytical solution is discretised.

    Returns
    -------
    rewards : list
        List of simulated episode rewards.
    """

    env = SimpleEnv(**args_environment, printing=False, debug=False, analytical=True)

    data_bid = generate_optimal_depth(**args_generating, bid=True, discrete=discrete)
    data_ask = generate_optimal_depth(**args_generating, bid=False, discrete=discrete)

    rewards = list()

    Q = 3  # args_environment['Q']

    for _ in range(int(n)):
        env.reset()
        disc_reward = 0

        while env.t < env.T:
            state = env.state()
            action = data_bid[state[0] + Q, env.t], data_ask[state[0] + Q, env.t]

            _, action_reward = env.step(
                np.array(action)
            )  # Get the new state and the reward

            disc_reward += action_reward  # * (gamma ** env.t)  # Discounting with gamma

        rewards.append(disc_reward)

    return rewards


def evaluate_constant_strategy(args_environment, n=1000, c=2):
    """
    Simulate n episodes and return the reward distribution obtained under the constant strategy, quoting at c ticks from the mid price.

    Parameters
    ----------
    args_environment : dict
        Parameters used to initialise the environment.
    n : int
        Number of episodes to simulate.
    c : int
        Quote distance in ticks from the mid price.

    Returns
    -------
    rewards : list
        List of simulated episode rewards.
    """

    env = SimpleEnv(**args_environment, printing=False, debug=False, analytical=False)

    rewards = list()

    for _ in range(int(n)):
        env.reset()
        disc_reward = 0

        while env.t < env.T:
            action = np.array([c, c]) - env.min_dp

            _, action_reward = env.step(
                np.array(action)
            )  # Get the new state and the reward

            disc_reward += action_reward  # * (gamma ** env.t)  # Discounting with gamma

        rewards.append(disc_reward)

    return rewards


def evaluate_random_strategy(args_environment, n=1000):
    """
    Simulate n episodes and return the reward distribution obtained under a random strategy.

    Parameters
    ----------
    args_environment : dict
         Parameters used to initialise the environment.
    n : int
        Number of episodes to simulate.

    Returns
    -------
    rewards : list
        List of simulated episode rewards.
    """

    env = SimpleEnv(**args_environment, printing=False, debug=False, analytical=False)

    rewards = list()

    for _ in range(int(n)):
        env.reset()
        disc_reward = 0

        while env.t < env.T:
            action = env.action_space.sample()

            _, action_reward = env.step(
                np.array(action)
            )  # Get the new state and the reward

            disc_reward += action_reward  # * (gamma ** env.t)  # Discounting with gamma

        rewards.append(disc_reward)

    return rewards


def evaluate_strategies(
    path, n=1000, T=5, Q=3, dp=0.01, alpha=1e-4, phi=1e-5, min_dp=0, d=4, c=2
):
    """
    Evaluate a saved Q-table against benchmark strategies over n simulated episodes.

    Parameters
    ----------
    path : str
        Path where the Q-table is stored.
    n : int
        Number of episodes to simulate.
    T : int
        Episode length.
    Q : int
        Maximum permitted inventory, i.e. inventory is constrained to (-Q, Q).
    dp : float
        Tick size.
    alpha : float
        Penalty term applied at liquidation.
    phi : float
        Running inventory penalty.
    min_dp : int
        Minimum quote depth.
    d : int
        Number of quote depths available to the market maker.
    c : int
        Quote depth used by the constant strategy.

    Returns
    -------
    None
    """

    args_environment = {
        "T": T,
        "Q": Q,
        "dp": dp,
        "alpha": alpha,
        "phi": phi,
        "use_all_times": True,
        "min_dp": min_dp,
        "d": d,
        "breaching_penalty": False,
    }
    args_generating = {"T": T, "Q": Q, "dp": dp, "phi": phi}

    # Get the analytical solutions
    rewards_analytical_discrete = evaluate_analytical_strategy(
        args_environment, args_generating, n=n, discrete=True
    )
    rewards_analytical_continuous = evaluate_analytical_strategy(
        args_environment, args_generating, n=n, discrete=False
    )

    # Get the constant rewards
    rewards_constant = evaluate_constant_strategy(args_environment, n=n, c=c)

    # Get the random rewards
    rewards_random = evaluate_random_strategy(args_environment, n=n)

    # Get the Q-learning rewards
    rewards_Q_learning, _, _ = evaluate_Q_matrix(path, n=n)

    data = [
        rewards_analytical_discrete,
        rewards_analytical_continuous,
        rewards_constant,
        rewards_random,
        rewards_Q_learning,
    ]

    labels = [
        "analytical_discrete",
        "analytical_continuous",
        "constant (d=" + str(c) + ")",
        "random",
        "Q_learning",
    ]

    headers = ["strategy", "mean reward", "std reward"]
    rows = []
    for i, label in enumerate(labels):
        rows.append([label, np.mean(data[i]), np.std(data[i])])

    print("Results:\n")
    print(tabulate(rows, headers=headers))

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel("reward")
    plt.show()


def evaluate_strategies_multiple_Q(
    file_names,
    args,
    mean_rewards,
    Q_mean,
    n_test=1e2,
    c=2,
    folder_mode=False,
    folder_name=None,
    save_mode=False,
):
    """
    Compare multiple strategies by producing boxplots and a summary table of reward means and standard deviations.

    Parameters
    ----------
    file_names : list
        List of file paths to the evaluated Q-tables.
    args : dict
        Parameters used to initialise the environment.
    mean_rewards : list
        Mean rewards obtained for the evaluated Q-tables.
    Q_mean : dict
        Mean Q-table.
    n_test : int
        Number of episodes used for strategy evaluation.
    c : int
        Quote distance in ticks used by the constant strategy.
    folder_mode : bool
         Flag indicating whether outputs are loaded from and/or saved to files.
    folder_name : str
        Directory used for loading/saving files.
    save_mode : bool
        Flag indicating whether figures and tables are saved.

    Returns
    -------
    None
    """

    args_environment = args
    args_generating = {"T": args["T"], "Q": 3, "dp": args["dp"], "phi": args["phi"]}

    # Get the analytical solutions
    rewards_analytical_discrete = evaluate_analytical_strategy(
        args_environment, args_generating, n=n_test, discrete=True
    )
    rewards_analytical_continuous = evaluate_analytical_strategy(
        args_environment, args_generating, n=n_test, discrete=False
    )

    # Get the constant rewards
    rewards_constant = evaluate_constant_strategy(args_environment, n=n_test, c=c)

    # Get the random rewards
    rewards_random = evaluate_random_strategy(args_environment, n=n_test)

    # Get the best Q-learning rewards
    best_idx = np.argmax(mean_rewards)
    rewards_Q_learning_best, _, _ = evaluate_Q_matrix(
        file_names[best_idx], n=n_test, folder_mode=folder_mode, folder_name=folder_name
    )

    # Get the average Q-learning rewards
    rewards_Q_learning_average, _, _ = evaluate_Q_matrix(
        None,
        n=n_test,
        folder_mode=folder_mode,
        folder_name=folder_name,
        Q_tab=Q_mean,
        args=args,
    )

    data = [
        rewards_analytical_discrete,
        rewards_analytical_continuous,
        rewards_constant,
        rewards_random,
        rewards_Q_learning_best,
        rewards_Q_learning_average,
    ]

    labels = [
        "analytical_discrete",
        "analytical_continuous",
        "constant (d=" + str(c) + ")",
        "random",
        "Q_learning (best run)",
        "Q_learning (average)",
    ]

    headers = ["strategy", "mean reward", "std reward"]
    rows = []
    for i, label in enumerate(labels):
        rows.append([label, np.mean(data[i]), np.std(data[i])])

    if save_mode:
        with open(
            "results/simple_model/" + folder_name + "/" "table_benchmarking", "w"
        ) as f:
            f.write(tabulate(rows, headers=headers))
    else:
        print("Results:\n")
        print(tabulate(rows, headers=headers))

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels)
    plt.title("Comparison of different strategies")
    plt.ylabel("reward")

    if save_mode:
        plt.savefig("results/simple_model/" + folder_name + "/" "box_plot_benchmarking")
        plt.close()
    else:
        plt.show()


def compare_Q_learning_runs(
    file_names, n_test=1e2, folder_mode=False, folder_name=None, save_mode=False
):
    """
    Compare multiple Q-learning runs by generating boxplots and a summary table of reward means and standard deviations.

    Parameters
    ----------
    file_names : list
        List of file paths to the evaluated Q-tables.
    n_test : int
        Number of episodes used to evaluate each run.
    folder_mode : bool
        Flag indicating whether outputs are loaded from and/or saved to files.
    folder_name : str
        Directory used for loading/saving files.
    save_mode : bool
        Flag indicating whether figures and tables are saved.

    Returns
    -------
    mean_reward : np.array
        Mean reward obtained for each run.
    """

    data = []
    actions = []
    q_values = []

    for file_name in file_names:
        reward, action, q_value = evaluate_Q_matrix(
            file_name, n=n_test, folder_mode=folder_mode, folder_name=folder_name
        )
        data.append(reward)
        actions.append(action)
        q_values.append(q_value)

        labels = ["run " + str(i + 1) for i in range(len(file_names))]

    headers = ["run", "mean reward", "std reward", "opt action", "q-value"]
    rows = []

    for i, label in enumerate(labels):
        rows.append([label, np.mean(data[i]), np.std(data[i]), actions[i], q_values[i]])

    if save_mode:
        with open(
            "results/simple_model/" + folder_name + "/" "table_different_runs", "w"
        ) as f:
            f.write(tabulate(rows, headers=headers))
    else:
        print("Results:\n")
        print(tabulate(rows, headers=headers))

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels)
    plt.title("Comparison of different Q-learning runs")
    plt.ylabel("reward")

    if save_mode:
        plt.savefig(
            "results/simple_model/" + folder_name + "/" "box_plot_different_runs"
        )
        plt.close()
    else:
        plt.show()

    return np.mean(data, axis=1)


def plot_rewards_multiple(
    file_names, folder_mode=False, folder_name=None, save_mode=False
):
    """
    Plot average rewards and Q-values with confidence intervals across multiple runs.

    Parameters
    ----------
    file_names : list
        List of file paths containing rewards and Q-values.
    folder_mode : bool
        Flag indicating whether outputs are loaded from and/or saved to files.
    folder_name : str
        Directory used for loading/saving files.
    save_mode : bool
        Flag indicating whether figures and tables are saved.

    Returns
    -------
    None
    """
    reward_matrix = []
    Q_zero_matrix = []

    # Open and save the values
    for file_name in file_names:
        _, _, _, rewards_average, Q_zero_average, x_values = load_Q(
            file_name, folder_mode=folder_mode, folder_name=folder_name
        )

        reward_matrix.append(rewards_average)
        Q_zero_matrix.append(Q_zero_average)

    reward_matrix = np.array(reward_matrix)
    Q_zero_matrix = np.array(Q_zero_matrix)

    # Calculate mean, std and area
    reward_mean = np.mean(reward_matrix, axis=0)
    reward_std = np.std(reward_matrix, axis=0)

    Q_zero_mean = np.mean(Q_zero_matrix, axis=0)
    Q_zero_std = np.std(Q_zero_matrix, axis=0)

    reward_area = np.array([reward_std, -reward_std]) + reward_mean
    Q_zero_area = np.array([Q_zero_std, -Q_zero_std]) + Q_zero_mean

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot the rewards
    ax1.fill_between(
        x_values,
        reward_area[0, :],
        reward_area[1, :],
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    ax1.plot(x_values, reward_mean, linewidth=0.2, color="purple", label="mean reward")
    ax1.set_xlabel("episode")
    ax1.set_xticks(np.linspace(0, x_values[-1], 6))
    ax1.set_ylabel("reward")
    ax1.set_title("average reward during training")

    # Plot the Q-values
    ax2.fill_between(
        x_values,
        Q_zero_area[0, :],
        Q_zero_area[1, :],
        alpha=0.3,
        color="purple",
        label="±$\sigma$",
    )
    ax2.plot(
        x_values,
        Q_zero_mean,
        linewidth=0.2,
        color="purple",
        label="mean Q[(0,0)]-value",
    )
    ax2.set_xlabel("episode")
    ax2.set_xticks(np.linspace(0, x_values[-1], 6))
    ax2.set_ylabel("Q[(0,0)]")
    ax2.set_title("average Q[(0,0)] during training")

    ax1.legend()
    ax2.legend()

    if save_mode:
        plt.savefig("results/simple_model/" + folder_name + "/" "results_graph")
        plt.close()
    else:
        plt.show()


def calculate_mean_Q(file_names, folder_mode=False, folder_name=None):
    """
    Compute an average Q-table across multiple runs.

    Parameters
    ----------
    file_names : list
        List of file paths to the Q-tables.
    folder_mode : bool
        Flag indicating whether inputs are loaded from and/or saved to files.
    folder_name : str
        Directory used for loading/saving files.

    Returns
    -------
    Q_mean : defaultdict
        Defaultdict with states as keys and average Q-values as values.
    Q_tables : list
        List of defaultdicts with states as keys and Q-values as values.
    """

    # Fetch the model parameters
    _, args, _, _, _, _ = load_Q(
        file_names[0], folder_mode=folder_mode, folder_name=folder_name
    )

    Q_mean = defaultdict(lambda: np.zeros((args["d"], args["d"])))

    Q_tables = []

    # Load the files
    for file_name in file_names:
        Q_tables.append(
            load_Q(file_name, folder_mode=folder_mode, folder_name=folder_name)[0]
        )

    # Calculate the mean
    for state in list(Q_tables[0].keys()):
        Q_mean[state] = np.mean(
            [Q_tables[i][state] for i in range(len(Q_tables))], axis=0
        )

    return Q_mean, Q_tables


def calculate_std_Q(Q_mean, Q_tables):
    """
    Compute the standard deviation, across runs, of the Q-value associated with the mean-optimal action in each state.

    Parameters
    ----------
    Q_mean : defaultdict
        Defaultdict with states as keys and average Q-values as values.
    Q_tables : list
        List of defaultdicts with states as keys and Q-values as values.

    Returns
    -------
    Q_std : defaultdict
        Defaultdict with states as keys and the standard deviation of Q-values as values.
    """

    Q_std = Q_mean

    for state in list(Q_mean.keys()):
        # Find the optimal action based on mean
        optimal_action = np.array(
            np.unravel_index(Q_mean[state].argmax(), Q_mean[state].shape)
        )

        # Calculate the standard deviation of the q-value of that action
        Q_std[state] = np.std(
            [Q_tables[i][state][tuple(optimal_action)] for i in range(len(Q_tables))]
        )

    return Q_std


def args_to_file_names(args, n_runs, n):
    """
    Generate a list of file names based on the model parameters.

    Parameters
    ----------
    args : dict
        Dictionary of model parameters.
    n_runs : int
        Number of distinct runs performed.
    n : int
        Number of training episodes per run.

    Returns
    -------
    file_names : list
        List of file name strings.
    """

    suffixes = np.arange(n_runs) + 1

    file_names = []

    for suffix in suffixes:
        file_names.append(fetch_table_name(args, n, suffix))

    return file_names


def Q_learning_comparison(
    n_train=1e2,
    n_test=1e1,
    n_runs=2,
    file_names=None,
    args=None,
    Q_learning_args=None,
    folder_mode=False,
    folder_name=None,
    save_mode=False,
):
    """
    Run tabular Q-learning multiple times, compare runs against each other and against benchmark strategies, and produce summary plots.

    Parameters
    ----------
    n_train : int
        Number of episodes used to train Q-learning.
    n_test : int
        Number of episodes used to evaluate the strategies.
    n_runs : int
        Number of independent Q-learning runs performed.
    file_names : list
        List of save locations for the Q-tables.
    args : dict
        Parameters used to initialise the environment.
    Q_learning_args : dict
         Parameters used for Q-learning.
    folder_mode : bool
        Flag indicating whether inputs/outputs are loaded from and/or saved to files.
    folder_name : str
        Directory used for loading/saving files.
    save_mode : bool
        Flag indicating whether figures and tables are saved.

    Returns
    -------
    None
    """

    if file_names == None:
        file_names = Q_learning_multiple(
            args,
            Q_learning_args,
            n_train,
            n_runs,
            folder_mode=folder_mode,
            folder_name=folder_name,
        )

    # Using mean Q instead
    Q_mean, Q_tables = calculate_mean_Q(
        file_names, folder_mode=folder_mode, folder_name=folder_name
    )

    plot_rewards_multiple(
        file_names,
        folder_mode=folder_mode,
        folder_name=folder_name,
        save_mode=save_mode,
    )

    mean_rewards = compare_Q_learning_runs(
        file_names,
        n_test,
        folder_mode=folder_mode,
        folder_name=folder_name,
        save_mode=save_mode,
    )

    evaluate_strategies_multiple_Q(
        file_names,
        args,
        mean_rewards,
        Q_mean,
        n_test,
        folder_mode=folder_mode,
        folder_name=folder_name,
        save_mode=save_mode,
    )

    env = SimpleEnv(**args, printing=False, debug=False, analytical=False)

    file_path = "results/simple_model/" + folder_name + "/" if save_mode else None

    Q_mean = remove_last_t(Q_mean, env.T)

    show_Q(Q_mean, env, file_path=file_path)
    heatmap_Q(Q_mean, file_path=file_path)

    heatmap_Q_n_errors(
        Q_mean.copy(), Q_tables.copy(), n_unique=True, file_path=file_path
    )
    heatmap_Q_n_errors(
        Q_mean.copy(), Q_tables.copy(), n_unique=False, file_path=file_path
    )

    Q_std = calculate_std_Q(Q_mean, Q_tables)
    heatmap_Q_std(Q_std, file_path=file_path)


def get_args_from_txt(folder_name):
    """
    Load arguments from a parameters.txt file.

    Parameters
    ----------
    folder_name : str
         Directory in which the txt file is stored.

    Returns
    -------
    args : dict
        Parameters used for the environment.
    """

    f = open("results/simple_model/" + folder_name + "/parameters.txt")
    lines = f.readlines()

    args = {}

    for line in lines:
        if line == "\n":
            return args

        if line != "MODEL PARAMETERS\n":
            key, value = line.split(":")
            key = key.strip()
            value = value.strip()

            if value in ["True", "False"]:
                value = bool(value)
            else:
                value = float(value)
                if value == int(value):
                    value = int(value)

            args[key] = value
