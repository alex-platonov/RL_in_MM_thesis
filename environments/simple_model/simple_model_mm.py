import random

import gym
import numpy as np
from scipy.linalg import expm  # O(n^3)


class SimpleEnv(gym.Env):
    """
    SimpleEnv is an object that simulates the simple probabilistic environment.
    
    Parameters
    ----------
    T : integer
        The number of time steps the model is run for.
    dt : integer
        The size of the time steps.
    Q : integer
        The maximum (absolute) allowed held volume.
    dq : integer
        Volume increments.
    Q_0 : integer
        The initial inventory of the market maker.
    dp : float
        The tick size.
    min_dp : integer
        The minimum number of ticks from the mid price at which the market maker must quote bid/ask prices.
    mu : float
        The initial price level.
    std : float
        The standard deviation of the price movement.
    lambda_pos : float
        Intensity of the Poisson process governing arrivals of sell market orders.
    lambda_neg : float
        Intensity of the Poisson process governing arrivals of buy market orders.
    kappa : float
        Parameter governing the execution probability of market orders.
    alpha : float
        Fee for taking liquidity (placing market orders).
    phi : float
        Running inventory penalty parameter.
    pre_run : integer
        Number of time steps for which the price process is advanced before simulation starts.
    printing : bool
        Flag indicating whether information is printed during simulation.
    debug : bool
        Flag indicating whether debug information is printed during simulation.
    d : int
        Number of ticks away from the mid price at which the market maker can quote.
    use_all_times : bool
        Flag indicating whether all time steps are used, or a time indicator is used instead.
    analytical : bool
        Flag indicating whether analytical depths are used.
    breaching_penalty : bool
        Flag indicating whether an inventory-breaching penalty is applied.
    breach_penalty : float
        Penalty applied when the inventory limit is breached.
    reward_scale : float
        Scaling factor applied to the reward.
    breach_penalty_function : function
        Function used to compute the breach penalty.
    """

    def __init__(
        self,
        T=10,
        dt=1,
        Q=3,
        dq=1,
        Q_0=0,
        dp=0.1,
        min_dp=1,
        mu=100,
        std=0.01,
        lambda_pos=1,
        lambda_neg=1,
        kappa=100,
        alpha=1e-4,
        phi=1e-5,
        pre_run=None,
        printing=False,
        debug=False,
        d=5,
        use_all_times=True,
        analytical=False,
        breaching_penalty=False,
        breach_penalty=20,
        reward_scale=1,
        breach_penalty_function=np.square,
    ):
        super(SimpleEnv, self).__init__()

        self.T = T  # maximal time
        self.dt = dt  # time increments

        self.Q = Q  # maximal volume
        self.dq = dq  # volume increments the agent can work with
        self.Q_0 = Q_0  # the starting volume

        self.dp = dp  # tick size
        self.min_dp = min_dp  # the minimum number of ticks from the mid price the MMr has to put their ask/bid price
        self.d = d  # Number of ticks away the MMr is allowed to quote

        self.mu = mu  # average price of the stock
        self.std = std  # std of the price movement

        self.alpha = alpha  # penalty for holding volume
        self.phi = phi  # the running inventory penalty parameter

        self.use_all_times = use_all_times

        # Set the action and observation space
        self.set_action_space()
        self.set_observation_space()

        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.kappa = kappa
        self.mm_bid = None
        self.mm_ask = None

        self.z = None
        self.A = None
        self.init_analytically_optimal()

        self.printing = printing
        self.debug = debug

        self.analytical = analytical
        self.breach_penalty = breach_penalty
        self.breach = False
        self.breaching_penalty = breaching_penalty
        self.breach_penalty_function = breach_penalty_function

        self.reward_scale = reward_scale

        # Reset the environment
        self.reset()

        # Pre-running the price for pre_run time steps
        if pre_run != None:
            self.pre_run(pre_run)  # Ha kvar? Troligtvis inte

        # Remembering the start price for the reward
        self.start_mid = self.mid

    def set_action_space(self):
        """
        Set the action space.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        low = np.array([0, 0])
        high = np.array(
            [self.d - 1, self.d - 1]
        )  # d-1 due to actions defined on [0, d-1]^2, where action [i,j]
        # corresponds to quoting i+1 and j+1 ticks away as bid and ask prices

        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.int16)

    def set_observation_space(self):
        """
        Set the observation space.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        if self.use_all_times:
            low = np.array([-self.Q, 0])
            high = np.array([self.Q, self.T / self.dt])
        else:
            low = np.array([-self.Q, 0])
            high = np.array([self.Q, 1])

        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.int16)

    def state(self):
        """
        Return the observation space.

        Parameters
        ----------
        None

        Returns
        -------
        obs : tuple
            Observation space expressed either as (volume, time) = (Q_t, t) or as (volume, end_of_trading_day) = (Q_t, self.t / self.T >= 0.9).

        """

        if self.use_all_times:
            return self.Q_t, self.t
        else:
            return self.Q_t, int(self.t / self.T >= 0.9)

    def pre_run(self, n_steps=100):
        """
        Update the price n_steps times.

        Parameters
        ----------
        n_steps : int
            Number of time steps for which the price process is advanced.

        Returns
        -------
        None
        """

        for _ in range(n_steps):
            self.update_price()

    def update_price(self):
        """
        Update the mid price once and makes sure it's within bounds.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # The change rounded to the closest tick size
        self.mid += self.round_to_tick(np.random.normal(0, self.std))

    def init_analytically_optimal(self):
        """
        Calculates `z` and `A` which will be used for the optimal solution.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.z = np.exp(
            -self.alpha
            * self.kappa
            * np.square(np.array(range(self.Q, -self.Q - 1, -1)))
        )
        self.A = np.zeros((self.Q * 2 + 1, self.Q * 2 + 1))
        for i in range(-self.Q, self.Q + 1):
            for q in range(-self.Q, self.Q + 1):
                if i == q:
                    self.A[i + self.Q, q + self.Q] = -self.phi * self.kappa * (q**2)
                elif i == q - 1:
                    self.A[i + self.Q, q + self.Q] = self.lambda_pos * np.exp(-1)
                elif i == q + 1:
                    self.A[i + self.Q, q + self.Q] = self.lambda_neg * np.exp(-1)

    def calc_analytically_optimal(self):
        """
        Calculates the analytically optimal bid and ask depths for the current time step.

        Parameters
        ----------
        None

        Returns
        -------
        action : np.array
            Array with bid/ask depths.
        """

        omega = np.matmul(expm(self.A * (self.T - self.t)), self.z)
        h = 1 / self.kappa * np.log(omega)

        if self.Q_t != -self.Q:
            delta_pos = 1 / self.kappa - h[self.Q_t + self.Q - 1] + h[self.Q_t + self.Q]
        if self.Q_t != self.Q:
            delta_neg = 1 / self.kappa - h[self.Q_t + self.Q + 1] + h[self.Q_t + self.Q]

        if self.Q_t == -self.Q:
            d_ask = np.Inf
            d_bid = delta_neg
        elif self.Q_t == self.Q:
            d_ask = delta_pos
            d_bid = np.Inf
        else:
            d_ask = delta_pos
            d_bid = delta_neg

        action = np.array([d_bid, d_ask])

        return action

    def discrete_analytically_optimal(self):
        """
        alculate the analytically optimal bid and ask depths for the current time step, expressed in ticks.

        Parameters
        ----------
        None

        Returns
        -------
        action : np.array
            Array containing the number of ticks away from the mid price.
        """

        action = np.rint(self.calc_analytically_optimal() / self.dp) - self.min_dp

        return action

    def round_to_tick(self, p):
        """
        Round a price to the closest tick.

        Parameters
        ----------
        p : float
            Input price.

        Returns
        -------
        p : float
            Rounded price.
        """

        return np.round(p / self.dp, decimals=0) * self.dp

    def transform_action(self, action):
        """
        Transform an action expressed in ticks into the corresponding bid/ask offsets from the mid price.
        The minimum distance to the mid price is enforced, and the ask offset is represented with a negative sign.
        
        Parameters
        ----------
        action : np.array
            Number of ticks away from the mid price for the ask and bid quotes.
        
        Returns
        -------
        action : np.array
            Offsets from the mid price for the chosen ask and bid quotes.
        """

        return (action + self.min_dp) * np.array([-1, 1]) * self.dp

    def step(self, action):
        """
        Advance the environment by one step given a market maker action.
        
        Parameters
        ----------
        action : np.array
            Number of ticks away from the mid price for the ask and bid quotes.
        
        Returns
        -------
        obs : tuple
            Observation at time step t.
        reward : float
            Reward at time step t.
        """

        self.t += self.dt

        # ----- UPDATING THE PRICE -----
        self.update_price()

        # Update bid and ask to the given number of ticks away from the mid price
        if self.analytical:
            [self.mm_bid, self.mm_ask] = self.mid + np.array([-1, 1]) * action
        else:
            [self.mm_bid, self.mm_ask] = self.mid + self.transform_action(action)

        if self.debug:
            print("Starting volume for time step:", self.Q_t)
            print("The mid price is:", self.mid)
            print("The action is:", action)
            print("The choice is:", self.mm_bid, "|", self.mm_ask)

        # ----- TAKING THE ACTION -----
        # In this case the MMr always updates the bid and ask to two ticks away from the mid with volume = 1

        # If not at the final time step the MM can do what it wants
        if self.t <= self.T:
            # ----- SAMPLE ORDERS -----

            # The number of orders that arrive
            n_MO_buy = np.random.poisson(self.lambda_neg)
            n_MO_sell = np.random.poisson(self.lambda_pos)

            # The probability that the orders get executed
            p_MO_buy = np.exp(-self.kappa * (self.mm_ask - self.mid))
            p_MO_sell = np.exp(-self.kappa * (self.mid - self.mm_bid))

            # Sample the number of orders executed
            n_exec_MO_buy = np.random.binomial(n_MO_buy, p_MO_buy)
            n_exec_MO_sell = np.random.binomial(n_MO_sell, p_MO_sell)

            # Step 1: add cash from the arriving buy and sell MOs that perfectly cancel each other
            if n_exec_MO_buy * n_exec_MO_sell > 0:
                self.X_t += (self.mm_ask - self.mm_bid) * np.min(
                    [n_exec_MO_buy, n_exec_MO_sell]
                )

            # NB: there shouldn't be any issues with infinite bids and asks since the probability of a filled LO at +/- inf is zero

            # Step 2: compute the net balance from time step t for the market maker, and make adjustments
            # Net balance for the market maker is the difference in arriving MO sell and arriving MO buy orders
            n_MO_net = n_exec_MO_sell - n_exec_MO_buy

            if (
                n_MO_net != 0
            ):  # Saving computational power, if net above is zero, skip below
                # Determine if net balance would result in a limit breach, and if so, adjust accordingly to keep
                # within limits
                if n_MO_net + self.Q_t > self.Q:  # long inventory limit breach
                    self.breach = self.breach_penalty_function(
                        n_MO_net + self.Q_t - self.Q
                    )
                    n_MO_net = (
                        self.Q - self.Q_t
                    )  # the maximum allowed net increase is given by Q - Q_t
                    n_exec_MO_sell -= n_MO_net + self.Q_t - self.Q
                elif n_MO_net + self.Q_t < -self.Q:  # short inventory limit breach
                    self.breach = self.breach_penalty_function(
                        -self.Q - (n_MO_net + self.Q_t)
                    )
                    n_MO_net = (
                        -self.Q - self.Q_t
                    )  # the maximum allowed net decrease is given by -Q + Q_t
                    n_exec_MO_buy -= -self.Q - (n_MO_net + self.Q_t)
                else:
                    self.breach = False

                # Step 3: add cash from net trading
                if n_MO_net > 0:
                    self.X_t -= self.mm_bid * n_MO_net
                elif n_MO_net < 0:
                    self.X_t -= self.mm_ask * n_MO_net

                self.Q_t += n_MO_net

            if self.debug:
                print("Arrvials:")
                print(n_MO_buy)
                print(n_MO_sell)

                print("\nProbabilities:")
                print(p_MO_buy)
                print(p_MO_sell)

                print("\nExecutions:")
                print(n_exec_MO_buy)
                print(n_exec_MO_sell)
                print("Net:", n_MO_net)
                print("Net:", n_exec_MO_sell - n_exec_MO_buy)

                print("\nX_t:", self.X_t)
                print("Q_t:", self.Q_t)

                print("_" * 20)

        # The time is up!
        if self.t == self.T:
            # The MM liquidates their position at a worse price than the mid price

            # The MM has to buy at their ask and sell at their bid with an additional unfavorable discount/increase
            self.X_t += self.final_liquidation()

            self.Q_t = 0

        # ----- THE REWARD -----
        V_t_new = self.X_t + self.H_t()
        if self.t <= self.T:
            reward = self._get_reward(V_t_new)
        else:
            reward = 0

        # ----- UPDATE VALUES -----
        self.V_t = V_t_new  # the cash value + the held value

        # ----- USEFUL INFORMATION -----
        if self.printing:
            print("The reward is:", reward)
            self.render()

        return self.state(), reward

    def _get_reward(self, V_t):
        """
        Return the reward for the current time step.
        
        Parameters
        ----------
        V_t : float
            Value process at the previous time step.
        
        Returns
        -------
        reward : float
            Reward at time step t.
        """

        # New value minus old
        # Subtract penalty for held volume

        if self.breaching_penalty:
            return (
                self.reward_scale * (V_t - self.V_t)
                + self.inventory_penalty()
                - self.breach_penalty * self.breach
            )
        else:
            return self.reward_scale * (V_t - self.V_t) + self.inventory_penalty()

    def H_t(self):
        """
        Return the value of the currently held inventory.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        H_t : float
            Value of the currently held inventory.
        """

        return self.Q_t * self.mid

    def inventory_penalty(self):
        """
        Return the running inventory penalty.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        g_t : float
            Running inventory penalty.
        """

        return -self.phi * (self.Q_t**2)

    def final_liquidation(self):
        """
        Return the value of the held inventory at time T.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        value : float
            Value of the held inventory at time T.
        """

        return self.Q_t * (self.mid - self.alpha * self.Q_t)

    def reset(self):
        """
        Reset the environment.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.mid = self.mu
        self.Q_t = self.Q_0
        self.V_t = self.H_t()  # the value process involves no cash at the start
        self.t = 0
        self.X_t = 0  # the cash process

    def render(self, mode="human", close=False):
        """
        Prints useful stats of the environment and the agent for debugging.

        Parameters
        ----------
        mode : string
            Rendering mode. This argument is accepted for compatibility; the output is printed to stdout.
        close : bool
            Close flag accepted for compatibility; no resources are released by this method.

        Returns
        -------
        None
        """

        print(20 * "-")
        print(f"End of t = {self.t}")
        print(f"Current mid price: {np.round(self.mid, 2)}")
        print(f"Current held volume: {self.Q_t}")
        print(f"Current held value: {np.round(self.H_t(), 2)}")
        print(f"Current cash process: {np.round(self.X_t, 2)}")
        print(f"Current value process: {np.round(self.V_t, 2)}")
        print(20 * "-" + "\n")
