# RL_in_MM_thesis
Git Repository for the MSc thesis "Reinforcement Learning for Dynamic Market Making: Inventory Management and Adaptive Quoting in Modern Electronic Markets"

# Abstract ENG
This thesis investigates how reinforcement learning (RL) can be used to design profitable and risk-aware market-making strategies in modern electronic limit order book markets. The central focus is on inventory management and adaptive quoting, framed by two research questions: (i) whether RL methods can match or outperform analytically derived strategies, and (ii) whether RL can manage inventory dynamically in a way that controls risk while preserving profitability.
The empirical analysis proceeds in two stages. First, a simple probabilistic model (SPM) of a single asset is used as a controlled benchmark where the optimal market-making policy is known in closed form. In this setting, a tabular Q-learning agent is trained on penalized profit-and-loss (P&L), combining mark-to-market gains with a quadratic inventory penalty. The learned strategy is shown to match the performance of the analytical Avellaneda - Stoikov solution and, in point estimates, to marginally exceed its penalized P&L, while decisively outperforming simple constant-depth and random-depth quoting rules. This demonstrates that model-free RL can recover economically meaningful policies even when a closed-form optimum is available.
Second, the thesis turns to a Markov-chain limit order book (MC LOB) model, which captures realistic microstructure features but does not admit an analytical solution. Here, both tabular Q-learning and Double Deep Q-Networks (DDQN) are tested in short- and long-horizon configurations, and their strategies are evaluated against heuristic benchmarks. In the short-horizon regime, RL agents achieve small but positive average P&L while keeping inventory fluctuations modest, outperforming naïve strategies that are systematically loss-making. In the long-horizon regime, tabular Q-learning becomes unstable under coarse state aggregation, whereas DDQN learns robustly profitable policies across random seeds, with inventory paths that exhibit intuitive build-and-unwind behavior over the trading day.
Overall, the results indicate that RL is a viable and often superior alternative to both analytical and heuristic approaches to market making, provided that state representation and function approximation are chosen carefully. The thesis concludes with an outlook on integrating RL-based market makers with transformer-based large language models, highlighting how multimodal and text-driven signals could further enhance state information, risk assessment, and the realism of simulated trading environments.
Key words: Reinforcement learning, Deep reinforcement learning, Market making, Inventory management, Adaptive quoting, Limit order book, Markov chains, Simple probabilistic model, High-frequency trading, Electronic financial markets, Profit-and-loss optimization, Risk-aware trading strategies, Large Language Models in trading

# Repository 
```
├───environments                           <- python code for setting up the simulation
│   ├───mc_model
│   ├───simple_model
├───images                                 <- various images plotted and used in the pdf
├───other notebooks                        <- some other notebooks
├───results                                <- results for each experiment
│   ├───mc_model
│   ├───mc_model_deep
│   └───simple_model
├───utils                                  <- helping utilities for each model
│   ├───mc_model
│   ├───simple_model
│   └── .py                                <- files for reinforcement learning and evaluation
└───Readme.md
```

# Reproductibility and compatibility
The technology stack used for the present project is not the most crip and new, even can be regarded as ancient. The reason for that is partlly my personal preference, partly - plethora of answers to my questions already present in the Stackoveflow. 

