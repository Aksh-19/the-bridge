"""Week 5 checkpoint: MCTS vs the greedy baseline on the trading
environment, over many episodes, with 95% confidence intervals.

Each episode uses the same market path for both agents (paired
comparison), so differences in return reflect decision quality rather
than luck.
"""

import math
import random
from collections.abc import Callable

import numpy as np

from bridge.envs.trading import State, TradingEnv
from bridge.mcts.stochastic_agent import mcts_trading_move
from bridge.mcts.trading_baselines import greedy_trading_move

Policy = Callable[[TradingEnv, State, random.Random], str]


def mcts_policy(n_iterations: int, c: float = 3.0) -> Policy:
    """Wrap the MCTS search so it matches the Policy signature."""

    def policy(env: TradingEnv, state: State, rng: random.Random) -> str:
        return mcts_trading_move(env, state, n_iterations=n_iterations, c=c, rng=rng)

    return policy


def run_episode(
    env: TradingEnv, policy: Policy, env_seed: int, agent_seed: int
) -> float:
    """Play one full episode and return the total reward.

    The market uses its own RNG (env_seed), separate from the agent's
    (agent_seed), so two different agents see the same price path.
    """
    env_rng = random.Random(env_seed)
    agent_rng = random.Random(agent_seed)

    state = env.initial_state(env_rng)
    total = 0.0
    while not env.is_terminal(state):
        action = policy(env, state, agent_rng)
        state, reward = env.step(state, action, env_rng)
        total += reward
    return total


def mean_and_ci(values: np.ndarray) -> tuple[float, float]:
    """Return (mean, 95% half-width): mean +/- 1.96 * sigma_hat / sqrt(n).

    Same formula as the Week 1 estimator (section 6 of the derivation).
    """
    n = len(values)
    mean = float(np.mean(values))
    std_error = float(np.std(values, ddof=1) / math.sqrt(n))
    return mean, 1.96 * std_error


def evaluate(
    n_episodes: int = 300,
    n_iterations: int = 500,
    c: float = 3.0,
    seed_offset: int = 10_000,
) -> None:
    env = TradingEnv()
    mcts = mcts_policy(n_iterations, c)

    mcts_returns = np.empty(n_episodes)
    greedy_returns = np.empty(n_episodes)

    for i in range(n_episodes):
        env_seed = seed_offset + i
        mcts_returns[i] = run_episode(env, mcts, env_seed, agent_seed=i)
        greedy_returns[i] = run_episode(
            env, greedy_trading_move, env_seed, agent_seed=i
        )

    diffs = mcts_returns - greedy_returns
    wins = int(np.sum(diffs > 1e-9))
    losses = int(np.sum(diffs < -1e-9))
    ties = n_episodes - wins - losses

    m_mean, m_half = mean_and_ci(mcts_returns)
    g_mean, g_half = mean_and_ci(greedy_returns)
    d_mean, d_half = mean_and_ci(diffs)

    print(f"Episodes: {n_episodes}, MCTS iterations per move: {n_iterations}")
    print(f"MCTS   mean return: {m_mean:+.3f} +/- {m_half:.3f} (95% CI)")
    print(f"Greedy mean return: {g_mean:+.3f} +/- {g_half:.3f} (95% CI)")
    print(f"MCTS - greedy (paired): {d_mean:+.3f} +/- {d_half:.3f} (95% CI)")
    print(f"MCTS better in {wins}, tied in {ties}, worse in {losses} episodes")


if __name__ == "__main__":
    evaluate()
