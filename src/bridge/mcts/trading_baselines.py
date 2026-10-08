"""Week 5 baseline: a greedy trader that maximizes the exact one-step
expected reward and never plans beyond it."""

import random

from bridge.envs.trading import State, TradingEnv


def greedy_trading_move(env: TradingEnv, state: State, rng: random.Random) -> str:
    """Pick the action with the highest one-step expected reward.

    `rng` is unused (greedy is deterministic) but kept so every policy
    shares the signature (env, state, rng) -> action.
    """
    return max(env.actions(state), key=lambda a: env.expected_reward(state, a))
