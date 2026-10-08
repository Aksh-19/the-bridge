import random

import numpy as np

from bridge.envs.trading import State, TradingEnv
from bridge.mcts.trading_baselines import greedy_trading_move
from bridge.mcts.trading_eval import mean_and_ci, run_episode


def test_greedy_skips_trade_when_fee_exceeds_one_step_gain():
    env = TradingEnv()
    rng = random.Random(0)
    # At price 7 the one-step gain from buying is 0.3, below the 0.5 fee.
    assert greedy_trading_move(env, State(0, 7, 0), rng) == "flat"


def test_greedy_keeps_profitable_position():
    env = TradingEnv()
    rng = random.Random(0)
    assert greedy_trading_move(env, State(0, 3, 1), rng) == "long"


def test_run_episode_is_reproducible():
    env = TradingEnv()
    a = run_episode(env, greedy_trading_move, env_seed=1, agent_seed=1)
    b = run_episode(env, greedy_trading_move, env_seed=1, agent_seed=1)
    assert a == b


def test_mean_and_ci_on_constant_values():
    mean, half = mean_and_ci(np.array([2.0, 2.0, 2.0, 2.0]))
    assert mean == 2.0
    assert half == 0.0
