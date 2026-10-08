import random

from bridge.envs.trading import ACTIONS, State, TradingEnv
from bridge.mcts.stochastic_agent import mcts_trading_move


def test_mcts_returns_a_valid_action():
    env = TradingEnv()
    rng = random.Random(0)
    action = mcts_trading_move(env, State(0, 10, 0), n_iterations=200, rng=rng)
    assert action in ACTIONS


def test_mcts_keeps_a_profitable_position():
    # Already long at a low price, where the price tends to drift up:
    # selling would cost a fee and forfeit the drift, so "long" should win.
    env = TradingEnv()
    rng = random.Random(0)
    action = mcts_trading_move(env, State(0, 3, 1), n_iterations=1000, rng=rng)
    assert action == "long"
