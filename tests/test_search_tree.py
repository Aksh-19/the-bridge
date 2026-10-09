import random

from bridge.envs.trading import State, TradingEnv
from bridge.mcts.stochastic_agent import mcts_trading_search


def test_search_tree_bookkeeping_is_consistent():
    env = TradingEnv()
    root = mcts_trading_search(
        env, State(0, 8, 0), n_iterations=300, rng=random.Random(0)
    )
    # Every iteration passes through the root exactly once.
    assert root.visits == 300
    assert sum(root.action_visits.values()) == 300
    assert sum(root.edge_visits.values()) == 300
