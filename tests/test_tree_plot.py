import random

import matplotlib.pyplot as plt

from bridge.envs.trading import State, TradingEnv
from bridge.mcts.stochastic_agent import mcts_trading_search
from bridge.viz.tree_plot import draw_tree

plt.switch_backend("Agg")


def test_draw_tree_writes_a_png(tmp_path):
    env = TradingEnv()
    root = mcts_trading_search(env, State(0, 6, 0), 200, rng=random.Random(0))
    out = tmp_path / "tree.png"
    draw_tree(env, root, "test", str(out))
    assert out.exists()
    assert out.stat().st_size > 0
