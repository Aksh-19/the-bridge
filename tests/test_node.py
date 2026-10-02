import math

from bridge.mcts.node import Node
from bridge.mcts.tictactoe import EMPTY_BOARD


def test_unvisited_child_has_infinite_ucb1():
    root = Node(EMPTY_BOARD, player_to_move="X")
    child = root.expand()
    assert child.ucb1_score(total_visits=1) == math.inf


def test_expand_adds_one_child_per_call():
    root = Node(EMPTY_BOARD, player_to_move="X")
    assert len(root.children) == 0
    root.expand()
    assert len(root.children) == 1
    root.expand()
    assert len(root.children) == 2


def test_best_child_prefers_higher_mean_value_once_visited():
    root = Node(EMPTY_BOARD, player_to_move="X")
    child_a = root.expand()
    child_b = root.expand()
    # Manually set visit/value stats, as if simulations had already run
    # (real values will come from Week 4's simulation step).
    child_a.visits, child_a.value_sum = 10, 2.0  # mean 0.2
    child_b.visits, child_b.value_sum = 10, 8.0  # mean 0.8
    root.visits = 20
    assert root.best_child() is child_b
