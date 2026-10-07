import random

from bridge.mcts.node import Node, backpropagate
from bridge.mcts.simulate import random_rollout
from bridge.mcts.tictactoe import EMPTY_BOARD


def test_random_rollout_returns_valid_outcome():
    rng = random.Random(0)
    result = random_rollout(EMPTY_BOARD, "X", rng)
    assert result in ("X", "O", None)


def test_backpropagate_updates_every_node_on_path():
    root = Node(EMPTY_BOARD, player_to_move="X")
    child = root.expand()
    grandchild = child.expand()

    backpropagate(grandchild, winning_player="X")

    assert root.visits == 1
    assert child.visits == 1
    assert grandchild.visits == 1


def test_backpropagate_rewards_correct_player():
    # root: X to move. child: created by X's move, O to move.
    root = Node(EMPTY_BOARD, player_to_move="X")
    child = root.expand()

    backpropagate(child, winning_player="X")  # X (who made this move) won
    assert child.value_sum == 1.0  # full credit: the mover (X) won

    backpropagate(child, winning_player="O")  # X made the move, O won instead
    assert child.value_sum == 1.0  # unchanged: this call adds 0.0
