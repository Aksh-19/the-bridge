import random

from bridge.mcts.agent import mcts_search
from bridge.mcts.tictactoe import EMPTY_BOARD, legal_moves


def test_mcts_chooses_a_legal_move():
    rng = random.Random(0)
    move = mcts_search(EMPTY_BOARD, "X", n_iterations=200, rng=rng)
    assert move in legal_moves(EMPTY_BOARD)


def test_mcts_takes_an_obvious_winning_move():
    # X has two in a row (0,1) and can win by playing 2.
    board = "XX......."
    rng = random.Random(0)
    move = mcts_search(board, "X", n_iterations=200, rng=rng)
    assert move == 2
