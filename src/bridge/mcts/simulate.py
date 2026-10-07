"""Week 4: the simulation (rollout) phase of MCTS -- play a game out to
completion using uniform-random moves for both players, and return the
outcome."""

import random

from bridge.mcts.tictactoe import apply_move, is_terminal, legal_moves, winner


def random_rollout(board: str, player_to_move: str, rng: random.Random) -> str | None:
    """Play uniformly-random moves from `board` until the game ends.

    Returns the winning player ('X' or 'O'), or None for a draw.
    """
    current_board = board
    current_player = player_to_move

    while not is_terminal(current_board):
        move = rng.choice(legal_moves(current_board))
        current_board = apply_move(current_board, move, current_player)
        current_player = "O" if current_player == "X" else "X"

    return winner(current_board)
