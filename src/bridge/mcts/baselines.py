"""Baseline opponents for Week 4's checkpoint: a random-move player, and
a greedy player that takes an immediate win if available, blocks an
opponent's immediate win otherwise, and moves randomly if neither
applies."""

import random

from bridge.mcts.tictactoe import apply_move, legal_moves, other_player, winner


def random_move(board: str, player_to_move: str, rng: random.Random) -> int:
    return rng.choice(legal_moves(board))


def greedy_move(board: str, player_to_move: str, rng: random.Random) -> int:
    """Take any move that wins immediately; otherwise block any move
    that would let the opponent win immediately; otherwise move
    randomly. Still "greedy" in the roadmap's sense -- one ply of
    lookahead only, no planning further ahead than that.
    """
    opponent = other_player(player_to_move)

    # 1. Win now if possible.
    for move in legal_moves(board):
        if winner(apply_move(board, move, player_to_move)) == player_to_move:
            return move

    # 2. Otherwise, block the opponent's immediate win, if they have one.
    for move in legal_moves(board):
        if winner(apply_move(board, move, opponent)) == opponent:
            return move

    # 3. Otherwise, no urgent move either way -- play randomly.
    return rng.choice(legal_moves(board))
