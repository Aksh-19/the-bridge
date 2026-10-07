"""Week 4: the complete MCTS agent -- select, expand, simulate,
backpropagate, repeated for N iterations, then pick the most-visited
root child as the move to play."""

import random

from bridge.mcts.node import Node, backpropagate
from bridge.mcts.select import select_and_expand
from bridge.mcts.simulate import random_rollout
from bridge.mcts.tictactoe import is_terminal, winner


def mcts_search(
    board: str,
    player_to_move: str,
    n_iterations: int = 500,
    c: float = 1.41421356,
    rng: random.Random | None = None,
) -> int:
    """Run MCTS from `board` for n_iterations, return the chosen move.

    Each iteration: select+expand down to a frontier node, simulate a
    random rollout from there, then backpropagate the result up to the
    root. After all iterations, pick the root child that was visited
    most -- the standard MCTS move-selection rule (more robust than
    picking by mean value alone, since visit count reflects sustained
    confidence, not a single lucky rollout).
    """
    if rng is None:
        rng = random.Random()

    root = Node(board, player_to_move)

    for _ in range(n_iterations):
        leaf = select_and_expand(root, c)

        if is_terminal(leaf.board):
            outcome = winner(leaf.board)
        else:
            outcome = random_rollout(leaf.board, leaf.player_to_move, rng)

        backpropagate(leaf, outcome)

    best_move = max(root.children, key=lambda move: root.children[move].visits)
    return best_move
