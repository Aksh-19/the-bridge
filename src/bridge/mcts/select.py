"""Week 3: the selection + expansion phases of MCTS, combined into one
tree-descent step. Simulation and backpropagation come in Week 4."""

from bridge.mcts.node import Node
from bridge.mcts.tictactoe import is_terminal


def select_and_expand(root: Node, c: float = 1.41421356) -> Node:
    """Descend the tree from `root`, following UCB1 at each fully-expanded
    node, until reaching either a terminal board or a node with an
    untried move -- then expand once and return the new (or terminal)
    node.
    """
    node = root

    while not is_terminal(node.board):
        if not node.is_fully_expanded():
            return node.expand()
        node = node.best_child(c)

    return node
