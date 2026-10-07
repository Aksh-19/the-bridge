"""MCTS tree node, with UCB1-based child selection (Week 3) and expansion.
Simulation and backpropagation are added in Week 4."""

import math

from bridge.mcts.tictactoe import apply_move, legal_moves, other_player


class Node:
    """One board state in the search tree.

    Attributes
    ----------
    board: the board string this node represents.
    player_to_move: 'X' or 'O' -- whose turn it is FROM this state.
    parent: the Node that led here (None for the root).
    move_from_parent: the move index that produced this node from its parent.
    children: dict mapping move -> child Node, filled in by expand().
    visits: how many times this node has been visited (n in the UCB1 formula).
    value_sum: sum of simulation outcomes backpropagated through this node
        (used in Week 4; total so far, divide by visits to get the mean).
    """

    def __init__(
        self,
        board: str,
        player_to_move: str,
        parent: Node | None = None,
        move_from_parent: int | None = None,
    ):
        self.board = board
        self.player_to_move = player_to_move
        self.parent = parent
        self.move_from_parent = move_from_parent
        self.children: dict[int, Node] = {}
        self.visits = 0
        self.value_sum = 0.0

    def is_fully_expanded(self) -> bool:
        """True once every legal move from this board has a child node."""
        return len(self.children) == len(legal_moves(self.board))

    def mean_value(self) -> float:
        """X-bar_n from the UCB1 formula: average outcome seen so far."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def ucb1_score(self, total_visits: int, c: float = math.sqrt(2)) -> float:
        """UCB1 score: exploitation term + exploration bonus.

        `c` scales the exploration term; sqrt(2) is the classic choice
        from the Hoeffding-based derivation (UCB1 original paper).
        An unvisited node returns infinity, so it's always selected first.
        """
        if self.visits == 0:
            return math.inf
        exploitation = self.mean_value()
        exploration = c * math.sqrt(math.log(total_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float = math.sqrt(2)) -> Node:
        """Select the child with the highest UCB1 score."""
        return max(
            self.children.values(),
            key=lambda child: child.ucb1_score(self.visits, c),
        )

    def expand(self) -> Node:
        """Create one new child for an untried move, return it.

        Picks the first legal move that doesn't already have a child --
        good enough for now; which untried move to pick first doesn't
        affect correctness, just the order nodes get created in.
        """
        tried_moves = set(self.children.keys())
        for move in legal_moves(self.board):
            if move not in tried_moves:
                child_board = apply_move(self.board, move, self.player_to_move)
                child = Node(
                    board=child_board,
                    player_to_move=other_player(self.player_to_move),
                    parent=self,
                    move_from_parent=move,
                )
                self.children[move] = child
                return child
        raise RuntimeError("expand() called on a fully expanded node")


def backpropagate(node: Node, winning_player: str | None) -> None:
    """Walk from `node` up to the root, updating visits and value_sum at
    every node on the path.

    Value convention: a node's value_sum represents the win rate for
    whichever player made the move that LED to this node -- i.e. the
    player who was to move at this node's parent. That's always
    other_player(node.player_to_move), since node.player_to_move is
    whoever moves NEXT, not whoever just moved.
    """
    from bridge.mcts.tictactoe import other_player

    current = node
    while current is not None:
        current.visits += 1
        mover = other_player(current.player_to_move)
        if winning_player is None:
            reward = 0.5  # draw
        elif winning_player == mover:
            reward = 1.0
        else:
            reward = 0.0
        current.value_sum += reward
        current = current.parent
