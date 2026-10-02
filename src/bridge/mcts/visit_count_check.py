"""Week 3 checkpoint: run repeated UCB1 selection, with some branches
given artificially good/bad outcomes, and show visits concentrate on
the good branch while still occasionally revisiting the others.

Real outcomes come from simulation in Week 4 -- here we hand-assign
value_sum/visits to a few children to stand in for that, purely to
demonstrate UCB1's exploration/exploitation behavior in isolation.
"""

import matplotlib.pyplot as plt

from bridge.mcts.node import Node
from bridge.mcts.tictactoe import EMPTY_BOARD, legal_moves


def seed_fake_outcomes(root: Node) -> None:
    """Fully expand the root, then hand-assign mean outcomes: move 4
    (center) looks clearly best, move 0 looks clearly worst, the rest
    are mediocre -- mimicking what real simulation results might show."""
    for _ in legal_moves(root.board):
        root.expand()

    fake_means = {
        0: 0.1,
        1: 0.4,
        2: 0.4,
        3: 0.4,
        4: 0.9,
        5: 0.4,
        6: 0.4,
        7: 0.4,
        8: 0.4,
    }
    for move, child in root.children.items():
        child.visits = 5
        child.value_sum = fake_means[move] * 5
        root.visits += child.visits


def run_checkpoint(n_iterations: int = 500, c: float = 1.41421356) -> dict[int, int]:
    """Repeatedly pick root's best child via UCB1 and record a visit each
    time, as a stand-in for a full MCTS iteration landing on that branch."""
    root = Node(EMPTY_BOARD, player_to_move="X")
    seed_fake_outcomes(root)

    visit_log = {move: 0 for move in root.children}
    for _ in range(n_iterations):
        chosen = root.best_child(c)
        chosen.visits += 1
        root.visits += 1
        visit_log[chosen.move_from_parent] += 1

    return visit_log


def main() -> None:
    visit_log = run_checkpoint()
    moves = sorted(visit_log.keys())
    counts = [visit_log[m] for m in moves]

    plt.figure(figsize=(7, 5))
    plt.bar([str(m) for m in moves], counts)
    plt.xlabel("move (child index)")
    plt.ylabel("visit count")
    plt.title("UCB1 visit counts: move 4 seeded best, move 0 seeded worst")
    plt.tight_layout()
    plt.savefig("results/figures/week3_visit_counts.png", dpi=150)
    plt.show()
    print(visit_log)


if __name__ == "__main__":
    main()
