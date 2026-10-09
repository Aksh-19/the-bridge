"""Week 6: draw MCTS search trees for hand-picked decisions.

Each figure shows the tree grown from one decision. Edge thickness is how
often the search went down that branch; edge colour says whether greedy
would also have taken that action; dashed edges are the less likely
price moves.
"""

import math
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bridge.envs.trading import State, TradingEnv
from bridge.mcts.stochastic_agent import (
    StateNode,
    mcts_trading_search,
    most_visited_action,
)
from bridge.mcts.trading_baselines import greedy_trading_move
from bridge.mdp.trading_dp import solve

SHARED_COLOR = "#4c78a8"  # greedy would take this action too
NON_GREEDY_COLOR = "#e45756"  # greedy would reject this action
LONG_FILL = "#c7e9c0"
FLAT_FILL = "#eeeeee"

# (name, episode, time step): decisions found by find_disagreements.
EXAMPLES = [
    ("A_price6_t0", 4, 0),
    ("B_price7_t1", 12, 1),
    ("C_price9_t0", 9, 0),
]

# (parent, child, visits, probability of that price move, greedy agrees)
Edge = tuple[StateNode, StateNode, int, float, bool]


def outcome_probability(env: TradingEnv, state: State, next_state: State) -> float:
    """Probability of the price move that led from `state` to `next_state`."""
    p_up = env.p_up(state.price)
    went_up = next_state.price > state.price or (
        next_state.price == state.price == env.max_price
    )
    return p_up if went_up else 1.0 - p_up


class TreeLayout:
    """Tidy-tree layout: leaves get consecutive x positions, and every
    parent sits at the average x of its children."""

    def __init__(self, env: TradingEnv, max_depth: int = 3, min_visits: int = 8):
        self.env = env
        self.max_depth = max_depth
        self.min_visits = min_visits
        self.nodes: dict[int, StateNode] = {}
        self.positions: dict[int, tuple[float, int]] = {}
        self.edges: list[Edge] = []
        self._next_leaf_x = 0

    def build(self, node: StateNode, depth: int = 0) -> float:
        """Lay out the subtree under `node`; return the node's x position."""
        greedy_action = greedy_trading_move(self.env, node.state, random.Random(0))

        kids = []
        if depth < self.max_depth:
            for (action, next_state), visits in sorted(
                node.edge_visits.items(), key=lambda kv: (kv[0][0], kv[0][1].price)
            ):
                if visits >= self.min_visits:
                    kids.append((action, next_state, visits))

        if not kids:
            x = float(self._next_leaf_x)
            self._next_leaf_x += 1
        else:
            child_xs = []
            for action, next_state, visits in kids:
                child = node.children[(action, next_state)]
                child_xs.append(self.build(child, depth + 1))
                prob = outcome_probability(self.env, node.state, next_state)
                self.edges.append((node, child, visits, prob, action == greedy_action))
            x = sum(child_xs) / len(child_xs)

        self.nodes[id(node)] = node
        self.positions[id(node)] = (x, -depth)
        return x


def draw_tree(
    env: TradingEnv,
    root: StateNode,
    title: str,
    path: str,
    max_depth: int = 3,
    min_visits: int = 8,
) -> None:
    layout = TreeLayout(env, max_depth, min_visits)
    layout.build(root)

    fig, ax = plt.subplots(figsize=(14, 7))

    for parent, child, visits, prob, shared in layout.edges:
        x0, y0 = layout.positions[id(parent)]
        x1, y1 = layout.positions[id(child)]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=SHARED_COLOR if shared else NON_GREEDY_COLOR,
            linewidth=0.5 + 9 * math.sqrt(visits / root.visits),
            linestyle="-" if prob >= 0.5 else "--",
            alpha=0.85,
            zorder=1,
        )
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            str(visits),
            fontsize=7,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
            zorder=2,
        )

    for key, (x, y) in layout.positions.items():
        node = layout.nodes[key]
        fill = LONG_FILL if node.state.position == 1 else FLAT_FILL
        ax.text(
            x,
            y,
            str(node.state.price),
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "circle,pad=0.3", "fc": fill, "ec": "black"},
            zorder=3,
        )

    xs = [x for x, _ in layout.positions.values()]
    max_level = -min(y for _, y in layout.positions.values())
    x_min, x_max = min(xs) - 1.2, max(xs) + 0.8
    for level in range(max_level + 1):
        ax.text(x_min, -level, f"t = {root.state.t + level}", fontsize=9, va="center")
    ax.set_xlim(x_min - 0.3, x_max)
    ax.set_ylim(-max_level - 0.6, 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=10)

    handles = [
        Line2D(
            [0], [0], color=SHARED_COLOR, lw=3, label="greedy would take this action"
        ),
        Line2D([0], [0], color=NON_GREEDY_COLOR, lw=3, label="greedy would reject it"),
        Line2D([0], [0], color="black", lw=2, ls="--", label="less likely price move"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=LONG_FILL,
            markeredgecolor="black",
            markersize=10,
            label="holding after action",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=FLAT_FILL,
            markeredgecolor="black",
            markersize=10,
            label="flat after action",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
    )

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def replay_search(
    env: TradingEnv,
    episode: int,
    target_t: int,
    n_iterations: int = 500,
    c: float = 3.0,
    seed_offset: int = 20_000,
) -> tuple[State, StateNode]:
    """Replay an episode from find_disagreements (same seeds, so the same
    trees) and return the state and search tree at time step `target_t`."""
    env_rng = random.Random(seed_offset + episode)
    agent_rng = random.Random(episode)
    state = env.initial_state(env_rng)
    while True:
        root = mcts_trading_search(env, state, n_iterations, c, agent_rng)
        if state.t == target_t:
            return state, root
        state, _ = env.step(state, most_visited_action(root), env_rng)


def main() -> None:
    env = TradingEnv()
    _, q_opt = solve(env)

    for name, episode, t in EXAMPLES:
        state, root = replay_search(env, episode, t)
        mcts_action = most_visited_action(root)
        greedy_action = greedy_trading_move(env, state, random.Random(0))
        q = q_opt[(state.t, state.price, state.position)]
        position = "long" if state.position else "flat"

        title = (
            f"Example {name}: t={state.t}, price={state.price}, "
            f"currently {position}\n"
            f"MCTS chose {mcts_action} "
            f"(visits: long={root.action_visits.get('long', 0)}, "
            f"flat={root.action_visits.get('flat', 0)}); "
            f"greedy chose {greedy_action}; "
            f"exact Q*: long={q['long']:+.2f}, flat={q['flat']:+.2f}"
        )
        path = f"results/figures/week6_tree_{name}.png"
        draw_tree(env, root, title, path)
        print(f"saved {path}")
        print(f"  {title.splitlines()[1]}")


if __name__ == "__main__":
    main()
