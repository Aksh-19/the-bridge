"""Week 6: find decisions where MCTS and the greedy baseline disagree.

We play episodes following MCTS and, at every decision, also ask what
greedy would do. Disagreements are the candidate "non-obvious move"
examples for the visualization and the report.
"""

import random
from collections import Counter

from bridge.envs.trading import ACTIONS, State, TradingEnv
from bridge.mcts.stochastic_agent import (
    StateNode,
    mcts_trading_search,
    most_visited_action,
)
from bridge.mcts.trading_baselines import greedy_trading_move

Disagreement = tuple[int, State, str, str, StateNode]


def find_disagreements(
    n_episodes: int = 30,
    n_iterations: int = 500,
    c: float = 3.0,
    seed_offset: int = 20_000,
) -> tuple[list[Disagreement], int]:
    """Return (disagreements, total_decisions).

    Each disagreement is (episode, state, mcts_action, greedy_action, tree).
    Uses a different seed range than the Week 5 evaluation, so these
    exploratory runs don't touch the reported test episodes.
    """
    env = TradingEnv()
    found: list[Disagreement] = []
    total_decisions = 0

    for i in range(n_episodes):
        env_rng = random.Random(seed_offset + i)
        agent_rng = random.Random(i)
        state = env.initial_state(env_rng)

        while not env.is_terminal(state):
            root = mcts_trading_search(env, state, n_iterations, c, agent_rng)
            mcts_action = most_visited_action(root)
            greedy_action = greedy_trading_move(env, state, agent_rng)

            total_decisions += 1
            if mcts_action != greedy_action:
                found.append((i, state, mcts_action, greedy_action, root))

            state, _ = env.step(state, mcts_action, env_rng)

    return found, total_decisions


def describe(entry: Disagreement) -> str:
    episode, state, mcts_action, greedy_action, root = entry
    stats = " | ".join(
        f"{a}: visits={root.action_visits.get(a, 0)}, mean={root.mean_value(a):+.2f}"
        for a in ACTIONS
    )
    return (
        f"ep {episode:2d}, t={state.t:2d}, price={state.price:2d}, "
        f"position={state.position}: MCTS={mcts_action}, "
        f"greedy={greedy_action}  [{stats}]"
    )


def main() -> None:
    found, total = find_disagreements()
    print(f"{len(found)} disagreements out of {total} decisions")

    pairs = Counter((mcts, greedy) for _, _, mcts, greedy, _ in found)
    for (mcts, greedy), count in pairs.most_common():
        print(f"  MCTS={mcts}, greedy={greedy}: {count}")

    print("\nEarliest disagreements (longest remaining horizon):")
    for entry in sorted(found, key=lambda e: e[1].t)[:10]:
        print(" ", describe(entry))


if __name__ == "__main__":
    main()
