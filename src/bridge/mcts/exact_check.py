"""Week 6: compare MCTS and greedy decisions to the exact optimum.

Replays the same episodes as find_disagreements (same seeds), so the
decisions examined here are the same ones.
"""

import random

from bridge.envs.trading import State, TradingEnv
from bridge.mcts.stochastic_agent import mcts_trading_move
from bridge.mcts.trading_baselines import greedy_trading_move
from bridge.mdp.trading_dp import QTable, solve


def q_for(q_opt: QTable, state: State) -> dict[str, float]:
    return q_opt[(state.t, state.price, state.position)]


def regret(q: dict[str, float], action: str) -> float:
    return max(q.values()) - q[action]


def main() -> None:
    env = TradingEnv()
    v_opt, q_opt = solve(env)
    v_greedy, _ = solve(env, greedy_trading_move)

    starts = range(env.mean_price - 5, env.mean_price + 6)
    opt_start = sum(v_opt[(0, p, 0)] for p in starts) / len(starts)
    greedy_start = sum(v_greedy[(0, p, 0)] for p in starts) / len(starts)
    print("Exact expected return from the start state:")
    print(f"  optimal policy: {opt_start:+.3f}")
    print(f"  greedy policy:  {greedy_start:+.3f}")

    # Replay MCTS-driven episodes (same seeds as find_disagreements).
    decisions: list[tuple[int, State, str, str]] = []
    for i in range(30):
        env_rng = random.Random(20_000 + i)
        agent_rng = random.Random(i)
        state = env.initial_state(env_rng)
        while not env.is_terminal(state):
            mcts_action = mcts_trading_move(env, state, 500, 3.0, agent_rng)
            greedy_action = greedy_trading_move(env, state, agent_rng)
            decisions.append((i, state, mcts_action, greedy_action))
            state, _ = env.step(state, mcts_action, env_rng)

    n = len(decisions)
    mcts_regrets = [regret(q_for(q_opt, s), m) for _, s, m, _ in decisions]
    greedy_regrets = [regret(q_for(q_opt, s), g) for _, s, _, g in decisions]
    mcts_optimal = sum(r < 1e-9 for r in mcts_regrets)
    greedy_optimal = sum(r < 1e-9 for r in greedy_regrets)

    print(f"\nOver {n} decisions on MCTS-driven trajectories:")
    print(f"  MCTS picks an optimal action:   {mcts_optimal}/{n}")
    print(f"  greedy picks an optimal action: {greedy_optimal}/{n}")
    print(f"  mean regret, MCTS:   {sum(mcts_regrets) / n:.3f}")
    print(f"  mean regret, greedy: {sum(greedy_regrets) / n:.3f}")

    disagreements = [d for d in decisions if d[2] != d[3]]
    mcts_right = sum(regret(q_for(q_opt, s), m) < 1e-9 for _, s, m, _ in disagreements)
    print(f"\nWhere they disagree ({len(disagreements)} decisions):")
    print(
        f"  MCTS was optimal in {mcts_right}, greedy in "
        f"{len(disagreements) - mcts_right}"
    )

    print("\nEarliest disagreements with exact optimal Q-values:")
    for i, s, m, g in sorted(disagreements, key=lambda d: d[1].t)[:10]:
        q = q_for(q_opt, s)
        print(
            f"  ep {i:2d}, t={s.t:2d}, price={s.price:2d}, pos={s.position}: "
            f"MCTS={m}, greedy={g} | Q*(long)={q['long']:+.2f}, "
            f"Q*(flat)={q['flat']:+.2f}"
        )


if __name__ == "__main__":
    main()
