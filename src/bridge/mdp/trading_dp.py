"""Week 6: exact solution of the trading MDP by backward induction.

This is the finite-horizon Bellman backup from Week 2 (gamma = 1): start
at the horizon, where value is 0, and work backward one step at a time.
The same code can also evaluate a fixed, deterministic policy exactly.
"""

import random
from collections.abc import Callable

from bridge.envs.trading import ACTIONS, State, TradingEnv

Key = tuple[int, int, int]  # (t, price, position)
ValueTable = dict[Key, float]
QTable = dict[Key, dict[str, float]]
Policy = Callable[[TradingEnv, State, random.Random], str]


def q_value(
    env: TradingEnv,
    V: ValueTable,
    t: int,
    price: int,
    pos: int,
    action: str,
) -> float:
    """Exact Q: expected reward plus value of the next state, averaged
    over the two possible price moves."""
    new_pos = 1 if action == "long" else 0
    cost = env.cost if new_pos != pos else 0.0
    p = env.p_up(price)
    up = min(price + 1, env.max_price)
    down = max(price - 1, env.min_price)
    up_total = new_pos * (up - price) - cost + V[(t + 1, up, new_pos)]
    down_total = new_pos * (down - price) - cost + V[(t + 1, down, new_pos)]
    return p * up_total + (1 - p) * down_total


def solve(env: TradingEnv, policy: Policy | None = None) -> tuple[ValueTable, QTable]:
    """Return (V, Q) tables keyed by (t, price, position).

    With policy=None, V is the optimal value (max over actions). With a
    deterministic policy, V is that policy's exact value. Q always holds
    the one-step-then-follow-V values for both actions.
    """
    prices = range(env.min_price, env.max_price + 1)
    dummy_rng = random.Random(0)  # deterministic policies ignore it

    V: ValueTable = {
        (env.horizon, price, pos): 0.0 for price in prices for pos in (0, 1)
    }
    Q: QTable = {}

    for t in reversed(range(env.horizon)):
        for price in prices:
            for pos in (0, 1):
                key = (t, price, pos)
                Q[key] = {a: q_value(env, V, t, price, pos, a) for a in ACTIONS}
                if policy is None:
                    V[key] = max(Q[key].values())
                else:
                    V[key] = Q[key][policy(env, State(t, price, pos), dummy_rng)]
    return V, Q
