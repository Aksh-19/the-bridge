import numpy as np

from bridge.envs.trading import ACTIONS, State, TradingEnv
from bridge.mcts.trading_baselines import greedy_trading_move
from bridge.mcts.trading_eval import mean_and_ci, run_episode
from bridge.mdp.trading_dp import solve


def test_last_step_value_matches_one_step_expected_reward():
    env = TradingEnv()
    v, _ = solve(env)
    t = env.horizon - 1
    for price in range(env.min_price, env.max_price + 1):
        for pos in (0, 1):
            state = State(t, price, pos)
            best = max(env.expected_reward(state, a) for a in ACTIONS)
            assert abs(v[(t, price, pos)] - best) < 1e-9


def test_optimal_value_is_at_least_greedy_value_everywhere():
    env = TradingEnv()
    v_opt, _ = solve(env)
    v_greedy, _ = solve(env, greedy_trading_move)
    for key, value in v_greedy.items():
        assert v_opt[key] >= value - 1e-9


def test_exact_greedy_value_matches_simulation():
    env = TradingEnv()
    v_greedy, _ = solve(env, greedy_trading_move)
    starts = range(env.mean_price - 5, env.mean_price + 6)
    exact = sum(v_greedy[(0, p, 0)] for p in starts) / len(starts)

    returns = np.array(
        [
            run_episode(env, greedy_trading_move, env_seed=i, agent_seed=i)
            for i in range(20_000)
        ]
    )
    mean, half = mean_and_ci(returns)
    assert abs(mean - exact) < 3 * half
