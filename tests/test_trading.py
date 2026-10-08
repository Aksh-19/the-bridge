import random

from bridge.envs.trading import State, TradingEnv


class AlwaysUp(random.Random):
    """Fake RNG: random() returns 0.0, so the price always ticks up."""

    def random(self) -> float:
        return 0.0


def test_cost_charged_only_when_position_changes():
    env = TradingEnv()
    rng = AlwaysUp()

    # Flat -> long at price 10: gain 1, pay 0.5.
    next_state, reward = env.step(State(0, 10, 0), "long", rng)
    assert next_state == State(1, 11, 1)
    assert reward == 0.5

    # Staying long: gain 1, no cost.
    _, reward = env.step(next_state, "long", rng)
    assert reward == 1.0


def test_episode_terminates_at_horizon():
    env = TradingEnv()
    assert not env.is_terminal(State(14, 10, 0))
    assert env.is_terminal(State(15, 10, 0))
    assert env.actions(State(15, 10, 0)) == ()


def test_expected_reward_at_mean_price():
    env = TradingEnv()
    # At the mean, p_up = 0.5, so expected price change is 0.
    assert env.expected_reward(State(0, 10, 0), "flat") == 0.0
    assert env.expected_reward(State(0, 10, 0), "long") == -0.5
