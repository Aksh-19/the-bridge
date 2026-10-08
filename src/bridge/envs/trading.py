"""Week 5: a stochastic trading environment. A mean-reverting integer
price, a flat/long position, and a transaction cost for switching.
Designed so that multi-step planning matters: one step of drift is
smaller than the cost of trading, but several steps of drift are not."""

import random
from dataclasses import dataclass

ACTIONS = ("flat", "long")


@dataclass(frozen=True)
class State:
    t: int  # current time step
    price: int  # current price level
    position: int  # 0 = flat, 1 = long


@dataclass
class TradingEnv:
    horizon: int = 15
    min_price: int = 0
    max_price: int = 20
    mean_price: int = 10
    reversion_strength: float = 0.05
    cost: float = 0.5

    def initial_state(self, rng: random.Random) -> State:
        """Start flat, at a random price near the mean."""
        price = rng.randint(self.mean_price - 5, self.mean_price + 5)
        return State(t=0, price=price, position=0)

    def is_terminal(self, state: State) -> bool:
        return state.t >= self.horizon

    def actions(self, state: State) -> tuple[str, ...]:
        return () if self.is_terminal(state) else ACTIONS

    def p_up(self, price: int) -> float:
        """Probability the price ticks up: higher when below the mean."""
        p = 0.5 + self.reversion_strength * (self.mean_price - price)
        return min(max(p, 0.05), 0.95)

    def step(
        self, state: State, action: str, rng: random.Random
    ) -> tuple[State, float]:
        """Apply `action`, let the price move randomly, return
        (next_state, reward)."""
        new_position = 1 if action == "long" else 0
        trade_cost = self.cost if new_position != state.position else 0.0

        move = 1 if rng.random() < self.p_up(state.price) else -1
        next_price = min(max(state.price + move, self.min_price), self.max_price)

        reward = new_position * (next_price - state.price) - trade_cost
        return State(state.t + 1, next_price, new_position), reward

    def expected_reward(self, state: State, action: str) -> float:
        """Exact one-step expected reward (no sampling). The greedy
        baseline will use this, since it only ever looks one step ahead."""
        new_position = 1 if action == "long" else 0
        trade_cost = self.cost if new_position != state.position else 0.0

        p = self.p_up(state.price)
        up_price = min(state.price + 1, self.max_price)
        down_price = max(state.price - 1, self.min_price)
        expected_change = p * (up_price - state.price) + (1 - p) * (
            down_price - state.price
        )
        return new_position * expected_change - trade_cost
