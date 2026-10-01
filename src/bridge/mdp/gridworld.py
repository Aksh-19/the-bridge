"""A small deterministic gridworld MDP, used to test value/policy iteration
against a brute-force-verifiable optimum (Week 2 checkpoint)."""

from dataclasses import dataclass, field

# Actions: (row_delta, col_delta)
ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class GridWorld:
    """A rectangular grid. The agent starts anywhere, moves one cell per
    step, and the episode ends on reaching `goal`. Moving into a wall or
    off the grid just keeps the agent in place (still costs a step)."""

    n_rows: int = 4
    n_cols: int = 4
    goal: tuple[int, int] = (0, 3)
    walls: set[tuple[int, int]] = field(default_factory=lambda: {(1, 1), (2, 1)})
    step_reward: float = -1.0
    goal_reward: float = 10.0

    def states(self) -> list[tuple[int, int]]:
        """All non-wall grid cells, including the goal (a terminal state)."""
        return [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if (r, c) not in self.walls
        ]

    def actions(self, state: tuple[int, int]) -> list[str]:
        """Every state (except the goal) allows all four moves."""
        if state == self.goal:
            return []
        return list(ACTIONS.keys())

    def step(
        self, state: tuple[int, int], action: str
    ) -> tuple[tuple[int, int], float]:
        """Deterministic transition: returns (next_state, reward).

        This environment has no randomness, so P(s'|s,a) is 1 for exactly
        one s' -- but value/policy iteration below are written generally,
        as if P were a distribution, so they'd work unchanged on a
        stochastic environment too.
        """
        dr, dc = ACTIONS[action]
        r, c = state[0] + dr, state[1] + dc

        if not (0 <= r < self.n_rows and 0 <= c < self.n_cols) or (r, c) in self.walls:
            r, c = state  # bounced off a wall or the grid edge

        next_state = (r, c)
        reward = self.goal_reward if next_state == self.goal else self.step_reward
        return next_state, reward
