"""Week 5: MCTS adapted to a one-player, stochastic environment.

Differences from the tic-tac-toe agent: a node's value is plain total
reward (no per-player perspective), and because the same action can lead
to different next states, children are keyed by (action, next_state).
"""

import math
import random

from bridge.envs.trading import State, TradingEnv


class StateNode:
    """One state in the search tree, with statistics PER ACTION."""

    def __init__(self, state: State):
        self.state = state
        self.visits = 0
        self.action_visits: dict[str, int] = {}
        self.action_value_sum: dict[str, float] = {}
        self.children: dict[tuple[str, State], StateNode] = {}

    def untried_action(self, actions: tuple[str, ...]) -> str | None:
        """First action never tried from this node, or None if all tried."""
        for action in actions:
            if self.action_visits.get(action, 0) == 0:
                return action
        return None

    def ucb_action(self, actions: tuple[str, ...], c: float) -> str:
        """Pick the action with the highest UCB1 score."""

        def score(action: str) -> float:
            n = self.action_visits[action]
            mean = self.action_value_sum[action] / n
            return mean + c * math.sqrt(math.log(self.visits) / n)

        return max(actions, key=score)

    def update(self, action: str, ret: float) -> None:
        """Record one more visit to `action` with total return `ret`."""
        self.visits += 1
        self.action_visits[action] = self.action_visits.get(action, 0) + 1
        self.action_value_sum[action] = self.action_value_sum.get(action, 0.0) + ret


def rollout_return(env: TradingEnv, state: State, rng: random.Random) -> float:
    """Play uniformly random actions to the horizon; return total reward."""
    total = 0.0
    while not env.is_terminal(state):
        action = rng.choice(env.actions(state))
        state, reward = env.step(state, action, rng)
        total += reward
    return total


def mcts_trading_move(
    env: TradingEnv,
    state: State,
    n_iterations: int = 500,
    c: float = 3.0,
    rng: random.Random | None = None,
) -> str:
    """Run MCTS from `state`, return the most-visited root action."""
    if rng is None:
        rng = random.Random()

    root = StateNode(state)

    for _ in range(n_iterations):
        node = root
        path: list[tuple[StateNode, str, float]] = []
        rollout_value = 0.0

        # Select / expand: descend until we create a brand-new node.
        while not env.is_terminal(node.state):
            actions = env.actions(node.state)
            action = node.untried_action(actions)
            if action is None:
                action = node.ucb_action(actions, c)

            next_state, reward = env.step(node.state, action, rng)
            path.append((node, action, reward))

            key = (action, next_state)
            if key not in node.children:
                node.children[key] = StateNode(next_state)
                # Simulate: random playout from the new node.
                rollout_value = rollout_return(env, next_state, rng)
                break
            node = node.children[key]

        # Backpropagate: return-to-go, G_t = r_t + G_{t+1}.
        ret = rollout_value
        for visited, action, reward in reversed(path):
            ret = reward + ret
            visited.update(action, ret)

    return max(root.action_visits, key=lambda a: root.action_visits[a])
