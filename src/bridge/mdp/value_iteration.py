"""Value iteration: repeatedly apply the Bellman optimality backup until V
stops changing, then read off the greedy policy (Week 2 checkpoint)."""

from bridge.mdp.gridworld import GridWorld


def value_iteration(
    env: GridWorld, gamma: float = 0.9, theta: float = 1e-8
) -> tuple[dict, dict]:
    """Solve for V* and the optimal policy pi* by iterating the Bellman
    optimality backup:  V(s) <- max_a [ R(s,a,s') + gamma * V(s') ]

    Parameters
    ----------
    gamma: discount factor.
    theta: stop when the largest change in V across all states drops
        below this threshold (convergence tolerance).

    Returns
    -------
    (V, policy): V maps state -> float value. policy maps state -> best
    action name (empty for the terminal/goal state).
    """
    V = {s: 0.0 for s in env.states()}

    while True:
        delta = 0.0  # largest change in V this sweep, for the stopping check
        for s in env.states():
            actions = env.actions(s)
            if not actions:  # terminal state: no decision to make, V stays 0
                continue

            action_values = []
            for a in actions:
                next_state, reward = env.step(s, a)
                action_values.append(reward + gamma * V[next_state])

            best_value = max(action_values)
            delta = max(delta, abs(best_value - V[s]))
            V[s] = best_value

        if delta < theta:
            break

    # Extract the greedy policy: at each state, the action with highest value.
    policy = {}
    for s in env.states():
        actions = env.actions(s)
        if not actions:
            policy[s] = None
            continue
        action_values = {}
        for a in actions:
            next_state, reward = env.step(s, a)
            action_values[a] = reward + gamma * V[next_state]
        policy[s] = max(action_values, key=action_values.get)

    return V, policy
