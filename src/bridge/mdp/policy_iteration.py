"""Policy iteration: alternate between evaluating a fixed policy exactly
and improving it greedily, until the policy stops changing."""

from bridge.mdp.gridworld import GridWorld


def policy_evaluation(
    env: GridWorld, policy: dict, gamma: float = 0.9, theta: float = 1e-8
) -> dict:
    """Compute V^pi for a FIXED policy, by repeatedly applying the Bellman
    equation for V^pi (no max -- just follow the policy's chosen action)."""
    V = {s: 0.0 for s in env.states()}

    while True:
        delta = 0.0
        for s in env.states():
            a = policy[s]
            if a is None:  # terminal state
                continue
            next_state, reward = env.step(s, a)
            new_value = reward + gamma * V[next_state]
            delta = max(delta, abs(new_value - V[s]))
            V[s] = new_value

        if delta < theta:
            break

    return V


def policy_improvement(env: GridWorld, V: dict, gamma: float = 0.9) -> dict:
    """Given V, build the greedy policy: at each state, pick the action
    with the highest one-step lookahead value."""
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
    return policy


def policy_iteration(env: GridWorld, gamma: float = 0.9) -> tuple[dict, dict]:
    """Alternate evaluation and improvement until the policy stops
    changing (this is the convergence criterion for policy iteration,
    distinct from value iteration's "V stops changing")."""
    # Start with an arbitrary policy: always "up" wherever possible.
    policy = {s: (env.actions(s)[0] if env.actions(s) else None) for s in env.states()}

    while True:
        V = policy_evaluation(env, policy, gamma)
        new_policy = policy_improvement(env, V, gamma)

        if new_policy == policy:  # no state's best action changed
            break
        policy = new_policy

    return V, policy
