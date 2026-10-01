"""Week 2 checkpoint: brute-force verify that a solved (V, policy) pair is
truly optimal, by independently recomputing the Bellman-optimal action
value at every state and checking nothing beats the policy's choice."""

from bridge.mdp.gridworld import GridWorld


def verify_optimal(
    env: GridWorld, V: dict, policy: dict, gamma: float = 0.9, tol: float = 1e-6
) -> bool:
    """Returns True if, for every state, the policy's action attains the
    max one-step-lookahead value, and V matches that max."""
    for s in env.states():
        actions = env.actions(s)
        if not actions:  # terminal state: nothing to check
            continue

        action_values = {
            a: env.step(s, a)[1] + gamma * V[env.step(s, a)[0]] for a in actions
        }
        best_value = max(action_values.values())
        policy_value = action_values[policy[s]]

        if abs(policy_value - best_value) > tol:
            print(
                f"FAIL at {s}: policy picked {policy[s]} (value {policy_value:.4f}), "
                f"but best was {best_value:.4f}"
            )
            return False

        if abs(V[s] - best_value) > tol:
            print(
                f"FAIL at {s}: V[s]={V[s]:.4f} "
                f"does not match best_value={best_value:.4f}"
            )
            return False

    return True
