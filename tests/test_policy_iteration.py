from bridge.mdp.gridworld import GridWorld
from bridge.mdp.policy_iteration import policy_iteration
from bridge.mdp.value_iteration import value_iteration


def test_policy_iteration_matches_value_iteration():
    env = GridWorld()
    V_vi, policy_vi = value_iteration(env)
    V_pi, policy_pi = policy_iteration(env)

    # Both algorithms solve the same MDP, so they should agree.
    for s in env.states():
        assert abs(V_vi[s] - V_pi[s]) < 1e-6
        assert policy_vi[s] == policy_pi[s]
