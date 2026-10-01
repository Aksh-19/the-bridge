from bridge.mdp.gridworld import GridWorld
from bridge.mdp.value_iteration import value_iteration
from bridge.mdp.verify_optimal import verify_optimal


def test_value_iteration_solution_is_optimal():
    env = GridWorld()
    V, policy = value_iteration(env)
    assert verify_optimal(env, V, policy)
