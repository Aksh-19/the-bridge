from bridge.mdp.gridworld import GridWorld
from bridge.mdp.value_iteration import value_iteration


def test_value_iteration_solves_gridworld():
    env = GridWorld()
    V, policy = value_iteration(env)

    # Value should strictly decrease with distance from goal (this grid has
    # no shortcuts through walls), since it costs one step_reward per move.
    assert V[(0, 2)] > V[(0, 0)]

    # The cell right next to the goal should prefer moving directly onto it.
    assert policy[(0, 2)] == "right"
