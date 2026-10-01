from bridge.mdp.gridworld import GridWorld


def test_step_into_wall_bounces_back():
    env = GridWorld()
    next_state, reward = env.step((0, 1), "down")  # (1,1) is a wall
    assert next_state == (0, 1)
    assert reward == env.step_reward


def test_reaching_goal_gives_goal_reward():
    env = GridWorld()
    next_state, reward = env.step((0, 2), "right")  # -> (0,3), the goal
    assert next_state == env.goal
    assert reward == env.goal_reward
