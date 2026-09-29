import numpy as np

from bridge.monte_carlo.estimator import die_roll_sampler, estimate_mean


def test_die_mean_converges_near_3_5():
    rng = np.random.default_rng(0)
    mean, se = estimate_mean(die_roll_sampler, n_samples=100_000, rng=rng)
    assert abs(mean - 3.5) < 0.05
    assert se > 0
