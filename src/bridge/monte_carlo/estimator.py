"""A simple Monte Carlo estimator for the expected value of a random process."""

from collections.abc import Callable

import numpy as np


def estimate_mean(
    sampler: Callable[[np.random.Generator, int], np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Estimate E[X] by drawing n_samples i.i.d. samples from `sampler`.

    Parameters
    ----------
    sampler:
        A function (rng, n) -> array of n samples of the random variable.
    n_samples:
        Number of samples to draw.
    rng:
        A numpy random Generator, so results are reproducible.

    Returns
    -------
    (mean, standard_error): the sample mean and its estimated standard error
    (sigma_hat / sqrt(n)), from section 6 of the Week 1 derivation.
    """
    samples = sampler(rng, n_samples)
    mean = float(np.mean(samples))
    std_error = float(np.std(samples, ddof=1) / np.sqrt(n_samples))
    return mean, std_error


def die_roll_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n rolls of a fair six-sided die."""
    return rng.integers(1, 7, size=n)
