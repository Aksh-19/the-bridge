"""Week 1 checkpoint: empirical estimation error vs. N, compared to the
theoretical 1/sqrt(N) curve derived in docs/derivations/week1_expectation_lln.md.
"""

import matplotlib.pyplot as plt
import numpy as np

from bridge.monte_carlo.estimator import die_roll_sampler

TRUE_MEAN = 3.5
TRUE_SIGMA = np.sqrt(35 / 12)  # derived in section 2 of the Week 1 writeup


def run_convergence_check(
    n_values: np.ndarray, n_repeats: int, rng: np.random.Generator
) -> np.ndarray:
    """For each N in n_values, estimate |sample_mean - true_mean| averaged
    over n_repeats independent trials, to smooth out single-run noise."""
    errors = np.zeros(len(n_values))
    for i, n in enumerate(n_values):
        trial_errors = np.empty(n_repeats)
        for r in range(n_repeats):
            samples = die_roll_sampler(rng, n)
            trial_errors[r] = abs(np.mean(samples) - TRUE_MEAN)
        errors[i] = np.mean(trial_errors)
    return errors


def main() -> None:
    rng = np.random.default_rng(42)
    n_values = np.array([10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000])
    empirical_error = run_convergence_check(n_values, n_repeats=200, rng=rng)
    theoretical_error = TRUE_SIGMA / np.sqrt(n_values)

    plt.figure(figsize=(7, 5))
    plt.loglog(n_values, empirical_error, "o-", label="empirical mean |error|")
    plt.loglog(
        n_values, theoretical_error, "--", label=r"theoretical $\sigma/\sqrt{N}$"
    )
    plt.xlabel("number of samples (N)")
    plt.ylabel("estimation error")
    plt.title("Monte Carlo estimation error vs. N (fair die)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/week1_convergence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
