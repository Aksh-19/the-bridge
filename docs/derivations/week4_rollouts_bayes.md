# Week 4: Rollouts as Unbiased Estimators, Backpropagation as Bayesian Updating

## 1. Unbiasedness of a rollout

Let $Z$ be the outcome of a single random rollout from state $s$ (e.g. $Z=1$ for a win, $0$ for a loss, $0.5$ for a draw), played out under a fixed rollout policy (uniform random moves by both sides). Define:

$$v(s) = E[Z \mid S_0 = s]$$

the true expected outcome of that rollout policy starting from $s$. By this definition, a single rollout's outcome $Z$ is an **unbiased estimator** of $v(s)$:

$$E[Z \mid S_0=s] = v(s)$$

A single rollout is noisy -- $Z$ only ever takes the values $0$, $0.5$, or $1$, while $v(s)$ is typically some number in between. But this is exactly the die-roll situation from Week 1: one roll is a noisy, unbiased estimate of $E[X]=3.5$, never landing on $3.5$ itself. By the law of large numbers (Week 1, section 7), averaging $n$ independent rollouts from the same node converges to the true mean:

$$\bar{Z}_n = \frac{1}{n}\sum_{i=1}^{n} Z_i \;\xrightarrow[n\to\infty]{}\; v(s)$$

This is why MCTS doesn't trust a single rollout -- it revisits each node many times (driven by UCB1's exploration term, section 2 of Week 3) and relies on the running average `value_sum / visits` converging toward $v(s)$, the same way the Week 1 die-roll average converged toward $3.5$.


## 2. Incremental mean update

By definition, the mean of $n+1$ samples is:

$$\bar{X}_{n+1} = \frac{1}{n+1}\sum_{i=1}^{n+1} Z_i$$

Split off the last term from the sum:

$$\bar{X}_{n+1} = \frac{1}{n+1}\left(\sum_{i=1}^{n} Z_i + Z_{n+1}\right)$$

Recognize that $\sum_{i=1}^{n} Z_i = n\bar{X}_n$, since $\bar{X}_n$ is by definition the mean of the first $n$ samples:

$$\bar{X}_{n+1} = \frac{n\bar{X}_n + Z_{n+1}}{n+1}$$

Split the fraction into two terms:

$$\bar{X}_{n+1} = \frac{n\bar{X}_n}{n+1} + \frac{Z_{n+1}}{n+1}$$

Rewrite $\dfrac{n}{n+1} = \dfrac{n+1-1}{n+1} = 1 - \dfrac{1}{n+1}$, so the first term becomes $\bar{X}_n - \dfrac{\bar{X}_n}{n+1}$:

$$\bar{X}_{n+1} = \bar{X}_n - \frac{\bar{X}_n}{n+1} + \frac{Z_{n+1}}{n+1}$$

Combine the last two terms:

$$\boxed{\bar{X}_{n+1} = \bar{X}_n + \frac{1}{n+1}\big(Z_{n+1} - \bar{X}_n\big)}$$

## 3. The Bayesian analogy

In Bayesian updating, a **prior** belief is combined with new **evidence** to form a **posterior** belief, and the posterior shifts toward the evidence by an amount that depends on how confident the prior already was. A prior built from very little data is easily swayed by one new observation; a prior built from a large, well-established body of evidence barely moves in response to a single new data point.

The incremental mean update from section 2,

$$\bar{X}_{n+1} = \bar{X}_n + \frac{1}{n+1}\big(Z_{n+1} - \bar{X}_n\big)$$

has exactly this shape. $\bar{X}_n$ is the belief about this node's value *before* seeing the new rollout -- the "prior." $Z_{n+1}$ is the new rollout outcome -- the "evidence." The update moves $\bar{X}_n$ toward $Z_{n+1}$, but by a step size of $\frac{1}{n+1}$, which shrinks as $n$ grows. Early on (small $n$), a single new rollout can swing the node's estimated value substantially -- the "prior" is weak, so new evidence dominates. After many visits (large $n$), the same single rollout barely moves the average at all -- the accumulated evidence is strong, and one more noisy sample is a drop in the bucket.

This is precisely the behavior a Bayesian posterior exhibits as the amount of prior evidence grows, which is why backpropagation's running-mean update is described as Bayesian-style, even though no explicit prior distribution or Bayes' rule is used here -- it's the same *qualitative* belief-revision pattern, implemented through a simple running average.
