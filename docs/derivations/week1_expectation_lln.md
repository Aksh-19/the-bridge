# Week 1: Expectation, Variance, and the Law of Large Numbers

## 1. Variance shortcut
Let $\mu = E[X]$. From the definition:

$$\mathrm{Var}(X) = E\big[(X-\mu)^2\big]$$

Expand the square, then apply linearity of expectation:

$$
\begin{aligned}
\mathrm{Var}(X) &= E\big[X^2 - 2\mu X + \mu^2\big] \\
&= E[X^2] - 2\mu\,E[X] + \mu^2 \\
&= E[X^2] - 2\mu^2 + \mu^2 \\
&= E[X^2] - \big(E[X]\big)^2
\end{aligned}
$$

The step from line 1 to line 2 uses that $\mu$ is a constant, so it pulls out of $E$, and $E[\mu^2] = \mu^2$.

## 2. Fair die: E[X], E[X^2], Var(X)

$$E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{21}{6} = \frac{7}{2}$$

$$E[X^2] = \sum_{x=1}^{6} x^2 \cdot \frac{1}{6} = \frac{1+4+9+16+25+36}{6} = \frac{91}{6}$$

$$\mathrm{Var}(X) = E[X^2] - \big(E[X]\big)^2 = \frac{91}{6} - \frac{49}{4} = \frac{182}{12} - \frac{147}{12} = \frac{35}{12} \approx 2.917$$

## 3. Linearity of expectation

Let $p(x,y) = P(X=x, Y=y)$ be the joint distribution, and $p_X(x) = \sum_y p(x,y)$, $p_Y(y) = \sum_x p(x,y)$ the marginals.

$$
\begin{aligned}
E[aX + bY] &= \sum_x \sum_y (ax + by)\, p(x,y) \\
&= a \sum_x x \sum_y p(x,y) + b \sum_y y \sum_x p(x,y) \\
&= a \sum_x x\, p_X(x) + b \sum_y y\, p_Y(y) \\
&= a\,E[X] + b\,E[Y]
\end{aligned}
$$

No independence assumption was used.

## 4. Var(aX + b)

Let $\mu = E[X]$. By linearity, $E[aX + b] = a\mu + b$.

$$
\begin{aligned}
\mathrm{Var}(aX + b) &= E\big[\big((aX + b) - (a\mu + b)\big)^2\big] \\
&= E\big[a^2 (X - \mu)^2\big] \\
&= a^2\, E\big[(X - \mu)^2\big] \\
&= a^2\, \mathrm{Var}(X)
\end{aligned}
$$

## 5. Variance of a sum of independent variables

Let $\mu_X = E[X]$ and $\mu_Y = E[Y]$. Using the shortcut formula:

$$
\begin{aligned}
\mathrm{Var}(X + Y) &= E\big[(X+Y)^2\big] - \big(E[X+Y]\big)^2 \\
&= E[X^2] + 2E[XY] + E[Y^2] - \big(\mu_X + \mu_Y\big)^2 \\
&= \big(E[X^2] - \mu_X^2\big) + \big(E[Y^2] - \mu_Y^2\big) + 2\big(E[XY] - \mu_X \mu_Y\big) \\
&= \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)
\end{aligned}
$$

Independence means $p(x,y) = p_X(x)\,p_Y(y)$, so
$$
E[XY] = \sum_x \sum_y xy\, p(x,y) = \sum_x \sum_y xy\, p_X(x)\, p_Y(y)
$$

$$
= \sum_x \sum_y xy\, p_X(x)\, p_Y(y) = \Big(\sum_x x\, p_X(x)\Big)\Big(\sum_y y\, p_Y(y)\Big) = E[X]\,E[Y]
$$

Hence $\mathrm{Cov}(X,Y) = 0$ and

$$\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$$

**Extension to $n$ variables (induction).** Suppose $\mathrm{Var}\big(\sum_{i=1}^{n-1} X_i\big) = \sum_{i=1}^{n-1} \mathrm{Var}(X_i)$. The sum $S_{n-1} = \sum_{i=1}^{n-1} X_i$ is independent of $X_n$, so applying the two-variable result to $S_{n-1} + X_n$ gives

$$\mathrm{Var}\Big(\sum_{i=1}^{n} X_i\Big) = \sum_{i=1}^{n} \mathrm{Var}(X_i)$$

## 6. Variance of the sample mean

Let $X_1, \dots, X_N$ be independent and identically distributed with $E[X_i] = \mu$ and $\mathrm{Var}(X_i) = \sigma^2$. Define the sample mean

$$\bar{X}_N = \frac{1}{N} \sum_{i=1}^{N} X_i$$

**Expectation** (linearity, section 3):

$$E[\bar{X}_N] = \frac{1}{N} \sum_{i=1}^{N} E[X_i] = \frac{1}{N}\cdot N\mu = \mu$$

So the sample mean is an unbiased estimator of $\mu$.

**Variance** (scaling with $a = 1/N$ from section 4, then independence from section 5):

$$
\begin{aligned}
\mathrm{Var}(\bar{X}_N) &= \mathrm{Var}\Big(\frac{1}{N} \sum_{i=1}^{N} X_i\Big) \\
&= \frac{1}{N^2}\, \mathrm{Var}\Big(\sum_{i=1}^{N} X_i\Big) \\
&= \frac{1}{N^2} \sum_{i=1}^{N} \mathrm{Var}(X_i) \\
&= \frac{1}{N^2}\cdot N\sigma^2 \\
&= \frac{\sigma^2}{N}
\end{aligned}
$$

**Standard error:**

$$\mathrm{SE}(\bar{X}_N) = \sqrt{\mathrm{Var}(\bar{X}_N)} = \frac{\sigma}{\sqrt{N}}$$

The typical error of a Monte Carlo estimate shrinks like $1/\sqrt{N}$. To cut the error by a factor of 10, you need 100 times more samples.

**Fair die example:** $\sigma = \sqrt{35/12} \approx 1.708$, so with $N = 10{,}000$ rolls the standard error is about $0.017$. That is the wobble you saw around 3.5 in the setup notebook.

## 7. Law of large numbers (via Chebyshev's inequality)

**Chebyshev's inequality.** For any random variable $Z$ with finite variance and any $\epsilon > 0$:

$$P\big(|Z - E[Z]| \geq \epsilon\big) \leq \frac{\mathrm{Var}(Z)}{\epsilon^2}$$

*Proof.* Since every term in $E[(Z-E[Z])^2]$ is non-negative, restricting the expectation to the event $|Z-E[Z]|\geq \epsilon$ can only lower it:

$$
\mathrm{Var}(Z) \;\geq\; E\big[(Z-E[Z])^2 \cdot \mathbb{1}\{|Z-E[Z]|\geq \epsilon\}\big] \;\geq\; \epsilon^2\, P\big(|Z-E[Z]|\geq \epsilon\big)
$$

using $(Z-E[Z])^2 \geq \epsilon^2$ on that event. Divide by $\epsilon^2$.

**Applying it to the sample mean.** Using $E[\bar X_N] = \mu$ and $\mathrm{Var}(\bar X_N) = \sigma^2/N$ from section 6:

$$P\big(|\bar{X}_N - \mu| \geq \epsilon\big) \leq \frac{\sigma^2}{N\epsilon^2} \;\xrightarrow[N\to\infty]{}\; 0$$

for any fixed $\epsilon > 0$. This is the **weak law of large numbers**: the sample mean converges in probability to the true mean as $N \to \infty$, and the bound above shows the convergence rate is governed by $1/N$.
