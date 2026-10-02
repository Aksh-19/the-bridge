# Week 3: UCB1 and Hoeffding's Inequality

## 1. Solving for epsilon

Start from:

$$e^{-2n\epsilon^2} = \delta$$

Take the natural log of both sides:

$$-2n\epsilon^2 = \ln \delta$$

Multiply both sides by $-1$, and use $-\ln \delta = \ln(1/\delta)$:

$$2n\epsilon^2 = -\ln \delta = \ln\!\left(\frac{1}{\delta}\right)$$

Divide by $2n$:

$$\epsilon^2 = \frac{\ln(1/\delta)}{2n}$$

Take the square root:

$$\epsilon = \sqrt{\frac{\ln(1/\delta)}{2n}}$$
## 2. From epsilon to the UCB1 formula

Recall $\delta$ is the probability we're willing to tolerate of our confidence bound being *wrong* — i.e., of the true mean secretly exceeding $\bar{X}_n + \epsilon$. A fixed $\delta$, like $\delta = 0.05$, would mean "I accept a flat 5% chance of being fooled on this one check."

But MCTS doesn't check a bound just once — it recomputes UCB1 at *every single visit*, for *every* arm, over potentially thousands or millions of visits as the search runs. If each check used a fixed error tolerance like $\delta=0.05$, then across enough checks, bad luck is virtually guaranteed to strike *somewhere* — some arm, at some point, will appear artificially promising purely by chance, and the algorithm would have no way to protect against that accumulating failure risk.

The fix is to make $\delta$ shrink as more total decisions get made, so the running risk of ever being fooled stays controlled. Setting $\delta = 1/t$, where $t$ is the *total* number of visits across *all* arms so far, does exactly this: as the search progresses and $t$ grows, the tolerance for error on each individual check becomes stricter and stricter — a kind of built-in, automatic caution that scales with how many chances-to-be-wrong have already occurred.

Substituting $\delta = 1/t$ into the formula from item 1:

$$\epsilon = \sqrt{\frac{\ln(1/\delta)}{2n}} = \sqrt{\frac{\ln(t)}{2n}}$$

This is close to the standard UCB1 term, but off by a constant factor of 2 inside the square root — $\sqrt{\ln t / (2n)}$ versus the usual $\sqrt{2\ln t / n}$. The difference comes from exactly how tight a confidence bound you demand (some derivations use a slightly different concentration constant, or bound both tails instead of one), which changes the constant but not the shape of the formula. What matters, and what every version agrees on, is the *qualitative* behavior: the exploration bonus grows with $\ln t$ (so it never stops rewarding exploration entirely, but very slowly) and shrinks with $1/\sqrt{n}$ (an arm you've tried a lot needs much less of a confidence cushion).

The standard UCB1 selection rule picks, at every decision, the arm maximizing:

$$\bar{X}_n + \sqrt{\frac{2\ln t}{n}}$$

where $\bar{X}_n$ is "how good this arm has looked so far" (exploitation) and $\sqrt{2\ln t/n}$ is "how uncertain I still am about this arm" (exploration) — an arm visited rarely (small $n$) gets a large bonus, while the $\ln t$ growth ensures that even a heavily-explored arm occasionally gets re-examined as the total visit count climbs.
