# Week 2: Markov Decision Processes and the Bellman Equation

## 1. Bellman equation for V^pi

By definition:

$$V^\pi(s) = E_\pi\Big[\sum_{t=0}^{\infty} \gamma^t R_t \,\Big|\, S_0 = s\Big]$$

Pull out the $t=0$ term from the sum:

$$
V^\pi(s) = E_\pi\Big[R_0 + \gamma \sum_{t=0}^{\infty} \gamma^t R_{t+1} \,\Big|\, S_0 = s\Big]
$$

The first action $A_0$ is chosen according to $\pi(a\mid s)$, and the next state $S_1=s'$ according to $P(s'\mid s, A_0)$. Averaging over both, and noting that once $S_1=s'$ the remaining discounted sum is by definition $V^\pi(s')$:

$$
\begin{aligned}
V^\pi(s) &= \sum_{a} \pi(a\mid s) \sum_{s'} P(s'\mid s,a) \Big[ R(s,a,s') + \gamma\, E_\pi\Big[\sum_{t=0}^{\infty}\gamma^t R_{t+1} \,\Big|\, S_1=s'\Big] \Big] \\
&= \sum_{a} \pi(a\mid s) \sum_{s'} P(s'\mid s,a) \big[ R(s,a,s') + \gamma\, V^\pi(s') \big]
\end{aligned}
$$

This is the **Bellman equation for $V^\pi$**: the value of state $s$ equals the immediate expected reward plus the discounted value of whatever state you land in next, averaged over the policy's action choice and the environment's transition randomness.

Equivalently, using the $Q^\pi$ notation from section 2:

$$V^\pi(s) = \sum_a \pi(a\mid s)\, Q^\pi(s,a)$$

where $Q^\pi(s,a) = \sum_{s'} P(s'\mid s,a)\big[R(s,a,s') + \gamma V^\pi(s')\big]$ — the two equations are the same statement viewed from either side.
## 2. Bellman equation for Q^pi

**Step A: V^pi(s) in terms of Q^pi(s,a).**

By definition, $\pi(a \mid s)$ is the probability that policy $\pi$ selects action $a$ in state $s$. Averaging $Q^\pi(s,a)$ over that choice of action recovers $V^\pi(s)$:

$$V^\pi(s) = \sum_{a} \pi(a \mid s)\, Q^\pi(s,a)$$

**Step B: recursive (Bellman) form of Q^pi.**

By definition:

$$Q^\pi(s,a) = E_\pi\Big[\sum_{t=0}^{\infty} \gamma^t R_t \,\Big|\, S_0 = s, A_0 = a\Big]$$

Pull out the $t=0$ term, and average over the possible next states $s'$ (weighted by $P(s' \mid s,a)$) for the remainder:

$$
\begin{aligned}
Q^\pi(s,a) &= \sum_{s'} P(s' \mid s,a) \Big[ R(s,a,s') + \gamma\, E_\pi\Big[\sum_{t=0}^{\infty}\gamma^t R_{t+1} \,\Big|\, S_1 = s'\Big] \Big] \\
&= \sum_{s'} P(s' \mid s,a) \big[ R(s,a,s') + \gamma\, V^\pi(s') \big]
\end{aligned}
$$

The inner expectation is exactly $V^\pi(s')$ by definition, since after reaching $s'$ the agent again follows $\pi$ for all future turns.

**Step C: substitute Step A into Step B.**

Since $V^\pi(s') = \sum_{a'} \pi(a' \mid s')\, Q^\pi(s',a')$, this gives $Q^\pi$ written entirely in terms of itself, one step later:

$$Q^\pi(s,a) = \sum_{s'} P(s' \mid s,a) \Big[ R(s,a,s') + \gamma \sum_{a'} \pi(a' \mid s')\, Q^\pi(s',a') \Big]$$
## 3. Bellman optimality equation

**Definition.** The optimal value function is the best value achievable from each state, over all possible policies:

$$V^*(s) = \max_\pi V^\pi(s), \qquad Q^*(s,a) = \max_\pi Q^\pi(s,a)$$

**Key relationship.** At the optimal policy, there is no benefit to randomizing over actions — the best choice is to always take whichever single action has the highest $Q^*$:

$$V^*(s) = \max_a Q^*(s,a)$$

**Recursive form.** Using the same Bellman relationship between $Q$ and $V$ derived in section 2 (now applied to the optimal functions):

$$Q^*(s,a) = \sum_{s'} P(s'\mid s,a)\big[R(s,a,s') + \gamma V^*(s')\big]$$

Substituting into $V^*(s) = \max_a Q^*(s,a)$:

$$V^*(s) = \max_{a} \sum_{s'} P(s'\mid s,a)\big[R(s,a,s') + \gamma V^*(s')\big]$$

This is the **Bellman optimality equation**: unlike the fixed-$\pi$ version, the $\max_a$ replaces the $\sum_a \pi(a\mid s)$ average — the optimal value only ever "follows" the single best action, not a weighted mix. The corresponding optimal policy is simply to always pick whichever action attains that max:

$$\pi^*(s) = \arg\max_{a} \sum_{s'} P(s'\mid s,a)\big[R(s,a,s') + \gamma V^*(s')\big]$$
