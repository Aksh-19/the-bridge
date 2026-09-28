# The Bridge

A predictive decision engine built from first principles: Markov Decision Processes, Monte Carlo estimation, Bayesian updating, and Monte Carlo Tree Search (MCTS).

**Goal:** an MCTS agent that measurably beats a greedy baseline on a nontrivial sequential decision problem, with a visualization of it exploring low-probability branches a greedy method would never check.

## Roadmap

| Week | Topic | Code | Derivation |
|------|-------|------|------------|
| 1 | Expectation, Monte Carlo, 1/√N convergence | `src/bridge/monte_carlo/` | `docs/derivations/week1_expectation_lln.md` |
| 2 | MDPs, Bellman equations, value/policy iteration | `src/bridge/mdp/` | `docs/derivations/week2_bellman_mdp.md` |
| 3 | MCTS selection and expansion, UCB1 | `src/bridge/mcts/` | `docs/derivations/week3_ucb1_hoeffding.md` |
| 4 | MCTS simulation and backpropagation, full agent | `src/bridge/mcts/` | `docs/derivations/week4_rollouts_bayes.md` |
| 5 | Harder problem and greedy baseline comparison | `src/bridge/envs/` | |
| 6 | Search-tree visualization and final report | `src/bridge/viz/`, `reports/` | |

## Setup

```bash
git clone https://github.com/Aksh-19/the-bridge.git
cd the-bridge
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

## Structure

```
src/bridge/     package code, one subpackage per topic
tests/          pytest tests
docs/           hand derivations (markdown + LaTeX)
notebooks/      exploration and experiments
results/        generated figures
reports/        final write-up
```

## Status

Work in progress. Currently: project setup (Phase 0).
