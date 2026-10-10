# Riftbound RL

A self-play reinforcement-learning agent for **Riftbound**, Riot Games' trading
card game. It's a two-player game with hidden information, a stack-style
response chain, and a pool of over 900 cards. The agent learns from scratch with
PPO, plays dozens of real tournament decklists, and makes every decision a human
player makes, including the between-games decisions of a best-of-three match
(sideboarding, battlefield choice, play order).

> **About the environment.** The agent trains against a rules engine I wrote
> from scratch. It implements the game's Core Rules closely enough to reproduce
> 2,000+ published judge rulings. The engine is kept private because of
> Riot's digital tools policy, which
> prohibits public projects that automate Riftbound gameplay. This repository
> holds the learning side: the observation encoder, network, trainer, opponent
> pool, parallel rollouts and evaluation tooling. It imports the engine, so it
> doesn't run by itself. The engine is available to view on request.

## Results

- **+133 Elo over a hand-written greedy baseline**, which itself sits 779 Elo
  above random play (round-robin ladder, ±50 Elo).
- **Each further gain was measured in a head-to-head duel** (600 Bo3 matches /
  2,400 Bo1 games, ±2pp). Learning the best-of-three decisions gave +40 Elo
  (p≈0.005). Adding two features to each action row ("is this mine", "is this in
  the current combat") gave +32 to +35 Elo (Bo1 p≈1e-5).
- **2.1× training throughput** (213 → 447 env steps/s) from multiprocess
  rollouts with shared-memory observations. The games are identical to serial
  execution, which a test checks.
- **Nulls are recorded, not hidden.** More self-play, higher entropy,
  matchup-aware opponent sampling and inference-time search each failed to move
  strength. Each failure is explained in [`docs/RUNS.md`](docs/RUNS.md).
  Opponent diversity was the one training lever that held up under measurement.

## Design

| decision | why |
|---|---|
| **PPO, a policy-gradient method, not Q-learning** | Optimal play in a hidden-information game is generally a *mixed* strategy. An argmax over Q-values can't represent bluffing and is maximally exploitable. The policy outputs a distribution and samples from it. |
| **Candidate-scoring action head** | The legal action set changes size and meaning every step. Each legal action is featurized as a row and scored against the trunk embedding, so the action space never has to be enumerated in advance. |
| **Cards encoded by behaviour, not ID** | No card-ID one-hots. Each card is described by attributes plus features derived from its scripted abilities, so the net generalizes to rarely-seen cards. A rules fix also changes a card's features instead of invalidating the checkpoint. |
| **DeepSets over zones** | One shared card encoder across hand, board, battlefields, chain and the rest, with masked mean and max pooling. Zones are permutation-invariant and variable-length. |
| **An honest critic, and a privileged one as a diagnostic only** | The critic that drives learning sees exactly what the policy sees. A second critic that also sees the opponent's hand is trained on a detached trunk, so it can't leak hidden information into the policy's features. A test enforces this: scrambling hidden information leaves every action logit bit-identical. |
| **Terminal ±1 reward, γ = 1** | No reward shaping, so the critic is literally a win probability. Score-based shaping was ruled out by analysis: points pair up into turns-to-win (at 8 to win, the 7th point buys no tempo), so turns-to-win is given to the network as an input feature instead. |
| **Self-play against a pool of frozen past selves, plus pinned ancestors** | A pool made only of recent self-snapshots has a flat difficulty surface, so prioritized sampling over it does nothing. Pinning earlier, weaker champions into the pool did measurably help. |
| **Observation layouts only grow** | New observation features are appended, and old checkpoints grow into the new layout and play identically. That makes warm-starting across feature changes safe. |

~1M parameters. The architecture, hyperparameters and the reasoning behind each
choice are in [`docs/LEARNING.md`](docs/LEARNING.md) and
[`docs/DESIGN.md`](docs/DESIGN.md).

## Measuring

The most transferable part of the project is probably the evaluation
discipline. Details are in [`docs/MEASURING.md`](docs/MEASURING.md).

- **One variable per run, against a named baseline.** Runs are named for their
  independent variable (`v17-targets`, `v19-warmup`).
- **Win rate against a fixed baseline saturates.** Policies are compared against
  each other in a ladder and in paired duels sized to the effect being tested.
- **Behaviour is measured separately from strength.** `rl/tools/behaviour.py`
  turns specific mistakes into rates, such as buffing an enemy unit or leaving
  an opponent's hold uncontested. One run changed behaviour sharply while its
  Elo didn't move.
- **Configs are fingerprinted into checkpoints.** One run was trained on the
  wrong rules variant, and nothing caught it because the tensor shapes were
  identical. A rules difference is invisible to a tensor shape, so checkpoints
  now carry their full config and loading refuses a mismatch.
- **A paired A/B test separates co-occurrence from causation.** An exact
  endgame solver appeared to be worth 3.8% in win rate. It proved 34 forced
  wins and changed zero outcomes.

## Layout

```
rl/
  env.py        gym-style wrapper around the engine
  obs.py        game state -> tensors (zones, globals, featurized legal actions)
  nets.py       DeepSets card encoder, trunk, action head, two critics
  ppo.py        trainer: GAE, clipped surrogate, LR warm-up/anneal, resumable runs
  pool.py       frozen-opponent pool, PFSP, Elo
  parallel.py   multiprocess rollouts with shared-memory observations
  vec.py        batched environments
  eval.py       evaluation against baselines and per-deck breakdowns
  ladder.py     round-robin Elo ladder over checkpoints
  deckeval.py   deck strength from the critic
  solver.py     exact search for forced wins
  agents/       hand-written baselines (greedy, holder)
  tools/        behavioural metrics, solver A/B, matchup analysis
  tests/        trainer and parallelism tests, head-to-head duels
docs/
  LEARNING.md   the algorithm, the network, how to read a run
  MEASURING.md  how to tell whether a change helped
  RUNS.md       every training run: setup, result, interpretation
  PARALLEL.md   profiling and the parallel-rollout design
  DESIGN.md     design reasoning, excerpted from the project plan
```

The docs are working documents written during the project. Some of them refer
to private modules (the engine and the play clients) that aren't included here.

---

Riftbound is a trademark of Riot Games, Inc. This is an unofficial research
project, not endorsed by or affiliated with Riot Games. No card data, card art
or rules text is distributed here.
