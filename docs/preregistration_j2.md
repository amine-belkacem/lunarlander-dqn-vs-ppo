# Pre-registration: why does DQN collapse?

Written on 2026-10-22, before running any of the experiments below. It is committed and pushed before the runs start, so the git history shows that the criteria were not chosen after seeing the results.

## What I observed (v0.1, 3 seeds)

With the RL Zoo hyperparameters and a 1M-step budget, all three DQN runs pass a mean return of 200 between 100k and 150k steps, then fall apart. Their learning curve first goes below 0 at 475k, 575k and 750k steps, and the final policies do worse than a random agent.

## Hypothesis

The replay buffer holds the last 50,000 transitions. Once the agent lands reliably, that window only contains near-perfect landings. The Q-network stops seeing the states where things go wrong (tilted, too fast, drifting sideways), its estimates for those states drift, and the policy starts reaching them and failing.

If this is the main cause, keeping the whole history in the buffer should prevent the collapse, or at least push it much later.

## Experiment

One change, everything else identical to `configs/dqn.yaml`:

| | Baseline | Variant |
|---|---|---|
| Config | `configs/dqn.yaml` | `configs/dqn_big_buffer.yaml` |
| `buffer_size` | 50,000 | 1,000,000 (= the whole training run) |
| Seeds | 0 to 9 (3 already run, 7 new) | 0 to 4 |

Same budget (1M steps), same evaluation grid, same held-out final evaluation (100 episodes, seeds 1,000,000 to 1,000,099).

The variant gets 5 seeds rather than 10 because each DQN run costs about 2 hours on my laptop. If the result falls in the inconclusive zone below, I will add seeds 5 to 9 before concluding anything.

## Definitions, fixed now

- **Reached**: the learning curve (mean of the 10 evaluation episodes at a grid point) is ≥ 200 at least once.
- **Collapsed**: the run reached 200, and the mean of its last 5 curve points (900k, 925k, 950k, 975k, 1M) is below 0.
- Primary measure: number of collapsed runs. Secondary: success rate on the 100 held-out final episodes.

The 3 existing baseline runs are all "collapsed" under this definition (last-5 means -848, -391 and -542).

## Decision rule, fixed now

- **Hypothesis supported**: at most 1 of the 5 big-buffer runs collapses, and at least 8 of the 10 baseline runs collapse.
- **Hypothesis not supported**: 4 or 5 of the 5 big-buffer runs collapse.
- **Anything else**: inconclusive. I add seeds 5 to 9 to the variant and apply the same thresholds scaled to 10 runs (≤ 2 collapses supported, ≥ 8 not supported).
- If fewer than 8 of the 10 baseline runs collapse, the collapse is not systematic enough to test this way, and I will say so instead of forcing a conclusion.

## What this experiment cannot tell

A bigger buffer changes two things at once: the variety of states in the buffer, and how old the data is (with 1M transitions, most samples come from much older policies). A positive result shows that buffer size matters, not precisely which of those two effects does the work. Other plausible causes, such as the constant learning rate or Q-value overestimation, are not tested here.
