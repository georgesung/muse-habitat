# GRPO training logs

The logs of the 5 runs that make up the 300 updates, each resumed from the previous run's adapter and optimizer state.

| Run | Updates | Notes |
|---|---|---|
| r1 | 1–27 | from the original Glimmer |
| r2 | 28–101 | |
| r3 | 102–200 | ends at the released adapter (`p173`) |
| r4 | 201–275 | |
| r5 | 276–300 | ends at `p273` |

Inside a run, adapter `pN` is the policy after N updates of that run. From r2 on, the names count from r1's last
adapter, so add 27 to get the total number of updates (`p173` = 200). `glimmer` is the original model.

- `<run>_train_log.jsonl`: one line per update: loss, gradient norm, mean log-prob and entropy of the sampled tokens,
  importance-ratio deviation and clip fraction, packed tokens, wall time, and the group it trained on (episode id,
  the 8 rewards, and the adapters that played it).
- `<run>_groups.jsonl`: every completed group of 8 rollouts, trained on or not (groups where all 8 got the same
  reward have no learning signal and are skipped by the trainer).

Settings, the same in every run: 8 rollouts per group at temperature 1, reward = task success, one group per update,
LoRA rank 32 (alpha 64), learning rate 1e-5.

`scripts/training_curves.py` draws `docs/assets/training_curves.png` from these files.
