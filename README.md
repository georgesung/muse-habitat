# muse-habitat

Muse Glimmer 30B as the LLM planner on Habitat 3.0 / PARTNR (facebookresearch/partnr-planner), and reinforcement
learning to make it better there. Baseline: **782 / 1000 on PARTNR val** (setup C1: native thinking + one tool call
per turn, reasoning strength low; `docs/reports/baseline.md`).

## Layout

| Path | What |
|---|---|
| `src/muse_habitat/vllm_llm.py` | PARTNR LLM backend: the prompt string to vLLM's raw `/v1/completions`, planner grammar via structured outputs |
| `src/muse_habitat/native_tools.py` | `NativeToolsPlanner`: PARTNR skills as Glimmer tools (ATEM calls), setups C and C1 |
| `src/muse_habitat/results.py` | summarize `episode_result_log.csv` files (success, percent complete) |
| `src/muse_habitat/splits.py`, `data/splits/` | train pool / dev split of PARTNR train, kept apart from val (excluded: the val scene, val instructions) |
| `conf/llm`, `conf/instruct` | Hydra configs on PARTNR's search path (`glimmer_vllm_tools1` + `zero_shot_native_tools` = C1) |
| `scripts/` | serve Glimmer / Llama, one run, sharded runs, full val, val_mini check |
| `baselines/glimmer_c1_low_val/` | the full-val C1 low per-episode results (24 shards, 994 rows) |
| `docs/reports/` | `baseline.md` (the PARTNR report so far) |
| `runs/` | run logs (not in git) |

Runs use the sibling dirs `../partnr` (PARTNR at `ddfff19`, unmodified, with its data) and `../envs/partnr`,
`../envs/vllm`. Our code goes on `PYTHONPATH` (`src/`) and the Hydra search path (`conf/`) in `scripts/partnr_run.sh`.

```
scripts/glimmer_serve.sh 0 8011 &          # Glimmer on GPU 0; GLIMMER_MEM=0.85 when Habitat shares the GPU
scripts/val_mini_check.sh 0                # C1 low on the 10-episode val_mini study set
PYTHONPATH=src python3 -m muse_habitat.results baselines/glimmer_c1_low_val/*.csv --expect 1000
```
