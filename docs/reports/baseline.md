# PARTNR baseline: Muse Glimmer 30B as the household-task planner

**Status:** 2026-10-29 to 11-01. Muse Glimmer 30B evaluated on Meta's PARTNR benchmark (Habitat 3.0: a robot and a
humanoid rearrange objects in simulated homes from a natural-language task), first on small samples to choose the
interface, then on the full val split.

**Full val, 2026-11-01:** Glimmer C1 (native tool calls, one generation, low reasoning strength) succeeds on
**782 / 1000** val episodes (0.782, 0.871 of the task conditions done), above the paper's ReAct Llama-3.1-70B (0.73).
This is the baseline the RL work starts from.

## Setup

- PARTNR = facebookresearch/partnr-planner `ddfff19`, unmodified; our backends and planner live in `src/muse_habitat/`
  and `conf/`, on PYTHONPATH and the Hydra search path via `scripts/partnr_run.sh`. PARTNR's Python 3.9 env with
  habitat-sim 0.3.3 (headless + Bullet) and torch 2.7.1+cu128.
- Config: `baselines/decentralized_zero_shot_react_summary.yaml`: each agent has its own planner, partial
  observability, oracle motor skills. Greedy decoding throughout.
- Serving: vLLM, bf16, DFlash speculative decoding with the 5-layer assistant (`meta-models/Muse-Glimmer-30B-assistant`)
  (`scripts/glimmer_serve.sh`).

## The four Glimmer interfaces

PARTNR's planner builds one prompt string from four chat tags in the llm config and parses `Tool[args]` from the
reply. `vllm_llm.py` sends that string to vLLM's raw `/v1/completions` with Glimmer's tags and passes the planner's
grammar to vLLM's structured outputs (xgrammar parses PARTNR's EBNF).

- **A, ReAct:** the paper's prompt and grammar ("Thought: ..." line, then the action). Assistant turns open the
  `to=user` channel, so Glimmer's own thinking is skipped and the Thought line is the only reasoning.
- **B, native thinking:** two requests per turn: the `to=self` thinking channel, free, then the `to=user` answer under
  the grammar. (vLLM's "grammar after reasoning" relies on the reasoning parser, and Glimmer's ends reasoning only at
  a tool channel, so one request can't do it.)
- **C, native tool calls, constrained** (`native_tools.py`, `NativeToolsPlanner`): the skills plus Done become
  functions in the system block, rendered byte-identically to Glimmer's chat template with `tools=`; results come back
  as `tool` messages. Per turn: think, then a grammar-constrained recipient (`to=<tool>`), then grammar-constrained
  ATEM arguments from the world graph's names.
- **C1, native tool calls, one generation** (`conf/llm/glimmer_vllm_tools1.yaml`): from `<|start|>assistant` the model
  thinks and writes the call, stopping at `</atem:function_calls>`; parsed and passed to PARTNR as written. 422 calls
  on the 10-episode study set, 0 malformed.

## Small samples (val_mini)

10-episode study set (`random.Random(0)` sample of val_mini: `[20, 132, 155, 183, 197, 207, 215, 244, 248, 261]`):
A 8/10, B 8/10, C 9/10, **C1 10/10**. On 50 episodes (the first 50 of the same draw), A and C1 tie at 41/50 (0.879
complete each; each wins 5 episodes the other loses), and C1 at medium reasoning strength does 37/50. Ten episodes
cannot separate setups one or two episodes apart; the 10/10 did not survive 50.

C1 was kept: it is the model's native interface (thinking, then one tool call), needs no grammar, and its calls were
all well formed (1,727 calls on 40 more episodes, no fallback).

For reference, Llama 3.1 8B Instruct (ReAct, our vLLM backend) does 8/10 on the 10-episode set.

## Full val, Glimmer C1 low (2026-10-31 / 11-01)

**782 / 1000 succeed (0.782), 0.871 of the propositions done on average.** The paper's full-val numbers
(decentralized, partial observability, oracle skills): ReAct Llama-3.1-70B 0.73, ReAct 8B 0.64, fine-tuned 8B 0.70,
humans 0.93. Our harness is not identical to the paper's (our serving path, batched greedy decoding), so this is a
comparison, not a reproduction. Per-episode results: `baselines/glimmer_c1_low_val/`.

- **6 episodes (113, 282, 628, 644, 836, 837) crashed in PARTNR** with `'NoneType' object has no attribute
  'object_id'` (`oracle_object_state_skill.py:262`); PARTNR skips them and they count as failures. A PARTNR bug, not
  a planner error (as is PARTNR's occasional `Object <x> has no parent`).
- **Per scene:** 0.60 to 0.88.
- **Failures:** of the 212 logged failures, 54 made no progress at all; on average they reach 0.42 of the
  propositions. 169 episodes hit the 50-replan limit on one agent (93 of them still succeed: the other agent
  finishes). 193 hit the 20,000-simulator-step limit (137 of them succeed).
- **Cost:** median 279 s per episode, 38 replans on average for the two agents together.
- val_mini is slightly easier than val (0.80 vs 0.78 on the same run).

## What we learned

- For a thinking, tool-trained model the native interface is a single generation: reasoning channel, then a tool
  call. Grammar constraints were not needed.
- vLLM's reasoning + structured-output integration is parser-specific; raw completions with the model's own tags
  sidestep it.
- Glimmer 30B in its native interface is above the paper's ReAct 70B on full val (0.782 vs 0.73) and about 0.15 below
  humans: room to improve, which is what the RL work targets.
