"""GRPO data side on synthetic runs (CPU only)."""
import json

import torch

from muse_habitat.grpo import Turn, chunks, groups, load_rollouts, pack_objective, packs, Pack


def write_run(tmp_path, rewards):
    """rewards[k][episode] -> one rollout per (sample k, episode), two agents, append-only histories."""
    for k, per_ep in enumerate(rewards):
        (tmp_path / f"traj_k{k}").mkdir()
        res = tmp_path / f"k{k}_s0" / "results"
        res.mkdir(parents=True)
        rows = ["episode_id,task_percent_complete,task_state_success"]
        with open(tmp_path / f"traj_k{k}" / "traj_1.jsonl", "w") as f:
            for eid, r in per_ep.items():
                for agent in (0, 1):
                    prompt = [1, 2, 3]
                    for turn in range(3):
                        gen = [10 + turn, 11 + turn]
                        f.write(json.dumps({"episode_id": eid, "agent": agent, "turn": turn, "prompt_token_ids": prompt,
                                            "token_ids": gen, "logprobs": [-0.1, -0.2], "grammar": False}) + "\n")
                        prompt = prompt + gen + [99, 98]
                rows.append(f"{eid},{r},{r}")
        (res / "episode_result_log.csv").write_text("\n".join(rows) + "\n")


def test_groups_and_advantages(tmp_path):
    write_run(tmp_path, [{"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 1.0}])
    ros = load_rollouts(tmp_path)
    assert len(ros) == 6 and all(len(r.agents) == 2 and len(r.agents[0]) == 3 for r in ros.values())
    gs = groups(ros)
    assert [g[0].episode_id for g in gs] == ["a"]  # b: all equal, no signal
    adv = sorted(r.advantage for r in gs[0])
    assert abs(sum(adv)) < 1e-5 and adv[0] < 0 < adv[-1]


def test_chunks_and_packs(tmp_path):
    write_run(tmp_path, [{"a": 1.0}, {"a": 0.0}])
    gs = groups(load_rollouts(tmp_path))
    pl = packs(gs, budget=10_000)
    assert len(pl) == 4 and all(len(p.turns) == 3 and p.chain == p.turns[-1].prompt for p in pl)
    turns = gs[0][0].agents[0]
    assert [len(c) for c in chunks(turns, budget=12)] == [2, 1]  # 2nd chunk: last context 11 + 2 > 12 with 3 turns


def test_objective_clips():
    p = Pack([1], [Turn([1], [5, 6], [0.0, 0.0])], advantage=1.0)
    lp = torch.tensor([0.0, 1.0], requires_grad=True)  # ratios 1 and e
    loss, st = pack_objective(lp, torch.zeros(2), p)
    loss.backward()
    assert st["clipped"] == 1 and lp.grad[1] == 0 and lp.grad[0] == -1
