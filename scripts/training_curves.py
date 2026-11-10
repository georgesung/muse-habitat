"""Plot the GRPO training curves from data/training_logs/ (any Python with matplotlib).

The 300 updates ran as 5 runs, each resumed from the previous run's adapter and optimizer state:
r1 (updates 1-27), r2 (28-101), r3 (102-200, ends at the released adapter), r4 (201-275), r5 (276-300).
Inside a run, adapter pN is the policy after N updates of that run; r2 and later count from r1's last adapter, so
they are offset by 27 (the `p173` / `p273` names in run directories are 200 / 300 updates).

Top: rollout success, the share of the 8 samples in each group that finished the task. Rollouts keep running while
the trainer updates, so one group's episodes are played by 10-15 successive adapters; each group is placed at the
median adapter it used and averaged per block of updates. Each group is a different training task, so this is noisy. Bottom: mean policy entropy of the sampled tokens at each update.

    python3 scripts/training_curves.py [--out docs/assets/training_curves.png]
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path(__file__).resolve().parent.parent / "data" / "training_logs"
OFFSET = {"r1": 0, "r2": 27, "r3": 27, "r4": 27, "r5": 27}
RELEASED, BLOCK = 200, 50


def load():
    groups, steps = [], []
    for run, off in OFFSET.items():
        for line in open(LOGS / f"{run}_groups.jsonl"):
            g = json.loads(line)
            if not g["rewards"]:
                continue
            ps = sorted(0 if p == "glimmer" else int(m.group(1)) + off
                        for p in g["policies"] if p == "glimmer" or (m := re.fullmatch(r"p(\d+)", p)))
            if not ps:  # a few groups resumed after a restart have no adapter record
                continue
            upd = ps[len(ps) // 2]  # median adapter: rollouts keep running while the trainer updates
            groups.append((upd, sum(g["rewards"]) / len(g["rewards"])))
        for line in open(LOGS / f"{run}_train_log.jsonl"):
            s = json.loads(line)
            steps.append((s["step"] + 1 + off, s["entropy"]))
    return sorted(groups), sorted(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(LOGS.parent.parent / "docs" / "assets" / "training_curves.png"))
    a = ap.parse_args()
    groups, steps = load()

    xs, ys, ns, se = [], [], [], []
    for lo in range(0, 300, BLOCK):
        b = [r for u, r in groups if lo <= u < lo + BLOCK]
        if b:
            m = sum(b) / len(b)
            xs.append(lo + BLOCK / 2); ys.append(m); ns.append(len(b))
            se.append((sum((r - m) ** 2 for r in b) / (len(b) - 1) / len(b)) ** 0.5 if len(b) > 1 else 0)
    ex = [u for u, _ in steps]; ey = [e for _, e in steps]
    k = 10
    sm = [sum(ey[max(0, i - k + 1):i + 1]) / len(ey[max(0, i - k + 1):i + 1]) for i in range(len(ey))]

    SURF, T1, T2, GRID, BLUE, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#a3a29c"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "text.color": T1, "axes.labelcolor": T2,
                         "xtick.color": T2, "ytick.color": T2})
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 7.2), dpi=120, sharex=True, facecolor=SURF,
                                   gridspec_kw={"hspace": 0.45})
    for ax in (top, bot):
        ax.set_facecolor(SURF)
        ax.grid(axis="y", color=GRID, lw=1, zorder=0)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(length=0)
        ax.axvline(RELEASED, color=T2, lw=1.2, ls="--", zorder=1)
    top.fill_between(xs, [y - e for y, e in zip(ys, se)], [y + e for y, e in zip(ys, se)], color=BLUE, alpha=0.15,
                     lw=0, zorder=2)
    top.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=6, zorder=3)
    top.set_ylim(0.6, 1.0)
    top.set_title("Rollout success on training tasks", loc="left", fontsize=15, fontweight="bold", pad=24)
    top.text(0, 1.03, f"Share of 8 samples per group that finished the task, mean per {BLOCK} updates; band: ±1 SE "
             f"({len(groups)} groups)", transform=top.transAxes, fontsize=10, color=T2)
    top.text(RELEASED + 3, 0.62, "released adapter (200 updates)", fontsize=10, color=T2)
    bot.plot(ex, ey, color=GRAY, lw=1, zorder=2)
    bot.plot(ex, sm, color=BLUE, lw=2, zorder=3)
    bot.set_ylim(0, max(ey) * 1.1)
    bot.set_title("Policy entropy", loc="left", fontsize=15, fontweight="bold", pad=24)
    bot.text(0, 1.03, "Mean entropy of the sampled tokens at each update (gray), 10-update moving average (blue)",
             transform=bot.transAxes, fontsize=10, color=T2)
    bot.set_xlabel("GRPO updates")
    bot.set_xlim(0, 300)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.09)
    fig.savefig(a.out, facecolor=SURF)
    print("wrote", a.out)
    for x, y, n in zip(xs, ys, ns):
        print(f"updates {x - BLOCK / 2:3.0f}-{x + BLOCK / 2 - 1:3.0f}: {n:3d} groups, success {y:.3f}")


if __name__ == "__main__":
    main()
