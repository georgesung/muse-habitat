"""Summarize PARTNR runs from their episode_result_log.csv files (stdlib only, so any Python >= 3.9 runs it).

    python -m muse_habitat.results <csv or run dir> ... [--expect N]

A run dir is searched for results/episode_result_log.csv (a sharded run: give every shard). Episodes are counted by
episode_id; a duplicate id is an error. With --expect N, episodes missing from the logs (PARTNR skips episodes that
crash and writes no row) are counted as failures out of N.
"""
import argparse
import csv
import sys
from pathlib import Path


def load(paths):
    """{episode_id: row} over the given CSVs / run dirs."""
    rows = {}
    for p in map(Path, paths):
        if p.is_dir():
            p = p / "results" / "episode_result_log.csv"
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                eid = int(row["episode_id"])
                if eid in rows:
                    raise SystemExit(f"duplicate episode {eid} in {p}")
                rows[eid] = row
    return rows


def summarize(rows, expect=None):
    n = len(rows) if expect is None else expect
    success = sum(float(r["task_state_success"]) == 1.0 for r in rows.values())
    complete = sum(float(r["task_percent_complete"]) for r in rows.values())
    return {"logged": len(rows), "episodes": n, "success": success, "success_rate": success / n,
            "percent_complete": complete / n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--expect", type=int, help="episodes in the run; missing ones count as failures")
    a = ap.parse_args(argv)
    s = summarize(load(a.paths), a.expect)
    print(f"{s['success']} / {s['episodes']} = {s['success_rate']:.3f} success, "
          f"{s['percent_complete']:.3f} complete ({s['logged']} logged)")


if __name__ == "__main__":
    sys.exit(main())
