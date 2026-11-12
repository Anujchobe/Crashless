"""Aggregate CARLA transfer results across seeds into results/carla.md.

    python scripts/carla_report.py
Reads runs/transfer/{sac_2d,sac_finetune,sac_scratch}_s*/carla_eval.json (100-episode
evals), the training logs, and results/carla_lane-follow.json.
"""
import csv
import json
import re
from pathlib import Path

from math import comb

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "runs/transfer"
ROWS = [("sac_2d", "SAC trained in 2D only (zero-shot)", "none"),
        ("sac_scratch", "SAC from scratch in CARLA", "30k steps"),
        ("sac_finetune", "**SAC 2D-pretrained + fine-tuned**", "30k steps")]
COLS = [("success_rate", "Success", "{:.0%}"), ("collision_rate", "Collision", "{:.0%}"),
        ("offroad_rate", "Off-road", "{:.0%}"), ("timeout_rate", "Timeout", "{:.0%}"),
        ("mean_speed_mps", "Speed m/s", "{:.1f}"), ("mean_abs_steer_change", "Abs steer change", "{:.3f}")]


def fmt(v, f):
    v = np.asarray(v, float)
    return f.format(v[0]) if len(v) == 1 else f"{f.format(v.mean())} ± {f.format(v.std())}"


def sign_test(a: dict, b: dict):
    """Paired one-sided sign test that a beats b, over seeds present in both.
    Returns (wins, n_untied, p). Ties are dropped, as usual for a sign test."""
    pairs = [(a[k], b[k]) for k in sorted(a) if k in b]
    wins = sum(x > y for x, y in pairs)
    n = sum(x != y for x, y in pairs)
    p = sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n if n else 1.0
    return wins, n, p


def curve(log):
    out = {}
    for line in open(log):
        m = re.match(r"step=(\d+) .*?success=([\d.]+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out


def main():
    lines = ["# Results: CARLA (Town02, CARLA 0.9.16, `-nullrhi`)", "",
             "100 episodes per trained policy, deterministic actions, eval seeds 1,000,000+, same reward terms and "
             "observation contract as the 2D sim (progress and lateral offset are measured differently; "
             "see README). Mean ± std across training seeds.", "",
             "**Not a held-out test.** Eval seeds change the random draws, but routes still start from "
             "Town02's spawn points on the map the CARLA agents trained on; only obstacle placement is new. "
             "Rows are comparable (same episodes for every policy), but these are not unseen roads.", "",
             "| Policy | CARLA training | Seeds | " + " | ".join(c[1] for c in COLS) + " |",
             "|" + "---|" * (3 + len(COLS))]
    lf = ROOT / "results/carla_lane-follow.json"
    if lf.exists():
        e = json.load(open(lf))
        lines.append("| Lane-follow PD (no obstacle sensing) | none | n/a | "
                     + " | ".join(f.format(e[k]) for k, _, f in COLS) + " |")
    summary, by_seed = {}, {}
    for key, label, budget in ROWS:
        evs = []
        by_seed[key] = {}
        for d in sorted(T.glob(f"{key}_s*")):
            ev = d / "carla_eval.json"
            if ev.exists() and json.load(open(ev)).get("episodes") == 100:
                e = json.load(open(ev))
                evs.append(e)
                by_seed[key][d.name.rsplit("_s", 1)[1]] = e["success_rate"]
        if not evs:
            continue
        summary[key] = [e["success_rate"] for e in evs]
        lines.append(f"| {label} | {budget} | {len(evs)} | "
                     + " | ".join(fmt([e[k] for e in evs], f) for k, _, f in COLS) + " |")
    per_seed = {k: ", ".join(f"{v:.0%}" for v in vals) for k, vals in summary.items()}
    lines += ["", "Success per seed: " + "; ".join(f"{k}: {v}" for k, v in per_seed.items()), ""]
    if by_seed.get("sac_finetune") and by_seed.get("sac_scratch"):
        w, n, p = sign_test(by_seed["sac_finetune"], by_seed["sac_scratch"])
        verdict = "significant at 0.05" if p < 0.05 else "not significant at 0.05"
        lines += [f"Fine-tuned beat scratch on {w}/{n} seeds: one-sided sign test p = {p:.3f} ({verdict}).", ""]

    steps = [5000, 10000, 15000, 20000, 25000, 30000]
    lines += ["Training-time success (stochastic actions, last 50 episodes), mean across seeds:", "",
              "| CARLA steps | " + " | ".join(f"{s // 1000}k" for s in steps) + " |", "|" + "---|" * (1 + len(steps))]
    for kind, label in (("finetune", "Fine-tuned"), ("scratch", "From scratch")):
        cs = [curve(p) for p in sorted(T.glob(f"{kind}_s*.log"))]
        cs = [c for c in cs if all(s in c for s in steps)]
        if cs:
            lines.append(f"| {label} ({len(cs)} seeds) | "
                         + " | ".join(f"{np.mean([c[s] for c in cs]):.0%}" for s in steps) + " |")
    out = ROOT / "results/carla.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
