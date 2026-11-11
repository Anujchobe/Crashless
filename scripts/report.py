"""Aggregate a benchmark suite into results/<suite>.md, results/<suite>.json and plots.

    python scripts/report.py main
All numbers come from runs/<suite>/**/eval.json and train_log.csv; nothing is typed in by hand.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METRICS = [("success_rate", "Success", "{:.0%}"), ("collision_rate", "Collision", "{:.0%}"),
           ("offroad_rate", "Off-road", "{:.0%}"), ("mean_abs_lateral_m", "Abs lateral (m)", "{:.2f}"),
           ("mean_speed_mps", "Speed m/s", "{:.1f}"), ("mean_abs_steer_change", "Abs steer change", "{:.3f}")]


def load(suite):
    groups = defaultdict(list)
    for ev in sorted((ROOT / "runs" / suite).glob("*/*/eval.json")):
        groups[(ev.parent.parent.name, json.load(open(ev))["algo"])].append(ev.parent)
    return groups


def curve(run):
    rows = list(csv.DictReader(open(run / "train_log.csv")))
    return np.array([[float(r["step"]), float(r.get("success") or "nan")] for r in rows])


def steps_to(c, thr):
    hit = np.where(c[:, 1] >= thr)[0]
    return c[hit[0], 0] if len(hit) else np.nan


def fmt(vals, f):
    vals = np.asarray(vals, float)
    return f"{f.format(vals.mean())} ± {f.format(vals.std())}" if len(vals) > 1 else f.format(vals[0])


def baselines(env_cfg):
    from crashless.baselines import LaneFollowPolicy, RandomPolicy
    from crashless.evaluate import evaluate
    return {"random": evaluate(RandomPolicy(0), env_cfg, 100),
            "lane-follow PD (no obstacle sensing)": evaluate(LaneFollowPolicy(9), env_cfg, 100)}


def main(suite, plot=True):
    import yaml
    groups = load(suite)
    if not groups:
        sys.exit(f"no finished runs under runs/{suite}")
    out = {}
    lines = [f"# Results: `{suite}` suite", "",
             "Deterministic policy, 100 held-out road layouts (seeds 1,000,000+) per trained model. "
             "Values are mean ± std across training seeds.", "",
             "| Variant | Algo | Seeds | Train steps | Wall-clock s | " + " | ".join(m[1] for m in METRICS)
             + " | Steps to 80% train success |",
             "|" + "---|" * (6 + len(METRICS))]
    for (var, algo), runs in sorted(groups.items()):
        evs = [json.load(open(r / "eval.json")) for r in runs]
        curves = [curve(r) for r in runs]
        s80 = [steps_to(c, 0.8) for c in curves]
        reached = [x for x in s80 if not np.isnan(x)]
        s80_txt = (f"{np.mean(reached)/1e3:.0f}k ({len(reached)}/{len(runs)} seeds)" if reached else "never")
        out[f"{var}/{algo}"] = {"seeds": len(runs), **{m: [e[m] for e in evs] for m, *_ in METRICS},
                                "train_seconds": [e["train_seconds"] for e in evs],
                                "steps_to_80pct_train_success": s80}
        lines.append(f"| {var} | {algo.upper()} | {len(runs)} | {evs[0]['train_steps']:,} | "
                     f"{np.mean([e['train_seconds'] for e in evs]):.0f} | "
                     + " | ".join(fmt([e[m] for e in evs], f) for m, _, f in METRICS) + f" | {s80_txt} |")
    env_cfg = yaml.safe_load(open(next(iter(groups.values()))[0] / "env.yaml"))
    if suite == "main":
        lines += ["", "Reference controllers on the same 100 layouts:", "",
                  "| Controller | " + " | ".join(m[1] for m in METRICS) + " |", "|" + "---|" * (1 + len(METRICS))]
        for name, ev in baselines(env_cfg).items():
            out[f"baseline/{name}"] = ev
            lines.append(f"| {name} | " + " | ".join(f.format(ev[m]) for m, _, f in METRICS) + " |")
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(out, open(ROOT / f"results/{suite}.json", "w"), indent=2, default=float)
    if plot:
        lines += ["", f"![learning curves]({suite}_curves.png)"]
        plot_curves(suite, groups)
    (ROOT / f"results/{suite}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def plot_curves(suite, groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for (var, algo), runs in sorted(groups.items()):
        cs = [curve(r) for r in runs]
        n = min(len(c) for c in cs)
        x = cs[0][:n, 0]
        y = np.stack([c[:n, 1] for c in cs])
        label = f"{algo.upper()}" + (f" / {var}" if len({v for v, _ in groups}) > 1 else "")
        for a, xs in ((ax[0], x), (ax[1], x)):
            a.plot(xs, y.mean(0), label=label)
            a.fill_between(xs, y.mean(0) - y.std(0), y.mean(0) + y.std(0), alpha=0.2)
    ax[1].set_xscale("log")
    for a, t in zip(ax, ("linear steps", "log steps")):
        a.set_xlabel(f"environment steps ({t})")
        a.set_ylabel("training success rate (last 50 episodes)")
        a.set_ylim(0, 1)
        a.grid(alpha=0.3)
    ax[0].legend()
    fig.suptitle(f"Crashless: {suite} suite (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(ROOT / f"results/{suite}_curves.png", dpi=120)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    main(a.suite, not a.no_plot)
