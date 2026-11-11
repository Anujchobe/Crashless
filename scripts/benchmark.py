"""Run a suite of training jobs in parallel, one process per (variant, algo, seed).

    python scripts/benchmark.py main        --seeds 0 1 2 3 4
    python scripts/benchmark.py smoothness  --seeds 0 1 2
    python scripts/benchmark.py shaping     --seeds 0 1 2
Output: runs/<suite>/<variant>/<algo>_s<seed>/ ; then run scripts/report.py <suite>.
Finished runs (eval.json present) are skipped, so the command is resumable.
"""
import argparse
import copy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = yaml.safe_load(open(ROOT / "configs/env.yaml"))


def variant(**overrides):
    cfg = copy.deepcopy(BASE)
    for k, v in overrides.items():
        if k in cfg["rewards"]:
            cfg["rewards"][k] = v
        else:
            cfg[k] = v
    return cfg


SUITES = {
    # final configuration, more seeds, full budgets
    "main": {"variants": {"final": variant()}, "steps": {"ppo": None, "sac": None}},
    # the steering-chatter investigation (docs/FAILURE_LOG.md)
    "smoothness": {"variants": {
        "smooth0.1": variant(smooth=0.1),
        "smooth1.0": variant(smooth=1.0),
        "smooth0.1_ratelimit": variant(smooth=0.1, steer_rate=0.15),
        "smooth1.0_ratelimit": variant(smooth=1.0, steer_rate=0.15),
    }, "steps": {"ppo": 300_000, "sac": 40_000}},
    # does the dense shaping (lateral + heading terms) matter?
    "shaping": {"variants": {
        "full": variant(),
        "sparse": variant(lateral=0.0, heading=0.0, smooth=0.0),
    }, "steps": {"ppo": 300_000, "sac": 40_000}},
}


def run(job):
    suite, name, cfg, algo, seed, steps = job
    out = ROOT / "runs" / suite / name / f"{algo}_s{seed}"
    if (out / "eval.json").exists():
        return f"skip {out.relative_to(ROOT)}"
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "env.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"))
    cmd = [sys.executable, str(ROOT / "train.py"), "--algo", algo, "--seed", str(seed),
           "--env-config", str(cfg_path), "--out", str(out), "--quiet"]
    if steps:
        cmd += ["--steps", str(steps)]
    with open(out / "stdout.txt", "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    return f"{'ok  ' if rc == 0 else 'FAIL'} {out.relative_to(ROOT)}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=SUITES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--algos", nargs="+", default=["ppo", "sac"])
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    s = SUITES[a.suite]
    jobs = [(a.suite, n, c, algo, seed, s["steps"][algo])
            for n, c in s["variants"].items() for algo in a.algos for seed in a.seeds]
    # SAC jobs are the slowest; start them first
    jobs.sort(key=lambda j: j[3] != "sac")
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(run, jobs):
            print(msg, flush=True)
