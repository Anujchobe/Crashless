"""Re-evaluate saved models (e.g. after adding a metric). Usage:
    python scripts/reeval.py runs/abl_* --episodes 100
Prints one line per run; env config is read from the run's env.yaml if present,
else from the ablation config matching its name."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from crashless.evaluate import evaluate
from crashless.ppo import PPO, PPOConfig
from crashless.sac import SAC, SACConfig


def load_agent(run: Path, env_cfg=None):
    """Load a policy for acting. Its internal 2D env is never stepped, so default
    settings are used; only the 14-dim observation shape has to match."""
    ck = torch.load(run / "model.pt", map_location="cpu")
    if ck["algo"] == "ppo":
        return PPO(None, PPOConfig(**ck["cfg"])).load(run / "model.pt")
    return SAC(None, SACConfig(**ck["cfg"])).load(run / "model.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    for r in (Path(x) for x in a.runs if Path(x, 'model.pt').exists()):
        env_cfg = yaml.safe_load(open(r / "env.yaml"))
        res = evaluate(load_agent(r, env_cfg), env_cfg, a.episodes)
        json.dump(res, open(r / "eval.json", "w"), indent=2)
        print(f"{r.name:28s} succ={res['success_rate']:.2f} coll={res['collision_rate']:.2f} "
              f"off={res['offroad_rate']:.2f} "
              f"dsteer={res['mean_abs_steer_change']:.3f} lat={res['mean_abs_lateral_m']:.2f} "
              f"v={res['mean_speed_mps']:.1f}")
