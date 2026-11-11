"""Evaluate a trained model (2D or CARLA) or a baseline in CARLA.

    python scripts/carla_eval.py runs/transfer/sac_2d_s0      # zero-shot transfer
    python scripts/carla_eval.py lane-follow
Writes <run>/carla_eval.json (or results/carla_<baseline>.json).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crashless.baselines import LaneFollowPolicy, RandomPolicy
from crashless.carla_env import CarlaDrivingEnv
from crashless.evaluate import evaluate
from scripts.reeval import load_agent
from train import load_carla_cfg

ap = argparse.ArgumentParser()
ap.add_argument("target", help="run dir with model.pt, or 'lane-follow' / 'random'")
ap.add_argument("--episodes", type=int, default=20)
a = ap.parse_args()

cfg = load_carla_cfg()
env = CarlaDrivingEnv(cfg)
try:
    if a.target == "lane-follow":
        pol, out = LaneFollowPolicy(env.cfg.n_rays), ROOT / "results/carla_lane-follow.json"
    elif a.target == "random":
        pol, out = RandomPolicy(0), ROOT / "results/carla_random.json"
    else:
        pol, out = load_agent(Path(a.target), cfg), Path(a.target) / "carla_eval.json"
    res = evaluate(pol, cfg, episodes=a.episodes, env=env)
    res["target"] = a.target
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
finally:
    env.close()
