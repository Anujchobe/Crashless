"""Train one agent.  Examples:
    python train.py --algo sac --seed 0                         # 2D sim
    python train.py --algo ppo --seed 1 --steps 200000
    python train.py --algo sac --env carla                      # CARLA, from scratch
    python train.py --algo sac --env carla --init-from runs/main/final/sac_s0/model.pt \
                    --learning-starts 1000                      # fine-tune a 2D policy in CARLA
Writes <out>/{model.pt, train_log.csv, eval.json, env.yaml}; default out is runs/<env>_<algo>_s<seed>.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from crashless.evaluate import evaluate
from crashless.logger import CSVLogger
from crashless.ppo import PPO, PPOConfig
from crashless.sac import SAC, SACConfig

ROOT = Path(__file__).parent


def load_yaml(p):
    return yaml.safe_load(open(p)) or {}


def load_carla_cfg(path=ROOT / "configs/carla.yaml") -> dict:
    """CARLA settings, with reward weights taken from the 2D config so both sims share one reward."""
    cfg = load_yaml(path)
    cfg.setdefault("rewards", load_yaml(ROOT / "configs/env.yaml").get("rewards", {}))
    return cfg


def make_env(kind: str, env_cfg: dict):
    if kind == "2d":
        return None                       # agents build their own 2D envs
    from crashless.carla_env import CarlaDrivingEnv
    return CarlaDrivingEnv(env_cfg)


def build(algo: str, seed: int, steps: int | None, env_cfg: dict, env=None, carla: bool = False):
    if algo == "ppo":
        cfg = PPOConfig(**load_yaml(ROOT / "configs/ppo.yaml"))
        if steps:
            cfg.total_steps = steps
        return PPO(env_cfg, cfg, seed=seed, envs=[env] if env is not None else None)
    cfg = SACConfig(**load_yaml(ROOT / ("configs/sac_carla.yaml" if carla else "configs/sac.yaml")))
    if steps:
        cfg.total_steps = steps
    return SAC(env_cfg, cfg, seed=seed, env=env)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "sac"], required=True)
    ap.add_argument("--env", choices=["2d", "carla"], default="2d")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--env-config", default=None, help="default: configs/env.yaml or configs/carla.yaml")
    ap.add_argument("--init-from", default=None, help="SAC checkpoint to fine-tune (needs critics saved)")
    ap.add_argument("--learning-starts", type=int, default=None, help="override SAC random warm-up steps")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-episodes", type=int, default=None, help="default 100 (2D) / 20 (CARLA)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    torch.set_num_threads(1)
    carla = a.env == "carla"
    if carla:
        env_cfg = load_carla_cfg(a.env_config) if a.env_config else load_carla_cfg()
    else:
        env_cfg = load_yaml(a.env_config or ROOT / "configs/env.yaml")
    out = Path(a.out or ROOT / "runs" / f"{a.env}_{a.algo}_s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(env_cfg, open(out / "env.yaml", "w"))

    env = make_env(a.env, env_cfg)
    try:
        agent = build(a.algo, a.seed, a.steps, env_cfg, env, carla)
        if a.init_from:
            if a.algo != "sac":
                raise SystemExit("--init-from is implemented for SAC only")
            agent.load(a.init_from, full=True)
        if a.learning_starts is not None:
            agent.cfg.learning_starts = a.learning_starts
        log = CSVLogger(out / "train_log.csv", echo_every=0 if a.quiet else (1 if carla else 10))
        t0 = time.time()
        try:
            agent.train(log)
        finally:                      # keep what was learned even if CARLA dies mid-run
            log.flush()
            agent.save(out / "model.pt")
        episodes = a.eval_episodes or (20 if carla else 100)
        res = evaluate(agent, env_cfg, episodes=episodes, env=env)
        res.update(algo=a.algo, env=a.env, seed=a.seed, train_steps=agent.cfg.total_steps,
                   init_from=a.init_from, train_seconds=round(time.time() - t0, 1))
        json.dump(res, open(out / "eval.json", "w"), indent=2)
        print(json.dumps(res, indent=2))
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
