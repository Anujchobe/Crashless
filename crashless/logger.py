"""Episode bookkeeping and CSV logging."""
from __future__ import annotations

import csv
from collections import deque
from pathlib import Path


class EpisodeTracker:
    """Accumulates returns per parallel env; summarises the last `window` episodes."""

    def __init__(self, n_envs: int, window: int = 50):
        self.ret = [0.0] * n_envs
        self.len = [0] * n_envs
        self.done = deque(maxlen=window)
        self.episodes = 0

    def step(self, i: int, reward: float, ended: bool, info: dict):
        self.ret[i] += reward
        self.len[i] += 1
        if ended:
            self.done.append((self.ret[i], self.len[i], info.get("success", False),
                              info.get("crashed", False), info.get("offroad", False)))
            self.ret[i], self.len[i] = 0.0, 0
            self.episodes += 1

    def summary(self) -> dict:
        if not self.done:
            return {"episodes": 0}
        n = len(self.done)
        return {"episodes": self.episodes,
                "ep_return": sum(d[0] for d in self.done) / n,
                "ep_len": sum(d[1] for d in self.done) / n,
                "success": sum(d[2] for d in self.done) / n,
                "crash": sum(d[3] for d in self.done) / n,
                "offroad": sum(d[4] for d in self.done) / n}


class CSVLogger:
    def __init__(self, path: str | Path, echo_every: int = 0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.echo_every, self._n = echo_every, 0

    def __call__(self, step: int, stats: dict):
        row = {"step": step, **stats}
        self.rows.append(row)
        self._n += 1
        if self.echo_every and self._n % self.echo_every == 0:
            msg = " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
            print(msg, flush=True)

    def flush(self):
        keys = sorted({k for r in self.rows for k in r}, key=lambda k: (k != "step", k))
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)
