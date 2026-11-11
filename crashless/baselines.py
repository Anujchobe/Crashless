"""Non-learned reference controllers, used to put RL numbers in context."""
from __future__ import annotations

import numpy as np


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, deterministic=True):
        return self.rng.uniform(-1, 1, 2).astype(np.float32)


class LaneFollowPolicy:
    """PD controller on lateral offset and heading error, fixed target speed.
    It keeps the lane well but has no notion of obstacles, so its crash rate
    shows how much of the task is obstacle avoidance."""

    def __init__(self, n_rays: int, k_lat: float = 1.2, k_head: float = 3.0, target_speed: float = 0.6):
        self.n, self.k_lat, self.k_head, self.vt = n_rays, k_lat, k_head, target_speed

    def act(self, obs, deterministic=True):
        lat, head, speed = obs[self.n], obs[self.n + 1], obs[self.n + 2]
        steer = -self.k_lat * lat - self.k_head * head
        throttle = 2.0 * (self.vt - speed)
        return np.clip([steer, throttle], -1, 1).astype(np.float32)
