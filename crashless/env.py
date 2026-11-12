"""Lightweight 2D driving environment for lane-keeping and obstacle avoidance.

Why not CARLA directly: CARLA needs a Linux/Windows GPU server and minutes per
episode, which makes multi-seed benchmarking and unit tests impractical. This
env keeps the parts of the problem the RL agent actually sees (a ray-based
range sensor, lane geometry, continuous steering/throttle, collisions) and runs
~20k steps/s on one laptop CPU core (measured with random actions). The observation/action contract is documented in
README.md so a CARLA wrapper can expose the same interface.

Dynamics: kinematic bicycle model, dt = 0.1 s.
Road: a winding centerline y = f(x) built from two sinusoids, fixed half-width.
Obstacles: static discs placed on one side of the lane so a free gap always
exists on the other side.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import gymnasium as gym
import numpy as np


@dataclass
class RewardWeights:
    progress: float = 1.0      # per metre of forward progress along x
    lateral: float = 0.3       # * (lateral offset / half_width)^2, per step
    heading: float = 0.1       # * (heading error / pi)^2, per step
    smooth: float = 0.1        # * |delta steer action|, per step
    crash: float = 50.0        # one-off penalty on collision or leaving the road
    success: float = 20.0      # one-off bonus for reaching the end of the road


@dataclass
class EnvConfig:
    road_length: float = 200.0
    half_width: float = 4.0
    n_obstacles: int = 4
    obstacle_radius: float = 1.0
    obstacle_offset: float = 2.0     # lateral offset of obstacle centre from centerline
    car_radius: float = 1.0
    wheelbase: float = 2.5
    max_steer: float = 0.5           # rad
    max_accel: float = 4.0           # m/s^2
    max_speed: float = 15.0          # m/s
    dt: float = 0.1
    max_steps: int = 400
    n_rays: int = 9
    ray_fov_deg: float = 120.0
    ray_range: float = 25.0
    ray_step: float = 0.5
    # Max change of the normalised steer command per step (actuator rate limit).
    # None = the policy sets the wheel angle directly each step.
    steer_rate: float | None = None
    rewards: RewardWeights = field(default_factory=RewardWeights)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        rw = RewardWeights(**d.pop("rewards", {}))
        return cls(rewards=rw, **d)

    def to_dict(self) -> dict:
        return asdict(self)


class DrivingEnv(gym.Env):
    """Observation (float32, len n_rays + 5), all roughly in [-1, 1]:
        [0:n_rays]   normalised free distance along each ray (1 = nothing within range)
        n_rays + 0   lateral offset from centerline / half_width
        n_rays + 1   heading error relative to road tangent / pi
        n_rays + 2   speed / max_speed
        n_rays + 3   previous steer action
        n_rays + 4   previous throttle action
    Action (float32, len 2, clipped to [-1, 1]): [steer, throttle(+)/brake(-)].
    With steer_rate set, the applied steer moves toward the commanded one by at
    most steer_rate per step; obs and info report the APPLIED action.
    Episode ends (terminated) on collision, leaving the road, or reaching
    x >= road_length; it is truncated at max_steps.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | dict | None = None):
        super().__init__()
        self.cfg = config if isinstance(config, EnvConfig) else EnvConfig.from_dict(config)
        c = self.cfg
        self.obs_dim = c.n_rays + 5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        half_fov = np.deg2rad(c.ray_fov_deg) / 2
        self._ray_angles = np.linspace(-half_fov, half_fov, c.n_rays)
        self._ray_ts = np.arange(c.ray_step, c.ray_range + 1e-9, c.ray_step)
        self._rng = np.random.default_rng()

    # ---------------------------------------------------------------- road
    def _center(self, x):
        a1, w1, a2, w2, ph = self._road
        return a1 * np.sin(w1 * x) + a2 * np.sin(w2 * x + ph)

    def _center_slope(self, x):
        a1, w1, a2, w2, ph = self._road
        return a1 * w1 * np.cos(w1 * x) + a2 * w2 * np.cos(w2 * x + ph)

    def _lateral(self, x, y):
        """Signed perpendicular-ish offset from the centerline (left positive)."""
        return (y - self._center(x)) * np.cos(np.arctan(self._center_slope(x)))

    # ---------------------------------------------------------------- api
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        c, r = self.cfg, self._rng
        self._road = (r.uniform(2, 6), 2 * np.pi / r.uniform(60, 120),
                      r.uniform(0.5, 2), 2 * np.pi / r.uniform(25, 50), r.uniform(0, 2 * np.pi))
        # One obstacle per equal road segment, >= 16 m apart, first >= 33 m ahead,
        # so every layout is passable.
        edges = np.linspace(25, c.road_length - 10, c.n_obstacles + 1)
        xs = r.uniform(edges[:-1] + 8, edges[1:] - 8)
        sides = r.choice([-1.0, 1.0], c.n_obstacles)
        slopes = np.arctan(self._center_slope(xs))
        off = sides * c.obstacle_offset
        self._obs_xy = np.stack([xs - off * np.sin(slopes), self._center(xs) + off * np.cos(slopes)], 1) \
            if c.n_obstacles else np.zeros((0, 2))
        self.x, self.y = 0.0, float(self._center(0.0))
        self.yaw = float(np.arctan(self._center_slope(0.0)))
        self.v = 5.0
        self.prev_action = np.zeros(2, np.float32)
        self.steps = 0
        return self._observe(), {}

    def step(self, action):
        c, w = self.cfg, self.cfg.rewards
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        if c.steer_rate is not None:
            a[0] = self.prev_action[0] + np.clip(a[0] - self.prev_action[0], -c.steer_rate, c.steer_rate)
        steer = a[0] * c.max_steer
        self.v = float(np.clip(self.v + a[1] * c.max_accel * c.dt, 0.0, c.max_speed))
        x0 = self.x
        self.x += self.v * np.cos(self.yaw) * c.dt
        self.y += self.v * np.sin(self.yaw) * c.dt
        self.yaw += self.v / c.wheelbase * np.tan(steer) * c.dt
        self.yaw = float((self.yaw + np.pi) % (2 * np.pi) - np.pi)
        self.steps += 1

        lat = float(self._lateral(self.x, self.y))
        head_err = self._heading_error()
        crashed = self._collided()
        offroad = abs(lat) > c.half_width - c.car_radius * 0.5
        success = self.x >= c.road_length and not (crashed or offroad)

        reward = (w.progress * (self.x - x0)
                  - w.lateral * (lat / c.half_width) ** 2
                  - w.heading * (head_err / np.pi) ** 2
                  - w.smooth * abs(float(a[0] - self.prev_action[0])))
        if crashed or offroad:
            reward -= w.crash
        elif success:
            reward += w.success
        self.prev_action = a
        terminated = bool(crashed or offroad or success)
        truncated = bool(not terminated and self.steps >= c.max_steps)
        info = {"crashed": bool(crashed), "offroad": bool(offroad and not crashed),
                "success": bool(success), "lateral": lat, "speed": self.v,
                "steer": float(a[0]), "progress": self.x / c.road_length}
        return self._observe(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- helpers
    def _heading_error(self) -> float:
        road_yaw = np.arctan(self._center_slope(self.x))
        return float((self.yaw - road_yaw + np.pi) % (2 * np.pi) - np.pi)

    def _collided(self) -> bool:
        if len(self._obs_xy) == 0:
            return False
        d = np.hypot(self._obs_xy[:, 0] - self.x, self._obs_xy[:, 1] - self.y)
        return bool(np.any(d < self.cfg.obstacle_radius + self.cfg.car_radius))

    def ray_distances(self) -> np.ndarray:
        """Distance along each ray to the first road edge or obstacle (<= ray_range)."""
        c = self.cfg
        ang = self.yaw + self._ray_angles                       # (R,)
        px = self.x + np.cos(ang)[:, None] * self._ray_ts       # (R, T)
        py = self.y + np.sin(ang)[:, None] * self._ray_ts
        hit = np.abs(self._lateral(px, py)) > c.half_width
        if len(self._obs_xy):
            dx = px[..., None] - self._obs_xy[:, 0]
            dy = py[..., None] - self._obs_xy[:, 1]
            hit |= np.any(dx * dx + dy * dy < c.obstacle_radius ** 2, axis=-1)
        first = np.where(hit.any(1), hit.argmax(1), len(self._ray_ts))
        ts = np.append(self._ray_ts, c.ray_range)
        return ts[first]

    def _observe(self) -> np.ndarray:
        c = self.cfg
        rays = self.ray_distances() / c.ray_range
        lat = self._lateral(self.x, self.y) / c.half_width
        return np.concatenate([rays, [lat, self._heading_error() / np.pi, self.v / c.max_speed],
                               self.prev_action]).astype(np.float32)
