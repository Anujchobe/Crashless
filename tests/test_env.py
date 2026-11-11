import numpy as np
import pytest

from crashless.baselines import LaneFollowPolicy
from crashless.env import DrivingEnv


def rollout(env, policy, seed, max_steps=1000):
    obs, _ = env.reset(seed=seed)
    for _ in range(max_steps):
        obs, r, te, tr, info = env.step(policy(obs))
        if te or tr:
            return info, te, tr
    raise AssertionError("episode never ended")


def test_observation_shape_and_range():
    env = DrivingEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32
    for _ in range(50):
        obs, *_ = env.step(env.action_space.sample())
        assert np.all(obs[: env.cfg.n_rays] >= 0) and np.all(obs[: env.cfg.n_rays] <= 1)
        assert np.all(np.isfinite(obs))


def test_same_seed_same_episode():
    a, b = DrivingEnv(), DrivingEnv()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa, ob)
    rng = np.random.default_rng(0)
    for _ in range(100):
        act = rng.uniform(-1, 1, 2)
        ra, rb = a.step(act), b.step(act)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1] == rb[1]


def test_different_seeds_change_layout():
    env = DrivingEnv()
    env.reset(seed=1)
    l1 = env._obs_xy.copy()
    env.reset(seed=2)
    assert not np.allclose(l1, env._obs_xy)


def test_hard_left_leaves_road_and_terminates():
    info, te, tr = rollout(DrivingEnv({"n_obstacles": 0}), lambda o: np.array([1.0, 0.5]), seed=0)
    assert te and not tr and info["offroad"]


def test_driving_straight_into_obstacle_is_a_collision():
    env = DrivingEnv({"n_obstacles": 1, "road_length": 100})
    env.reset(seed=0)
    ox, oy = env._obs_xy[0]
    env.x, env.y, env.yaw = ox - 5.0, oy, 0.0
    env._road = (0.0, 1.0, 0.0, 1.0, 0.0)          # straight road at y=0
    env._obs_xy[0, 1] = 0.0
    env.y = 0.0
    for _ in range(20):
        _, r, te, _, info = env.step([0.0, 0.0])
        if te:
            break
    assert info["crashed"] and r < -env.cfg.rewards.crash / 2


def test_lane_follower_finishes_empty_road():
    """Without obstacles a PD lane follower must always succeed: proves the road is drivable."""
    env = DrivingEnv({"n_obstacles": 0})
    pol = LaneFollowPolicy(env.cfg.n_rays)
    for seed in range(20):
        info, te, _ = rollout(env, lambda o: pol.act(o), seed)
        assert info["success"], f"seed {seed}"


@pytest.mark.parametrize("seed", range(200))
def test_obstacle_layout_always_passable(seed):
    """Every generated layout leaves >= 16 m between consecutive obstacles."""
    env = DrivingEnv()
    env.reset(seed=seed)
    xs = np.sort(env._obs_xy[:, 0])
    assert xs[0] > 30 and np.all(np.diff(xs) > 15)


def test_ray_sees_obstacle_ahead():
    env = DrivingEnv({"n_obstacles": 1})
    env.reset(seed=0)
    env._road = (0.0, 1.0, 0.0, 1.0, 0.0)
    env.x, env.y, env.yaw = 0.0, 0.0, 0.0
    env._obs_xy = np.array([[10.0, 0.0]])
    mid = env.cfg.n_rays // 2
    d = env.ray_distances()
    assert d[mid] == pytest.approx(9.0, abs=env.cfg.ray_step)   # 10 m minus radius 1


def test_steer_rate_limit():
    env = DrivingEnv({"steer_rate": 0.1, "n_obstacles": 0})
    env.reset(seed=0)
    _, _, _, _, info = env.step([1.0, 0.0])
    assert info["steer"] == pytest.approx(0.1)
    _, _, _, _, info = env.step([-1.0, 0.0])
    assert info["steer"] == pytest.approx(0.0)


def test_truncation_is_not_termination():
    env = DrivingEnv({"n_obstacles": 0, "max_steps": 5})
    env.reset(seed=0)
    for _ in range(5):
        _, _, te, tr, _ = env.step([0.0, -1.0])
    assert tr and not te
