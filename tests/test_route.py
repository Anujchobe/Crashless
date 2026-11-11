"""Geometry used by the CARLA env, checked against cases with known answers."""
import numpy as np
import pytest

from crashless.route import Route, carla_to_local, lidar_ray_distances, wrap_angle


def straight(length=100.0, hw=3.5):
    xs = np.arange(0.0, length + 1e-9, 1.0)
    return Route(np.stack([xs, np.zeros_like(xs)], 1), hw)


def circle(radius=30.0, hw=3.5, n=200):
    th = np.linspace(0, np.pi, n)                       # counter-clockwise half circle
    return Route(np.stack([radius * np.sin(th), radius * (1 - np.cos(th))], 1), hw)


def test_straight_route_projection():
    r = straight()
    pr = r.project(np.array([[10.5, 2.0], [40.0, -1.0]]))
    np.testing.assert_allclose(pr["s"], [10.5, 40.0])
    np.testing.assert_allclose(pr["lateral"], [2.0, -1.0])      # left is positive
    np.testing.assert_allclose(pr["yaw"], [0.0, 0.0])
    assert r.length == pytest.approx(100.0)


def test_circle_lateral_is_radial_distance():
    R = 30.0
    r = circle(R)
    # a point 2 m inside the circle (towards the centre = left of a CCW route)
    th = 0.8
    centre = np.array([0.0, R])
    on = np.array([R * np.sin(th), R * (1 - np.cos(th))])
    inside = on + 2.0 * (centre - on) / R
    pr = r.project(inside)
    assert pr["lateral"] == pytest.approx(2.0, abs=0.02)
    assert pr["s"] == pytest.approx(R * th, abs=0.05)
    assert pr["yaw"] == pytest.approx(th, abs=0.02)


def test_window_prevents_snapping_to_a_nearby_later_part_of_the_route():
    # U-turn: out along y=0, back along y=6
    out = np.stack([np.arange(0, 50.0), np.zeros(50)], 1)
    back = np.stack([np.arange(49.0, -1, -1), np.full(50, 6.0)], 1)
    r = Route(np.concatenate([out, np.array([[50.0, 3.0]]), back]), 3.5)
    p = np.array([10.0, 3.2])                      # slightly closer to the return leg
    assert r.project(p)["s"] > 60                  # global search jumps to the return leg
    assert r.project(p, hint=10, window=15)["s"] == pytest.approx(10.0, abs=0.01)


def test_half_width_is_interpolated():
    r = Route(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([2.0, 4.0]))
    assert r.project(np.array([5.0, 0.0]))["hw"] == pytest.approx(3.0)


def test_ray_edge_distances_on_straight_corridor():
    r = straight(hw=3.5)
    angles = np.deg2rad([-60, -30, 0, 30, 60])
    ts = np.arange(0.5, 25.0 + 1e-9, 0.5)
    d = r.ray_edge_distances(20.0, 0.0, 0.0, angles, ts)
    expected = np.minimum(3.5 / np.abs(np.sin(angles[[0, 1, 3, 4]])), 25.0)
    # sampled every 0.5 m, so the hit is within one step past the true edge
    assert np.all(d[[0, 1, 3, 4]] >= expected - 1e-9)
    assert np.all(d[[0, 1, 3, 4]] <= expected + 0.5)
    assert d[2] == 25.0                            # straight ahead never leaves the road


def test_lidar_binning_picks_nearest_point_per_ray():
    angles = np.deg2rad([-60, 0, 60])
    pts = np.array([[10.0, 0.0], [5.0, 0.2],                  # ahead: nearest is ~5 m
                    [5.0 * np.cos(np.deg2rad(60)), 5.0 * np.sin(np.deg2rad(60))],   # left ray
                    [-3.0, 0.0]])                              # behind: ignored
    d = lidar_ray_distances(pts, angles, 25.0)
    assert d[0] == 25.0
    assert d[1] == pytest.approx(np.hypot(5.0, 0.2))
    assert d[2] == pytest.approx(5.0)


def test_lidar_respects_max_range_and_empty_cloud():
    angles = np.deg2rad([-60, 0, 60])
    assert np.all(lidar_ray_distances(np.zeros((0, 2)), angles, 25.0) == 25.0)
    assert lidar_ray_distances(np.array([[40.0, 0.0]]), angles, 25.0)[1] == 25.0


def test_carla_frame_flip():
    # CARLA: y to the right, yaw clockwise. A car at y=+2 facing yaw=+90 deg
    # is 2 m to the RIGHT and facing RIGHT; in our frame that is y=-2, yaw=-pi/2.
    x, y, yaw = carla_to_local(1.0, 2.0, 90.0)
    assert (x, y) == (1.0, -2.0) and yaw == pytest.approx(-np.pi / 2)


def test_wrap_angle():
    np.testing.assert_allclose(wrap_angle([np.pi + 0.1, -np.pi - 0.1, 0.3]), [-np.pi + 0.1, np.pi - 0.1, 0.3])


def test_carla_config_matches_2d_contract():
    """The CARLA env must expose the same observation/action contract as the 2D env,
    otherwise a 2D-trained policy would silently read the wrong numbers."""
    from crashless.carla_env import CarlaConfig
    from crashless.env import EnvConfig
    c, e = CarlaConfig(), EnvConfig()
    for k in ("dt", "max_steps", "max_steer", "max_speed", "n_rays", "ray_fov_deg", "ray_range", "ray_step"):
        assert getattr(c, k) == getattr(e, k), k
