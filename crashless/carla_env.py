"""CARLA version of the Crashless driving task, with the same contract as DrivingEnv.

Same observation (14 floats), same action (2 floats), same reward formula, same
terminated/truncated rules. That is deliberate: a policy trained in the 2D
simulator can be dropped in here unchanged, and the gap between how it drives
there and here is the sim-to-sim transfer gap.

How each part of the observation is produced in CARLA:
    range rays, obstacles  -> semantic LiDAR on the front bumper, keeping only
                              points that hit our spawned obstacle cars
                              (or "ground_truth": exact obstacle boxes, for debugging)
    range rays, road edges -> map geometry: the route corridor from `route.py`
    lateral / heading      -> projection of the car onto the route
    speed                  -> vehicle velocity
Crash = CARLA collision sensor fired (any object). Off-road = left the corridor.

The corridor: CARLA's Town02 roads are two lanes, one per direction. We treat
the whole carriageway as the corridor (centre = the road's centre line,
half-width = one lane width: 4 m in Town02, same as the 2D sim). The car
must stay on the road and pass parked cars by using the other lane. With no
oncoming traffic that is a lane-keeping / obstacle-avoidance task, not
traffic-legal driving.

Coordinates: CARLA is left-handed (y right, yaw clockwise, degrees). Everything
is converted to our right-handed frame (y left, yaw counter-clockwise, radians)
on the way in, and steering is negated on the way out.

Server: run CARLA first, e.g.  ./CarlaUE4.sh -RenderOffScreen -quality-level=Low
This env switches the world to synchronous mode (fixed dt) and, by default,
no-rendering mode; `close()` switches both back.
"""
from __future__ import annotations

import queue
from dataclasses import asdict, dataclass, field

import gymnasium as gym
import numpy as np

from .env import RewardWeights
from .route import Route, carla_to_local, lidar_ray_distances, wrap_angle

try:
    import carla
except ImportError:  # lets the module import on machines without CARLA (docs, tests)
    carla = None

LIDAR_DTYPE = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"),
                        ("cos", "f4"), ("obj", "u4"), ("tag", "u4")])


@dataclass
class CarlaConfig:
    host: str = "localhost"
    port: int = 2000
    timeout: float = 30.0
    town: str = "Town02"
    no_rendering: bool = True
    ego_blueprint: str = "vehicle.tesla.model3"
    obstacle_blueprint: str = "vehicle.audi.a2"
    route_length: float = 200.0
    junction_choice: str = "straightest"     # or "random"
    n_obstacles: int = 4
    obstacle_sensing: str = "lidar"           # or "ground_truth"
    lidar_height: float = 0.7
    # Each env ray = nearest lidar point within +-this angle. Must be narrow: the 2D
    # sim casts thin rays, and a wide cone sees obstacles a thin ray would miss
    # (first CARLA run with +-7.5 deg: 0.93 m mean / 7.5 m p95 error vs ground truth).
    lidar_half_beam_deg: float = 1.0
    settle_ticks: int = 10
    start_speed: float = 5.0
    # ---- shared contract with DrivingEnv (keep in sync) ----
    dt: float = 0.1
    max_steps: int = 400
    max_steer: float = 0.5        # rad at the wheels for action = +-1
    max_speed: float = 15.0
    n_rays: int = 9
    ray_fov_deg: float = 120.0
    ray_range: float = 25.0
    ray_step: float = 0.5
    offroad_margin: float = 0.5   # off-road when |lateral| > half_width - margin
    rewards: RewardWeights = field(default_factory=RewardWeights)

    @classmethod
    def from_dict(cls, d: dict | None) -> "CarlaConfig":
        d = dict(d or {})
        return cls(rewards=RewardWeights(**d.pop("rewards", {})), **d)

    def to_dict(self) -> dict:
        return asdict(self)


class CarlaDrivingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: CarlaConfig | dict | None = None):
        if carla is None:
            raise ImportError("the `carla` package is not installed. On the CARLA machine: "
                              "pip install carla==0.9.15 (or the .whl in CARLA/PythonAPI/carla/dist)")
        super().__init__()
        self.cfg = config if isinstance(config, CarlaConfig) else CarlaConfig.from_dict(config)
        c = self.cfg
        self.obs_dim = c.n_rays + 5
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        half_fov = np.deg2rad(c.ray_fov_deg) / 2
        self._ray_angles = np.linspace(-half_fov, half_fov, c.n_rays)
        self._ray_ts = np.arange(c.ray_step, c.ray_range + 1e-9, c.ray_step)
        self._rng = np.random.default_rng()

        self.client = carla.Client(c.host, c.port)
        self.client.set_timeout(c.timeout)
        self.world = self.client.get_world()
        if not self.world.get_map().name.endswith(c.town):
            self.world = self.client.load_world(c.town)
        self._orig_settings = self.world.get_settings()
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = c.dt
        s.no_rendering_mode = c.no_rendering
        self.world.apply_settings(s)
        self.map = self.world.get_map()
        self._spawn_points = self.map.get_spawn_points()
        self._actors: list = []
        self._ego = None

    # ------------------------------------------------------------ episode setup
    def _build_route(self) -> tuple[Route, list]:
        """Follow waypoints 1 m apart from a random spawn point until the route is long enough."""
        c = self.cfg
        need = c.route_length + 15.0
        for _ in range(50):
            sp = self._spawn_points[self._rng.integers(len(self._spawn_points))]
            wp = self.map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            wps = [wp]
            while len(wps) < need:
                nxt = wps[-1].next(1.0)
                if not nxt:
                    break
                if len(nxt) == 1:
                    wps.append(nxt[0])
                elif c.junction_choice == "straightest":
                    yaw0 = wps[-1].transform.rotation.yaw
                    wps.append(min(nxt, key=lambda w: abs(wrap_angle(np.deg2rad(w.transform.rotation.yaw - yaw0)))))
                else:
                    wps.append(nxt[self._rng.integers(len(nxt))])
            if len(wps) >= need:
                break
        else:
            raise RuntimeError("could not find a long enough route; lower route_length")

        pts, hws = [], []
        off = None
        for w in wps:
            x, y, yaw = carla_to_local(w.transform.location.x, w.transform.location.y, w.transform.rotation.yaw)
            left = w.get_left_lane()
            two_way = left is not None and left.lane_type == carla.LaneType.Driving and left.lane_id * w.lane_id < 0
            if two_way or off is None or not w.is_junction:
                # corridor centre = this lane's left edge (the road centre line on a two-lane road)
                off = w.lane_width / 2 if two_way else 0.0
                hw = w.lane_width if two_way else w.lane_width / 2
            # inside junctions there is often no left lane: keep the last offset, avoid a jump
            pts.append((x - np.sin(yaw) * off, y + np.cos(yaw) * off))
            hws.append(hw)
        pts = np.asarray(pts)
        keep = np.r_[True, np.hypot(*np.diff(pts, axis=0).T) > 1e-3]
        return Route(pts[keep], np.asarray(hws)[keep]), [w for w, k in zip(wps, keep) if k]

    def _to_carla_transform(self, x, y, yaw, z):
        return carla.Transform(carla.Location(x=float(x), y=float(-y), z=float(z)),
                               carla.Rotation(yaw=float(-np.rad2deg(yaw))))

    def _spawn_obstacles(self, route: Route, z: float):
        c, lib = self.cfg, self.world.get_blueprint_library()
        bp = lib.find(c.obstacle_blueprint)
        self._obstacles = []
        if c.n_obstacles == 0:
            return
        # same placement rule as the 2D env: one per segment, >= 16 m apart
        edges = np.linspace(25, c.route_length - 10, c.n_obstacles + 1)
        for s in self._rng.uniform(edges[:-1] + 8, edges[1:] - 8):
            k = int(np.searchsorted(route.s_at, s)) - 1
            k = min(max(k, 0), len(route.seg_yaw) - 1)
            yaw = route.seg_yaw[k]
            base = route.p[k] + route.seg_dir[k] * (s - route.s_at[k])
            side = self._rng.choice([-1.0, 1.0])
            off = side * route.hw[k] / 2                          # centre of one lane
            x, y = base[0] - np.sin(yaw) * off, base[1] + np.cos(yaw) * off
            a = self.world.try_spawn_actor(bp, self._to_carla_transform(x, y, yaw, z + 0.5))
            if a is None:
                continue
            self._actors.append(a)
            self._obstacles.append(a)

    def _spawn_ego_and_sensors(self, wp0):
        c, lib = self.cfg, self.world.get_blueprint_library()
        bp = lib.find(c.ego_blueprint)
        bp.set_attribute("role_name", "hero")
        t = wp0.transform
        t.location.z += 0.5
        self._ego = self.world.try_spawn_actor(bp, t)
        if self._ego is None:
            return False
        self._actors.append(self._ego)
        phys = self._ego.get_physics_control()
        max_deg = max(w.max_steer_angle for w in phys.wheels)
        self._steer_scale = np.rad2deg(c.max_steer) / max_deg   # action 1.0 -> max_steer rad at the wheels

        col = self.world.spawn_actor(lib.find("sensor.other.collision"), carla.Transform(), attach_to=self._ego)
        col.listen(lambda e: setattr(self, "_collided", True))
        self._actors.append(col)

        ext = self._ego.bounding_box.extent
        self._lidar_x = ext.x + 0.1                               # front bumper
        lb = lib.find("sensor.lidar.ray_cast_semantic")
        for k, v in {"channels": 1, "range": c.ray_range, "upper_fov": 0, "lower_fov": 0,
                     "rotation_frequency": 1.0 / c.dt, "points_per_second": int(720 / c.dt)}.items():
            lb.set_attribute(k, str(v))
        if lb.has_attribute("horizontal_fov"):
            lb.set_attribute("horizontal_fov", "360")
        self._lidar = self.world.spawn_actor(
            lb, carla.Transform(carla.Location(x=self._lidar_x, z=c.lidar_height)), attach_to=self._ego)
        self._lidar_q = queue.Queue()
        self._lidar.listen(self._lidar_q.put)
        self._actors.append(self._lidar)
        return True

    def _destroy(self):
        for a in reversed(self._actors):
            try:
                if a.type_id.startswith("sensor."):
                    a.stop()
                a.destroy()
            except RuntimeError:
                pass
        self._actors, self._ego = [], None

    # ------------------------------------------------------------ gym api
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._destroy()
        for _ in range(20):
            self.route, wps = self._build_route()
            if self._spawn_ego_and_sensors(wps[0]):
                break
            self._destroy()
        else:
            raise RuntimeError("could not spawn the ego vehicle")
        self._spawn_obstacles(self.route, wps[0].transform.location.z)
        # let the car drop onto the road, then give it the same start speed as the 2D env
        for _ in range(self.cfg.settle_ticks):
            self._ego.apply_control(carla.VehicleControl(brake=1.0))
            self._tick()
        for o in self._obstacles:
            o.set_simulate_physics(False)       # parked cars: settled on the road, now frozen
        fwd = self._ego.get_transform().get_forward_vector()
        self._ego.set_target_velocity(carla.Vector3D(fwd.x * self.cfg.start_speed, fwd.y * self.cfg.start_speed, 0))
        self._lidar_pts = self._tick()
        self._collided = False
        self._seg = 0
        self._update_state()
        self._s0 = self._s
        self.prev_action = np.zeros(2, np.float32)
        self.steps = 0
        return self._observe(), {}

    def step(self, action):
        c, w = self.cfg, self.cfg.rewards
        a = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
        speed = self._speed()
        throttle = float(a[1]) if a[1] > 0 and speed < c.max_speed else 0.0
        brake = float(-a[1]) if a[1] < 0 else 0.0
        steer = float(np.clip(-a[0] * self._steer_scale, -1.0, 1.0))   # CARLA: +steer = right
        self._ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        s_prev = self._s
        self._lidar_pts = self._tick()
        self._update_state()
        self.steps += 1

        crashed = self._collided
        offroad = abs(self._lat) > self._hw - c.offroad_margin
        success = (self._s - self._s0) >= c.route_length and not (crashed or offroad)
        reward = (w.progress * (self._s - s_prev)
                  - w.lateral * (self._lat / self._hw) ** 2
                  - w.heading * (self._head_err / np.pi) ** 2
                  - w.smooth * abs(float(a[0] - self.prev_action[0])))
        if crashed or offroad:
            reward -= w.crash
        elif success:
            reward += w.success
        self.prev_action = a
        terminated = bool(crashed or offroad or success)
        truncated = bool(not terminated and self.steps >= c.max_steps)
        info = {"crashed": bool(crashed), "offroad": bool(offroad and not crashed),
                "success": bool(success), "lateral": float(self._lat), "speed": float(self._speed()),
                "steer": float(a[0]), "progress": float((self._s - self._s0) / c.route_length)}
        return self._observe(), float(reward), terminated, truncated, info

    def close(self):
        self._destroy()
        try:
            self.world.apply_settings(self._orig_settings)
        except RuntimeError:
            pass

    # ------------------------------------------------------------ internals
    def _tick(self) -> np.ndarray:
        """Advance one fixed step and return this frame's obstacle lidar points (our frame, from car centre)."""
        frame = self.world.tick()
        while True:
            data = self._lidar_q.get(timeout=self.cfg.timeout)
            if data.frame >= frame:
                break
        pts = np.frombuffer(data.raw_data, dtype=LIDAR_DTYPE)
        ids = np.array([o.id for o in self._obstacles], dtype=np.uint32) if getattr(self, "_obstacles", None) else []
        pts = pts[np.isin(pts["obj"], ids)]
        # sensor frame (x fwd, y right) -> ours (x fwd, y left), origin moved to the car centre
        return np.stack([pts["x"] + self._lidar_x, -pts["y"]], axis=1).astype(np.float64)

    def _speed(self) -> float:
        v = self._ego.get_velocity()
        return float(np.hypot(v.x, v.y))

    def _update_state(self):
        t = self._ego.get_transform()
        self._x, self._y, self._yaw = carla_to_local(t.location.x, t.location.y, t.rotation.yaw)
        pr = self.route.project(np.array([self._x, self._y]), hint=self._seg, window=30)
        self._seg = int(pr["seg"])
        self._s, self._lat, self._hw = float(pr["s"]), float(pr["lateral"]), float(pr["hw"])
        self._head_err = float(wrap_angle(self._yaw - pr["yaw"]))

    def _obstacle_rays(self) -> np.ndarray:
        c = self.cfg
        if c.obstacle_sensing == "lidar":
            return lidar_ray_distances(self._lidar_pts, self._ray_angles, c.ray_range,
                                       half_beam=np.deg2rad(c.lidar_half_beam_deg))
        # ground truth: sample the same rays and test against each obstacle's oriented box
        ang = self._yaw + self._ray_angles
        px = self._x + np.cos(ang)[:, None] * self._ray_ts
        py = self._y + np.sin(ang)[:, None] * self._ray_ts
        hit = np.zeros(px.shape, bool)
        for o in self._obstacles:
            t, e = o.get_transform(), o.bounding_box.extent
            ox, oy, oyaw = carla_to_local(t.location.x, t.location.y, t.rotation.yaw)
            dx, dy = px - ox, py - oy
            u = np.cos(oyaw) * dx + np.sin(oyaw) * dy
            v = -np.sin(oyaw) * dx + np.cos(oyaw) * dy
            hit |= (np.abs(u) < e.x) & (np.abs(v) < e.y)
        first = np.where(hit.any(1), hit.argmax(1), len(self._ray_ts) - 1)
        d = self._ray_ts[first]
        d[~hit.any(1)] = c.ray_range
        return d

    def _observe(self) -> np.ndarray:
        c = self.cfg
        edges = self.route.ray_edge_distances(self._x, self._y, self._yaw, self._ray_angles,
                                              self._ray_ts, hint=self._seg, window=40)
        rays = np.minimum(edges, self._obstacle_rays()) / c.ray_range
        return np.concatenate([rays, [self._lat / self._hw, self._head_err / np.pi, self._speed() / c.max_speed],
                               self.prev_action]).astype(np.float32)
