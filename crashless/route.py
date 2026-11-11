"""Pure-NumPy geometry for driving along a polyline route.

Nothing here imports CARLA, so it is unit-tested on any machine. The CARLA
environment turns the map's waypoints into a `Route`, then asks it the same
questions the 2D simulator answers analytically:
    - how far along the route am I?           -> progress reward, success
    - how far am I from the corridor centre?   -> lateral obs, off-road check
    - which way does the road point here?      -> heading-error obs
    - where do my range rays leave the road?   -> ray obs (road-edge part)

Frame convention: right-handed, x forward / y LEFT, yaw counter-clockwise.
CARLA's world is left-handed (y to the right), so the CARLA env flips y and
yaw before anything reaches this module. See `carla_to_local`.
"""
from __future__ import annotations

import numpy as np


def carla_to_local(x: float, y: float, yaw_deg: float) -> tuple[float, float, float]:
    """CARLA (left-handed, yaw in degrees, clockwise-positive seen from above)
    -> our right-handed frame (yaw in radians, counter-clockwise-positive)."""
    return x, -y, -np.deg2rad(yaw_deg)


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


class Route:
    """A driving corridor: centreline points, one half-width per point.

    points:      (N, 2) centreline, roughly evenly spaced (CARLA: 1 m)
    half_width:  (N,) corridor half-width at each point
    """

    def __init__(self, points: np.ndarray, half_width: np.ndarray | float):
        self.p = np.asarray(points, dtype=np.float64)
        if len(self.p) < 2:
            raise ValueError("route needs at least 2 points")
        self.hw = np.broadcast_to(np.asarray(half_width, dtype=np.float64), (len(self.p),)).copy()
        seg = np.diff(self.p, axis=0)                      # (N-1, 2)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        if np.any(self.seg_len < 1e-9):
            raise ValueError("route has duplicate consecutive points")
        self.seg_dir = seg / self.seg_len[:, None]          # unit tangents
        self.seg_yaw = np.arctan2(self.seg_dir[:, 1], self.seg_dir[:, 0])
        self.s_at = np.concatenate([[0.0], np.cumsum(self.seg_len)])   # arc length at each point
        self.length = float(self.s_at[-1])

    def project(self, xy: np.ndarray, hint: int | None = None, window: int | None = None):
        """Project points onto the route.

        xy: (..., 2). Returns dict of arrays shaped like xy[..., 0]:
            s        arc length of the closest point on the route
            lateral  signed distance, positive = LEFT of the direction of travel
            yaw      route tangent at the closest point
            seg      index of the closest segment
            hw       corridor half-width there (linearly interpolated)
        `hint`/`window` restrict the search to segments [hint-window, hint+window],
        which keeps per-step cost constant on long routes and stops a point
        from snapping onto a different part of a route that loops back near itself.
        """
        xy = np.asarray(xy, dtype=np.float64)
        flat = xy.reshape(-1, 2)
        lo, hi = 0, len(self.seg_len)
        if hint is not None and window is not None:
            lo, hi = max(0, hint - window), min(len(self.seg_len), hint + window + 1)
        a = self.p[lo:hi]                                   # segment starts (S, 2)
        d = self.seg_dir[lo:hi]
        L = self.seg_len[lo:hi]
        rel = flat[:, None, :] - a[None]                    # (M, S, 2)
        t = np.clip(np.einsum("msk,sk->ms", rel, d), 0.0, L[None])
        closest = a[None] + t[..., None] * d[None]
        dist2 = np.sum((flat[:, None, :] - closest) ** 2, axis=-1)
        k = np.argmin(dist2, axis=1)                        # (M,)
        m = np.arange(len(flat))
        tk = t[m, k]
        relk = rel[m, k]
        dk = d[k]
        lateral = dk[:, 0] * relk[:, 1] - dk[:, 1] * relk[:, 0]   # 2D cross product
        seg = k + lo
        frac = tk / self.seg_len[seg]
        hw = self.hw[seg] * (1 - frac) + self.hw[seg + 1] * frac
        shape = xy.shape[:-1]
        return {"s": (self.s_at[seg] + tk).reshape(shape),
                "lateral": lateral.reshape(shape),
                "yaw": self.seg_yaw[seg].reshape(shape),
                "seg": seg.reshape(shape),
                "hw": hw.reshape(shape)}

    def ray_edge_distances(self, x, y, yaw, ray_angles, ray_ts, hint=None, window=None):
        """Distance along each ray until it leaves the corridor (ray_ts[-1] if never).
        Same sampling scheme as the 2D env: points every ray_step metres."""
        ang = yaw + np.asarray(ray_angles)
        pts = np.stack([x + np.cos(ang)[:, None] * ray_ts,
                        y + np.sin(ang)[:, None] * ray_ts], axis=-1)       # (R, T, 2)
        pr = self.project(pts, hint, window)
        out = np.abs(pr["lateral"]) > pr["hw"]
        first = np.where(out.any(1), out.argmax(1), len(ray_ts) - 1)
        return np.asarray(ray_ts)[first]


def lidar_ray_distances(points_local: np.ndarray, ray_angles: np.ndarray, max_range: float,
                        half_beam: float | None = None) -> np.ndarray:
    """Turn a lidar point cloud into the env's fixed set of range rays.

    points_local: (M, 2) obstacle points in the SENSOR frame, already in our
                  right-handed convention (x forward, y left).
    Each ray reports the nearest point within +-half_beam of its angle
    (default: half the spacing between rays), or max_range if none.
    """
    ray_angles = np.asarray(ray_angles)
    if half_beam is None:
        half_beam = 0.5 * float(np.min(np.diff(ray_angles))) if len(ray_angles) > 1 else np.pi
    out = np.full(len(ray_angles), float(max_range))
    if len(points_local) == 0:
        return out
    pts = np.asarray(points_local, dtype=np.float64)
    r = np.hypot(pts[:, 0], pts[:, 1])
    th = np.arctan2(pts[:, 1], pts[:, 0])
    diff = np.abs(wrap_angle(th[None, :] - ray_angles[:, None]))   # (R, M)
    in_beam = (diff <= half_beam) & (r[None, :] <= max_range)
    masked = np.where(in_beam, r[None, :], np.inf)
    return np.minimum(out, masked.min(axis=1))
