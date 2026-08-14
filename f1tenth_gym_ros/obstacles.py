# MIT License

# Copyright (c) 2026 NTU Autonomous Racing Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Virtual obstacles injected into the simulated laser scans.

The gym raycasts scans against the track's occupancy map, which only contains
walls. This module adds cylindrical obstacles that exist purely on the bridge
side: each published scan is min-merged with an exact ray/circle intersection
per beam, so obstacles occlude walls and are occluded by them, exactly like a
box placed on a real track. The gym itself never sees them — driving through
one is not a collision in the sim (the bridge logs a warning instead).
"""

import math

import numpy as np


def parse_obstacles(spec, default_radius):
    """Parse an obstacle spec string into a list of (x, y, r) tuples.

    Format: semicolon-separated obstacles, each 'x,y' or 'x,y,r'. Whitespace
    is ignored; empty entries are skipped. Raises ValueError on bad input so
    a typo'd parameter set is rejected instead of silently dropped.
    """
    circles = []
    for entry in str(spec).split(';'):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p.strip() for p in entry.split(',')]
        if len(parts) not in (2, 3):
            raise ValueError(
                f"obstacle entry '{entry}' must be 'x,y' or 'x,y,r'")
        x, y = float(parts[0]), float(parts[1])
        r = float(parts[2]) if len(parts) == 3 else float(default_radius)
        if r <= 0.0:
            raise ValueError(f"obstacle entry '{entry}' has radius <= 0")
        circles.append((x, y, r))
    return circles


class ObstacleField:
    """A set of cylindrical obstacles and the scan math to make them visible."""

    def __init__(self, default_radius=0.2):
        self.default_radius = float(default_radius)
        self.circles = []

    def set_from_spec(self, spec):
        """Replace all obstacles from a spec string (see parse_obstacles)."""
        self.circles = parse_obstacles(spec, self.default_radius)

    def to_spec(self):
        return '; '.join(f'{x:.3f},{y:.3f},{r:.3f}' for x, y, r in self.circles)

    def add(self, x, y, r=None):
        self.circles.append((float(x), float(y),
                             float(r) if r else self.default_radius))

    def remove_at(self, x, y):
        """Remove the obstacle whose disc contains (x, y). True if one was."""
        for i, (cx, cy, r) in enumerate(self.circles):
            if math.hypot(x - cx, y - cy) <= r:
                del self.circles[i]
                return True
        return False

    def clear(self):
        self.circles = []

    def hit_index(self, x, y, margin=0.0):
        """Index of the first obstacle within margin of point (x, y), or -1."""
        for i, (cx, cy, r) in enumerate(self.circles):
            if math.hypot(x - cx, y - cy) <= r + margin:
                return i
        return -1

    def inject(self, ranges, lidar_x, lidar_y, lidar_yaw, beam_angles,
               range_min, noise_std=0.0, rng=None):
        """Min-merge obstacle returns into a scan.

        ranges: iterable of map-raycast ranges, one per beam.
        beam_angles: per-beam angles in the lidar frame (numpy array).
        Returns a numpy array of the merged ranges.
        """
        merged = np.asarray(ranges, dtype=np.float64)
        if not self.circles:
            return merged
        theta = lidar_yaw + beam_angles
        ux = np.cos(theta)
        uy = np.sin(theta)
        for cx, cy, r in self.circles:
            dx = cx - lidar_x
            dy = cy - lidar_y
            # Distance along each beam to the closest point to the centre,
            # then back off by the half-chord to the near surface.
            t_mid = dx * ux + dy * uy
            disc = r * r - ((dx * dx + dy * dy) - t_mid * t_mid)
            hit = (disc >= 0.0) & (t_mid >= 0.0)
            if not hit.any():
                continue
            t_hit = t_mid - np.sqrt(np.where(hit, disc, 0.0))
            if noise_std > 0.0 and rng is not None:
                t_hit = t_hit + rng.normal(0.0, noise_std, size=t_hit.shape)
            # Below the lidar's minimum range there is no return, same as a
            # real scanner pressed up against a box.
            t_hit = np.where(hit & (t_hit >= range_min), t_hit, np.inf)
            merged = np.minimum(merged, t_hit)
        return merged
