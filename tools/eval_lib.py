#!/usr/bin/env python3
"""Shared primitives for offline evaluation of kf_object_tracker.

Bag reading, timestamp handling, interpolation, ground-truth extraction and the
camera-visibility test, factored out of what used to be three scripts that each
reimplemented them (analyze_bag.py, compare_bags.py, replay_tracker.py).

Nothing here imports rclpy beyond the message deserialisers, so it runs without
a live ROS graph.
"""

import math
import os
from collections import defaultdict

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

# Topics the harness knows about.
DET3D = '/object_localization/detections3d'
TRACKED = '/tracked_objects'
MOCAP = '/pose_modelcars'
KIN = '/kinematic_state'
DET2D = '/detectnet/detections'

# Camera envelope. The colour HFOV of the D435 is 69 deg; range limits come from
# the localizer's own config so the two cannot drift apart.
HFOV_DEG = 69.0
_DEFAULT_MIN_RANGE = 1.2
_DEFAULT_MAX_RANGE = 5.0


def camera_range_limits():
    """(min, max) usable range in metres, read from rgbd3d/config.py if present.

    Falls back to the values those parameters currently default to, so the
    harness still runs from a checkout without the localizer package.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.join(here, '..', '..', '3D_object_projection', 'rgbd3d',
                       'config.py')
    lo, hi = _DEFAULT_MIN_RANGE, _DEFAULT_MAX_RANGE
    try:
        for line in open(cfg):
            if line.startswith('MAX_DEPTH_M'):
                hi = float(line.split('=')[1].split('#')[0].strip())
    except (OSError, ValueError, IndexError):
        pass
    return lo, hi


# --------------------------------------------------------------------------
# Bag access
# --------------------------------------------------------------------------
def read_bag(path, topics=None):
    """Deserialise a rosbag2 into {topic: [msg, ...]}."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3'),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    out = defaultdict(list)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topics is not None and topic not in topics:
            continue
        out[topic].append(deserialize_message(data, get_message(types[topic])))
    return out


def bag_span(path):
    """(start, end) header-stamp seconds across every message, for pairing."""
    d = read_bag(path)
    ts = [stamp(m.header) for msgs in d.values() for m in msgs
          if hasattr(m, 'header')]
    return (min(ts), max(ts)) if ts else (None, None)


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------
class Series:
    """Time-indexed samples with linear interpolation.

    Values may be scalar or vector; angles must be unwrapped by the caller
    before construction, since interpolating across a +-pi wrap is meaningless.
    """

    def __init__(self, t, values):
        t = np.asarray(t, dtype=float)
        v = np.asarray(values, dtype=float)
        order = np.argsort(t)
        self.t = t[order]
        self.v = v[order]

    def __len__(self):
        return len(self.t)

    @property
    def span(self):
        return (self.t[0], self.t[-1]) if len(self.t) else (None, None)

    def at(self, t):
        if len(self.t) == 0 or t < self.t[0] or t > self.t[-1]:
            return None
        i = np.searchsorted(self.t, t)
        if i == 0:
            return self.v[0]
        t0, t1 = self.t[i - 1], self.t[i]
        f = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return self.v[i - 1] + f * (self.v[i] - self.v[i - 1])

    def deriv(self, t, win=0.25):
        """Central difference over +-win/2 — for mocap, which is pose-only."""
        a, b = self.at(t - win / 2.0), self.at(t + win / 2.0)
        return None if a is None or b is None else (b - a) / win


# --------------------------------------------------------------------------
# Ego and ground truth
# --------------------------------------------------------------------------
def ego_pose(ego_bag_data, ego_id=None):
    """Ego position Series (x, y) and unwrapped yaw Series.

    Prefers the ego's own /kinematic_state (measured against its mocap body at
    0.036 m, so it is effectively exact) and falls back to the mocap rigid body
    named by ROS_DOMAIN_ID for bags recorded without it.
    """
    if ego_id is None:
        ego_id = os.environ.get('ROS_DOMAIN_ID', '11')

    kin = ego_bag_data.get(KIN, [])
    if kin:
        t = [stamp(m.header) for m in kin]
        p = [[m.pose_with_covariance.pose.position.x,
              m.pose_with_covariance.pose.position.y] for m in kin]
        y = [quat_yaw(m.pose_with_covariance.pose.orientation) for m in kin]
        o = np.argsort(t)
        return (Series(np.asarray(t)[o], np.asarray(p)[o]),
                Series(np.asarray(t)[o], np.unwrap(np.asarray(y)[o])))

    rows = []
    for m in ego_bag_data.get(MOCAP, []):
        t = stamp(m.header)
        for rb in m.rigidbodies:
            if rb.rigid_body_name == str(ego_id):
                p = rb.pose.position
                rows.append((t, p.x, p.y, quat_yaw(rb.pose.orientation)))
    if not rows:
        return None, None
    a = np.array(sorted(rows))
    return Series(a[:, 0], a[:, 1:3]), Series(a[:, 0], np.unwrap(a[:, 3]))


def gt_objects(ego_bag_data, target_bag_data=None, ego_id=None):
    """Ground truth as {name: {'pos', 'vel', 'yaw', ...}} of Series.

    The target car's own /kinematic_state is THE ground truth when a target bag
    is given — used directly, not matched onto a mocap body first. It is the
    better source on its own merits: it carries map-frame velocity in
    twist.linear (localization.py's state is [x, y, vx, vy, ax, ay, psi] in ENU
    and _publish_kinematic writes vx/vy straight through), and heading, neither
    of which mocap provides without differentiating a pose-only stream.

    Note what it describes: the centre of the target's **base_link**, whereas
    /tracked_objects reports the centre of the detected 3D **box**. Those are
    two different points on the same car, so part of any measured error is a
    fixed lever arm rather than tracking error — see lever_arm(), which needs
    the 'yaw' series to separate them.

    Mocap is only the fallback, for ego bags recorded without a target
    counterpart. There every rigid body except the ego becomes an object, with
    bodies parked at exactly (0, 0, 0) dropped — that is mocap's "not currently
    seen" placeholder, not an object at the origin.
    """
    if ego_id is None:
        ego_id = os.environ.get('ROS_DOMAIN_ID', '11')

    if target_bag_data:
        kin = target_bag_data.get(KIN, [])
        if kin:
            t = [stamp(m.header) for m in kin]
            p = [[m.pose_with_covariance.pose.position.x,
                  m.pose_with_covariance.pose.position.y,
                  m.pose_with_covariance.pose.position.z] for m in kin]
            v = [[m.twist_with_covariance.twist.linear.x,
                  m.twist_with_covariance.twist.linear.y,
                  m.twist_with_covariance.twist.linear.z] for m in kin]
            yw = [quat_yaw(m.pose_with_covariance.pose.orientation) for m in kin]
            o = np.argsort(t)
            tt = np.asarray(t)[o]
            pos = Series(tt, np.asarray(p)[o])
            span = float(np.linalg.norm(pos.v[:, :2].max(axis=0)
                                        - pos.v[:, :2].min(axis=0)))
            return {'target': {'pos': pos,
                               'vel': Series(tt, np.asarray(v)[o]),
                               'yaw': Series(tt, np.unwrap(np.asarray(yw)[o])),
                               'moved': span,
                               'from_target_bag': True}}

    rows = defaultdict(list)
    for m in ego_bag_data.get(MOCAP, []):
        t = stamp(m.header)
        for rb in m.rigidbodies:
            if rb.rigid_body_name == str(ego_id):
                continue
            p = rb.pose.position
            if abs(p.x) < 1e-9 and abs(p.y) < 1e-9 and abs(p.z) < 1e-9:
                continue
            rows[rb.rigid_body_name].append(
                (t, p.x, p.y, p.z, quat_yaw(rb.pose.orientation)))

    out = {}
    for name, r in rows.items():
        a = np.array(sorted(r))
        out[name] = {'pos': Series(a[:, 0], a[:, 1:4]), 'vel': None,
                     'yaw': Series(a[:, 0], np.unwrap(a[:, 4])),
                     'moved': float(np.linalg.norm(a[:, 1:3].max(axis=0)
                                                   - a[:, 1:3].min(axis=0)))}
    return out


def lever_arm(pairs):
    """Fit the constant offset between two points on the same rigid body.

    `pairs` is [(error_xyz, target_yaw)] where error is (tracked - truth) in the
    map frame. /kinematic_state gives the target's base_link centre while
    /tracked_objects gives the detected box centre, so a fixed body-frame
    offset between them shows up in map coordinates as an error that rotates
    with the target's heading — which is why it must be averaged in the body
    frame, not the map frame.

    Returns (offset_body_xyz, residual_map_errors) so a caller can report the
    error with the reference-point mismatch removed and see what is left.
    """
    if not pairs:
        return None, None
    body = []
    for e, yaw in pairs:
        c, s_ = math.cos(-float(yaw)), math.sin(-float(yaw))
        body.append([c * e[0] - s_ * e[1], s_ * e[0] + c * e[1], e[2]])
    body = np.asarray(body)
    off = body.mean(axis=0)
    resid = []
    for e, yaw in pairs:
        c, s_ = math.cos(float(yaw)), math.sin(float(yaw))
        off_map = np.array([c * off[0] - s_ * off[1],
                            s_ * off[0] + c * off[1], off[2]])
        resid.append(e - off_map)
    return off, np.asarray(resid)


def gt_velocity(gt, t, win=0.25):
    """GT velocity: the target bag's twist when available, else differentiated."""
    if gt.get('vel') is not None:
        return gt['vel'].at(t)
    return gt['pos'].deriv(t, win)


# --------------------------------------------------------------------------
# Visibility
# --------------------------------------------------------------------------
def visible(gt, ego_p, ego_y, t, rng_limits=None, hfov_deg=HFOV_DEG):
    """Is this GT object inside the camera's usable envelope at time t?

    Without this, recall and false-positive counts are meaningless: the parked
    mocap cars are legitimate objects, but scoring a miss while they are behind
    the car or past the depth cap would punish the tracker for the camera's
    field of view.
    """
    if rng_limits is None:
        rng_limits = camera_range_limits()
    lo, hi = rng_limits
    g = gt['pos'].at(t)
    e = ego_p.at(t)
    y = ego_y.at(t)
    if g is None or e is None or y is None:
        return False
    dx, dy = g[0] - e[0], g[1] - e[1]
    rng = math.hypot(dx, dy)
    if not (lo <= rng <= hi):
        return False
    return abs(math.degrees(wrap(math.atan2(dy, dx) - float(y)))) <= hfov_deg / 2.0


def to_body(p_xy, ego_p, ego_y, t):
    """Map-frame point -> ego body frame (x forward, y left) at time t."""
    e, y = ego_p.at(t), ego_y.at(t)
    if e is None or y is None:
        return None
    dx, dy = p_xy[0] - e[0], p_xy[1] - e[1]
    c, s = math.cos(-float(y)), math.sin(-float(y))
    return np.array([c * dx - s * dy, s * dx + c * dy])


# --------------------------------------------------------------------------
# Message unpacking
# --------------------------------------------------------------------------
def detections_of(msg):
    """[(id, position(3), velocity(3), size(3), class)] from a Detection3DArray.

    kf_object_tracker packs the KF velocity into results[0].pose.pose.position
    because Detection3DArray has nowhere else to put it; the raw localizer puts
    the box centre there instead, so velocity is only meaningful on
    /tracked_objects.
    """
    out = []
    for d in msg.detections:
        c = d.bbox.center.position
        pos = np.array([c.x, c.y, c.z])
        if d.results:
            v = d.results[0].pose.pose.position
            vel = np.array([v.x, v.y, v.z])
            cls = d.results[0].hypothesis.class_id
        else:
            vel, cls = np.zeros(3), ''
        sz = d.bbox.size
        out.append((d.id, pos, vel, np.array([sz.x, sz.y, sz.z]), cls))
    return out


def rate_of(msgs):
    """(hz, dt array) from header stamps."""
    if len(msgs) < 2:
        return 0.0, np.array([])
    t = np.sort(np.array([stamp(m.header) for m in msgs]))
    d = np.diff(t)
    d = d[(d > 0) & (d < 30)]
    if len(d) == 0:
        return 0.0, d
    return float(len(t) / (t[-1] - t[0])), d
