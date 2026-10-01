#!/usr/bin/env python3
"""
Kalman Filter Object Tracker — tracking pipeline only.

Subscribes: /object_localization/detections3d          (vision_msgs/Detection3DArray)

Publishes:  /tracked_objects                           (vision_msgs/Detection3DArray)
            /tracked_objects/predicted_path_array      (vision_msgs/Detection3DArray)
            /tracked_objects/predicted_pose_array      (geometry_msgs/PoseArray)
            /tracked_objects/predicted_snapshots       (vision_msgs/Detection3DArray)
            /tracked_objects/nav_path/track_<id>       (nav_msgs/Path)

State vector per track: [x, y, z, vx, vy, vz] — constant velocity.

What the filter is and is not good for, measured over four scripted scenario
runs (427 frames) plus nine freeform ones:

  * It does NOT improve position. Raw detections score 0.217 m rms, the filtered
    output 0.237 m — a gain of 0.918, below 1.0 in every configuration tried
    (R shape, R magnitude, motion model, bias augmentation). The localizer's
    residual has a lag-1 autocorrelation of 0.86-0.95, which violates the white-
    noise assumption a Kalman filter is built on; smooth drift is not separable
    from real motion, so there is nothing for the filter to average away.
  * It IS what provides association, stable ids across detection gaps, and the
    velocity that pure_pursuit extrapolates 1 s ahead to size its avoidance
    corridor. Those are the things worth protecting when tuning.
"""

import math
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import Path
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose


# ---------------------------------------------------------------------------
# Kalman Filter — Constant Velocity model
# ---------------------------------------------------------------------------
class KalmanFilter:
    """Constant-velocity filter stepped by the *measured* interval.

    dt is passed to predict() per step rather than baked into F at
    construction: the detection rate is neither fixed nor equal to the old
    0.1 s default (measured 0.6-2 Hz with multi-second gaps), and a filter
    told 0.1 s while 1 s of motion actually elapsed attributes the whole
    displacement to a tenth of the time — inflating every velocity estimate
    by dt_real/dt_assumed, which then drives unmatched tracks off into empty
    space at that same exaggerated speed.
    """

    def __init__(self, x, y, z, q_accel, r_diag):
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        # Process noise is a spectral density (m²/s³), not a fixed covariance,
        # so Q can be rebuilt for whatever dt each step actually spans.
        self.q = float(q_accel)
        self.R = np.diag([float(v) for v in r_diag])

        self.x = np.array([x, y, z, 0.0, 0.0, 0.0])
        # Position is known to measurement accuracy at birth; velocity is not
        # known at all, so start it loose and let the first updates shrink it.
        self.P = np.diag([self.R[0, 0], self.R[1, 1], self.R[2, 2],
                          10.0, 10.0, 10.0])

    def _Q(self, dt):
        """Discrete white-noise-acceleration Q for state [pos(3), vel(3)]."""
        I = np.eye(3)
        q11 = (dt ** 3) / 3.0 * self.q
        q12 = (dt ** 2) / 2.0 * self.q
        q22 = dt * self.q
        Q = np.zeros((6, 6))
        Q[:3, :3] = q11 * I
        Q[:3, 3:] = q12 * I
        Q[3:, :3] = q12 * I
        Q[3:, 3:] = q22 * I
        return Q

    def predict(self, dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    @property
    def position(self):
        return self.x[:3]

    @property
    def velocity(self):
        return self.x[3:]


# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------
class Track:
    _next_id = 1

    def __init__(self, detection: Detection3D, cfg, reclaim_id=None):
        pos = detection.bbox.center.position
        self.cfg = cfg
        self.kf = KalmanFilter(pos.x, pos.y, pos.z,
                               cfg['q_accel'], cfg['r_diag'])
        # reclaim_id lets an object that vanished and came back keep the
        # identity it had before, instead of being issued a fresh one — see
        # KFObjectTrackerNode._reclaim_id.
        if reclaim_id is not None:
            self.track_id = reclaim_id
        else:
            self.track_id = Track._next_id
            Track._next_id += 1
        self.hits      = 0
        # Seconds of coasting since the last matched detection. Frame counts
        # are meaningless when the input rate swings between 0.6 Hz and 10 Hz:
        # 10 missed frames is 1 s at design rate but 16 s at the rate actually
        # measured, which is long enough for a track to drift metres away on a
        # stale velocity and then be unreachable by its own association gate.
        self.coast_time = 0.0
        # Running mean of the box extent. The raw per-frame size jitters ~15%
        # of the car's length (measured 1.040 +- 0.152 m), and pure_pursuit
        # sizes its clearance corridor from this, so a jittering box means a
        # jittering safety margin.
        self._size = np.array([detection.bbox.size.x,
                               detection.bbox.size.y,
                               detection.bbox.size.z], dtype=float)
        self.bbox_size = detection.bbox.size
        self.bbox_orientation = detection.bbox.center.orientation
        self.class_id  = (detection.results[0].hypothesis.class_id
                          if detection.results else 'unknown')

    @property
    def size(self):
        return self._size

    def predict(self, dt: float):
        self.kf.predict(dt)

    def update(self, detection: Detection3D):
        pos = detection.bbox.center.position

        # Scale R by detection confidence when asked. The localizer's fg_prob
        # spans 0.48-0.97 across a run, so a weak detection genuinely deserves
        # less weight than a strong one; the tracker previously used one fixed
        # R for both and then overwrote the score with 1.0.
        conf = (detection.results[0].hypothesis.score
                if detection.results else 1.0)
        scale = 1.0
        if self.cfg['score_weight_r'] and 0.0 < conf <= 1.0:
            scale = min(1.0 / max(conf, 0.1), 5.0)
        R_saved = self.kf.R
        if scale != 1.0:
            self.kf.R = R_saved * scale

        self.kf.update(np.array([pos.x, pos.y, pos.z]))
        self.kf.R = R_saved

        self.coast_time = 0.0
        self.hits      += 1

        a = self.cfg['size_alpha']
        s = np.array([detection.bbox.size.x, detection.bbox.size.y,
                      detection.bbox.size.z], dtype=float)
        self._size = (1.0 - a) * self._size + a * s if a < 1.0 else s
        self.bbox_size = detection.bbox.size
        self.bbox_orientation = detection.bbox.center.orientation
        if detection.results:
            self.class_id = detection.results[0].hypothesis.class_id


# ---------------------------------------------------------------------------
# Data association — greedy nearest-neighbour
# ---------------------------------------------------------------------------
def associate(tracks: list[Track], detections: list[Detection3D], max_dist: float):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    tp = np.array([t.kf.position for t in tracks])
    dp = np.array([[d.bbox.center.position.x,
                    d.bbox.center.position.y,
                    d.bbox.center.position.z] for d in detections])

    cost = np.linalg.norm(tp[:, np.newaxis, :] - dp[np.newaxis, :, :], axis=2)

    matched, used_t, used_d = [], set(), set()
    for idx in np.argsort(cost.ravel()):
        ti, di = divmod(int(idx), len(detections))
        if ti in used_t or di in used_d:
            continue
        if cost[ti, di] > max_dist:
            break
        matched.append((ti, di))
        used_t.add(ti)
        used_d.add(di)

    unmatched_t = [i for i in range(len(tracks))     if i not in used_t]
    unmatched_d = [i for i in range(len(detections)) if i not in used_d]
    return matched, unmatched_t, unmatched_d


# ---------------------------------------------------------------------------
# ROS 2 node
# ---------------------------------------------------------------------------
class KFObjectTrackerNode(Node):

    def __init__(self):
        super().__init__('kf_object_tracker')

        # Fallback dt only — used for the very first detection of a track and
        # whenever the measured interval is unusable. Every normal step uses
        # the real elapsed time between message stamps.
        self.declare_parameter('dt',            0.1)
        self.declare_parameter('dt_max',        1.0)  # clamp: cap a huge gap rather than let one predict step fly
        self.declare_parameter('max_dist',      2.0)
        # Seconds a track may coast unmatched before deletion. Replaces the
        # old frame-count limit, which meant wildly different wall-clock
        # lifetimes as the input rate varied.
        self.declare_parameter('max_coast_time', 1.0)
        # Seconds a retired track's identity is remembered so a reappearing
        # object can reclaim it rather than being issued a new id.
        #
        # Deliberately separate from max_coast_time, which governs how long a
        # track keeps *publishing* an extrapolated position. Merging the two
        # forces a bad trade: measured on ego_7020446, stretching the coast
        # window to 3 s did cut id churn but degraded position 78% (0.294 ->
        # 0.522 m rms) because the filter free-ran three seconds on a stale
        # velocity. Retention costs nothing in position because a retired
        # track publishes nothing at all — it is only a remembered id.
        #
        # 3.0 s covers what the detector actually does: across 22 detection
        # gaps in two runs (longest 4.34 s) the object moved 0.01-0.77 m, so
        # its position when it reappears is still well inside reid_max_dist.
        # Swept over all 7 recorded pairs: 0 s -> 25 ids / 19 switches,
        # 3 s -> 14 / 9, 5 s -> 12 / 8, 10 s -> 12 / 9 (worse again, stale ids
        # start being reclaimed by the wrong object). 3 s sits on the plateau.
        self.declare_parameter('max_id_retention', 3.0)
        # How far a new detection may be from a retired track's last known
        # position and still be considered the same object. 1.5 m gives ~2x
        # margin on the largest observed gap displacement (0.77 m) while
        # staying below max_dist, so a retired identity can never be claimed
        # from further away than live association would allow.
        #
        # A 3.0 m gate scored better in the sweep (12 ids / 7 switches vs
        # 14 / 9) and is NOT used: every recorded bag contains a single moving
        # object, so nothing in the data can penalise handing an identity to
        # the wrong one — the metric is blind to precisely the failure this
        # gate exists to prevent. Revisit only with genuinely multi-object
        # ground truth.
        self.declare_parameter('reid_max_dist', 1.5)
        self.declare_parameter('min_hits',      1)    # publish on first confirmed detection
        self.declare_parameter('predict_steps', 20)
        self.declare_parameter('predict_dt',    0.1)
        # Measurement noise variance per axis (m²), measured as the scatter of
        # /object_localization/detections3d about ground truth (bias excluded —
        # a fixed offset is not noise). Measured [0.029, 0.012, 0.0008] and
        # [0.018, 0.031, 0.0014] on two runs.
        #
        # Do not expect much from retuning this. R was swept over shape and
        # magnitude across the four scenario runs and nothing beat the value
        # below on position: x1 to x32 inflation, and ray-aligned variants using
        # the genuinely measured anisotropy (sigma 0.137 m along the camera ray
        # vs 0.060 m across it), all scored 0.237-0.256 m rms against this
        # value's 0.237. Inflating R does halve the false-motion rate, 10% -> 5%,
        # but only by trading phantom velocity for lag. The ray-aligned variants
        # were dropped outright: an isotropic R of equal magnitude matched them,
        # so the effect was magnitude, never shape.
        self.declare_parameter('r_diag', [0.03, 0.03, 0.0015])
        # Process noise spectral density (m²/s³): how much unmodelled
        # acceleration the constant-velocity model must absorb. Swept in
        # simulation against the measured R at 2 Hz — position error vs raw
        # measurements and velocity error both improve monotonically as this
        # drops (q=1.0 -> 0.80x / 0.53 m/s, q=0.5 -> 0.76x / 0.42 m/s,
        # q=0.05 -> 0.63x / 0.19 m/s). Held at 0.5 rather than the sweep's
        # floor because that sweep's manoeuvre case is an instantaneous speed
        # step; a real sustained manoeuvre punishes an over-confident filter
        # harder than the test shows. Lower it if tracks look jittery, raise
        # it if they lag behind accelerating targets.
        self.declare_parameter('q_accel', 0.5)
        # EMA weight for the box extent; 1.0 keeps the old take-latest.
        self.declare_parameter('size_alpha', 0.3)
        # Inflate R on low-confidence detections.
        self.declare_parameter('score_weight_r', True)
        self.declare_parameter('target_classes', ['person', 'car'])
        self.declare_parameter('expected_frame', 'map')

        self.dt            = self.get_parameter('dt').value
        self.dt_max        = self.get_parameter('dt_max').value
        self.max_dist      = self.get_parameter('max_dist').value
        self.max_coast_time = self.get_parameter('max_coast_time').value
        self.max_id_retention = float(self.get_parameter('max_id_retention').value)
        self.reid_max_dist = float(self.get_parameter('reid_max_dist').value)
        # [(track_id, last_position, class_id, age_seconds)] for tracks that
        # died recently and may still come back.
        self._retired = []
        self.min_hits      = self.get_parameter('min_hits').value
        self.r_diag        = list(self.get_parameter('r_diag').value)
        self.q_accel       = float(self.get_parameter('q_accel').value)
        self._last_stamp   = None
        self.track_cfg = {
            'q_accel':        self.q_accel,
            'r_diag':         self.r_diag,
            'size_alpha':     float(self.get_parameter('size_alpha').value),
            'score_weight_r': bool(self.get_parameter('score_weight_r').value),
        }
        self.predict_steps = self.get_parameter('predict_steps').value
        self.predict_dt    = self.get_parameter('predict_dt').value
        self.target_classes = list(self.get_parameter('target_classes').value)
        self.expected_frame = str(self.get_parameter('expected_frame').value)

        self.tracks: list[Track] = []
        self.nav_path_pubs: dict[int, Any] = {}

        self.create_subscription(
            Detection3DArray, '/object_localization/detections3d', self._callback, 10)

        self.pub = self.create_publisher(
            Detection3DArray, '/tracked_objects', 10)
        self.pub_path_array = self.create_publisher(
            Detection3DArray, '/tracked_objects/predicted_path_array', 10)
        self.pub_pose_array = self.create_publisher(
            PoseArray, '/tracked_objects/predicted_pose_array', 10)
        self.pub_snapshots = self.create_publisher(
            Detection3DArray, '/tracked_objects/predicted_snapshots', 10)

        self.get_logger().info(
            f'KF Object Tracker started. Tracking classes: {self.target_classes}')

    # ------------------------------------------------------------------
    def _callback(self, msg: Detection3DArray):
        # The upstream 3D localizer degrades to publishing detections in the
        # camera-optical frame (instead of map) when its TF lookup to map
        # fails, so this frame can silently change at runtime. Tracking
        # camera-relative positions as if they were map-frame would corrupt
        # every track's Kalman-filtered position/velocity — and that velocity
        # feeds directly into pure_pursuit's obstacle-avoidance geometry — so
        # refuse the whole message rather than risk it. Existing tracks are
        # left frozen (not predicted/pruned) until valid frames resume; the
        # gap in /tracked_objects surfaces downstream via pure_pursuit's own
        # detection timeout.
        if msg.header.frame_id and msg.header.frame_id != self.expected_frame:
            self.get_logger().warn(
                f"Dropping detections in frame '{msg.header.frame_id}' "
                f"(expected '{self.expected_frame}') — localizer TF to "
                f"{self.expected_frame} is likely down",
                throttle_duration_sec=5.0,
            )
            return

        # Real elapsed time between detection messages. Falls back to the `dt`
        # parameter on the first message, and is clamped to dt_max so a single
        # multi-second dropout can't fling every track across the map in one
        # predict step (gaps of 3-6 s have been observed on this pipeline).
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_stamp is None or not (0.0 < stamp - self._last_stamp):
            dt = self.dt
        else:
            dt = min(stamp - self._last_stamp, self.dt_max)
        self._last_stamp = stamp

        dets = [d for d in msg.detections
                if d.results and d.results[0].hypothesis.class_id in self.target_classes]

        for t in self.tracks:
            t.predict(dt)

        matched, unmatched_t, unmatched_d = associate(self.tracks, dets, self.max_dist)

        for ti, di in matched:
            self.tracks[ti].update(dets[di])
        for ti in unmatched_t:
            self.tracks[ti].coast_time += dt
        # Age the retirement list before it is consulted, so an id that has
        # been gone longer than max_id_retention cannot be reclaimed.
        if self.max_id_retention > 0.0:
            self._retired = [(tid, p, c, age + dt)
                             for tid, p, c, age in self._retired
                             if age + dt <= self.max_id_retention]

        for di in unmatched_d:
            self.tracks.append(
                Track(dets[di], self.track_cfg,
                      reclaim_id=self._reclaim_id(dets[di])))

        # Prune dead tracks and their nav publishers. Age is in seconds, so a
        # track's lifetime no longer depends on how fast detections happen to
        # be arriving.
        active = {t.track_id for t in self.tracks
                  if t.coast_time <= self.max_coast_time}
        for tid in list(self.nav_path_pubs):
            if tid not in active:
                self.destroy_publisher(self.nav_path_pubs.pop(tid))

        # Remember the identity of anything about to be dropped. The track
        # itself stops here — nothing further is published for it — but its id
        # stays claimable for max_id_retention seconds.
        if self.max_id_retention > 0.0:
            for t in self.tracks:
                if t.coast_time > self.max_coast_time and t.hits >= self.min_hits:
                    self._retired.append((t.track_id,
                                          np.array(t.kf.position, dtype=float),
                                          t.class_id, 0.0))
        self.tracks = [t for t in self.tracks
                       if t.coast_time <= self.max_coast_time]

        # Build confirmed-track output
        out = Detection3DArray()
        out.header = msg.header
        for track in self.tracks:
            if track.hits < self.min_hits:
                continue
            sz = track.size          # EMA-filtered extent, not raw last frame
            d = Detection3D()
            d.header = msg.header
            d.id     = str(track.track_id)
            px, py, pz = track.kf.position
            d.bbox.center.position.x    = float(px)
            d.bbox.center.position.y    = float(py)
            d.bbox.center.position.z    = float(pz)
            d.bbox.center.orientation   = track.bbox_orientation
            d.bbox.size.x = float(sz[0]) if sz[0] > 0.0 else 1.0
            d.bbox.size.y = float(sz[1]) if sz[1] > 0.0 else 0.6
            d.bbox.size.z = float(sz[2]) if sz[2] > 0.0 else 0.6

            # Embed class_id and KF velocity — required by pure_pursuit DWA/TTC logic.
            # results[0].hypothesis.class_id  = object class (e.g. "person")
            # results[0].pose.pose.position   = (vx, vy, vz) map-frame velocity from KF
            vx, vy, vz = track.kf.velocity
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id     = track.class_id
            hyp.hypothesis.score        = 1.0
            hyp.pose.pose.position.x    = float(vx)
            hyp.pose.pose.position.y    = float(vy)
            hyp.pose.pose.position.z    = float(vz)
            hyp.pose.pose.orientation.w = 1.0
            d.results.append(hyp)

            out.detections.append(d)

        self.pub.publish(out)
        self._publish_path_array(out.header, out.detections)
        self._publish_pose_array(out.header, out.detections)
        self._publish_snapshots(out.header, out.detections)
        self._publish_nav_paths(out.header, out.detections)

    # ------------------------------------------------------------------
    def _reclaim_id(self, detection: Detection3D):
        """Identity of a recently-retired track this detection likely continues.

        Detection gaps are the dominant cost in this pipeline: on two recorded
        runs, misses accounted for far more lost MOTA than false positives or
        association errors, because a gap longer than max_coast_time kills the
        track and the object comes back as a brand-new id. Measured across 22
        such gaps (longest 4.34 s) the object had moved only 0.01-0.77 m, so
        its last known position identifies it reliably on return.

        Matching is on last-known position, not an extrapolation: the velocity
        that survives a multi-second gap is exactly the one least worth
        trusting, and the measurements above show the object barely moves
        anyway. Class must agree, so a person cannot inherit a car's id.

        Returns None when nothing plausible is retired, and the caller then
        allocates a fresh id as before.
        """
        if not self._retired or self.max_id_retention <= 0.0:
            return None
        pos = detection.bbox.center.position
        p = np.array([pos.x, pos.y, pos.z], dtype=float)
        cls = (detection.results[0].hypothesis.class_id
               if detection.results else 'unknown')
        best, best_d = None, self.reid_max_dist
        for i, (tid, last_p, last_c, _age) in enumerate(self._retired):
            if last_c != cls:
                continue
            d = float(np.linalg.norm(p - last_p))
            if d < best_d:
                best, best_d = i, d
        if best is None:
            return None
        tid = self._retired.pop(best)[0]
        self.get_logger().info(
            f"track {tid} reacquired {best_d:.2f} m from where it was lost",
            throttle_duration_sec=2.0)
        return tid

    # ------------------------------------------------------------------
    def _waypoints(self, track: Track) -> list[tuple[float, float, float]]:
        px, py, pz = track.kf.position
        vx, vy, vz = track.kf.velocity
        dt = self.predict_dt
        return [(px + vx * dt * i, py + vy * dt * i, pz + vz * dt * i)
                for i in range(self.predict_steps + 1)]

    # ------------------------------------------------------------------
    def _publish_path_array(self, header, confirmed):
        """Single flat Detection3DArray: all tracks × all future steps."""
        now_sec = header.stamp.sec + header.stamp.nanosec * 1e-9
        out = Detection3DArray()
        out.header = header

        for det in confirmed:
            track = next((t for t in self.tracks if t.track_id == int(det.id)), None)
            if track is None:
                continue
            sz = track.size
            sx = float(sz[0]) if sz[0] > 0.0 else 1.0
            sy = float(sz[1]) if sz[1] > 0.0 else 0.6
            sz_ = float(sz[2]) if sz[2] > 0.0 else 0.6

            for step, (x, y, z) in enumerate(self._waypoints(track)):
                future = now_sec + step * self.predict_dt
                d = Detection3D()
                d.id                        = det.id
                d.header.frame_id           = header.frame_id
                d.header.stamp.sec          = int(future)
                d.header.stamp.nanosec      = int((future % 1.0) * 1e9)
                d.bbox.center.position.x    = float(x)
                d.bbox.center.position.y    = float(y)
                d.bbox.center.position.z    = float(z)
                d.bbox.center.orientation   = track.bbox_orientation
                d.bbox.size.x = sx
                d.bbox.size.y = sy
                d.bbox.size.z = sz_
                out.detections.append(d)

        self.pub_path_array.publish(out)

    # ------------------------------------------------------------------
    def _publish_pose_array(self, header, confirmed):
        msg = PoseArray()
        msg.header = header
        for det in confirmed:
            track = next((t for t in self.tracks if t.track_id == int(det.id)), None)
            if track is None:
                continue
            yaw = math.atan2(float(track.kf.velocity[1]), float(track.kf.velocity[0]))
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            for x, y, z in self._waypoints(track):
                p = Pose()
                p.position.x    = float(x)
                p.position.y    = float(y)
                p.position.z    = float(z)
                p.orientation.z = qz
                p.orientation.w = qw
                msg.poses.append(p)
        self.pub_pose_array.publish(msg)

    # ------------------------------------------------------------------
    def _publish_snapshots(self, header, confirmed):
        """One Detection3DArray per future timestep (all tracks at that step)."""
        now_sec = header.stamp.sec + header.stamp.nanosec * 1e-9
        track_wps = []
        for det in confirmed:
            track = next((t for t in self.tracks if t.track_id == int(det.id)), None)
            if track:
                track_wps.append((det.id, self._waypoints(track)))

        for step in range(self.predict_steps + 1):
            future = now_sec + step * self.predict_dt
            snap = Detection3DArray()
            snap.header.frame_id      = header.frame_id
            snap.header.stamp.sec     = int(future)
            snap.header.stamp.nanosec = int((future % 1.0) * 1e9)
            for tid, wps in track_wps:
                x, y, z = wps[step]
                d = Detection3D()
                d.header = snap.header
                d.id     = tid
                d.bbox.center.position.x = float(x)
                d.bbox.center.position.y = float(y)
                d.bbox.center.position.z = float(z)
                snap.detections.append(d)
            self.pub_snapshots.publish(snap)

    # ------------------------------------------------------------------
    def _publish_nav_paths(self, header, confirmed):
        now_sec = header.stamp.sec + header.stamp.nanosec * 1e-9
        for det in confirmed:
            track_id = int(det.id)
            track = next((t for t in self.tracks if t.track_id == track_id), None)
            if track is None:
                continue
            if track_id not in self.nav_path_pubs:
                self.nav_path_pubs[track_id] = self.create_publisher(
                    Path, f'/tracked_objects/nav_path/track_{track_id}', 10)

            yaw = math.atan2(float(track.kf.velocity[1]), float(track.kf.velocity[0]))
            qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)

            path = Path()
            path.header = header
            for i, (x, y, z) in enumerate(self._waypoints(track)):
                future = now_sec + i * self.predict_dt
                ps = PoseStamped()
                ps.header.frame_id      = header.frame_id
                ps.header.stamp.sec     = int(future)
                ps.header.stamp.nanosec = int((future % 1.0) * 1e9)
                ps.pose.position.x      = float(x)
                ps.pose.position.y      = float(y)
                ps.pose.position.z      = float(z)
                ps.pose.orientation.z   = qz
                ps.pose.orientation.w   = qw
                path.poses.append(ps)
            self.nav_path_pubs[track_id].publish(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = KFObjectTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
