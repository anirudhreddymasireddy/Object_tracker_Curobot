#!/usr/bin/env python3
"""Score /tracked_objects against ground truth while a run is happening.

Ground truth is the target car's own /kinematic_state, bridged onto the ego's
domain as /target/kinematic_state (see config/target_gt_bridge.yaml — the two
cars are on different ROS domains, so nothing on the ego can see it directly).

Metrics deliberately mirror tools/evaluate_tracker.py, so a live reading and an
offline bag score of the same run are directly comparable. That includes the
reference-point split: /kinematic_state reports the centre of the target's
base_link while /tracked_objects reports the centre of the detected 3D box, so
part of the raw error is a fixed lever arm rather than tracking error. Both are
printed.

Run (on the ego's domain, after starting the bridge):
    python3 tools/live_eval.py
    python3 tools/live_eval.py --ros-args -p gt_topic:=/target/kinematic_state
"""

import collections
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from curobot_msgs.msg import KinematicState
from vision_msgs.msg import Detection3DArray


def quat_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LiveEval(Node):

    def __init__(self):
        super().__init__('live_eval')
        p = self.declare_parameter
        gt_topic = p('gt_topic', '/target/kinematic_state').value
        trk_topic = p('tracked_topic', '/tracked_objects').value
        self.report_period = float(p('report_period', 2.0).value)
        # Ignore a track further than this from truth: it is a different object,
        # not a bad estimate of this one.
        self.match_dist = float(p('match_dist', 3.0).value)
        # Direction error is meaningless when the target is barely moving, so
        # only accumulate it above this speed.
        self.min_speed = float(p('min_speed_for_dir', 0.25).value)
        self.window = int(p('window', 400).value)

        # localization.py publishes BEST_EFFORT; the tracker publishes RELIABLE.
        # A RELIABLE subscriber cannot match a BEST_EFFORT publisher, so the two
        # subscriptions must not share one profile.
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE, depth=10)
        rel = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE, depth=10)

        self._gt = None            # (stamp, pos(3), vel(3), yaw)
        self._n_gt = 0
        self.pos = collections.deque(maxlen=self.window)   # error vectors
        self.lev = collections.deque(maxlen=self.window)   # (err, target yaw)
        self.vel = collections.deque(maxlen=self.window)
        self.dirs = collections.deque(maxlen=self.window)
        self.ids = collections.deque(maxlen=self.window)
        self.ntrk = collections.deque(maxlen=self.window)
        self._unmatched = 0
        self._no_gt = 0
        self._last_id = None
        self._idsw = 0

        self.create_subscription(KinematicState, gt_topic, self._on_gt, be)
        self.create_subscription(Detection3DArray, trk_topic, self._on_trk, rel)
        self.create_timer(self.report_period, self._report)
        self.get_logger().info(
            f'live_eval: scoring {trk_topic} against {gt_topic}')

    # ------------------------------------------------------------------
    def _on_gt(self, msg):
        p = msg.pose_with_covariance.pose.position
        v = msg.twist_with_covariance.twist.linear
        self._gt = (np.array([p.x, p.y, p.z]), np.array([v.x, v.y, v.z]),
                    quat_yaw(msg.pose_with_covariance.pose.orientation))
        self._n_gt += 1

    def _on_trk(self, msg):
        if self._gt is None:
            self._no_gt += 1
            return
        g, gv, gyaw = self._gt
        if not msg.detections:
            return
        best, best_d = None, float('inf')
        for d in msg.detections:
            c = d.bbox.center.position
            pos = np.array([c.x, c.y, c.z])
            dist = float(np.linalg.norm(pos - g))
            if dist < best_d:
                best, best_d = d, dist
        if best_d > self.match_dist:
            self._unmatched += 1
            return

        c = best.bbox.center.position
        err = np.array([c.x, c.y, c.z]) - g
        self.pos.append(err)
        self.lev.append((err, gyaw))
        self.ntrk.append(len(msg.detections))
        self.ids.append(best.id)
        if self._last_id is not None and best.id != self._last_id:
            self._idsw += 1
        self._last_id = best.id

        if best.results:
            v = best.results[0].pose.pose.position
            ev = np.array([v.x, v.y, v.z])
            self.vel.append(float(np.linalg.norm(ev - gv)))
            ng, nv = np.linalg.norm(gv[:2]), np.linalg.norm(ev[:2])
            if ng > self.min_speed and nv > 1e-6:
                self.dirs.append(math.degrees(math.acos(np.clip(
                    float(np.dot(ev[:2], gv[:2])) / (nv * ng), -1.0, 1.0))))

    # ------------------------------------------------------------------
    def _report(self):
        if self._gt is None:
            self.get_logger().warn(
                'no ground truth yet — is the domain bridge running? '
                '(ros2 run domain_bridge domain_bridge '
                'src/object_tracker/config/target_gt_bridge.yaml)',
                throttle_duration_sec=5.0)
            return
        if not self.pos:
            self.get_logger().warn(
                f'ground truth OK ({self._n_gt} msgs) but no matched track '
                f'({self._unmatched} beyond {self.match_dist} m)',
                throttle_duration_sec=5.0)
            return

        P = np.array(self.pos)
        d = np.linalg.norm(P, axis=1)
        # Fixed body-frame offset between base_link centre and box centre.
        body = []
        for e, yaw in self.lev:
            c, s = math.cos(-yaw), math.sin(-yaw)
            body.append([c * e[0] - s * e[1], s * e[0] + c * e[1], e[2]])
        off = np.array(body).mean(axis=0)
        resid = np.array([b - off for b in body])
        rd = np.linalg.norm(resid, axis=1)

        v = f'{math.sqrt(float(np.mean(np.square(self.vel)))):.3f}' if self.vel else '  -  '
        dm = f'{np.median(self.dirs):5.1f}' if self.dirs else '  -  '
        self.get_logger().info(
            f'n={len(d):4d}  pos mean={d.mean():.3f} rms='
            f'{math.sqrt(float((d ** 2).mean())):.3f} max={d.max():.3f} m  |  '
            f'lever fwd={off[0]:+.2f} lat={off[1]:+.2f} -> tracking err '
            f'rms={math.sqrt(float((rd ** 2).mean())):.3f} m  |  '
            f'vel rms={v} m/s  dir={dm} deg  |  '
            f'ids={len(set(self.ids))} switches={self._idsw}  '
            f'tracks/frame={np.mean(self.ntrk):.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = LiveEval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
