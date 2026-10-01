#!/usr/bin/env python3
"""Compare the KF tracker's state against mocap ground truth.

Subscribes:
  /tracked_objects   (vision_msgs/Detection3DArray)   -- KF estimate
  /pose_modelcars    (mocap4r2_msgs/RigidBodies)      -- ground truth

Both topics are published in the map frame, so positions are compared
directly with no TF involved. The mocap publishes pose only, so ground-truth
velocity is finite-differenced across a short window (--gt-window) rather
than between adjacent samples, which would amplify mocap jitter.

Each track is matched to the nearest rigid body within --match-dist metres.
The ego car (ROS_DOMAIN_ID) is excluded, as are bodies parked exactly at the
origin -- that is mocap's placeholder for "not currently seen".

Run (no build needed):
    python3 tools/compare_kf_vs_mocap.py
    python3 tools/compare_kf_vs_mocap.py --csv /tmp/kf_vs_mocap.csv
"""

import argparse
import csv
import math
import os
from collections import defaultdict, deque

import rclpy
from rclpy.node import Node

from mocap4r2_msgs.msg import RigidBodies
from vision_msgs.msg import Detection3DArray


def _stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class KFvsMocap(Node):

    def __init__(self, args):
        super().__init__('compare_kf_vs_mocap')

        self.match_dist = args.match_dist
        self.gt_window = args.gt_window
        self.ego = args.ego

        # rigid_body_name -> deque[(t, x, y, z)], used for the GT derivative.
        self.gt_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=120))
        self.gt_latest: dict[str, tuple] = {}

        self.csv_writer = None
        self.csv_file = None
        if args.csv:
            self.csv_file = open(args.csv, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                't', 'track_id', 'class',
                'kf_x', 'kf_y', 'kf_z', 'kf_vx', 'kf_vy', 'kf_vz',
                'gt_name', 'gt_x', 'gt_y', 'gt_z', 'gt_vx', 'gt_vy', 'gt_vz',
                'err_pos', 'err_vel', 'kf_speed', 'gt_speed',
            ])

        self.create_subscription(
            RigidBodies, '/pose_modelcars', self._on_mocap, 10)
        self.create_subscription(
            Detection3DArray, '/tracked_objects', self._on_tracks, 10)

        self.get_logger().info(
            f'Comparing /tracked_objects vs /pose_modelcars  '
            f'(ego={self.ego}, match_dist={self.match_dist} m)')

    # ------------------------------------------------------------------
    def _on_mocap(self, msg: RigidBodies):
        t = _stamp_s(msg.header)
        for rb in msg.rigidbodies:
            name = rb.rigid_body_name
            if name == self.ego:
                continue
            p = rb.pose.position
            # Exactly-zero pose is the mocap's "body not seen" placeholder,
            # not a car sitting on the origin.
            if abs(p.x) < 1e-9 and abs(p.y) < 1e-9 and abs(p.z) < 1e-9:
                continue
            self.gt_latest[name] = (p.x, p.y, p.z)
            self.gt_hist[name].append((t, p.x, p.y, p.z))

    # ------------------------------------------------------------------
    def _gt_velocity(self, name: str) -> tuple:
        """Finite-difference the mocap pose over ~gt_window seconds."""
        hist = self.gt_hist.get(name)
        if not hist or len(hist) < 2:
            return (0.0, 0.0, 0.0)
        t_new, x_new, y_new, z_new = hist[-1]
        # Walk back to the oldest sample still inside the window.
        old = hist[0]
        for sample in reversed(hist):
            if t_new - sample[0] >= self.gt_window:
                old = sample
                break
        dt = t_new - old[0]
        if dt <= 1e-6:
            return (0.0, 0.0, 0.0)
        return ((x_new - old[1]) / dt,
                (y_new - old[2]) / dt,
                (z_new - old[3]) / dt)

    # ------------------------------------------------------------------
    def _on_tracks(self, msg: Detection3DArray):
        if not msg.detections:
            return
        t = _stamp_s(msg.header)
        rows = []

        for det in msg.detections:
            c = det.bbox.center.position
            kf = (c.x, c.y, c.z)

            # The tracker packs the KF velocity into results[0].pose.pose.position
            # (see kf_object_tracker.py's "Embed class_id and KF velocity" block).
            if det.results:
                v = det.results[0].pose.pose.position
                kf_v = (v.x, v.y, v.z)
                cls = det.results[0].hypothesis.class_id
            else:
                kf_v, cls = (0.0, 0.0, 0.0), '?'

            # Nearest ground-truth body within the gate.
            best_name, best_d = None, float('inf')
            for name, g in self.gt_latest.items():
                d = math.dist(kf, g)
                if d < best_d:
                    best_name, best_d = name, d

            if best_name is None:
                rows.append((det.id, cls, kf, kf_v, None, None, None, None))
                continue

            gt = self.gt_latest[best_name]
            gt_v = self._gt_velocity(best_name)
            matched = best_d <= self.match_dist
            rows.append((det.id, cls, kf, kf_v,
                         best_name if matched else f'{best_name}?',
                         gt, gt_v, best_d))

        self._report(t, rows)

    # ------------------------------------------------------------------
    def _report(self, t, rows):
        lines = []
        for tid, cls, kf, kf_v, gt_name, gt, gt_v, dist in rows:
            kf_sp = math.dist((0, 0, 0), kf_v)
            if gt is None:
                lines.append(
                    f'  id={tid:<3} {cls:<7} '
                    f'KF pos=({kf[0]:7.3f},{kf[1]:7.3f},{kf[2]:6.3f}) '
                    f'vel=({kf_v[0]:6.3f},{kf_v[1]:6.3f},{kf_v[2]:6.3f}) '
                    f'|v|={kf_sp:5.3f}   GT=none')
                continue

            gt_sp = math.dist((0, 0, 0), gt_v)
            err_v = math.dist(kf_v, gt_v)
            lines.append(
                f'  id={tid:<3} {cls:<7} '
                f'KF pos=({kf[0]:7.3f},{kf[1]:7.3f},{kf[2]:6.3f}) '
                f'vel=({kf_v[0]:6.3f},{kf_v[1]:6.3f},{kf_v[2]:6.3f}) |v|={kf_sp:5.3f}\n'
                f'{"":14}GT[{gt_name}] pos=({gt[0]:7.3f},{gt[1]:7.3f},{gt[2]:6.3f}) '
                f'vel=({gt_v[0]:6.3f},{gt_v[1]:6.3f},{gt_v[2]:6.3f}) |v|={gt_sp:5.3f}\n'
                f'{"":14}ERR pos={dist:6.3f} m   vel={err_v:6.3f} m/s')

            if self.csv_writer:
                self.csv_writer.writerow([
                    f'{t:.3f}', tid, cls,
                    f'{kf[0]:.4f}', f'{kf[1]:.4f}', f'{kf[2]:.4f}',
                    f'{kf_v[0]:.4f}', f'{kf_v[1]:.4f}', f'{kf_v[2]:.4f}',
                    gt_name,
                    f'{gt[0]:.4f}', f'{gt[1]:.4f}', f'{gt[2]:.4f}',
                    f'{gt_v[0]:.4f}', f'{gt_v[1]:.4f}', f'{gt_v[2]:.4f}',
                    f'{dist:.4f}', f'{err_v:.4f}', f'{kf_sp:.4f}', f'{gt_sp:.4f}',
                ])
                self.csv_file.flush()

        self.get_logger().info(f't={t:.2f}  tracks={len(rows)}\n' + '\n'.join(lines))

    # ------------------------------------------------------------------
    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        super().destroy_node()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', default=None, help='append a CSV log here')
    ap.add_argument('--match-dist', type=float, default=1.5,
                    help='max distance (m) to call a rigid body a match')
    ap.add_argument('--gt-window', type=float, default=0.25,
                    help='seconds to finite-difference mocap pose over')
    ap.add_argument('--ego', default=os.environ.get('ROS_DOMAIN_ID', '11'),
                    help='rigid body name of the ego car, excluded from matching')
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = KFvsMocap(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
