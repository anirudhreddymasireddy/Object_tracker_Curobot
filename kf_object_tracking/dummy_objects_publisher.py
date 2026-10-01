#!/usr/bin/env python3
"""
Dummy Objects Publisher
Publishes: /object_localization/detections3d  (vision_msgs/Detection3DArray)

Simulates moving objects whose speed varies sinusoidally so the KF tracker's
velocity estimates and predicted-path lengths change visibly over time.

Speed model per object:
    v(t)  =  v_base * (1 + amp * sin(freq * t))
    s(t)  =  v_base * (t  +  amp/freq  -  amp/freq * cos(freq * t))   [integral, s(0)=0]

With amp < 1 the speed stays positive; with amp close to 1 the object nearly
stops, then surges — clearly visible in RViz2 as the predicted-path shrinks
and grows and the speed label changes.

Scenarios
---------
  0  all four objects (default)
  1  linear   — X-axis, surges every ~12 s  (0.5 → 4.5 m/s)
  2  circular — orbit, angular speed varies every ~10 s
  3  diagonal — 30° bearing, speed varies every ~16 s
  4  zigzag   — forward speed + lateral amplitude both vary;
                disappears for 10 s every 30 s (track-lifecycle test)

Parameters
----------
publish_rate  (float, Hz) : publish frequency; must match tracker dt (default 10.0)
noise_std     (float, m)  : Gaussian position noise std (default 0.05)
frame_id      (str)       : header frame_id (default "map")
scenario      (int)       : which objects to publish (default 0)
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)


class DummyObjectsPublisher(Node):

    def __init__(self):
        super().__init__('dummy_objects_publisher')

        self.declare_parameter('publish_rate', 10.0)
        self.declare_parameter('noise_std',    0.05)
        self.declare_parameter('frame_id',     'map')
        self.declare_parameter('scenario',     0)
        # Class published on every detection. kf_object_tracker filters on
        # results[0].hypothesis.class_id being in its target_classes, so a
        # detection without a populated `results` is silently discarded — which
        # is what this publisher used to emit, making it a no-op against the
        # current tracker.
        self.declare_parameter('class_id',     'car')

        rate            = self.get_parameter('publish_rate').value
        self.noise_std  = self.get_parameter('noise_std').value
        self.frame_id   = self.get_parameter('frame_id').value
        self.scenario   = self.get_parameter('scenario').value
        self.class_id   = str(self.get_parameter('class_id').value)

        self.dt  = 1.0 / rate
        self.t   = 0.0
        self.rng = np.random.default_rng()   # one RNG instance, not rebuilt each call

        self.pub = self.create_publisher(Detection3DArray, '/object_localization/detections3d', 10)
        self.create_timer(self.dt, self._timer_cb)

        self.get_logger().info(
            f'Dummy objects publisher started  '
            f'rate={rate:.1f} Hz  noise_std={self.noise_std:.3f} m  scenario={self.scenario}'
        )

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------
    def _timer_cb(self):
        msg = Detection3DArray()
        msg.header.frame_id = self.frame_id
        msg.header.stamp    = self.get_clock().now().to_msg()

        t = self.t

        if self.scenario in (0, 1):
            msg.detections.append(self._make_det(*self._linear(t), traj=self._linear, t=t))

        if self.scenario in (0, 2):
            msg.detections.append(self._make_det(*self._circular(t), traj=self._circular, t=t))

        if self.scenario in (0, 3):
            msg.detections.append(self._make_det(*self._diagonal(t), traj=self._diagonal, t=t))

        if self.scenario in (0, 4):
            # Disappears for 10 s every 30 s — exercises track birth/death
            if t % 30.0 < 20.0:
                msg.detections.append(self._make_det(*self._zigzag(t), traj=self._zigzag, t=t))

        self.pub.publish(msg)
        self.t += self.dt

    # ------------------------------------------------------------------
    # Speed-profile helper
    # ------------------------------------------------------------------
    @staticmethod
    def _arc(t: float, v_base: float, amp: float, freq: float) -> float:
        """
        Cumulative arc length for v(t) = v_base * (1 + amp*sin(freq*t)).
        Integral: s(t) = v_base * (t + amp/freq*(1 - cos(freq*t)))
        s(0) = 0 always; speed oscillates between v_base*(1-amp) and v_base*(1+amp).
        """
        return v_base * (t + (amp / freq) * (1.0 - math.cos(freq * t)))

    @staticmethod
    def _speed(t: float, v_base: float, amp: float, freq: float) -> float:
        """Instantaneous speed at time t (for logging / debug)."""
        return v_base * (1.0 + amp * math.sin(freq * t))

    # ------------------------------------------------------------------
    # Trajectory functions — return (x, y, z)
    # ------------------------------------------------------------------

    def _linear(self, t):
        """
        X-axis.  Speed: 0.5 → 4.5 m/s, period ≈ 12.6 s.
        Short predicted-path  →  long predicted-path clearly visible in RViz2.
        """
        s = self._arc(t, v_base=2.5, amp=0.8, freq=0.5)
        x = s % 40.0 - 20.0
        return x, 0.0, 0.5

    def _circular(self, t):
        """
        Circle r=5 m around (0, 8).  Angular speed: 0.15 → 0.85 rad/s, period ≈ 10.5 s.
        Arc spacing on the circle stretches and compresses visibly.
        """
        theta = self._arc(t, v_base=0.5, amp=0.7, freq=0.6)
        x = 5.0 * math.cos(theta)
        y = 8.0 + 5.0 * math.sin(theta)
        return x, y, 0.5

    def _diagonal(self, t):
        """
        30° diagonal.  Speed: 0.45 → 3.15 m/s, period ≈ 15.7 s.
        """
        s     = self._arc(t, v_base=1.8, amp=0.75, freq=0.4)
        d     = s % 40.0 - 20.0
        angle = math.radians(30.0)
        return d * math.cos(angle), d * math.sin(angle), 0.3

    def _zigzag(self, t):
        """
        Forward speed varies (0.4 → 3.6 m/s, period ≈ 14 s).
        Lateral sway amplitude also oscillates (0.5 → 3.5 m) independently,
        so both the path shape and speed change at different rates.
        """
        s         = self._arc(t, v_base=2.0, amp=0.8, freq=0.45)
        x         = s % 40.0 - 20.0
        sway_amp  = 2.0 + 1.5 * math.sin(0.2 * t)   # 0.5 → 3.5 m, period ≈ 31 s
        y         = sway_amp * math.sin(0.6 * t)
        return x, y, 0.4

    # ------------------------------------------------------------------
    # Detection builder
    # ------------------------------------------------------------------
    def _make_det(self, x: float, y: float, z: float,
                  sx: float = 1.0, sy: float = 0.6, sz: float = 0.6,
                  traj=None, t: float = 0.0) -> Detection3D:
        """Wrap a 3D position (+ optional Gaussian noise) in a Detection3D.

        `traj`/`t` are used to derive a true heading by sampling the trajectory
        a short step ahead. Without it the box orientation stays identity, which
        would leave the heading unusable downstream — and the heading is
        exactly what that model exists to consume.
        """
        # Heading from the noise-free trajectory, so the published orientation
        # is ground truth rather than a differentiated noisy position.
        yaw = 0.0
        if traj is not None:
            ax, ay, _ = traj(t)
            bx, by, _ = traj(t + 0.05)
            if abs(bx - ax) > 1e-9 or abs(by - ay) > 1e-9:
                yaw = math.atan2(by - ay, bx - ax)

        if self.noise_std > 0.0:
            x += float(self.rng.normal(0.0, self.noise_std))
            y += float(self.rng.normal(0.0, self.noise_std))
            z += float(self.rng.normal(0.0, self.noise_std))

        det = Detection3D()
        det.bbox.center.position.x    = float(x)
        det.bbox.center.position.y    = float(y)
        det.bbox.center.position.z    = float(z)
        det.bbox.center.orientation.z = float(math.sin(yaw / 2.0))
        det.bbox.center.orientation.w = float(math.cos(yaw / 2.0))
        det.bbox.size.x = float(sx)
        det.bbox.size.y = float(sy)
        det.bbox.size.z = float(sz)

        # Without this the tracker's class filter drops the detection outright.
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = self.class_id
        hyp.hypothesis.score    = 1.0
        det.results.append(hyp)
        return det


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = DummyObjectsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
