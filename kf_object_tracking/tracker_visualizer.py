#!/usr/bin/env python3
"""
Tracker Visualizer — subscribes to data topics, publishes all RViz2 markers.

Subscribes:
  /tracked_objects                              (vision_msgs/Detection3DArray)
  /tracked_objects/predicted_path_array         (vision_msgs/Detection3DArray)

Publishes:
  /tracked_objects/bounding_boxes               (visualization_msgs/MarkerArray)
  /tracked_objects/history                      (visualization_msgs/MarkerArray)
  /tracked_objects/predicted_paths              (visualization_msgs/MarkerArray)
  /tracked_objects/predicted_path_array/markers (visualization_msgs/MarkerArray)
"""

import colorsys
import math
from collections import deque

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


def _track_color(track_id: int, past: bool = False) -> tuple[float, float, float]:
    """Stable per-track colour; `past` fades it for already-travelled history.

    Hue encodes identity only (the golden-ratio step keeps consecutive ids far
    apart on the wheel), so a track's trail and its prediction stay the same
    colour and differ just in saturation/brightness.
    """
    hue = (track_id * 0.618033988749895) % 1.0
    s, v = (0.35, 0.65) if past else (0.85, 1.0)
    return colorsys.hsv_to_rgb(hue, s, v)


def _rgba(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = float(r), float(g), float(b), float(a)
    return c


def _point(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x, p.y, p.z = x, y, z
    return p


class TrackerVisualizerNode(Node):

    def __init__(self):
        super().__init__('tracker_visualizer')

        self.declare_parameter('history_len',     50)
        # Sized against the worst-case inter-message gap seen by a *remote*
        # RViz (Wi-Fi jitter + dropped frames), not the nominal publish period.
        # Too short and markers expire before their replacement arrives, which
        # reads as flicker rather than as a dropout.
        self.declare_parameter('marker_lifetime', 3.0)

        self.history_len = self.get_parameter('history_len').value
        ml = self.get_parameter('marker_lifetime').value
        self.lifetime = Duration(sec=int(ml), nanosec=int((ml % 1.0) * 1e9))

        self.history: dict[int, deque] = {}

        self.create_subscription(
            Detection3DArray, '/tracked_objects', self._on_tracked, 10)
        self.create_subscription(
            Detection3DArray, '/tracked_objects/predicted_path_array', self._on_predicted, 10)

        self.pub_boxes      = self.create_publisher(
            MarkerArray, '/tracked_objects/bounding_boxes', 10)
        self.pub_history    = self.create_publisher(
            MarkerArray, '/tracked_objects/history', 10)
        self.pub_paths      = self.create_publisher(
            MarkerArray, '/tracked_objects/predicted_paths', 10)
        self.pub_footprints = self.create_publisher(
            MarkerArray, '/tracked_objects/predicted_path_array/markers', 10)

        self.get_logger().info('Tracker Visualizer started.')

    # ------------------------------------------------------------------
    def _on_tracked(self, msg: Detection3DArray):
        """Record history, emit bounding-box cubes and history trails."""
        header = msg.header
        active_ids: set[int] = set()

        boxes_ma   = MarkerArray()
        history_ma = MarkerArray()

        # Clear all stale markers from previous frame
        del_m = Marker()
        del_m.header = header
        del_m.action = Marker.DELETEALL
        boxes_ma.markers.append(del_m)

        del_m2 = Marker()
        del_m2.header = header
        del_m2.action = Marker.DELETEALL
        history_ma.markers.append(del_m2)

        for det in msg.detections:
            tid = int(det.id)
            active_ids.add(tid)
            pos = det.bbox.center.position
            sz  = det.bbox.size
            r, g, b = _track_color(tid)

            # Append to history
            if tid not in self.history:
                self.history[tid] = deque(maxlen=self.history_len)
            self.history[tid].append((float(pos.x), float(pos.y), float(pos.z)))

            # Current-position bounding-box cube
            box = Marker()
            box.header        = header
            box.ns            = 'bounding_boxes'
            box.id            = tid
            box.type          = Marker.CUBE
            box.action        = Marker.ADD
            box.lifetime      = self.lifetime
            box.pose          = det.bbox.center
            box.scale.x       = float(sz.x) if sz.x > 0.0 else 1.0
            box.scale.y       = float(sz.y) if sz.y > 0.0 else 0.6
            box.scale.z       = float(sz.z) if sz.z > 0.0 else 0.6
            box.color         = _rgba(r, g, b, 0.55)
            boxes_ma.markers.append(box)

            # Past-position trail — same hue, faded, so it recedes behind the
            # live box and the prediction without losing the id colour coding.
            hist = self.history[tid]
            if len(hist) >= 2:
                pr, pg, pb = _track_color(tid, past=True)
                line = Marker()
                line.header   = header
                line.ns       = 'history'
                line.id       = tid
                line.type     = Marker.LINE_STRIP
                line.action   = Marker.ADD
                line.lifetime = self.lifetime
                line.scale.x  = 0.06
                line.color    = _rgba(pr, pg, pb, 0.7)
                line.points   = [_point(hx, hy, hz) for hx, hy, hz in hist]
                history_ma.markers.append(line)

        # Drop history for tracks that have been deleted
        for tid in list(self.history):
            if tid not in active_ids:
                del self.history[tid]

        self.pub_boxes.publish(boxes_ma)
        self.pub_history.publish(history_ma)

    # ------------------------------------------------------------------
    def _on_predicted(self, msg: Detection3DArray):
        """Regroup flat path array by track, emit CUBE_LIST footprints and path lines."""
        header = msg.header

        # Regroup — preserve insertion order (Python 3.7+ dict)
        waypoints: dict[int, list] = {}
        sizes:     dict[int, tuple] = {}
        for det in msg.detections:
            tid = int(det.id)
            if tid not in waypoints:
                waypoints[tid] = []
                sz = det.bbox.size
                sizes[tid] = (
                    float(sz.x) if sz.x > 0.0 else 1.0,
                    float(sz.y) if sz.y > 0.0 else 0.6,
                    float(sz.z) if sz.z > 0.0 else 0.6,
                )
            p = det.bbox.center.position
            waypoints[tid].append((float(p.x), float(p.y), float(p.z)))

        footprints_ma = MarkerArray()
        paths_ma      = MarkerArray()

        del_f = Marker(); del_f.header = header; del_f.action = Marker.DELETEALL
        del_p = Marker(); del_p.header = header; del_p.action = Marker.DELETEALL
        footprints_ma.markers.append(del_f)
        paths_ma.markers.append(del_p)

        for tid, wps in waypoints.items():
            r, g, b  = _track_color(tid)
            sx, sy, sz = sizes[tid]
            n = len(wps)

            # Predicted footprints — one cube per waypoint, alpha fades toward future
            cubes = Marker()
            cubes.header   = header
            cubes.ns       = 'predicted_footprints'
            cubes.id       = tid * 2
            cubes.type     = Marker.CUBE_LIST
            cubes.action   = Marker.ADD
            cubes.lifetime = self.lifetime
            cubes.scale.x  = sx
            cubes.scale.y  = sy
            cubes.scale.z  = sz

            spine = Marker()
            spine.header   = header
            spine.ns       = 'predicted_footprints'
            spine.id       = tid * 2 + 1
            spine.type     = Marker.LINE_STRIP
            spine.action   = Marker.ADD
            spine.lifetime = self.lifetime
            spine.scale.x  = 0.04
            spine.color    = _rgba(r, g, b, 0.6)

            for i, (x, y, z) in enumerate(wps):
                pt = _point(x, y, z)
                cubes.points.append(pt)
                spine.points.append(pt)
                alpha = float(1.0 - 0.85 * i / max(n - 1, 1))
                cubes.colors.append(_rgba(r, g, b, alpha))

            footprints_ma.markers.append(cubes)
            footprints_ma.markers.append(spine)

            # Predicted path line
            path_line = Marker()
            path_line.header   = header
            path_line.ns       = 'predicted_paths'
            path_line.id       = tid
            path_line.type     = Marker.LINE_STRIP
            path_line.action   = Marker.ADD
            path_line.lifetime = self.lifetime
            path_line.scale.x  = 0.05
            path_line.color    = _rgba(r, g, b, 0.9)
            path_line.points   = [_point(x, y, z) for x, y, z in wps]
            paths_ma.markers.append(path_line)

            # Speed label above current position (uses spacing of first two waypoints)
            if n >= 2:
                x0, y0, z0 = wps[0]
                x1, y1, z1 = wps[1]
                # waypoints are predict_dt apart; 10 Hz assumed for m/s display
                speed = math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) * 10.0
                label = Marker()
                label.header   = header
                label.ns       = 'predicted_paths_labels'
                label.id       = tid
                label.type     = Marker.TEXT_VIEW_FACING
                label.action   = Marker.ADD
                label.lifetime = self.lifetime
                label.pose.position.x    = x0
                label.pose.position.y    = y0
                label.pose.position.z    = z0 + 1.0
                label.pose.orientation.w = 1.0
                label.scale.z  = 0.4
                # Track-coloured, not white: with several objects close
                # together a white label is ambiguous about which box it names.
                label.color    = _rgba(r, g, b, 0.95)
                label.text     = f'ID:{tid}  {speed:.1f} m/s'
                paths_ma.markers.append(label)

                # Heading arrow: tail at the current position, tip at the end
                # of the prediction. Skipped for near-stationary tracks — a
                # zero-length ARROW renders as a degenerate blob in RViz.
                xe, ye, ze = wps[-1]
                if math.dist((x0, y0, z0), (xe, ye, ze)) > 0.05:
                    arrow = Marker()
                    arrow.header   = header
                    arrow.ns       = 'predicted_paths_arrow'
                    arrow.id       = tid
                    arrow.type     = Marker.ARROW
                    arrow.action   = Marker.ADD
                    arrow.lifetime = self.lifetime
                    # Two-point form: points = [tail, tip], and scale means
                    # (shaft diameter, head diameter, head length) — not the
                    # position/orientation form's bounding box.
                    arrow.points   = [_point(x0, y0, z0), _point(xe, ye, ze)]
                    arrow.scale.x  = 0.05
                    arrow.scale.y  = 0.12
                    arrow.scale.z  = 0.18
                    arrow.color    = _rgba(r, g, b, 0.95)
                    paths_ma.markers.append(arrow)

        self.pub_footprints.publish(footprints_ma)
        self.pub_paths.publish(paths_ma)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = TrackerVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
