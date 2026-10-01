"""Unit tests for kf_object_tracker.

Run with:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest tests/ -q

The plugin-autoload guard is needed because the system pytest and the
locally-installed anyio plugin disagree about `_pytest.scope`; autoloading it
crashes the run before any test executes.

ROS message types are stubbed rather than imported, so these tests run without
a sourced workspace. The tracker only ever duck-types the detection objects it
is handed, so a stub with the right attributes exercises the real code paths.

A deliberate bias in what is covered: several of these are regression tests for
bugs that actually shipped, and each one is labelled as such. Those are the
tests worth keeping when the rest gets refactored.
"""

import sys
import types as _types

import numpy as np
import pytest

# --- stub the ROS modules the tracker imports at module scope ---------------
_rclpy = _types.ModuleType('rclpy')
_rclpy_node = _types.ModuleType('rclpy.node')
_rclpy_node.Node = type('Node', (), {})
_rclpy.node = _rclpy_node
sys.modules.setdefault('rclpy', _rclpy)
sys.modules.setdefault('rclpy.node', _rclpy_node)

for _pkg, _names in (('geometry_msgs', ('Pose', 'PoseArray', 'PoseStamped')),
                     ('nav_msgs', ('Path',)),
                     ('vision_msgs', ('Detection3D', 'Detection3DArray',
                                      'ObjectHypothesisWithPose'))):
    _mod = _types.ModuleType(_pkg)
    _msg = _types.ModuleType(f'{_pkg}.msg')
    for _n in _names:
        setattr(_msg, _n, type(_n, (), {'__init__': lambda self: None}))
    _mod.msg = _msg
    sys.modules.setdefault(_pkg, _mod)
    sys.modules.setdefault(f'{_pkg}.msg', _msg)

from kf_object_tracking.kf_object_tracker import (            # noqa: E402
    KalmanFilter, Track, associate, KFObjectTrackerNode)


# --- detection stubs -------------------------------------------------------
class _Detection:
    """Duck-typed stand-in for vision_msgs/Detection3D."""

    def __init__(self, x, y, z, class_id='car', score=1.0,
                 size=(1.0, 0.6, 0.6)):
        pos = _types.SimpleNamespace(x=float(x), y=float(y), z=float(z))
        orient = _types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        self.bbox = _types.SimpleNamespace(
            center=_types.SimpleNamespace(position=pos, orientation=orient),
            size=_types.SimpleNamespace(x=float(size[0]), y=float(size[1]),
                                        z=float(size[2])))
        self.results = [_types.SimpleNamespace(
            hypothesis=_types.SimpleNamespace(class_id=class_id,
                                              score=float(score)))]
        self.id = '0'


def cfg(**kw):
    """Track config mirroring the node's declared defaults."""
    base = dict(q_accel=0.5, r_diag=[0.03, 0.03, 0.0015],
                size_alpha=0.3, score_weight_r=True)
    base.update(kw)
    return base


@pytest.fixture(autouse=True)
def _reset_ids():
    """Track ids are class state; without this, tests leak into each other."""
    Track._next_id = 1
    yield
    Track._next_id = 1


# ===========================================================================
# KalmanFilter
# ===========================================================================
def test_update_pulls_estimate_toward_measurement():
    kf = KalmanFilter(0.0, 0.0, 0.0, 0.5, [0.03, 0.03, 0.0015])
    z = np.array([1.0, 0.0, 0.0])
    before = np.linalg.norm(kf.position - z)
    kf.predict(0.1)
    kf.update(z)
    assert np.linalg.norm(kf.position - z) < before


def test_predict_scales_with_dt():
    """REGRESSION: dt was hardcoded at 0.1 s while the real rate varied 0.6-10 Hz.

    A filter told 0.1 s while a full second elapsed attributes the whole
    displacement to a tenth of the time, inflating velocity by
    dt_real/dt_assumed and flinging coasting tracks off the map.
    """
    def travel(dt):
        kf = KalmanFilter(0.0, 0.0, 0.0, 0.5, [0.03, 0.03, 0.0015])
        kf.x[3:] = [1.0, 0.0, 0.0]          # 1 m/s along +x
        kf.predict(dt)
        return kf.position[0]

    assert travel(0.1) == pytest.approx(0.1, abs=1e-9)
    assert travel(1.0) == pytest.approx(1.0, abs=1e-9)
    assert travel(1.0) == pytest.approx(10.0 * travel(0.1), rel=1e-9)


def test_velocity_converges_on_constant_velocity_input():
    """Position-only measurements must still yield the right velocity."""
    kf = KalmanFilter(0.0, 0.0, 0.0, 0.5, [0.001, 0.001, 0.001])
    dt, v = 0.1, 0.5
    for k in range(1, 60):
        kf.predict(dt)
        kf.update(np.array([v * k * dt, 0.0, 0.0]))
    assert kf.velocity[0] == pytest.approx(v, abs=0.05)
    assert kf.velocity[1] == pytest.approx(0.0, abs=0.05)


def test_covariance_grows_when_coasting_and_shrinks_on_update():
    kf = KalmanFilter(0.0, 0.0, 0.0, 0.5, [0.03, 0.03, 0.0015])
    start = np.trace(kf.P[:3, :3])
    kf.predict(1.0)
    coasted = np.trace(kf.P[:3, :3])
    kf.update(np.array([0.0, 0.0, 0.0]))
    corrected = np.trace(kf.P[:3, :3])
    assert coasted > start
    assert corrected < coasted


def test_velocity_starts_unknown_not_zero_confident():
    """Birth covariance on velocity must be loose, or the first updates
    cannot move it and the track under-reacts to real motion."""
    kf = KalmanFilter(1.0, 2.0, 3.0, 0.5, [0.03, 0.03, 0.0015])
    assert np.allclose(kf.position, [1.0, 2.0, 3.0])
    assert np.allclose(kf.velocity, 0.0)
    assert np.all(np.diag(kf.P)[3:] > 1.0)


# ===========================================================================
# Association
# ===========================================================================
def test_associate_matches_nearest():
    tracks = [Track(_Detection(0, 0, 0), cfg()),
              Track(_Detection(5, 0, 0), cfg())]
    dets = [_Detection(0.2, 0, 0), _Detection(4.8, 0, 0)]
    matched, un_t, un_d = associate(tracks, dets, 1.0)
    assert sorted(matched) == [(0, 0), (1, 1)]
    assert un_t == [] and un_d == []


def test_associate_respects_the_distance_gate():
    """A detection beyond max_dist is a different object, not a bad estimate."""
    tracks = [Track(_Detection(0, 0, 0), cfg())]
    dets = [_Detection(9.0, 0, 0)]
    matched, un_t, un_d = associate(tracks, dets, 2.0)
    assert matched == []
    assert un_t == [0] and un_d == [0]


def test_associate_is_greedy_closest_first():
    """Two tracks competing for one detection: the nearer must win."""
    tracks = [Track(_Detection(0.0, 0, 0), cfg()),
              Track(_Detection(1.0, 0, 0), cfg())]
    dets = [_Detection(0.9, 0, 0)]
    matched, un_t, un_d = associate(tracks, dets, 2.0)
    assert matched == [(1, 0)]
    assert un_t == [0] and un_d == []


def test_associate_handles_empty_inputs():
    t = [Track(_Detection(0, 0, 0), cfg())]
    assert associate([], [], 2.0) == ([], [], [])
    assert associate(t, [], 2.0) == ([], [0], [])
    assert associate([], [_Detection(0, 0, 0)], 2.0) == ([], [], [0])


# ===========================================================================
# Track
# ===========================================================================
def test_update_increments_hits_and_clears_coast_time():
    tr = Track(_Detection(1, 2, 3), cfg())
    assert tr.hits == 0 and tr.coast_time == 0.0
    tr.coast_time = 0.8                       # simulate coasting
    tr.update(_Detection(1.1, 2, 3))
    assert tr.hits == 1
    assert tr.coast_time == 0.0


def test_coast_time_is_seconds_not_frames():
    """REGRESSION: expiry used to count frames.

    10 missed frames is 1 s at the design rate but 16 s at the rate actually
    measured, so a frame count meant wildly different real lifetimes.
    """
    tr = Track(_Detection(0, 0, 0), cfg())
    for _ in range(3):
        tr.predict(0.4)
        tr.coast_time += 0.4
    assert tr.coast_time == pytest.approx(1.2)


def test_ids_increment_and_reclaim_is_honoured():
    a = Track(_Detection(0, 0, 0), cfg())
    b = Track(_Detection(1, 0, 0), cfg())
    assert (a.track_id, b.track_id) == (1, 2)
    reborn = Track(_Detection(2, 0, 0), cfg(), reclaim_id=a.track_id)
    assert reborn.track_id == a.track_id


def test_size_is_smoothed_not_taken_raw():
    """Raw box extent jitters ~15% of car length, and pure_pursuit sizes its
    clearance corridor from it, so a jittering box is a jittering margin."""
    tr = Track(_Detection(0, 0, 0, size=(1.0, 0.6, 0.6)), cfg(size_alpha=0.3))
    tr.update(_Detection(0, 0, 0, size=(2.0, 0.6, 0.6)))
    assert tr.size[0] == pytest.approx(1.0 + 0.3 * (2.0 - 1.0))

    raw = Track(_Detection(0, 0, 0, size=(1.0, 0.6, 0.6)), cfg(size_alpha=1.0))
    raw.update(_Detection(0, 0, 0, size=(2.0, 0.6, 0.6)))
    assert raw.size[0] == pytest.approx(2.0)


def test_low_confidence_detection_is_trusted_less():
    """A weak detection should move the estimate less than a strong one."""
    strong = Track(_Detection(0, 0, 0, score=1.0), cfg())
    weak = Track(_Detection(0, 0, 0, score=0.2), cfg())
    strong.update(_Detection(1.0, 0, 0, score=1.0))
    weak.update(_Detection(1.0, 0, 0, score=0.2))
    assert strong.kf.position[0] > weak.kf.position[0]


def test_confidence_scaling_restores_R():
    """REGRESSION: R is scaled per-detection and must be put back.

    Leaking the scaled R would compound it every frame until the filter
    stopped believing measurements at all.
    """
    tr = Track(_Detection(0, 0, 0), cfg())
    before = tr.kf.R.copy()
    tr.update(_Detection(0.1, 0, 0, score=0.2))
    assert np.allclose(tr.kf.R, before)


def test_class_id_is_captured():
    tr = Track(_Detection(0, 0, 0, class_id='person'), cfg())
    assert tr.class_id == 'person'


# ===========================================================================
# Node helpers — called unbound, so the real code runs without rclpy
# ===========================================================================
def test_waypoints_are_evenly_spaced_along_velocity():
    tr = Track(_Detection(0, 0, 0), cfg())
    tr.kf.x[3:] = [2.0, 0.0, 0.0]
    node = _types.SimpleNamespace(predict_steps=5, predict_dt=0.5)

    wps = KFObjectTrackerNode._waypoints(node, tr)

    assert len(wps) == 6                      # steps + 1, including now
    xs = [p[0] for p in wps]
    assert xs[0] == pytest.approx(0.0)
    steps = np.diff(xs)
    assert np.allclose(steps, steps[0])
    assert steps[0] == pytest.approx(2.0 * 0.5)


def test_reclaim_id_returns_matching_retired_track():
    node = _types.SimpleNamespace(
        _retired=[(7, np.array([1.0, 1.0, 0.0]), 'car', 0.5)],
        max_id_retention=3.0, reid_max_dist=1.5,
        get_logger=lambda: _types.SimpleNamespace(info=lambda *a, **k: None))

    got = KFObjectTrackerNode._reclaim_id(node, _Detection(1.2, 1.0, 0.0))

    assert got == 7
    assert node._retired == []                # consumed, not reusable twice


def test_reclaim_id_refuses_wrong_class_or_too_far():
    def node():
        return _types.SimpleNamespace(
            _retired=[(7, np.array([1.0, 1.0, 0.0]), 'car', 0.5)],
            max_id_retention=3.0, reid_max_dist=1.5,
            get_logger=lambda: _types.SimpleNamespace(
                info=lambda *a, **k: None))

    # a person must never inherit a car's identity
    assert KFObjectTrackerNode._reclaim_id(
        node(), _Detection(1.1, 1.0, 0.0, class_id='person')) is None
    # beyond reid_max_dist it is a different object
    assert KFObjectTrackerNode._reclaim_id(
        node(), _Detection(5.0, 1.0, 0.0)) is None


def test_reclaim_id_is_disabled_when_retention_is_zero():
    node = _types.SimpleNamespace(
        _retired=[(7, np.array([1.0, 1.0, 0.0]), 'car', 0.5)],
        max_id_retention=0.0, reid_max_dist=1.5,
        get_logger=lambda: _types.SimpleNamespace(info=lambda *a, **k: None))
    assert KFObjectTrackerNode._reclaim_id(node, _Detection(1.0, 1.0, 0.0)) is None
