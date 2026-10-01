#!/usr/bin/env python3
"""Single entry point for judging a kf_object_tracker change.

    evaluate_tracker.py EGO [TARGET]                  score the recorded run
    evaluate_tracker.py EGO [TARGET] --replay         re-run current code on the
                                                      recorded detections
    evaluate_tracker.py EGO TARGET --ab A B           two configs, same input
    evaluate_tracker.py --regression                  every discovered bag pair

Two families of numbers are reported.

PRIMARY-OBJECT accuracy answers "how well is the object we care about tracked":
for each frame the track nearest the designated ground-truth object is scored,
with no visibility gate. This is deliberately the same computation the ad-hoc
scripts used, so previously published figures remain comparable.

MULTI-OBJECT (CLEAR-MOT) answers "is the tracker managing tracks correctly":
ground truth and tracks are matched frame by frame with the Hungarian
algorithm, restricted to ground-truth objects actually inside the camera's
envelope. Without that restriction the parked mocap cars would score as
permanent misses and every number would be noise.
"""

import argparse
import glob
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_lib as E                                     # noqa: E402

# Replay reuses the node's own estimation classes, sliced out of the module so
# importing it never constructs a Node or pulls in rclpy.
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                   'kf_object_tracking', 'kf_object_tracker.py')


def load_tracker_classes():
    from vision_msgs.msg import Detection3D
    src = open(SRC).read()
    start = src.index('class KalmanFilter:')
    end = src.index('class KFObjectTrackerNode')
    ns = {'np': np, 'Detection3D': Detection3D, 'math': math}
    exec(compile(src[start:end], 'tracker', 'exec'), ns)
    return ns['Track'], ns['associate']


def reclaim_id(retired, detection, reid_max_dist):
    """Offline mirror of KFObjectTrackerNode._reclaim_id.

    The node method lives on the Node class, so replaying it here means
    duplicating ~15 lines. Kept deliberately literal against the original so
    the two cannot drift without the A/B numbers moving.
    """
    if not retired:
        return None
    pos = detection.bbox.center.position
    p = np.array([pos.x, pos.y, pos.z], dtype=float)
    cls = (detection.results[0].hypothesis.class_id
           if detection.results else 'unknown')
    best, best_d = None, reid_max_dist
    for i, (tid, last_p, last_c, _age) in enumerate(retired):
        if last_c != cls:
            continue
        d = float(np.linalg.norm(p - last_p))
        if d < best_d:
            best, best_d = i, d
    return retired.pop(best)[0] if best is not None else None


def default_cfg(**kw):
    """Mirrors kf_object_tracker's declared defaults."""
    cfg = dict(q_accel=0.5, r_diag=[0.03, 0.03, 0.0015],
               size_alpha=0.3, score_weight_r=True)
    cfg.update(kw)
    return cfg


def parse_cfg(spec):
    """'q_accel=0.1,size_alpha=1.0' -> dict, on top of the defaults."""
    kw = {}
    for part in (spec or '').split(','):
        if not part.strip():
            continue
        k, _, v = part.partition('=')
        k, v = k.strip(), v.strip()
        if k in ('score_weight_r',):
            kw[k] = v.lower() in ('1', 'true', 'yes')
        elif k == 'r_diag':
            kw[k] = [float(x) for x in v.strip('[]').replace(',', ' ').split()]
        else:
            kw[k] = float(v)
    return default_cfg(**kw)


# --------------------------------------------------------------------------
# Track sources
# --------------------------------------------------------------------------
def frames_recorded(ego):
    """[(t, [(id, pos, vel)])] straight from the recorded /tracked_objects."""
    return [(E.stamp(m.header),
             [(d[0], d[1], d[2]) for d in E.detections_of(m)])
            for m in ego.get(E.TRACKED, [])]


def frames_replayed(ego, cfg, *, dt_fallback=0.1, dt_max=1.0, max_dist=2.0,
                    max_coast=1.0, min_hits=1, classes=('person', 'car'),
                    id_retention=3.0, reid_max_dist=1.5, hook=None):
    """Feed the recorded detections through the current tracker code.

    `hook(t, tracks)`, if given, is called once per frame with the live Track
    objects. It exists so horizon evaluation can copy a filter and roll it
    forward from the real internal state — covariance included — rather than
    re-deriving a state from the published position and velocity. Only the
    live objects carry P, and the rollout is not faithful without it.
    """
    Track, associate = load_tracker_classes()
    Track._next_id = 1
    tracks, retired, last, frames = [], [], None, []
    for msg in ego.get(E.DET3D, []):
        t = E.stamp(msg.header)
        dt = dt_fallback if (last is None or not (0.0 < t - last)) \
            else min(t - last, dt_max)
        last = t
        dets = [d for d in msg.detections
                if d.results and d.results[0].hypothesis.class_id in classes]
        for tr in tracks:
            tr.predict(dt)
        matched, un_t, un_d = associate(tracks, dets, max_dist)
        for ti, di in matched:
            tracks[ti].update(dets[di])
        for ti in un_t:
            tracks[ti].coast_time += dt
        if id_retention > 0.0:
            retired = [(i, p, c, a + dt) for i, p, c, a in retired
                       if a + dt <= id_retention]
        for di in un_d:
            rid = reclaim_id(retired, dets[di], reid_max_dist) \
                if id_retention > 0.0 else None
            tracks.append(Track(dets[di], cfg, reclaim_id=rid))
        if id_retention > 0.0:
            for tr in tracks:
                if tr.coast_time > max_coast and tr.hits >= min_hits:
                    retired.append((tr.track_id,
                                    np.array(tr.kf.position, dtype=float),
                                    tr.class_id, 0.0))
        tracks = [tr for tr in tracks if tr.coast_time <= max_coast]
        frames.append((t, [(str(tr.track_id), tr.kf.position.copy(),
                            tr.kf.velocity.copy())
                           for tr in tracks if tr.hits >= min_hits]))
        if hook is not None:
            hook(t, [tr for tr in tracks if tr.hits >= min_hits])
    return frames


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def primary_metrics(frames, gt, min_speed_for_dir=0.25):
    """Accuracy against the ground-truth object, nearest track per frame."""
    pe, ve, dirs, ntr, ids, lev = [], [], [], [], set(), []
    for t, items in frames:
        g = gt['pos'].at(t)
        if g is None or not items:
            continue
        tid, p, v = min(items, key=lambda it: np.linalg.norm(it[1] - g))
        ids.add(tid)
        ntr.append(len(items))
        pe.append(p - g)
        if gt.get('yaw') is not None:
            yw = gt['yaw'].at(t)
            if yw is not None:
                lev.append((p - g, float(yw)))
        gv = E.gt_velocity(gt, t)
        if gv is not None:
            ve.append(np.linalg.norm(v - gv))
            ng, nv = np.linalg.norm(gv[:2]), np.linalg.norm(v[:2])
            if ng > min_speed_for_dir and nv > 1e-6:
                dirs.append(math.degrees(math.acos(np.clip(
                    np.dot(v[:2], gv[:2]) / (nv * ng), -1.0, 1.0))))
    if not pe:
        return None
    P = np.array(pe)
    d = np.linalg.norm(P, axis=1)
    # /kinematic_state is the target's base_link centre; /tracked_objects is
    # the detected box centre. The constant part of the gap between them is a
    # reference-point offset, not tracking error, and it only separates out in
    # the target's own body frame.
    off, resid = E.lever_arm(lev)
    rd = np.linalg.norm(resid, axis=1) if resid is not None else None
    V = np.array(ve) if ve else np.array([np.nan])
    D = np.array(dirs) if dirs else np.array([np.nan])
    return dict(
        n=len(P), pos_mean=d.mean(), pos_med=float(np.median(d)),
        pos_rms=math.sqrt(float((d ** 2).mean())),
        pos_p95=float(np.percentile(d, 95)), pos_max=d.max(),
        bias=P.mean(axis=0), std=P.std(axis=0),
        vel_rms=math.sqrt(float(np.nanmean(V ** 2))),
        dir_med=float(np.nanmedian(D)),
        dir_p90=float(np.nanpercentile(D, 90)) if dirs else float('nan'),
        backwards=100.0 * float(np.mean(D > 150.0)) if dirs else float('nan'),
        ids=len(ids), tracks_per_frame=float(np.mean(ntr)),
        multi=100.0 * float(np.mean([n > 1 for n in ntr])),
        lever=off,
        resid_mean=float(rd.mean()) if rd is not None else float('nan'),
        resid_rms=math.sqrt(float((rd ** 2).mean())) if rd is not None
        else float('nan'))


def mot_metrics(frames, gts, ego_p, ego_y, match_dist=2.0):
    """CLEAR-MOT over ground-truth objects inside the camera envelope."""
    lim = E.camera_range_limits()
    names = sorted(gts)
    last_id, was_matched = {}, {}
    idsw = frag = misses = fps = gt_total = 0
    dists = []
    for t, items in frames:
        vis = [n for n in names if E.visible(gts[n], ego_p, ego_y, t, lim)]
        gt_total += len(vis)
        if not vis and not items:
            continue
        if vis and items:
            C = np.zeros((len(vis), len(items)))
            for i, n in enumerate(vis):
                g = gts[n]['pos'].at(t)
                for j, (_, p, _) in enumerate(items):
                    C[i, j] = np.linalg.norm(p - g)
            ri, cj = linear_sum_assignment(C)
            pairs = [(i, j) for i, j in zip(ri, cj) if C[i, j] <= match_dist]
        else:
            pairs = []
        matched_g = {i for i, _ in pairs}
        matched_t = {j for _, j in pairs}
        for i, j in pairs:
            n, tid = vis[i], items[j][0]
            dists.append(C[i, j])
            if n in last_id and last_id[n] != tid:
                idsw += 1
            if was_matched.get(n) is False:
                frag += 1
            last_id[n] = tid
            was_matched[n] = True
        for i, n in enumerate(vis):
            if i not in matched_g:
                misses += 1
                if was_matched.get(n):
                    was_matched[n] = False
        fps += len(items) - len(matched_t)
    if gt_total == 0:
        return None
    return dict(gt_frames=gt_total, matches=len(dists), misses=misses,
                fps=fps, idsw=idsw, frag=frag,
                recall=100.0 * len(dists) / gt_total,
                motp=float(np.mean(dists)) if dists else float('nan'),
                mota=100.0 * (1.0 - (misses + fps + idsw) / gt_total))


def empirical_r(ego, gt):
    """Measurement-noise variance per axis from raw detections about GT.

    Bias is excluded on purpose: a fixed offset is not noise, and feeding it
    into R would make the filter distrust measurements for the wrong reason.
    """
    errs = []
    for m in ego.get(E.DET3D, []):
        t = E.stamp(m.header)
        g = gt['pos'].at(t)
        if g is None or not m.detections:
            continue
        p = min((d[1] for d in E.detections_of(m)),
                key=lambda q: np.linalg.norm(q - g))
        errs.append(p - g)
    if len(errs) < 3:
        return None, None
    A = np.array(errs)
    return A.std(axis=0) ** 2, A.mean(axis=0)


def pick_primary(gts):
    """The GT object to score accuracy against: the target bag's if present,
    else whichever mocap body actually moved."""
    for n, g in gts.items():
        if g.get('from_target_bag'):
            return n
    return max(gts, key=lambda n: gts[n].get('moved') or 0.0) if gts else None


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def report(label, prim, mot, rates=None, rvar=None, rbias=None):
    print(f"\n=== {label} ===")
    if rates:
        for name, (hz, dt) in rates.items():
            if len(dt):
                print(f"  {name:<38} {hz:6.2f} Hz   dt med={np.median(dt):.3f}"
                      f" max={dt.max():.3f} s")
    if prim:
        print(f"  PRIMARY OBJECT  n={prim['n']}")
        print(f"    position   mean={prim['pos_mean']:.3f}  med={prim['pos_med']:.3f}"
              f"  rms={prim['pos_rms']:.3f}  p95={prim['pos_p95']:.3f}"
              f"  max={prim['pos_max']:.3f} m")
        b, s = prim['bias'], prim['std']
        print(f"    bias=({b[0]:+.3f},{b[1]:+.3f},{b[2]:+.3f})  "
              f"std=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}) m")
        print(f"    velocity   rms={prim['vel_rms']:.3f} m/s   "
              f"dir med={prim['dir_med']:.1f} deg  p90={prim['dir_p90']:.1f} deg"
              f"  backwards={prim['backwards']:.1f}%")
        if prim.get('lever') is not None:
            lv = prim['lever']
            print(f"    reference-point offset (target body frame): "
                  f"fwd={lv[0]:+.3f}  lat={lv[1]:+.3f}  vert={lv[2]:+.3f} m")
            print(f"    error with that offset removed: "
                  f"mean={prim['resid_mean']:.3f}  rms={prim['resid_rms']:.3f} m"
                  f"   <- tracking error proper")
        print(f"    track ids={prim['ids']}  tracks/frame={prim['tracks_per_frame']:.2f}"
              f"  multi-track frames={prim['multi']:.0f}%")
    if mot:
        print(f"  MULTI-OBJECT (visible GT only)")
        print(f"    gt frames={mot['gt_frames']}  matches={mot['matches']}"
              f"  misses={mot['misses']}  false pos={mot['fps']}")
        print(f"    ID switches={mot['idsw']}  fragmentations={mot['frag']}")
        print(f"    recall={mot['recall']:.1f}%  MOTP={mot['motp']:.3f} m"
              f"  MOTA={mot['mota']:.1f}%")
    if rvar is not None:
        print(f"  empirical R diag = [{rvar[0]:.4f}, {rvar[1]:.4f}, {rvar[2]:.4f}]"
              f"   (raw detection bias ({rbias[0]:+.3f},{rbias[1]:+.3f},"
              f"{rbias[2]:+.3f}) m, excluded)")


def evaluate(ego_path, target_path=None, source='recorded', cfg=None,
             quiet=False, label=None):
    ego = E.read_bag(ego_path)
    tgt = E.read_bag(target_path) if target_path else None
    ego_p, ego_y = E.ego_pose(ego)
    gts = E.gt_objects(ego, tgt)
    if not gts or ego_p is None:
        if not quiet:
            print(f"\n=== {label or os.path.basename(ego_path)} ===")
            print("  no ground truth available (no mocap bodies / no ego pose)")
        return None
    prim_name = pick_primary(gts)
    frames = frames_recorded(ego) if source == 'recorded' \
        else frames_replayed(ego, cfg or default_cfg())
    prim = primary_metrics(frames, gts[prim_name])
    mot = mot_metrics(frames, gts, ego_p, ego_y)
    rvar, rbias = empirical_r(ego, gts[prim_name])
    if not quiet:
        rates = {t: E.rate_of(ego.get(t, []))
                 for t in (E.DET2D, E.DET3D, E.TRACKED) if ego.get(t)}
        report(label or f"{os.path.basename(ego_path)}  [{source}]  "
                        f"primary GT = body {prim_name}",
               prim, mot, rates, rvar, rbias)
    return dict(primary=prim, mot=mot, r=rvar, gt=prim_name,
                n_gt_objects=len(gts))


# --------------------------------------------------------------------------
# Bag pairing
# --------------------------------------------------------------------------
def discover_pairs(root='/home/af/ros2_ws'):
    """Pair ego and target bags by greatest time overlap, not by name.

    Recorded pairs start 2.4-5.3 s apart and the naming is inconsistent
    (ego_* vs avoid_*, mismatched numeric prefixes), so name matching would be
    fragile.
    """
    egos = sorted(glob.glob(os.path.join(root, 'ego_*'))
                  + glob.glob(os.path.join(root, 'avoid_*')))
    tgts = sorted(glob.glob(os.path.join(root, 'target_*')))
    spans = {}
    for b in egos + tgts:
        try:
            spans[b] = E.bag_span(b)
        except Exception:
            spans[b] = (None, None)
    pairs = []
    for e in egos:
        es, ee = spans[e]
        best, best_ov = None, 0.0
        if es is not None:
            for t in tgts:
                ts, te = spans[t]
                if ts is None:
                    continue
                ov = min(ee, te) - max(es, ts)
                if ov > best_ov:
                    best, best_ov = t, ov
        pairs.append((e, best if best_ov > 1.0 else None))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ego_bag', nargs='?')
    ap.add_argument('target_bag', nargs='?')
    ap.add_argument('--replay', action='store_true',
                    help='re-run the current tracker code on recorded detections')
    ap.add_argument('--ab', nargs=2, metavar=('A', 'B'),
                    help='two configs, e.g. --ab "q_accel=0.5" "q_accel=0.1"')
    ap.add_argument('--regression', action='store_true')
    ap.add_argument('--root', default='/home/af/ros2_ws')
    args = ap.parse_args()

    if args.regression:
        pairs = discover_pairs(args.root)
        print(f"{'bag':<27}{'GT':>7}{'n':>6}{'pos rms':>9}{'vel rms':>9}"
              f"{'dir med':>9}{'ids':>5}{'IDsw':>6}{'frag':>6}{'rec%':>7}{'MOTA%':>8}")
        print("-" * 89)
        for e, t in pairs:
            r = evaluate(e, t, quiet=True)
            nm = os.path.basename(e)
            if not r or not r['primary']:
                print(f"{nm:<27}{'-':>7}{'no ground truth / no tracks':>60}")
                continue
            p, m = r['primary'], r['mot']
            print(f"{nm:<27}{r['gt']:>7}{p['n']:>6}{p['pos_rms']:>9.3f}"
                  f"{p['vel_rms']:>9.3f}{p['dir_med']:>9.1f}{p['ids']:>5}"
                  f"{(m['idsw'] if m else 0):>6}{(m['frag'] if m else 0):>6}"
                  f"{(m['recall'] if m else float('nan')):>7.1f}"
                  f"{(m['mota'] if m else float('nan')):>8.1f}")
        return

    if not args.ego_bag:
        ap.error('give an EGO bag, or --regression')

    if args.ab:
        for spec in args.ab:
            evaluate(args.ego_bag, args.target_bag, source='replay',
                     cfg=parse_cfg(spec), label=f"replay [{spec}]")
        return

    evaluate(args.ego_bag, args.target_bag,
             source='replay' if args.replay else 'recorded')


if __name__ == '__main__':
    main()
