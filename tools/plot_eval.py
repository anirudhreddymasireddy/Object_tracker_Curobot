#!/usr/bin/env python3
"""Per-sample CSV and a diagnostic figure of /tracked_objects vs ground truth.

evaluate_tracker.py answers "how good is it" in one line of summary statistics.
This answers "where and when does it go wrong", which summaries hide: the error
here is strongly range-dependent (measured -2% beyond 2.5 m but -109% inside
1.5 m) and comes in bursts around detection gaps, neither of which shows up in
an RMS.

    plot_eval.py EGO_BAG TARGET_BAG [-o out.png] [--csv out.csv] [--replay]

Ground truth is the target car's own /kinematic_state. Note it reports the
centre of the target's base_link while /tracked_objects reports the centre of
the detected box, so a fixed body-frame offset between them is expected — the
figure separates that out rather than counting it as error.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')                      # no display needed on the car
import matplotlib.pyplot as plt            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_lib as E                       # noqa: E402
import evaluate_tracker as T               # noqa: E402


def floor_ylim(p, span, centre=None):
    """Stop matplotlib magnifying numerically negligible residuals.

    Autoscale fills the axes with whatever range the data happens to span. On a
    clean simulated run that span can be micrometres, which renders a fully
    converged filter as a dramatic curve and invites reading structure into
    rounding error. Holding a floor on the span keeps "flat" looking flat.
    """
    lo, hi = p.get_ylim()
    if hi - lo >= span:
        return
    c = 0.5 * (lo + hi) if centre is None else centre
    p.set_ylim(c - span / 2.0, c + span / 2.0)


def apply_lever(rows, spec):
    """Remove the base_link-centre vs box-centre offset from the estimate.

    /kinematic_state reports the centre of the target's base_link (rear axle);
    /tracked_objects reports the centre of the detected box (mid-body). The gap
    between them is a fixed offset in the TARGET's body frame — measured at
    +0.219 m forward, +0.004 m lateral across three runs, about half a
    wheelbase on an 0.80 m car. It is a definitional difference between the two
    signals, not tracking error, and it accounts for 28% of the raw rms.

    spec:
      'none'  leave the estimate alone
      'auto'  fit the constant body-frame offset from this run and remove it
      'fwd,lat,vert'  remove a fixed offset — use this when comparing several
              runs, since an auto fit differs per run and would make them
              incomparable.

    Returns (rows, offset_used) with err_* / err_norm / trk_* recomputed.
    """
    if spec == 'none':
        return rows, None
    usable = [r for r in rows if math.isfinite(r['gt_yaw'])]
    if not usable:
        return rows, None
    if spec == 'auto':
        body = []
        for r in usable:
            e = np.array([r['err_x'], r['err_y'], r['err_z']])
            y = r['gt_yaw']
            c, s_ = math.cos(-y), math.sin(-y)
            body.append([c * e[0] - s_ * e[1], s_ * e[0] + c * e[1], e[2]])
        off = np.asarray(body).mean(axis=0)
    else:
        off = np.array([float(x) for x in spec.split(',')], dtype=float)
        if off.size == 1:
            off = np.array([off[0], 0.0, 0.0])

    for r in rows:
        y = r['gt_yaw']
        if not math.isfinite(y):
            continue
        c, s_ = math.cos(y), math.sin(y)
        om = np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1], off[2]])
        for k, i in (('trk_x', 0), ('trk_y', 1), ('trk_z', 2)):
            r[k] -= om[i]
        for k, i in (('err_x', 0), ('err_y', 1), ('err_z', 2)):
            r[k] -= om[i]
        r['err_norm'] = float(math.sqrt(r['err_x'] ** 2 + r['err_y'] ** 2
                                        + r['err_z'] ** 2))
    return rows, off


def collect(ego_path, target_path, replay=False):
    """Per-sample rows aligning each track estimate with ground truth."""
    ego = E.read_bag(ego_path)
    tgt = E.read_bag(target_path) if target_path else None
    gts = E.gt_objects(ego, tgt)
    name = T.pick_primary(gts)
    if name is None:
        raise SystemExit('no ground truth in these bags')
    gt = gts[name]
    ego_p, ego_y = E.ego_pose(ego)

    frames = (T.frames_replayed(ego, T.default_cfg()) if replay
              else T.frames_recorded(ego))
    t0 = frames[0][0] if frames else 0.0

    rows = []
    for t, items in frames:
        g = gt['pos'].at(t)
        if g is None or not items:
            continue
        tid, p, v = min(items, key=lambda it: np.linalg.norm(it[1] - g))
        gv = E.gt_velocity(gt, t)
        gyaw = gt['yaw'].at(t) if gt.get('yaw') is not None else None
        # Range from the ego, because the error is range-dependent.
        rng = float('nan')
        if ego_p is not None:
            e = ego_p.at(t)
            if e is not None:
                rng = float(math.hypot(g[0] - e[0], g[1] - e[1]))
        err = p - g
        row = dict(
            t=t - t0, track_id=tid, n_tracks=len(items), gt_range=rng,
            gt_x=g[0], gt_y=g[1], gt_z=g[2],
            trk_x=p[0], trk_y=p[1], trk_z=p[2],
            err_x=err[0], err_y=err[1], err_z=err[2],
            err_norm=float(np.linalg.norm(err)),
            gt_yaw=float(gyaw) if gyaw is not None else float('nan'))
        if gv is not None:
            row.update(gt_speed=float(np.linalg.norm(gv[:2])),
                       trk_speed=float(np.linalg.norm(v[:2])),
                       vel_err=float(np.linalg.norm(v - gv)))
            ng, nv = np.linalg.norm(gv[:2]), np.linalg.norm(v[:2])
            row['dir_err'] = (math.degrees(math.acos(np.clip(
                float(np.dot(v[:2], gv[:2])) / (nv * ng), -1.0, 1.0)))
                if ng > 0.25 and nv > 1e-6 else float('nan'))
        else:
            row.update(gt_speed=float('nan'), trk_speed=float('nan'),
                       vel_err=float('nan'), dir_err=float('nan'))
        rows.append(row)
    return rows, name


def write_csv(rows, path):
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figure(rows, title, out_path):
    a = {k: np.array([r[k] for r in rows], dtype=float)
         for k in rows[0] if k != 'track_id'}
    ids = [r['track_id'] for r in rows]
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    fig.suptitle(title, fontsize=12)

    # 1. trajectories, the "does it follow the car" view
    p = ax[0][0]
    p.plot(a['gt_x'], a['gt_y'], '-', lw=2, label='ground truth', color='#1a7f37')
    p.plot(a['trk_x'], a['trk_y'], '-', lw=1, alpha=.85, label='tracked',
           color='#c0392b')
    p.plot(a['gt_x'][0], a['gt_y'][0], 'o', ms=7, color='#1a7f37')
    p.set_title('trajectory (map frame)'); p.set_xlabel('x [m]')
    p.set_ylabel('y [m]'); p.axis('equal'); p.legend(); p.grid(alpha=.3)

    # 2. error over time, with track changes marked
    p = ax[0][1]
    p.plot(a['t'], a['err_norm'], lw=1, color='#c0392b')
    for i in range(1, len(ids)):
        if ids[i] != ids[i - 1]:
            p.axvline(a['t'][i], color='#888', ls=':', lw=1)
    p.axhline(float(np.median(a['err_norm'])), color='#333', ls='--', lw=1,
              label=f"median {np.median(a['err_norm']):.2f} m")
    p.set_title('position error over time  (dotted = track id change)')
    p.set_xlabel('t [s]'); p.set_ylabel('|error| [m]'); p.legend(); p.grid(alpha=.3)
    floor_ylim(p, 0.02, centre=0.01)

    # 3. error vs range — the dependency summaries hide
    #
    # Settling samples are excluded. A freshly-created track spends ~0.25 s
    # converging from zero velocity, and that transient lands at whatever range
    # the pass happened to begin at — the far end for a lateral run, the near
    # end for a receding one. Left in, it masquerades as a range dependence
    # pointing in whichever direction the trajectory started from.
    p = ax[0][2]
    settling = np.array([bool(r.get('settling', False)) for r in rows])
    ok = np.isfinite(a['gt_range']) & ~settling
    if ok.any():
        p.scatter(a['gt_range'][ok], a['err_norm'][ok], s=8, alpha=.4,
                  color='#c0392b')
        lo, hi = np.nanmin(a['gt_range'][ok]), np.nanmax(a['gt_range'][ok])
        edges = np.linspace(lo, hi, 9)
        mid = 0.5 * (edges[:-1] + edges[1:])
        med = [np.nanmedian(a['err_norm'][ok][(a['gt_range'][ok] >= edges[i])
                                              & (a['gt_range'][ok] < edges[i + 1])])
               for i in range(len(edges) - 1)]
        p.plot(mid, med, '-o', color='#1a1a1a', lw=2, ms=4, label='median')
        if settling.any():
            p.scatter(a['gt_range'][settling], a['err_norm'][settling], s=10,
                      alpha=.5, color='#999', marker='x',
                      label=f'settling ({int(settling.sum())})')
        p.legend(fontsize=8)
    p.set_title('position error vs range'); p.set_xlabel('true range from ego [m]')
    p.set_ylabel('|error| [m]'); p.grid(alpha=.3)
    floor_ylim(p, 0.02, centre=0.01)

    # 4. per-axis error, to separate a fixed offset from scatter
    p = ax[1][0]
    for k, c, lb in (('err_x', '#c0392b', 'x'), ('err_y', '#1f6feb', 'y'),
                     ('err_z', '#8250df', 'z')):
        p.plot(a['t'], a[k], lw=1, color=c, alpha=.85,
               label=f'{lb}  bias {np.nanmean(a[k]):+.2f} m')
    p.axhline(0, color='#333', lw=.8)
    p.set_title('error by axis  (offset vs scatter)')
    p.set_xlabel('t [s]'); p.set_ylabel('error [m]'); p.legend(); p.grid(alpha=.3)
    floor_ylim(p, 0.02, centre=0.0)

    # 5. speed
    p = ax[1][1]
    p.plot(a['t'], a['gt_speed'], lw=2, color='#1a7f37', label='ground truth')
    p.plot(a['t'], a['trk_speed'], lw=1, color='#c0392b', alpha=.85, label='tracked')
    p.set_title('speed'); p.set_xlabel('t [s]'); p.set_ylabel('|v| [m/s]')
    p.legend(); p.grid(alpha=.3)
    floor_ylim(p, 0.2, centre=float(np.nanmean(a['gt_speed']))
               if np.isfinite(a['gt_speed']).any() else None)

    # 6. velocity direction error, only where the target actually moves
    p = ax[1][2]
    d = a['dir_err']; ok = np.isfinite(d)
    if ok.any():
        p.plot(a['t'][ok], d[ok], '.', ms=3, color='#c0392b')
        p.axhline(float(np.median(d[ok])), color='#333', ls='--', lw=1,
                  label=f'median {np.median(d[ok]):.0f}°')
        p.axhline(90, color='#888', ls=':', lw=1, label='90° = no better than random')
        p.set_ylim(0, 180); p.legend()
    p.set_title('velocity direction error  (moving samples only)')
    p.set_xlabel('t [s]'); p.set_ylabel('angle [deg]'); p.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ego_bag')
    ap.add_argument('target_bag', nargs='?')
    ap.add_argument('-o', '--out', default=None, help='figure path (.png)')
    ap.add_argument('--csv', default=None)
    ap.add_argument('--replay', action='store_true',
                    help='score the current code instead of the recorded output')
    ap.add_argument('--lever', default='auto',
                    help="reference-point offset to remove: 'auto' (fit this "
                         "run), 'none', or 'fwd,lat,vert' in the target's body "
                         "frame. Use a fixed value to compare runs.")
    ap.add_argument('--label', default=None, help='scenario name for the title')
    args = ap.parse_args()

    rows, gt_name = collect(args.ego_bag, args.target_bag, args.replay)
    if not rows:
        raise SystemExit('no overlapping samples between the bags')
    raw = np.array([r['err_norm'] for r in rows])
    rows, lever = apply_lever(rows, args.lever)

    base = os.path.basename(args.ego_bag.rstrip('/'))
    # Default under the workspace rather than /tmp: these are results worth
    # comparing across runs, and /tmp is cleared on reboot.
    outdir = os.path.expanduser(os.environ.get('EVAL_OUT_DIR',
                                               '~/ros2_ws/eval_results'))
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, f'{base}_eval.png')
    csv_path = args.csv or os.path.join(outdir, f'{base}_eval.csv')
    write_csv(rows, csv_path)
    lev = (f"  |  lever removed fwd={lever[0]:+.3f} lat={lever[1]:+.3f} m"
           if lever is not None else '  |  raw, no lever removed')
    figure(rows, f"{args.label or base}  vs  ground truth ({gt_name})"
                 f"{'  [replay]' if args.replay else ''}{lev}", out)

    e = np.array([r['err_norm'] for r in rows])
    print(f"  samples : {len(rows)}")
    if lever is not None:
        print(f"  lever   : fwd={lever[0]:+.3f} lat={lever[1]:+.3f} "
              f"vert={lever[2]:+.3f} m  (removed)")
        print(f"  |err|   : raw rms={math.sqrt(float((raw**2).mean())):.3f}"
              f"  ->  corrected rms={math.sqrt(float((e**2).mean())):.3f} m")
    print(f"  |err|   : mean={e.mean():.3f}  median={np.median(e):.3f}  "
          f"rms={math.sqrt(float((e**2).mean())):.3f}  max={e.max():.3f} m")
    print(f"  csv     : {csv_path}")
    print(f"  figure  : {out}")


if __name__ == '__main__':
    main()
