#!/usr/bin/env python3
"""Overlay several recorded runs so scenario-dependent effects are visible.

A single run cannot separate "the error grows with range" from "the error
depends on which face of the target the camera sees" — the two are confounded
whenever range and aspect change together, which they do in any normal drive.
Comparing runs that vary one and hold the other apart is what settles it.

    compare_scenarios.py ego1:target1 ego2:target2 ... [-o out.png]
    compare_scenarios.py --auto        # pair ego_<name>_* with target_<name>_*

Errors are reported RAW. Per-run bias fitting is deliberately not applied here:
it differs per run, so it would rescale each curve by a different amount and
destroy exactly the comparison this tool exists to make.
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_lib as E                       # noqa: E402
import evaluate_tracker as T               # noqa: E402

COLORS = ['#c0392b', '#1a7f37', '#1f6feb', '#8250df', '#d97706', '#0891b2']


def series(ego_path, target_path):
    """Per-sample (range, bearing, |err|, speed, dir_err) for one run."""
    ego = E.read_bag(ego_path)
    tgt = E.read_bag(target_path) if target_path else None
    gts = E.gt_objects(ego, tgt)
    name = T.pick_primary(gts)
    if name is None:
        return None
    gt = gts[name]
    ep, ey = E.ego_pose(ego)
    out = []
    for t, items in T.frames_recorded(ego):
        g = gt['pos'].at(t)
        if g is None or not items:
            continue
        _, p, v = min(items, key=lambda it: np.linalg.norm(it[1] - g))
        e = ep.at(t) if ep is not None else None
        y = ey.at(t) if ey is not None else None
        if e is None or y is None:
            continue
        dx, dy = g[0] - e[0], g[1] - e[1]
        rng = math.hypot(dx, dy)
        bearing = math.degrees(E.wrap(math.atan2(dy, dx) - float(y)))
        gv = E.gt_velocity(gt, t)
        sp = float(np.linalg.norm(gv[:2])) if gv is not None else float('nan')
        de = float('nan')
        if gv is not None:
            ng, nv = np.linalg.norm(gv[:2]), np.linalg.norm(v[:2])
            if ng > 0.25 and nv > 1e-6:
                de = math.degrees(math.acos(np.clip(
                    float(np.dot(v[:2], gv[:2])) / (nv * ng), -1.0, 1.0)))
        out.append((rng, bearing, float(np.linalg.norm(p - g)), sp, de))
    return np.array(out) if out else None


def binned(x, y, edges):
    m, c = [], []
    for i in range(len(edges) - 1):
        s = (x >= edges[i]) & (x < edges[i + 1])
        if s.sum() >= 5:
            m.append(0.5 * (edges[i] + edges[i + 1]))
            c.append(float(np.median(y[s])))
    return np.array(m), np.array(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='*', help='ego:target')
    ap.add_argument('--auto', action='store_true')
    ap.add_argument('--root', default='/home/af/ros2_ws')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    runs = []
    if args.auto:
        for e in sorted(glob.glob(os.path.join(args.root, 'ego_*_*'))):
            m = re.match(r'ego_([A-Za-z]+)_\d+$', os.path.basename(e))
            if not m:
                continue
            tg = glob.glob(os.path.join(args.root, f'target_{m.group(1)}_*'))
            if tg:
                runs.append((m.group(1), e, sorted(tg)[0]))
    else:
        for p in args.pairs:
            e, _, t = p.partition(':')
            runs.append((os.path.basename(e), e, t))
    if not runs:
        raise SystemExit('no runs found (try --auto)')

    outdir = os.path.expanduser(os.environ.get('EVAL_OUT_DIR',
                                               '~/ros2_ws/eval_results'))
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, 'scenario_comparison.png')

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    fig.suptitle('scenario comparison — raw error, no per-run bias removed',
                 fontsize=12)
    print(f"{'scenario':<18}{'n':>5}{'rms':>8}{'median':>9}{'range':>8}"
          f"{'|bearing|':>11}{'dir med':>9}")
    print('-' * 68)
    for i, (label, e, t) in enumerate(runs):
        a = series(e, t)
        if a is None:
            print(f'{label:<18} no overlapping samples')
            continue
        rng, bear, err, spd, de = a.T
        c = COLORS[i % len(COLORS)]
        ax[0].scatter(rng, err, s=7, alpha=.25, color=c)
        mx, my = binned(rng, err, np.linspace(1.0, 5.0, 11))
        if len(mx):
            ax[0].plot(mx, my, '-o', color=c, lw=2, ms=4, label=label)
        mx, my = binned(np.abs(bear), err, np.linspace(0, 35, 8))
        if len(mx):
            ax[1].plot(mx, my, '-o', color=c, lw=2, ms=4, label=label)
        ok = np.isfinite(de)
        if ok.sum() > 3:
            ax[2].plot(np.sort(de[ok]),
                       np.linspace(0, 100, ok.sum()), color=c, lw=2, label=label)
        dm = np.nanmedian(de) if ok.any() else float('nan')
        print(f'{label:<18}{len(err):>5}{math.sqrt(float((err**2).mean())):>8.3f}'
              f'{float(np.median(err)):>9.3f}{float(np.mean(rng)):>8.2f}'
              f'{float(np.mean(np.abs(bear))):>11.1f}{dm:>9.1f}')

    ax[0].set_title('position error vs range')
    ax[0].set_xlabel('true range from ego [m]'); ax[0].set_ylabel('|error| [m]')
    ax[1].set_title('position error vs |bearing|')
    ax[1].set_xlabel('|bearing| off boresight [deg]'); ax[1].set_ylabel('|error| [m]')
    ax[2].set_title('velocity direction error (cumulative)')
    ax[2].set_xlabel('angle [deg]'); ax[2].set_ylabel('% of samples below')
    ax[2].set_xlim(0, 180)
    for a_ in ax:
        a_.grid(alpha=.3); a_.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=110)
    print(f'\n  figure: {out}')


if __name__ == '__main__':
    main()
