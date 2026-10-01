#!/usr/bin/env python3
"""A/B the tracker across the four controlled scenarios, with plots.

The pooled regression over every bag mixes freeform driving, the two avoidance
runs and the scripted scenarios, and its median hides the thing that matters:
the error is strongly scenario-dependent. Approaching and receding move along
the camera's depth axis, where the localizer is noisiest; lateral and reverse
lateral move across it, where it is not. A change that helps one and hurts the
other shows up as "no change" in the pooled number.

So this scores the four scripted runs separately, under two configs, and plots
them side by side. Every metric that matters downstream is included: filtered
position error, the 1 s rollout error that avoidance actually consumes, and the
false-motion rate that the position metrics cannot see.

    scenario_regression.py
    scenario_regression.py --a "q_accel=0.5" --b "q_accel=0.1"
"""

import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_lib as E                       # noqa: E402
import evaluate_tracker as T               # noqa: E402
import predict_eval as P                   # noqa: E402

# Ordered so the two depth-axis runs sit together, then the two cross-axis
# ones — the split the anisotropy argument turns on.
SCENARIOS = [
    ('approaching',     'ego_approaching_164730',    'target_approaching_164735'),
    ('receding',        'ego_receding_164958',       'target_receding_164954'),
    ('lateral',         'ego_lateral_165157',        'target_lateral_165152'),
    ('reverse lateral', 'ego_reverselateral_165310', 'target_reverselateral_165306'),
]

STILL, MOVE = 0.05, 0.25          # gt speed thresholds, m/s


def score(ego, gt, cfg):
    """Every scenario metric under one config."""
    frames = T.frames_replayed(ego, cfg)
    err, gs, ts, raw = [], [], [], []
    for t, items in frames:
        g = gt['pos'].at(t)
        gv = E.gt_velocity(gt, t)
        if g is None or not items:
            continue
        _, p, v = min(items, key=lambda it: np.linalg.norm(it[1] - g))
        err.append(float(np.linalg.norm(p - g)))
        if gv is not None:
            gs.append(float(np.linalg.norm(gv[:2])))
            ts.append(float(np.linalg.norm(v[:2])))
    for msg in ego.get(E.DET3D, []):
        t = E.stamp(msg.header)
        g = gt['pos'].at(t)
        ds = E.detections_of(msg)
        if g is None or not ds:
            continue
        raw.append(float(np.linalg.norm(
            min([d[1] for d in ds], key=lambda q: np.linalg.norm(q - g)) - g)))
    if not err:
        return None
    err, gs, ts, raw = map(np.array, (err, gs, ts, raw))
    st = gs < STILL if len(gs) else np.zeros(0, bool)

    rolls = P.rollouts(ego, gt, cfg, steps=10, step_dt=0.1, period=0.3,
                       source='replay')
    fde = np.array([float(np.linalg.norm(pr[-1] - tr[-1]))
                    for _, pr, tr, _ in rolls]) if rolls else np.zeros(0)
    return dict(
        err=err, raw=raw, fde=fde,
        pos_rms=math.sqrt(float((err ** 2).mean())),
        raw_rms=math.sqrt(float((raw ** 2).mean())) if len(raw) else float('nan'),
        fde_med=float(np.median(fde)) if len(fde) else float('nan'),
        fde_p90=float(np.percentile(fde, 90)) if len(fde) else float('nan'),
        false_v=(100.0 * float((ts[st] > MOVE).mean())
                 if st.any() else float('nan')),
        phantom=(float(np.percentile(ts[st], 90))
                 if st.any() else float('nan')),
        n=len(err))


def bars(ax, labels, a, b, key, title, ylabel, la, lb, lower_better=True):
    x = np.arange(len(labels))
    w = 0.38
    va = [d[key] if d else np.nan for d in a]
    vb = [d[key] if d else np.nan for d in b]
    ax.bar(x - w / 2, va, w, label=la, color='#8c8c8c')
    ax.bar(x + w / 2, vb, w, label=lb, color='#1a7f37')
    for i, (p, q) in enumerate(zip(va, vb)):
        if not (np.isfinite(p) and np.isfinite(q)) or p == 0:
            continue
        d = 100.0 * (q - p) / p
        good = (d < 0) if lower_better else (d > 0)
        ax.annotate(f'{d:+.0f}%', (i + w / 2, q), ha='center', va='bottom',
                    fontsize=8.5, color='#1a7f37' if good else '#c0392b')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=10.5)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=.3, axis='y')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', default='q_accel=0.5', help='baseline config')
    ap.add_argument('--b', default='q_accel=0.1', help='candidate config')
    ap.add_argument('--la', default='baseline')
    ap.add_argument('--lb', default='candidate')
    ap.add_argument('--root', default='/home/af/ros2_ws')
    ap.add_argument('-o', '--out', default=None)
    args = ap.parse_args()

    cfg_a, cfg_b = T.parse_cfg(args.a), T.parse_cfg(args.b)
    outdir = os.path.expanduser(os.environ.get('EVAL_OUT_DIR',
                                               '~/ros2_ws/eval_results'))
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, 'scenario_regression.png')

    labels, A, B = [], [], []
    hdr = (f"{'scenario':<17}{'n':>5}{'raw':>8}"
           f"{'posRMS_a':>10}{'posRMS_b':>10}"
           f"{'FDE_a':>8}{'FDE_b':>8}{'falseV_a':>10}{'falseV_b':>10}")
    print(f'\nA = {args.la}   ({args.a})')
    print(f'B = {args.lb}   ({args.b})\n')
    print(hdr)
    print('-' * len(hdr))
    for lab, e, t in SCENARIOS:
        ep, tp = os.path.join(args.root, e), os.path.join(args.root, t)
        if not os.path.isdir(ep):
            print(f'{lab:<17} missing {e}')
            continue
        ego = E.read_bag(ep)
        tgt = E.read_bag(tp) if os.path.isdir(tp) else None
        gts = E.gt_objects(ego, tgt)
        name = T.pick_primary(gts)
        if name is None:
            print(f'{lab:<17} no ground truth')
            continue
        gt = gts[name]
        sa, sb = score(ego, gt, cfg_a), score(ego, gt, cfg_b)
        if sa is None or sb is None:
            print(f'{lab:<17} no scorable frames')
            continue
        labels.append(lab)
        A.append(sa)
        B.append(sb)
        print(f'{lab:<17}{sa["n"]:>5}{sa["raw_rms"]:>8.3f}'
              f'{sa["pos_rms"]:>10.3f}{sb["pos_rms"]:>10.3f}'
              f'{sa["fde_med"]:>8.3f}{sb["fde_med"]:>8.3f}'
              f'{sa["false_v"]:>10.0f}{sb["false_v"]:>10.0f}')

    if not labels:
        raise SystemExit('no scenarios scored')

    ea = np.concatenate([d['err'] for d in A])
    eb = np.concatenate([d['err'] for d in B])
    ra = np.concatenate([d['raw'] for d in A])
    fa = np.concatenate([d['fde'] for d in A])
    fb = np.concatenate([d['fde'] for d in B])
    print('-' * len(hdr))
    print(f'{"POOLED":<17}{len(ea):>5}{math.sqrt((ra**2).mean()):>8.3f}'
          f'{math.sqrt((ea**2).mean()):>10.3f}{math.sqrt((eb**2).mean()):>10.3f}'
          f'{np.median(fa):>8.3f}{np.median(fb):>8.3f}')
    # Gain is the number that has refused to move all session: the filter has
    # been sitting at pass-through (0.98-1.01) because a misspecified R gave it
    # nothing to work with.
    print(f'\ngain (raw rms / tracked rms)   A {math.sqrt((ra**2).mean())/math.sqrt((ea**2).mean()):.3f}'
          f'   B {math.sqrt((ra**2).mean())/math.sqrt((eb**2).mean()):.3f}')

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle(f'scenario regression — {args.la}  vs  {args.lb}', fontsize=13)
    bars(ax[0][0], labels, A, B, 'pos_rms',
         'filtered position error', 'rms [m]', args.la, args.lb)
    bars(ax[0][1], labels, A, B, 'fde_med',
         '1 s prediction error (what avoidance consumes)', 'median FDE [m]',
         args.la, args.lb)
    bars(ax[1][0], labels, A, B, 'false_v',
         'false-motion rate (stationary target reported moving)',
         '% of stationary samples', args.la, args.lb)
    p = ax[1][1]
    for arr, c, lb_ in ((ea, '#8c8c8c', args.la), (eb, '#1a7f37', args.lb)):
        p.plot(np.sort(arr), np.linspace(0, 100, len(arr)), lw=2, color=c,
               label=f'{lb_}  median {np.median(arr):.3f} m')
    p.plot(np.sort(ra), np.linspace(0, 100, len(ra)), lw=1.4, ls='--',
           color='#c0392b', label=f'raw detections  median {np.median(ra):.3f} m')
    p.set_title('pooled error distribution', fontsize=10.5)
    p.set_xlabel('|error| [m]')
    p.set_ylabel('% of samples below')
    p.legend(fontsize=8.5)
    p.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=110)
    print(f'\n  figure: {out}')


if __name__ == '__main__':
    main()
