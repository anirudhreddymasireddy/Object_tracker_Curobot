#!/usr/bin/env python3
"""Score the tracker's PREDICTIONS, not just its filtered position.

Every metric in evaluate_tracker.py asks "where does the tracker think the car
is right now". Nothing downstream consumes that alone. pure_persuit_node reads
`avoidance_predict_t` (1.0 s) and steers around where the obstacle is going to
be, so the number that decides whether the ego hits the car is the error of a
1 s rollout — and that error is dominated by velocity, which absolute position
error barely tests. A tracker can sit 30 cm off the truth with a perfect
velocity and avoid cleanly, or sit on the truth with velocity pointing the
wrong way and drive straight into it.

So: at each anchor time the filter is copied and run open-loop (predict only,
no measurements) for `--steps` steps of `--step-dt`, and every predicted step
is scored against ground truth at that future instant. The standard motion-
forecasting pair falls out:

    ADE  average displacement error, mean over the whole horizon
    FDE  final displacement error, at the last step — what avoidance uses

Rollouts start from the live Track objects, so the copied filter carries its
real covariance. Reconstructing a state from the published position and
velocity would drop P.

    predict_eval.py <ego_bag> [target_bag]
    predict_eval.py <ego> <tgt> --horizon 2.0 --anchor-period 0.5
    predict_eval.py <ego> <tgt> --source recorded    # constant-velocity rollout
    predict_eval.py --regression                     # every pair, one table
"""

import argparse
import copy
import csv
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


def rollouts(ego, gt, cfg, *, steps, step_dt, period, source='replay'):
    """[(t0, pred(steps,3), truth(steps,3), filt_err)] — one per anchor.

    `filt_err` is the error of the filtered estimate at the anchor itself, kept
    alongside so the horizon curve can be read against the zero-lookahead
    baseline. Without it there is no way to tell a rollout that degrades from
    one that merely inherits a bad starting point.
    """
    out, next_anchor = [], None

    def take(t, pos, vel, kf=None):
        nonlocal next_anchor
        if next_anchor is not None and t < next_anchor:
            return
        g0 = gt['pos'].at(t)
        if g0 is None:
            return
        # Ground truth must exist across the whole horizon, or the rollout is
        # scored against extrapolated nonsense at exactly the steps that matter
        # most. Anchors near the end of the bag are dropped for this reason.
        truth = [gt['pos'].at(t + (k + 1) * step_dt) for k in range(steps)]
        if any(g is None for g in truth):
            return
        if kf is not None:
            k = copy.deepcopy(kf)
            pred = []
            for _ in range(steps):
                k.predict(step_dt)
                pred.append(np.array(k.position, dtype=float))
        else:
            # No filter available (recorded source): the published position and
            # velocity are all there is, so this is a plain constant-velocity
            # rollout — exact, since the filter is constant-velocity too.
            pred = [pos + vel * ((k + 1) * step_dt) for k in range(steps)]
        out.append((t, np.array(pred), np.array(truth),
                    float(np.linalg.norm(pos - g0))))
        next_anchor = t + period

    if source == 'replay':
        def hook(t, tracks):
            if not tracks:
                return
            g = gt['pos'].at(t)
            if g is None:
                return
            tr = min(tracks, key=lambda x: np.linalg.norm(x.kf.position - g))
            take(t, np.array(tr.kf.position, dtype=float),
                 np.array(tr.kf.velocity, dtype=float), kf=tr.kf)
        T.frames_replayed(ego, cfg, hook=hook)
    else:
        for t, items in T.frames_recorded(ego):
            if not items:
                continue
            g = gt['pos'].at(t)
            if g is None:
                continue
            _, p, v = min(items, key=lambda it: np.linalg.norm(it[1] - g))
            take(t, np.asarray(p, dtype=float), np.asarray(v, dtype=float))
    return out


def horizon_stats(rolls, steps, step_dt):
    """Per-step error distribution across every anchor, plus ADE/FDE."""
    if not rolls:
        return None
    err = np.array([np.linalg.norm(p - g, axis=1) for _, p, g, _ in rolls])
    ade = err.mean(axis=1)
    fde = err[:, -1]
    return dict(
        n=len(rolls), err=err,
        lead=np.arange(1, steps + 1) * step_dt,
        med=np.median(err, axis=0),
        p90=np.percentile(err, 90, axis=0),
        mean=err.mean(axis=0),
        ade=ade, fde=fde,
        filt=np.array([f for _, _, _, f in rolls]),
        t=np.array([t for t, _, _, _ in rolls]))


def figure(rolls, st, gt, label, out_path, horizon):
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle(f'{label} — prediction horizon {horizon:.1f} s '
                 f'({st["n"]} rollouts)', fontsize=13)

    # 1. every rollout drawn on the map, against the truth it was aiming at
    p = ax[0][0]
    g = np.array([gt['pos'].at(t) for t in gt['pos'].t
                  if gt['pos'].at(t) is not None])
    p.plot(g[:, 0], g[:, 1], lw=2.5, color='#1a7f37', label='ground truth',
           zorder=1)
    for i, (_, pr, tr, _) in enumerate(rolls):
        p.plot(pr[:, 0], pr[:, 1], lw=1.1, color='#c0392b', alpha=.55,
               zorder=3, label='prediction' if i == 0 else None)
        p.plot(pr[0, 0], pr[0, 1], '.', ms=4, color='#1f1f1f', zorder=4)
    p.set_title('rollout fan  (dot = anchor, line = where it thought it would go)')
    p.set_xlabel('x [m]'); p.set_ylabel('y [m]')
    p.axis('equal'); p.legend(fontsize=9); p.grid(alpha=.3)

    # 2. how error grows with lookahead — the whole point of the exercise
    p = ax[0][1]
    for row in st['err']:
        p.plot(st['lead'], row, lw=.6, color='#c0392b', alpha=.13)
    p.plot(st['lead'], st['med'], '-o', lw=2.2, ms=4, color='#c0392b',
           label='median')
    p.plot(st['lead'], st['p90'], '--', lw=1.6, color='#d97706', label='p90')
    p.axhline(float(np.median(st['filt'])), color='#1f6feb', ls=':', lw=1.6,
              label=f"filtered, 0 s lookahead ({np.median(st['filt']):.2f} m)")
    p.set_title('prediction error vs lookahead')
    p.set_xlabel('lookahead [s]'); p.set_ylabel('|error| [m]')
    p.legend(fontsize=9); p.grid(alpha=.3)

    # 3. ADE and FDE over the run
    p = ax[1][0]
    p.plot(st['t'] - st['t'][0], st['ade'], '-o', ms=3, lw=1.3,
           color='#8250df', label=f"ADE  median {np.median(st['ade']):.2f} m")
    p.plot(st['t'] - st['t'][0], st['fde'], '-o', ms=3, lw=1.3,
           color='#c0392b', label=f"FDE  median {np.median(st['fde']):.2f} m")
    p.plot(st['t'] - st['t'][0], st['filt'], lw=1.2, color='#1f6feb',
           alpha=.8, label=f"filtered  median {np.median(st['filt']):.2f} m")
    p.set_title('displacement error per anchor')
    p.set_xlabel('t [s]'); p.set_ylabel('|error| [m]')
    p.legend(fontsize=9); p.grid(alpha=.3)

    # 4. what the horizon costs on top of the filtered estimate
    #
    # Plotted as a difference because the absolute FDE is mostly inherited
    # bias: a rollout starting 0.35 m off cannot end closer than that except by
    # luck. The growth is the part the tracker's velocity is responsible for.
    p = ax[1][1]
    grow = st['fde'] - st['filt']
    p.hist(grow, bins=max(8, min(30, st['n'] // 3)), color='#c0392b',
           alpha=.75, edgecolor='#7d1f14')
    p.axvline(0, color='#1a1a1a', lw=1.2)
    p.axvline(float(np.median(grow)), color='#1a7f37', ls='--', lw=1.8,
              label=f'median {np.median(grow):+.2f} m')
    p.set_title(f'FDE minus filtered error  '
                f'({100.0 * float((grow < 0).mean()):.0f}% of rollouts beat it)')
    p.set_xlabel(f'extra error from predicting {horizon:.1f} s ahead [m]')
    p.set_ylabel('rollouts'); p.legend(fontsize=9); p.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def write_csv(rolls, step_dt, path):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['anchor_t', 'step', 'lookahead', 'pred_x', 'pred_y',
                    'gt_x', 'gt_y', 'err', 'filt_err_at_anchor'])
        t0 = rolls[0][0]
        for t, pr, tr, fe in rolls:
            for k in range(len(pr)):
                w.writerow([f'{t - t0:.3f}', k + 1, f'{(k + 1) * step_dt:.2f}',
                            f'{pr[k][0]:.4f}', f'{pr[k][1]:.4f}',
                            f'{tr[k][0]:.4f}', f'{tr[k][1]:.4f}',
                            f'{np.linalg.norm(pr[k] - tr[k]):.4f}',
                            f'{fe:.4f}'])


def evaluate(ego_path, target_path, args, outdir, label=None):
    ego = E.read_bag(ego_path)
    tgt = E.read_bag(target_path) if target_path else None
    gts = E.gt_objects(ego, tgt)
    name = T.pick_primary(gts)
    if name is None:
        return None
    label = label or os.path.basename(ego_path)
    steps = max(1, int(round(args.horizon / args.step_dt)))
    rolls = rollouts(ego, gts[name], T.default_cfg(), steps=steps,
                     step_dt=args.step_dt, period=args.anchor_period,
                     source=args.source)
    st = horizon_stats(rolls, steps, args.step_dt)
    if st is None:
        return None
    if not args.no_plot:
        figure(rolls, st, gts[name], label,
               os.path.join(outdir, f'pred_{label}.png'), args.horizon)
        write_csv(rolls, args.step_dt,
                  os.path.join(outdir, f'pred_{label}.csv'))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ego', nargs='?')
    ap.add_argument('target', nargs='?')
    ap.add_argument('--horizon', type=float, default=1.0,
                    help='seconds to predict ahead (default 1.0, matching '
                         'pure_persuit_node avoidance_predict_t)')
    ap.add_argument('--step-dt', type=float, default=0.1,
                    help='rollout step, seconds (default 0.1 = tracker rate)')
    ap.add_argument('--anchor-period', type=float, default=1.0,
                    help='seconds between rollouts (default 1.0)')
    ap.add_argument('--source', choices=('replay', 'recorded'),
                    default='replay')
    ap.add_argument('--regression', action='store_true')
    ap.add_argument('--root', default='/home/af/ros2_ws')
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    outdir = os.path.expanduser(os.environ.get('EVAL_OUT_DIR',
                                               '~/ros2_ws/eval_results'))
    os.makedirs(outdir, exist_ok=True)
    steps = max(1, int(round(args.horizon / args.step_dt)))

    hdr = (f"{'run':<26}{'n':>4}{'filt':>8}{'ADE':>8}{'FDE':>8}"
           f"{'FDE-filt':>10}{'win%':>7}")
    print(f'\nhorizon {args.horizon:.1f} s = {steps} x {args.step_dt:.2f} s, '
          f'anchor every {args.anchor_period:.1f} s, source {args.source}\n')
    print(hdr); print('-' * len(hdr))

    if args.regression:
        rows = []
        for e, t in T.discover_pairs(args.root):
            lab = os.path.basename(e.rstrip('/'))
            st = evaluate(e, t, args, outdir, label=lab)
            if st is None:
                print(f'{lab:<26} no scorable rollouts')
                continue
            rows.append(st)
            g = st['fde'] - st['filt']
            print(f'{lab:<26}{st["n"]:>4}{np.median(st["filt"]):>8.3f}'
                  f'{np.median(st["ade"]):>8.3f}{np.median(st["fde"]):>8.3f}'
                  f'{np.median(g):>10.3f}{100.0 * float((g < 0).mean()):>7.0f}')
        if rows:
            af = np.concatenate([r['filt'] for r in rows])
            aa = np.concatenate([r['ade'] for r in rows])
            ad = np.concatenate([r['fde'] for r in rows])
            print('-' * len(hdr))
            print(f'{"ALL":<26}{len(af):>4}{np.median(af):>8.3f}'
                  f'{np.median(aa):>8.3f}{np.median(ad):>8.3f}'
                  f'{np.median(ad - af):>10.3f}'
                  f'{100.0 * float(((ad - af) < 0).mean()):>7.0f}')
    else:
        if not args.ego:
            raise SystemExit('need an ego bag, or --regression')
        lab = os.path.basename(args.ego.rstrip('/'))
        st = evaluate(args.ego, args.target, args, outdir, label=lab)
        if st is None:
            raise SystemExit('no scorable rollouts')
        g = st['fde'] - st['filt']
        print(f'{lab:<26}{st["n"]:>4}{np.median(st["filt"]):>8.3f}'
              f'{np.median(st["ade"]):>8.3f}{np.median(st["fde"]):>8.3f}'
              f'{np.median(g):>10.3f}{100.0 * float((g < 0).mean()):>7.0f}')
        print('\nerror by lookahead:')
        for l, m, q in zip(st['lead'], st['med'], st['p90']):
            print(f'  {l:4.1f} s   median {m:6.3f}   p90 {q:6.3f}')
        if not args.no_plot:
            print(f'\n  figure: {outdir}/pred_{lab}.png')
            print(f'  csv:    {outdir}/pred_{lab}.csv')

    print('\n  filt     = filtered error at the anchor (0 s lookahead)')
    print('  ADE/FDE  = average / final displacement error over the horizon')
    print('  win%     = rollouts whose FDE beat the filtered estimate')


if __name__ == '__main__':
    main()
