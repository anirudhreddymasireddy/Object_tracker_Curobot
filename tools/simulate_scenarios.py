#!/usr/bin/env python3
"""Synthesise object_projection_3d output for the four scenarios and score it.

Recorded runs cannot isolate the tracker: ground truth carries its own error,
the detector drops frames unevenly, and a single pass yields only tens of
samples (the approaching run had 19). Here the trajectory is exact, the sample
count is whatever you ask for, and the noise is the one thing that varies — so
any error in the result is the tracker's.

The noise model is calibrated from the recorded bags, not invented:

  * magnitude   sigma = sqrt(R) with R = [0.03, 0.03, 0.0015] m^2, measured as
                the scatter of detections3d about ground truth
  * COLOUR      lag-1 autocorrelation 0.8, measured at 0.77-0.81. This matters
                more than the magnitude: white noise of the same size would let
                the Kalman filter average it away, and the real detector's does
                not, which is why every q/R sweep failed to beat passing the
                measurement through. Modelled as AR(1).
  * range       error grows with distance (measured 0.07 m at 1.5 m to 0.28 m
                at 3.7 m on the baseline run), applied as a linear scale on
                sigma.
  * dropouts    detections3d misses frames; 13% on the best recorded run.

Ego is static at the origin facing +x, which is how the scenarios were driven.

    simulate_scenarios.py [--speed 0.5] [--rate 8.8] [--passes 4] [--seed 0]
    simulate_scenarios.py --white        # uncorrelated noise, for contrast
"""

import argparse
import math
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_tracker as T                 # noqa: E402
import plot_eval as PE                       # noqa: E402

from vision_msgs.msg import (Detection3D, Detection3DArray,      # noqa: E402
                             ObjectHypothesisWithPose)

# Fallbacks only. The simulator calibrates these from a real bag by default —
# guessing them is what made the first version wrong: rho was set to 0.8 when
# the data says 0.94, which doubled the simulated velocity noise (0.93 vs the
# measured 0.52 m/s) and inflated direction error from ~7 deg to ~27 deg.
SIGMA_FB = np.array([0.196, 0.169, 0.032])   # m, from ego_162135
RHO_FB = np.array([0.94, 0.95, 0.90])        # lag-1, per axis
BIAS_FB = np.array([-0.076, 0.127, -0.058])  # m
NEAR_R, FAR_R = 1.6, 5.2                     # usable envelope, m
# A new track converges from zero velocity in ~0.25 s (measured with perfect
# input); 0.5 s is a safe window to treat as transient.
SETTLE_S = 0.5

# Unscored lead-in, in seconds, before the segment that gets plotted. The
# birth transient decays with a time constant near 0.5 s, so 0.5 s of warm-up
# leaves ~2 mm of it — invisible physically, but the error panel autoscales to
# whatever range the data spans, so 2 mm gets magnified into what looks like a
# spike. Four time constants puts the residual below 10 um and the panel stays
# flat.
WARMUP_S = 2.0
CAL_EGO = '/home/af/ros2_ws/ego_162135'
CAL_TGT = '/home/af/ros2_ws/target_162152'


class Calibration:
    """Detection-noise statistics measured from a recorded run.

    Everything the simulator needs comes from data rather than assumption:
    magnitude, per-axis lag-1 correlation, the standing bias, and the raw
    residual sequence itself for replay mode.
    """

    def __init__(self, sigma, rho, bias, residuals, dt, src='fallback'):
        self.sigma = np.asarray(sigma, float)
        self.rho = np.clip(np.asarray(rho, float), 0.0, 0.999)
        self.bias = np.asarray(bias, float)
        self.residuals = residuals
        self.dt = dt
        self.src = src

    def describe(self):
        r = self.residuals
        return (f"calibration [{self.src}]  sigma="
                f"({self.sigma[0]:.3f},{self.sigma[1]:.3f},{self.sigma[2]:.3f}) m"
                f"  rho=({self.rho[0]:.2f},{self.rho[1]:.2f},{self.rho[2]:.2f})"
                f"  bias=({self.bias[0]:+.3f},{self.bias[1]:+.3f},"
                f"{self.bias[2]:+.3f}) m"
                f"  residual samples={0 if r is None else len(r)}")


def calibrate(ego_path=CAL_EGO, tgt_path=CAL_TGT):
    """Measure the detection-noise process from a recorded bag pair."""
    try:
        import eval_lib as EL
        ego = EL.read_bag(ego_path); tgt = EL.read_bag(tgt_path)
        g = EL.gt_objects(ego, tgt)['target']
        ts, errs = [], []
        for m in ego[EL.DET3D]:
            t = EL.stamp(m.header); gp = g['pos'].at(t)
            if gp is None or not m.detections:
                continue
            p = min((d[1] for d in EL.detections_of(m)),
                    key=lambda q: np.linalg.norm(q - gp))
            ts.append(t); errs.append(p - gp)
        if len(errs) < 50:
            raise ValueError('too few samples')
        ERR = np.asarray(errs)
        dt = float(np.median(np.diff(np.sort(np.asarray(ts)))))
        bias = ERR.mean(axis=0)
        D = ERR - bias
        rho = np.array([float(np.corrcoef(D[:-1, i], D[1:, i])[0, 1])
                        for i in range(3)])
        return Calibration(D.std(axis=0), rho, bias, D, dt,
                           src=os.path.basename(ego_path))
    except Exception as exc:                       # noqa: BLE001
        print(f"  (calibration from bags failed: {exc}; using fallback)")
        return Calibration(SIGMA_FB, RHO_FB, BIAS_FB, None, 0.133)


def noise_sequence(n, mode, cal, rng):
    """(n, 3) detection error sequence.

    'replay'  circular block bootstrap of the REAL residuals. Blocks are long
              (~2 s) so the multi-second drift the data shows — lag-20
              autocorrelation is still 0.71-0.77 — survives resampling. This is
              the honest option: it reproduces correlation, magnitude and any
              non-Gaussian structure without a model.
    'ar1'     AR(1) per axis at the measured rho. A model, but a calibrated one.
    'white'   uncorrelated, same magnitude. Only for the contrast that shows
              what the filter could do if the noise were white.
    """
    if mode == 'replay' and cal.residuals is not None:
        res = cal.residuals
        blk = max(4, int(round(2.0 / cal.dt)))
        out = np.empty((n, 3))
        i = 0
        while i < n:
            st = int(rng.integers(0, len(res)))
            take = min(blk, n - i)
            idx = (st + np.arange(take)) % len(res)
            out[i:i + take] = res[idx]
            i += take
        return out
    if mode == 'none':
        return np.zeros((n, 3))
    if mode == 'white':
        return rng.normal(0.0, 1.0, (n, 3)) * cal.sigma
    e = np.zeros(3)
    out = np.empty((n, 3))
    for k in range(n):
        w = rng.normal(0.0, 1.0, 3)
        e = cal.rho * e + np.sqrt(1.0 - cal.rho ** 2) * cal.sigma * w
        out[k] = e
    return out
COLORS = {'approaching': '#c0392b', 'receding': '#1f6feb',
          'lateral': '#1a7f37', 'reverse lateral': '#8250df'}


def trajectory(kind, speed, rate, warmup=WARMUP_S):
    """One continuous constant-velocity run: (t, pos(N,3), yaw(N), i0).

    The path is extended BACKWARD by `warmup` seconds before the segment that
    gets scored, and `i0` is the index where scoring should begin. A track is
    born with velocity zero — a single detection carries no velocity — so its
    first predict() advances position by nothing while the target really moved
    speed*dt. That produces a one-off transient of about speed*dt (0.05 m at
    0.5 m/s and 10 Hz) which decays within ~0.25 s.

    Warming up on unscored samples means the filter is already locked on when
    the measured segment starts, so ground truth and tracked plot as one
    continuous line instead of opening with a spike that is an artefact of
    where the run happened to begin.
    """
    dt = 1.0 / rate
    pre = max(0, int(round(warmup / dt)))
    if kind in ('approaching', 'receding'):
        span = FAR_R - NEAR_R
        n = max(4, int(span / speed / dt))
        step = span / (n - 1)
        s = np.arange(-pre, n) * step                 # negative = warm-up
        if kind == 'receding':
            x, yaw = NEAR_R + s, np.zeros(len(s))     # nose away from ego
        else:
            x, yaw = FAR_R - s, np.full(len(s), math.pi)
        y = np.zeros(len(s))
    else:
        half, rng_fixed = 1.5, 3.0
        n = max(4, int(2 * half / speed / dt))
        step = 2 * half / (n - 1)
        s = -half + np.arange(-pre, n) * step
        if kind == 'reverse lateral':
            s = -s
        x = np.full(len(s), rng_fixed)
        y = s
        yaw = np.full(len(s), math.pi / 2 if kind == 'lateral'
                      else -math.pi / 2)
    t = np.arange(len(x)) * dt
    return t, np.column_stack([x, y, np.full(len(x), 0.20)]), yaw, pre


def synth_detections(t, pos, yaw, rng, cal, mode='replay', drop=0.13,
                     range_scale=None, bias=True):
    """Detection3DArray per sample: truth + calibrated noise, with dropouts."""
    # Replayed residuals already contain the source run's range dependence;
    # scaling them again double-counts it and inflated position rms from ~0.3
    # to ~0.49 m. The parametric modes use a single averaged sigma, so they do
    # need the scaling.
    if range_scale is None:
        range_scale = (mode != 'replay')
    seq = noise_sequence(len(t), mode, cal, rng)
    msgs = []
    for i in range(len(t)):
        r = float(np.linalg.norm(pos[i, :2]))
        # Error grows with range. Normalised so scale = 1 at the calibration
        # run's typical range (~2.5 m); in replay mode the residuals already
        # carry the source run's range dependence, so this only extrapolates
        # beyond where it was measured.
        scale = max(0.35, 1.0 + 0.55 * (r - 2.5)) if range_scale else 1.0
        e = seq[i] * scale + (cal.bias if bias else 0.0)
        if rng.random() < drop:
            msgs.append((t[i], None))
            continue
        p = pos[i] + e
        d = Detection3D()
        d.bbox.center.position.x = float(p[0])
        d.bbox.center.position.y = float(p[1])
        d.bbox.center.position.z = float(p[2])
        d.bbox.center.orientation.z = float(math.sin(yaw[i] / 2.0))
        d.bbox.center.orientation.w = float(math.cos(yaw[i] / 2.0))
        d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = 0.80, 0.40, 0.40
        h = ObjectHypothesisWithPose()
        h.hypothesis.class_id = 'car'
        h.hypothesis.score = 0.9
        d.results.append(h)
        arr = Detection3DArray()
        arr.header.frame_id = 'map'
        arr.header.stamp.sec = int(t[i])
        arr.header.stamp.nanosec = int((t[i] % 1.0) * 1e9)
        arr.detections.append(d)
        msgs.append((t[i], arr))
    return msgs


def run_tracker(msgs, cfg, max_coast=1.0, max_dist=2.0, id_retention=3.0,
                reid=1.5):
    """Same stepping as KFObjectTrackerNode._callback."""
    Track, associate = T.load_tracker_classes()
    Track._next_id = 1
    tracks, retired, last, out = [], [], None, []
    for t, arr in msgs:
        dt = 0.1 if last is None or not (0.0 < t - last) else min(t - last, 1.0)
        last = t
        dets = list(arr.detections) if arr is not None else []
        for tr in tracks:
            tr.predict(dt)
        matched, un_t, un_d = associate(tracks, dets, max_dist)
        for ti, di in matched:
            tracks[ti].update(dets[di])
        for ti in un_t:
            tracks[ti].misses += 1
            tracks[ti].coast_time += dt
        retired = [(i, p, c, a + dt) for i, p, c, a in retired
                   if a + dt <= id_retention]
        for di in un_d:
            tracks.append(Track(dets[di], cfg,
                                reclaim_id=T.reclaim_id(retired, dets[di], reid)))
        for tr in tracks:
            if tr.coast_time > max_coast and tr.hits >= 1:
                retired.append((tr.track_id, np.array(tr.kf.position, float),
                                tr.class_id, 0.0))
        tracks = [tr for tr in tracks if tr.coast_time <= max_coast]
        out.append((t, [(str(tr.track_id), tr.kf.position.copy(),
                         tr.kf.velocity.copy()) for tr in tracks]))
    return out


def score(frames, t, pos, yaw, speed, i0=0):
    rows = []
    for i, (tt, items) in enumerate(frames):
        if not items or i < i0:          # skip the unscored warm-up
            continue
        g = pos[i]
        _, p, v = min(items, key=lambda it: np.linalg.norm(it[1] - g))
        # True velocity: along heading for approaching/receding, across for
        # lateral. Derived from the trajectory rather than assumed.
        # One-sided at the ends. Returning zeros there made ground-truth
        # speed read 0 on the first and last sample of every pass, which drew
        # a spurious step down to zero in the speed panel.
        if len(pos) < 2:
            gv = np.zeros(3)
        elif i == 0:
            gv = (pos[1] - pos[0]) / (t[1] - t[0])
        elif i == len(pos) - 1:
            gv = (pos[-1] - pos[-2]) / (t[-1] - t[-2])
        else:
            gv = (pos[i + 1] - pos[i - 1]) / (t[i + 1] - t[i - 1])
        ng, nv = np.linalg.norm(gv[:2]), np.linalg.norm(v[:2])
        de = (math.degrees(math.acos(np.clip(
            float(np.dot(v[:2], gv[:2])) / (nv * ng), -1, 1)))
            if ng > 1e-6 and nv > 1e-6 else float('nan'))
        err = p - g
        rows.append(dict(
            t=tt - t[i0], track_id=items[0][0], n_tracks=len(items),
            gt_range=float(np.linalg.norm(g[:2])),
            gt_x=g[0], gt_y=g[1], gt_z=g[2],
            trk_x=p[0], trk_y=p[1], trk_z=p[2],
            err_x=err[0], err_y=err[1], err_z=err[2],
            err_norm=float(np.linalg.norm(err)),
            gt_yaw=float(yaw[i]),
            gt_speed=ng, trk_speed=nv,
            vel_err=float(np.linalg.norm(v - gv)), dir_err=de,
            # aliases kept for the summary table below
            rng=float(np.linalg.norm(g[:2])),
            err=float(np.linalg.norm(p - g)),
            verr=float(np.linalg.norm(v - gv)), dir=de, tid=items[0][0]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=0.5)
    ap.add_argument('--rate', type=float, default=8.8)
    ap.add_argument('--passes', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', default='replay',
                    choices=('replay', 'ar1', 'white', 'none'),
                    help="replay = resample the REAL residual sequence "
                         "(default, most faithful); ar1 = calibrated model; "
                         "white = uncorrelated; none = perfect input, which "
                         "isolates the tracker's own lag and convergence")
    ap.add_argument('--white', action='store_true',
                    help='shorthand for --noise white')
    ap.add_argument('--cal-ego', default=CAL_EGO)
    ap.add_argument('--cal-target', default=CAL_TGT)
    ap.add_argument('--no-bias', action='store_true',
                    help='omit the standing detection bias')
    ap.add_argument('--drop', type=float, default=None,
                    help='detection dropout fraction (default 0.13; 0 = none)')
    ap.add_argument('-o', '--out', default=None)
    ap.add_argument('--per-scenario', action='store_true',
                    help='also write the 6-panel figure and CSV per scenario, '
                         'in the same layout plot_eval.py produces for real '
                         'bags, so simulated and recorded runs can be read '
                         'side by side')
    args = ap.parse_args()

    outdir = os.path.expanduser(os.environ.get('EVAL_OUT_DIR',
                                               '~/ros2_ws/eval_results'))
    os.makedirs(outdir, exist_ok=True)
    mode = 'white' if args.white else args.noise
    cal = calibrate(args.cal_ego, args.cal_target)
    print(f"  {cal.describe()}\n")
    out = args.out or os.path.join(outdir, f'simulated_{mode}.png')

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0))
    kind = {'replay': 'replayed real residuals',
            'ar1': f'AR(1) rho={cal.rho[0]:.2f}',
            'white': 'white',
            'none': 'NONE — perfect input'}[mode]
    # Perfect input implies a perfect detector: no dropped frames, no standing
    # bias. Leaving those on would confound the tracker's own behaviour with
    # detector artefacts, which is the whole point of this mode.
    drop = args.drop if args.drop is not None else (0.0 if mode == 'none' else 0.13)
    if mode == 'none':
        args.no_bias = True
    fig.suptitle(f'simulated object_projection_3d — target {args.speed} m/s, '
                 f'{args.rate} Hz, noise {kind}', fontsize=12)

    print(f"target {args.speed} m/s   rate {args.rate} Hz   noise {kind}\n")
    print(f"{'scenario':<18}{'n':>5}{'pos rms':>9}{'raw det':>9}{'gain':>7}"
          f"{'vel rms':>9}{'dir med':>9}{'ids':>5}")
    print('-' * 71)
    for si, name in enumerate(('approaching', 'receding',
                               'lateral', 'reverse lateral')):
        # Each scenario draws its own noise. Re-seeding all four identically
        # gave receding the byte-identical residual sequence to approaching
        # (it is the same path reversed, same sample count), so their scores
        # matched to three decimals and looked like a suspicious coincidence
        # instead of four independent measurements. Derived from args.seed, so
        # the whole set is still reproducible.
        rng = np.random.default_rng(args.seed + 1000 * si)
        t, pos, yaw, i0 = trajectory(name, args.speed, args.rate)
        msgs = synth_detections(t, pos, yaw, rng, cal, mode=mode,
                                drop=drop, bias=not args.no_bias)
        frames = run_tracker(msgs, T.default_cfg())
        rows = score(frames, t, pos, yaw, args.speed, i0=i0)
        for row in rows:
            row['settling'] = False      # warm-up already removed
        raw_l = [float(np.linalg.norm(
            np.array([m.detections[0].bbox.center.position.x,
                      m.detections[0].bbox.center.position.y,
                      m.detections[0].bbox.center.position.z]) - pos[i]))
            for i, (_, m) in enumerate(msgs) if m is not None and i >= i0]
        if not rows:
            print(f'{name:<18} no tracks'); continue
        e = np.array([r['err'] for r in rows])
        ve = np.array([r['verr'] for r in rows])
        d = np.array([r['dir'] for r in rows]); ok = np.isfinite(d)
        raw = np.array(raw_l)
        rms = math.sqrt(float((e ** 2).mean()))
        rraw = math.sqrt(float((raw ** 2).mean()))
        c = COLORS[name]
        ax[0].plot([r['t'] for r in rows], e, lw=1, color=c, alpha=.8, label=name)
        srt = np.argsort([r['rng'] for r in rows])
        rr = np.array([r['rng'] for r in rows])[srt]
        ee = e[srt]
        edges = np.linspace(rr.min(), rr.max(), 9)
        mid = 0.5 * (edges[:-1] + edges[1:])
        med = [np.median(ee[(rr >= edges[i]) & (rr < edges[i + 1])])
               if ((rr >= edges[i]) & (rr < edges[i + 1])).sum() > 3 else np.nan
               for i in range(8)]
        ax[1].plot(mid, med, '-o', color=c, lw=2, ms=4, label=name)
        if ok.any():
            ax[2].plot(np.sort(d[ok]), np.linspace(0, 100, ok.sum()),
                       color=c, lw=2, label=name)
        if args.per_scenario:
            tag = name.replace(' ', '')
            clean = [{k: v for k, v in r.items()
                      if k not in ('rng', 'err', 'verr', 'dir', 'tid')}
                     for r in rows]
            noise = mode
            PE.write_csv(clean, os.path.join(
                outdir, f'sim_{tag}_{noise}.csv'))
            PE.figure(clean,
                      f'SIMULATED {name} — {args.speed} m/s, {args.rate} Hz, '
                      f'noise {kind}',
                      os.path.join(outdir, f'sim_{tag}_{noise}.png'))
        print(f"{name:<18}{len(rows):>5}{rms:>9.3f}{rraw:>9.3f}"
              f"{rraw / rms:>7.2f}{math.sqrt(float((ve**2).mean())):>9.3f}"
              f"{np.median(d[ok]) if ok.any() else float('nan'):>9.1f}"
              f"{len(set(r['tid'] for r in rows)):>5}")

    ax[0].set_title('position error over time')
    ax[0].set_xlabel('t [s]'); ax[0].set_ylabel('|error| [m]')
    ax[1].set_title('position error vs range')
    ax[1].set_xlabel('range from ego [m]'); ax[1].set_ylabel('|error| [m]')
    ax[2].set_title('velocity direction error (cumulative)')
    ax[2].set_xlabel('angle [deg]'); ax[2].set_ylabel('% of samples below')
    ax[2].set_xlim(0, 90)
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=110)
    print(f"\n  gain = raw detection rms / tracked rms  (>1 means the filter helped)")
    print(f"  figure: {out}")


if __name__ == '__main__':
    main()
