#!/usr/bin/env python3
"""Generate a combined extended statsheet with trace analytics.

This script uses the original extended statsheet as the baseline, then appends
extra pages from the high-frequency Gate trace section.

Usage:
    python generate_trace_extended_statsheet.py --csv "../../new-statsheet/18APR-race - trace-work.csv"
"""

import argparse
import csv
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "statsheet" / "output"
DATA_DIR = ROOT_DIR / "statsheet" / "data"
BASE_SCRIPT = ROOT_DIR / "legacy-statsheet" / "code" / "generate_extended_statsheet.py"


def _load_base_module():
    spec = importlib.util.spec_from_file_location("generate_extended_statsheet", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load baseline script: {BASE_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_base_module()

# Temporarily sideline ML speed-factor analysis pages in the baseline report.
DISABLE_BASELINE_ML = True

TRACE_BIN_EDGES = np.linspace(-50.0, 50.0, 101)
ANGLE_BIN_EDGES = np.linspace(-80.0, 30.0, 111)  # gate angle range (deg)


def _interp_nan_1d(y):
    """Linearly interpolate NaNs in a 1D array, holding endpoints."""
    y = np.array(y, dtype=float)
    n = len(y)
    if n == 0:
        return y
    x = np.arange(n)
    ok = np.isfinite(y)
    if ok.sum() == 0:
        return np.zeros_like(y)
    if ok.sum() == 1:
        return np.full_like(y, y[ok][0])
    out = y.copy()
    out[~ok] = np.interp(x[~ok], x[ok], y[ok])
    return out


def _fill_internal_nan_1d(y):
    """Fill NaN gaps between valid points; keep leading/trailing NaNs unchanged."""
    y = np.array(y, dtype=float)
    if len(y) == 0:
        return y

    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y

    x = np.arange(len(y))
    first = int(np.where(ok)[0][0])
    last = int(np.where(ok)[0][-1])
    gaps = (~ok) & (x > first) & (x < last)
    if np.any(gaps):
        y[gaps] = np.interp(x[gaps], x[ok], y[ok])
    return y


def _smooth_1d(y, window=7):
    """Simple low-pass smoothing (triangular-like) on 1D arrays."""
    y = _interp_nan_1d(y)
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    half = w // 2
    weights = np.arange(1, half + 2, dtype=float)
    weights = np.r_[weights, weights[-2::-1]]
    weights /= weights.sum()
    pad = np.pad(y, (half, half), mode="edge")
    out = np.convolve(pad, weights, mode="valid")
    return out


def _repair_zero_gap(x, y, gap_width=2.0):
    """Repair NaN gaps around normalized-time zero by local interpolation."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    near = (x >= -gap_width) & (x <= gap_width)
    if not np.any(near):
        return y
    if np.all(np.isfinite(y[near])):
        return y

    ok = np.isfinite(y)
    if ok.sum() < 3:
        return _interp_nan_1d(y)

    out = y.copy()
    miss = near & ~ok
    out[miss] = np.interp(x[miss], x[ok], y[ok])
    return out


def _smooth_phase_curve(x, y, window=9):
    """Fill sparse bins and smooth phase curves for plotting/phase analytics."""
    y2 = _repair_zero_gap(x, y)
    y2 = _interp_nan_1d(y2)
    y2 = _smooth_1d(y2, window=window)
    return y2


def _smooth_trace_matrix(mat, window=7):
    """Smooth each column of a 2D trace matrix."""
    arr = np.array(mat, dtype=float)
    if arr.ndim != 2:
        return arr
    out = np.zeros_like(arr)
    for c in range(arr.shape[1]):
        out[:, c] = _smooth_1d(arr[:, c], window=window)
    return out


def _to_float(v, default=0.0):
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def _to_int(v, default=0):
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _parse_baseline_strokes(rows):
    """Parse stroke-summary section (SwivelPower block) with resilient casting."""
    header_row = None
    for i, row in enumerate(rows):
        if len(row) > 1 and row[1] == "SwivelPower":
            header_row = i
            break
    if header_row is None:
        raise RuntimeError("Could not find stroke-summary section (SwivelPower)")

    header = rows[header_row]

    def find_cols(name):
        for idx, col in enumerate(header):
            if col == name:
                return idx, idx + 8
        raise RuntimeError(f"Metric '{name}' not found in header")

    metric_names = [
        "SwivelPower", "MinAngle", "MaxAngle", "CatchSlip", "FinishSlip",
        "Drive Start T", "Rower Swivel Power", "Drive Time", "Recovery Time",
        "Angle Max F", "Angle 0.7 F",
        "Work PC Q1", "Work PC Q2", "Work PC Q3", "Work PC Q4",
        "Max Force PC",
    ]
    metrics = {name: find_cols(name) for name in metric_names}

    data_start = header_row + 2
    swivel_start, swivel_end = metrics["SwivelPower"]

    strokes = {name: [] for name in metrics}
    strokes["stroke_num"] = []
    strokes["rating"] = []
    strokes["boat_speed"] = []

    for row in rows[data_start:]:
        if len(row) < 9 or row[0] == "Time":
            break
        if all((c >= len(row) or row[c].strip() == "") for c in range(swivel_start, swivel_end)):
            continue

        for name, (start, end) in metrics.items():
            vals = [_to_float(row[c] if c < len(row) else "", default=0.0) for c in range(start, end)]
            strokes[name].append(vals)

        strokes["stroke_num"].append(_to_int(row[-4] if len(row) >= 4 else "", default=0))
        strokes["rating"].append(_to_float(row[-6] if len(row) >= 6 else "", default=0.0))
        strokes["boat_speed"].append(_to_float(row[-5] if len(row) >= 5 else "", default=0.0))

    for name in metrics:
        strokes[name] = np.array(strokes[name], dtype=float)

    strokes = base._remove_outliers(strokes, list(metrics.keys()))
    strokes["dead_seats"] = base._detect_dead_seats(strokes, list(metrics.keys()))
    return strokes


def _extract_crew_names(rows):
    """Extract seat names from Crew Info section (Position, Name)."""
    seat_names = [f"Rower {i}" for i in range(1, 9)] + ["Cox"]

    header_idx = None
    for i, row in enumerate(rows):
        if len(row) > 1 and row[0] == "Position" and row[1] == "Name":
            header_idx = i
            break

    if header_idx is None:
        return seat_names

    for row in rows[header_idx + 1:]:
        if len(row) < 2:
            continue
        # Crew section ends at the next metadata block marker.
        if row[0].strip().startswith("#ERROR!"):
            break

        pos = row[0].strip()
        name = row[1].strip()

        if not pos:
            continue

        if pos.isdigit() and name:
            p = int(pos)
            if 1 <= p <= 8:
                seat_names[p - 1] = name
        elif pos.lower() == "cox" and name:
            seat_names[8] = name

    return seat_names


def _parse_trace_section(rows):
    """Parse section with GateAngle/GateForceX/GateAngleVel and boat channels."""
    header_row = None
    for i, row in enumerate(rows):
        if len(row) > 1 and row[1] == "GateAngle":
            header_row = i
            break

    if header_row is None:
        return None

    header = rows[header_row]
    sub = rows[header_row + 1] if header_row + 1 < len(rows) else []

    def seat_cols(metric):
        cols = []
        for c, h in enumerate(header):
            if h != metric:
                continue
            s = sub[c].strip() if c < len(sub) else ""
            if s.isdigit() and 1 <= int(s) <= 8:
                cols.append((int(s) - 1, c))
        cols.sort(key=lambda x: x[0])
        if len(cols) != 8:
            return None
        return [c for _, c in cols]

    def boat_col(metric):
        for c, h in enumerate(header):
            if h != metric:
                continue
            s = sub[c].strip() if c < len(sub) else ""
            if s == "Boat":
                return c
        return None

    angle_cols = seat_cols("GateAngle")
    force_cols = seat_cols("GateForceX")
    vel_cols = seat_cols("GateAngleVel")
    speed_col = boat_col("Speed")
    dist_col = boat_col("Distance")
    accel_col = boat_col("Accel")
    roll_col = boat_col("Roll Angle")
    pitch_col = boat_col("Pitch Angle")
    yaw_col = boat_col("Yaw Angle")
    norm_col = boat_col("Normalized Time")

    if angle_cols is None or force_cols is None or vel_cols is None:
        return None

    times = []
    angles, forces, vels = [], [], []
    speed, dist, accel, roll, pitch, yaw, norm_t = [], [], [], [], [], [], []

    for row in rows[header_row + 2:]:
        if len(row) > 0 and row[0] == "Time":
            break
        if not any(cell.strip() for cell in row):
            continue

        try:
            t = float(row[0])
        except Exception:
            continue

        def _get(col):
            if col is None or col >= len(row) or not row[col].strip():
                return np.nan
            try:
                return float(row[col])
            except Exception:
                return np.nan

        ang = [_get(c) for c in angle_cols]
        frc = [_get(c) for c in force_cols]
        vel = [_get(c) for c in vel_cols]

        if np.all(np.isnan(ang)) and np.all(np.isnan(frc)):
            continue

        times.append(t)
        angles.append(ang)
        forces.append(frc)
        vels.append(vel)
        speed.append(_get(speed_col))
        dist.append(_get(dist_col))
        accel.append(_get(accel_col))
        roll.append(_get(roll_col))
        pitch.append(_get(pitch_col))
        yaw.append(_get(yaw_col))
        norm_t.append(_get(norm_col))

    if not times:
        return None

    trace = {
        "time": np.array(times, dtype=float),
        "angle": np.array(angles, dtype=float),
        "force": np.array(forces, dtype=float),
        "angle_vel": np.array(vels, dtype=float),
        "speed": np.array(speed, dtype=float),
        "distance": np.array(dist, dtype=float),
        "accel": np.array(accel, dtype=float),
        "roll": np.array(roll, dtype=float),
        "pitch": np.array(pitch, dtype=float),
        "yaw": np.array(yaw, dtype=float),
        "norm_time": np.array(norm_t, dtype=float),
        "sample_ms": np.nanmedian(np.diff(np.array(times, dtype=float))),
    }

    # Smooth all trace channels to reduce high-frequency jaggedness and small dropouts.
    trace["angle"] = _smooth_trace_matrix(trace["angle"], window=7)
    trace["force"] = _smooth_trace_matrix(trace["force"], window=7)
    trace["angle_vel"] = _smooth_trace_matrix(trace["angle_vel"], window=7)
    for k in ["speed", "distance", "accel", "roll", "pitch", "yaw"]:
        trace[k] = _smooth_1d(trace[k], window=7)

    return trace


def _trace_segments(norm_time):
    """Split trace into stroke-like segments using normalized-time resets."""
    n = len(norm_time)
    if n < 5:
        return []
    d = np.diff(norm_time)
    resets = np.where(d < -30)[0]
    starts = np.r_[0, resets + 1]
    ends = np.r_[resets, n - 1]
    segs = []
    for s, e in zip(starts, ends):
        if e - s + 1 >= 30:
            segs.append((int(s), int(e)))
    return segs


def _binned_mean(x, y, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = (x >= edges[i]) & (x < edges[i + 1]) & np.isfinite(y)
        if np.any(m):
            out[i] = np.nanmean(y[m])
    return centers, out


def _compute_trace_metrics(trace):
    """Compute seat-wise curve means and rich per-stroke trace-derived metrics."""
    nt = trace["norm_time"]
    force = trace["force"]
    angle = trace["angle"]
    angle_vel = trace["angle_vel"]
    time = trace["time"]
    speed = trace["speed"]
    accel = trace["accel"]
    roll = trace["roll"]
    yaw = trace["yaw"]
    pitch = trace["pitch"]
    sample_ms = trace["sample_ms"]

    segs = _trace_segments(nt)
    centers = 0.5 * (TRACE_BIN_EDGES[:-1] + TRACE_BIN_EDGES[1:])

    def _nanmean(v):
        return float(np.nanmean(v)) if len(v) else np.nan

    def _safe_corr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20:
            return np.nan
        aa = a[ok]
        bb = b[ok]
        if np.nanstd(aa) == 0 or np.nanstd(bb) == 0:
            return np.nan
        return float(np.corrcoef(aa, bb)[0, 1])

    def _interp_curve(x, y, grid):
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 6:
            return np.full(len(grid), np.nan)
        xs = x[ok]
        ys = y[ok]
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        uniq_x, uniq_idx = np.unique(xs, return_index=True)
        if len(uniq_x) < 6:
            return np.full(len(grid), np.nan)
        uniq_y = ys[uniq_idx]
        return np.interp(grid, uniq_x, uniq_y, left=np.nan, right=np.nan)

    force_curves, angle_curves, vel_curves = [], [], []
    force_by_angle = []
    for seat in range(8):
        cx, cy = _binned_mean(nt, force[:, seat], TRACE_BIN_EDGES)
        cy = _smooth_phase_curve(cx, cy, window=9)
        force_curves.append((cx, cy))
        cx, cy = _binned_mean(nt, angle[:, seat], TRACE_BIN_EDGES)
        cy = _smooth_phase_curve(cx, cy, window=9)
        angle_curves.append((cx, cy))
        cx, cy = _binned_mean(nt, angle_vel[:, seat], TRACE_BIN_EDGES)
        cy = _smooth_phase_curve(cx, cy, window=9)
        vel_curves.append((cx, cy))
        # Force vs gate angle (binned by actual angle value)
        ax_c, ay_c = _binned_mean(angle[:, seat], force[:, seat], ANGLE_BIN_EDGES)
        ay_c = _smooth_phase_curve(ax_c, ay_c, window=9)
        force_by_angle.append((ax_c, ay_c))

    per_seat = {
        "peak_force": [[] for _ in range(8)],
        "peak_phase": [[] for _ in range(8)],
        "rise_rate": [[] for _ in range(8)],
        "decay_rate": [[] for _ in range(8)],
        "pos_impulse": [[] for _ in range(8)],
        "neg_ratio": [[] for _ in range(8)],
        "force_roughness": [[] for _ in range(8)],
        "vel_peak_phase": [[] for _ in range(8)],
        "vel_smoothness": [[] for _ in range(8)],
        "catch_sharpness": [[] for _ in range(8)],
        "finish_release_smoothness": [[] for _ in range(8)],
        "drive_time_re": [[] for _ in range(8)],
        "recovery_time_re": [[] for _ in range(8)],
        "catch_phase": [[] for _ in range(8)],
        "finish_phase": [[] for _ in range(8)],
    }

    catch_times_by_seg = []
    seg_force_curves = []

    for s, e in segs:
        ts = time[s:e + 1]
        nts = nt[s:e + 1]
        seg_catch_times = [np.nan] * 8
        seg_curves_this = []

        for seat in range(8):
            fs = force[s:e + 1, seat]
            vs = angle_vel[s:e + 1, seat]

            seg_curves_this.append(_interp_curve(nts, fs, centers))

            ok = np.isfinite(fs) & np.isfinite(nts) & np.isfinite(ts)
            if ok.sum() < 8:
                continue

            fsv = fs[ok]
            ntsv = nts[ok]
            tsv = ts[ok]
            vsv = vs[ok] if len(vs) == len(fs) else np.full_like(fsv, np.nan)

            pk_i = int(np.nanargmax(fsv))
            pk = float(fsv[pk_i])
            per_seat["peak_force"][seat].append(pk)
            per_seat["peak_phase"][seat].append(float(ntsv[pk_i]))

            thr = max(8.0, 0.10 * pk)
            above = np.where(fsv >= thr)[0]
            if len(above) == 0:
                continue
            catch_i = int(above[0])
            finish_i = int(above[-1])
            if finish_i <= catch_i:
                continue

            catch_t = float(tsv[catch_i])
            finish_t = float(tsv[finish_i])
            catch_phase = float(ntsv[catch_i])
            finish_phase = float(ntsv[finish_i])

            seg_catch_times[seat] = catch_t
            per_seat["catch_phase"][seat].append(catch_phase)
            per_seat["finish_phase"][seat].append(finish_phase)

            # Force rise and decay rates around peak/catch/finish.
            rise_dt = max((tsv[pk_i] - tsv[catch_i]) / 1000.0, 1e-3)
            decay_dt = max((tsv[finish_i] - tsv[pk_i]) / 1000.0, 1e-3)
            per_seat["rise_rate"][seat].append(float((pk - fsv[catch_i]) / rise_dt))
            per_seat["decay_rate"][seat].append(float((pk - fsv[finish_i]) / decay_dt))

            # Positive impulse and negative force tail share.
            pos = np.clip(fsv, 0, None)
            neg = np.clip(-fsv, 0, None)
            if len(tsv) >= 2:
                per_seat["pos_impulse"][seat].append(float(np.trapezoid(pos, tsv)))
            den = float(np.nansum(np.abs(fsv)))
            if den > 0:
                per_seat["neg_ratio"][seat].append(float(np.nansum(neg) / den * 100.0))

            # Roughness and kinematic smoothness.
            d_force = np.diff(fsv)
            d_vel = np.diff(vsv) if np.isfinite(vsv).sum() > 3 else np.array([])
            if len(d_force) > 0:
                per_seat["force_roughness"][seat].append(float(np.nanstd(d_force)))
            if len(d_vel) > 0:
                per_seat["vel_smoothness"][seat].append(float(np.nanstd(d_vel)))

            # GateAngleVel peak timing and catch sharpness proxy.
            if np.isfinite(vsv).sum() > 4:
                vel_pk_i = int(np.nanargmax(vsv))
                per_seat["vel_peak_phase"][seat].append(float(ntsv[vel_pk_i]))

            if pk_i > catch_i:
                win_fs = fsv[catch_i:pk_i + 1]
                win_ts = tsv[catch_i:pk_i + 1]
                if len(win_fs) >= 2:
                    d = np.diff(win_fs) / np.maximum(np.diff(win_ts) / 1000.0, 1e-3)
                    if len(d):
                        per_seat["catch_sharpness"][seat].append(float(np.nanmax(d)))

            rel_start = catch_i + int(max(1, 0.8 * (finish_i - catch_i)))
            rel_fs = fsv[rel_start:finish_i + 1]
            rel_ts = tsv[rel_start:finish_i + 1]
            if len(rel_fs) >= 2:
                d_rel = np.diff(rel_fs) / np.maximum(np.diff(rel_ts) / 1000.0, 1e-3)
                per_seat["finish_release_smoothness"][seat].append(float(np.nanstd(d_rel)))

            # Re-derived drive/recovery timing from trace events.
            drive_t = max((finish_t - catch_t) / 1000.0, 0.0)
            cycle_t = max((tsv[-1] - tsv[0]) / 1000.0, 1e-3)
            rec_t = max(cycle_t - drive_t, 0.0)
            per_seat["drive_time_re"][seat].append(drive_t)
            per_seat["recovery_time_re"][seat].append(rec_t)

        catch_times_by_seg.append(seg_catch_times)
        seg_force_curves.append(seg_curves_this)

    # Catch spread and sync stability across piece.
    catch_spreads = []
    for row in catch_times_by_seg:
        vals = np.array(row, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) >= 2:
            catch_spreads.append(float(np.nanmax(vals) - np.nanmin(vals)))

    catch_spread_mean = float(np.nanmean(catch_spreads)) if catch_spreads else np.nan
    catch_spread_std = float(np.nanstd(catch_spreads)) if catch_spreads else np.nan

    # Catch lag vs stroke seat by stroke index.
    stroke_seat = 7
    catch_lag_mean = [np.nan] * 8
    catch_lag_std = [np.nan] * 8
    ref = np.array([row[stroke_seat] for row in catch_times_by_seg], dtype=float)
    for seat in range(8):
        cur = np.array([row[seat] for row in catch_times_by_seg], dtype=float)
        ok = np.isfinite(ref) & np.isfinite(cur)
        if ok.sum() >= 5:
            lag = cur[ok] - ref[ok]
            catch_lag_mean[seat] = float(np.nanmean(lag))
            catch_lag_std[seat] = float(np.nanstd(lag))

    # Seat-to-seat phase lag from cross-correlation against stroke seat.
    xcorr_lag_ms = [np.nan] * 8
    xcorr_peak = [np.nan] * 8
    lag_limit = int(max(1, round(500.0 / sample_ms))) if np.isfinite(sample_ms) and sample_ms > 0 else 25
    ref_force = force[:, stroke_seat]
    for seat in range(8):
        a = force[:, seat]
        ok = np.isfinite(a) & np.isfinite(ref_force)
        if ok.sum() < 200:
            continue
        av = a[ok] - np.nanmean(a[ok])
        bv = ref_force[ok] - np.nanmean(ref_force[ok])
        if np.nanstd(av) == 0 or np.nanstd(bv) == 0:
            continue
        corr = np.correlate(av, bv, mode="full")
        lags = np.arange(-len(av) + 1, len(av))
        keep = np.abs(lags) <= lag_limit
        corr_k = corr[keep]
        lags_k = lags[keep]
        best = int(np.nanargmax(corr_k))
        lag_samp = int(lags_k[best])
        xcorr_lag_ms[seat] = float(lag_samp * sample_ms)
        norm = np.sqrt(np.sum(av ** 2) * np.sum(bv ** 2))
        xcorr_peak[seat] = float(corr_k[best] / norm) if norm > 0 else np.nan

    # Drive vs recovery acceleration and phase speed windows.
    drive_mask = (nt >= -5) & (nt <= 35) & np.isfinite(accel)
    rec_mask = np.isfinite(accel) & ~drive_mask
    drive_accel_mean = float(np.nanmean(accel[drive_mask])) if np.any(drive_mask) else np.nan
    rec_accel_mean = float(np.nanmean(accel[rec_mask])) if np.any(rec_mask) else np.nan

    t_sec = (time - time[0]) / 1000.0
    speed_rate = np.gradient(speed, t_sec) if len(speed) > 2 else np.full(len(speed), np.nan)
    cx, speed_gain_phase = _binned_mean(nt, speed_rate, TRACE_BIN_EDGES)
    speed_gain_phase = _smooth_phase_curve(cx, speed_gain_phase, window=9)
    cx, roll_phase = _binned_mean(nt, roll, TRACE_BIN_EDGES)
    roll_phase = _smooth_phase_curve(cx, roll_phase, window=9)
    cx, pitch_phase = _binned_mean(nt, pitch, TRACE_BIN_EDGES)
    pitch_phase = _smooth_phase_curve(cx, pitch_phase, window=9)
    cx, yaw_phase = _binned_mean(nt, yaw, TRACE_BIN_EDGES)
    yaw_phase = _smooth_phase_curve(cx, yaw_phase, window=9)

    # Stability/set metrics and coupling.
    roll_rms = float(np.sqrt(np.nanmean(roll ** 2)))
    roll_p2p = float(np.nanmax(roll) - np.nanmin(roll))
    yaw_drift = float(yaw[-1] - yaw[0]) if len(yaw) else np.nan
    yaw_osc = float(np.nanstd(yaw - np.nanmean(yaw)))

    abs_force_mean = np.nanmean(np.abs(force), axis=1)
    force_asym = np.nanstd(force, axis=1) / np.maximum(abs_force_mean, 1e-6)
    asym_roll_corr = _safe_corr(force_asym, np.abs(roll))
    asym_yaw_corr = _safe_corr(force_asym, np.abs(yaw))

    # Shape consistency and crew coherence.
    seg_force_curves = np.array(seg_force_curves, dtype=float) if seg_force_curves else np.empty((0, 8, len(centers)))
    seat_repeatability = [np.nan] * 8
    seat_shape_cv = [np.nan] * 8
    for seat in range(8):
        if len(seg_force_curves) == 0:
            continue
        seat_cur = seg_force_curves[:, seat, :]
        mean_curve = np.nanmean(seat_cur, axis=0)
        corr_vals = []
        area_vals = []
        for c in seat_cur:
            ok = np.isfinite(c) & np.isfinite(mean_curve)
            if ok.sum() >= 20 and np.nanstd(c[ok]) > 0 and np.nanstd(mean_curve[ok]) > 0:
                corr_vals.append(float(np.corrcoef(c[ok], mean_curve[ok])[0, 1]))
            if np.isfinite(c).sum() >= 20:
                area_vals.append(float(np.nansum(c)))
        if corr_vals:
            seat_repeatability[seat] = float(np.nanmean(corr_vals) * 100.0)
        if area_vals and np.nanmean(area_vals) != 0:
            seat_shape_cv[seat] = float(np.nanstd(area_vals) / abs(np.nanmean(area_vals)) * 100.0)

    pair_corrs = []
    if len(seg_force_curves) > 0:
        for k in range(len(seg_force_curves)):
            block = seg_force_curves[k]
            for i in range(8):
                for j in range(i + 1, 8):
                    a = block[i]
                    b = block[j]
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() >= 20 and np.nanstd(a[ok]) > 0 and np.nanstd(b[ok]) > 0:
                        pair_corrs.append(float(np.corrcoef(a[ok], b[ok])[0, 1]))
    crew_coherence = float(np.nanmean(pair_corrs) * 100.0) if pair_corrs else np.nan

    # Fatigue/degradation (early/mid/late thirds).
    thirds = {
        "early": range(0, len(segs) // 3),
        "mid": range(len(segs) // 3, 2 * len(segs) // 3),
        "late": range(2 * len(segs) // 3, len(segs)),
    }
    peak_force_by_third = {k: [np.nan] * 8 for k in thirds}
    timing_sd_by_third = {k: [np.nan] * 8 for k in thirds}

    # Build per-segment peak/catch lag arrays first.
    seg_peak = np.full((len(segs), 8), np.nan)
    seg_catch_lag = np.full((len(segs), 8), np.nan)
    for si, row in enumerate(catch_times_by_seg):
        r = np.array(row, dtype=float)
        if np.isfinite(r[stroke_seat]):
            seg_catch_lag[si, :] = r - r[stroke_seat]

    for si, (s, e) in enumerate(segs):
        for seat in range(8):
            fs = force[s:e + 1, seat]
            if np.isfinite(fs).sum() >= 8:
                seg_peak[si, seat] = float(np.nanmax(fs))

    for part, idxs in thirds.items():
        idxs = list(idxs)
        if not idxs:
            continue
        for seat in range(8):
            peak_force_by_third[part][seat] = float(np.nanmean(seg_peak[idxs, seat]))
            timing_sd_by_third[part][seat] = float(np.nanstd(seg_catch_lag[idxs, seat]))

    fade_peak_pct = [np.nan] * 8
    fade_timing_sd = [np.nan] * 8
    for seat in range(8):
        e0 = peak_force_by_third["early"][seat]
        el = peak_force_by_third["late"][seat]
        if np.isfinite(e0) and abs(e0) > 1e-6 and np.isfinite(el):
            fade_peak_pct[seat] = float((el - e0) / e0 * 100.0)
        t0 = timing_sd_by_third["early"][seat]
        tl = timing_sd_by_third["late"][seat]
        if np.isfinite(t0) and np.isfinite(tl):
            fade_timing_sd[seat] = float(tl - t0)

    metrics = {
        "segments": segs,
        "force_curves": force_curves,
        "force_by_angle": force_by_angle,
        "angle_curves": angle_curves,
        "vel_curves": vel_curves,
        "centers": centers,
        "peak_force": [_nanmean(v) for v in per_seat["peak_force"]],
        "peak_phase": [_nanmean(v) for v in per_seat["peak_phase"]],
        "rise_rate": [_nanmean(v) for v in per_seat["rise_rate"]],
        "decay_rate": [_nanmean(v) for v in per_seat["decay_rate"]],
        "pos_impulse": [_nanmean(v) for v in per_seat["pos_impulse"]],
        "neg_ratio": [_nanmean(v) for v in per_seat["neg_ratio"]],
        "force_roughness": [_nanmean(v) for v in per_seat["force_roughness"]],
        "vel_peak_phase": [_nanmean(v) for v in per_seat["vel_peak_phase"]],
        "vel_smoothness": [_nanmean(v) for v in per_seat["vel_smoothness"]],
        "catch_sharpness": [_nanmean(v) for v in per_seat["catch_sharpness"]],
        "finish_release_smoothness": [_nanmean(v) for v in per_seat["finish_release_smoothness"]],
        "drive_time_re": [_nanmean(v) for v in per_seat["drive_time_re"]],
        "recovery_time_re": [_nanmean(v) for v in per_seat["recovery_time_re"]],
        "catch_phase": [_nanmean(v) for v in per_seat["catch_phase"]],
        "finish_phase": [_nanmean(v) for v in per_seat["finish_phase"]],
        "catch_lag_mean": catch_lag_mean,
        "catch_lag_std": catch_lag_std,
        "xcorr_lag_ms": xcorr_lag_ms,
        "xcorr_peak": xcorr_peak,
        "catch_spread_mean": catch_spread_mean,
        "catch_spread_std": catch_spread_std,
        "drive_accel_mean": drive_accel_mean,
        "recovery_accel_mean": rec_accel_mean,
        "speed_gain_phase": speed_gain_phase,
        "roll_phase": roll_phase,
        "pitch_phase": pitch_phase,
        "yaw_phase": yaw_phase,
        "roll_rms": roll_rms,
        "roll_p2p": roll_p2p,
        "yaw_drift": yaw_drift,
        "yaw_osc": yaw_osc,
        "asym_roll_corr": asym_roll_corr,
        "asym_yaw_corr": asym_yaw_corr,
        "seat_repeatability": seat_repeatability,
        "seat_shape_cv": seat_shape_cv,
        "crew_coherence": crew_coherence,
        "peak_force_by_third": peak_force_by_third,
        "timing_sd_by_third": timing_sd_by_third,
        "fade_peak_pct": fade_peak_pct,
        "fade_timing_sd": fade_timing_sd,
    }
    return metrics


def _seat_name(i, seat_names):
    if i < len(seat_names):
        return seat_names[i]
    return f"Rower {i + 1}"


def _draw_breaker(fig):
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.52, "Trace Analytics", fontsize=38, fontweight="bold",
             ha="center", va="center", color="#1f2d3d")
    fig.text(0.5, 0.44, "High-frequency gate force, angle, velocity, and boat dynamics",
             fontsize=14, ha="center", color="#555")


def _draw_force_curves_time(fig, trace, metrics, seat_names):
    fig.patch.set_facecolor("white")

    ax_t = fig.add_axes([0.06, 0.24, 0.90, 0.64])
    ax_t.set_title("Force Curve — Normalized Time", fontsize=16, pad=10)
    for seat in range(8):
        x, y = metrics["force_curves"][seat]
        ax_t.plot(x, y, color=base.SEAT_COLORS[seat], linewidth=2.1,
                  label=f"{_seat_name(seat, seat_names)}")
    ax_t.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax_t.axhline(0, color="#444", linewidth=0.8)
    ax_t.set_xlabel("Normalized Time")
    ax_t.set_ylabel("Gate Force X")
    ax_t.grid(True, alpha=0.25)
    ax_t.legend(loc="upper right", fontsize=9, ncol=2)

    sample_hz = 1000.0 / trace["sample_ms"] if trace["sample_ms"] > 0 else np.nan
    fig.text(0.06, 0.19,
             f"Samples: {len(trace['time'])}  |  Approx sample rate: {sample_hz:.1f} Hz  |  Parsed strokes: {len(metrics['segments'])}",
             fontsize=10, color="#555")

    ax_tb = fig.add_axes([0.06, 0.03, 0.90, 0.13])
    ax_tb.axis("off")
    headers = ["Rower", "Peak Phase", "Rise Rate", "Decay Rate", "Pos Impulse", "Neg Tail %"]
    rows = []
    for seat in range(8):
        rows.append([
            _seat_name(seat, seat_names),
            f"{metrics['peak_phase'][seat]:.1f}" if np.isfinite(metrics['peak_phase'][seat]) else "-",
            f"{metrics['rise_rate'][seat]:.0f}" if np.isfinite(metrics['rise_rate'][seat]) else "-",
            f"{metrics['decay_rate'][seat]:.0f}" if np.isfinite(metrics['decay_rate'][seat]) else "-",
            f"{metrics['pos_impulse'][seat]:.0f}" if np.isfinite(metrics['pos_impulse'][seat]) else "-",
            f"{metrics['neg_ratio'][seat]:.1f}" if np.isfinite(metrics['neg_ratio'][seat]) else "-",
        ])
    tbl = ax_tb.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1f2d3d")
        tbl[0, j].set_text_props(color="white", weight="bold")


def _draw_force_curves_angle(fig, trace, metrics, seat_names):
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.06, 0.20, 0.90, 0.68])
    ax.set_title("Force Curve — Gate Angle (deg)", fontsize=16, pad=10)
    for seat in range(8):
        x, y = metrics["force_by_angle"][seat]
        ax.plot(x, y, color=base.SEAT_COLORS[seat], linewidth=2.1,
                label=f"{_seat_name(seat, seat_names)}")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Gate Angle (deg)")
    ax.set_ylabel("Gate Force X")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=2)

    sample_hz = 1000.0 / trace["sample_ms"] if trace["sample_ms"] > 0 else np.nan
    fig.text(0.06, 0.14,
             f"Samples: {len(trace['time'])}  |  Approx sample rate: {sample_hz:.1f} Hz  |  Parsed strokes: {len(metrics['segments'])}",
             fontsize=10, color="#555")



def _draw_angle_kinematics(fig, metrics, seat_names):
    fig.patch.set_facecolor("white")

    ax2 = fig.add_axes([0.07, 0.42, 0.88, 0.46])
    ax2.set_title("Mean Gate Angle Velocity by Rower", fontsize=14)
    for seat in range(8):
        x, y = metrics["vel_curves"][seat]
        ax2.plot(x, y, color=base.SEAT_COLORS[seat], linewidth=2,
                 label=f"{_seat_name(seat, seat_names)}")
    ax2.axvline(0, color="#888", linestyle="--", linewidth=1)
    ax2.axhline(0, color="#444", linewidth=0.8)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Gate Angle Velocity")
    ax2.legend(fontsize=8, ncol=2, loc="upper right")

    ax3 = fig.add_axes([0.07, 0.12, 0.88, 0.22])
    ax3.axis("off")
    headers = ["Rower", "Vel Peak Phase", "Vel Smoothness", "Catch Sharpness", "Finish Release Smoothness"]
    rows = []
    for seat in range(8):
        rows.append([
            _seat_name(seat, seat_names),
            f"{metrics['vel_peak_phase'][seat]:.1f}" if np.isfinite(metrics['vel_peak_phase'][seat]) else "-",
            f"{metrics['vel_smoothness'][seat]:.2f}" if np.isfinite(metrics['vel_smoothness'][seat]) else "-",
            f"{metrics['catch_sharpness'][seat]:.0f}" if np.isfinite(metrics['catch_sharpness'][seat]) else "-",
            f"{metrics['finish_release_smoothness'][seat]:.2f}" if np.isfinite(metrics['finish_release_smoothness'][seat]) else "-",
        ])
    tbl = ax3.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1.0, 1.5)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1f2d3d")
        tbl[0, j].set_text_props(color="white", weight="bold")



def _draw_sync_page(fig, metrics, seat_names):
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Crew Timing & Sync (Trace-Derived)", ha="center", fontsize=18, fontweight="bold")

    rowers = [_seat_name(i, seat_names) for i in range(8)]
    x = np.arange(8)
    ax1 = fig.add_axes([0.06, 0.55, 0.42, 0.33])
    ax1.bar(x, metrics["xcorr_lag_ms"], color=base.SEAT_COLORS)
    ax1.axhline(0, color="#333", linewidth=0.8)
    ax1.set_title("Phase Lag vs Stroke Rower (Cross-Correlation)", fontsize=12)
    ax1.set_xlabel("Rower")
    ax1.set_ylabel("Lag (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rowers, rotation=30, ha="right", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = fig.add_axes([0.54, 0.55, 0.40, 0.33])
    ax2.bar(x, metrics["xcorr_peak"], color=base.SEAT_COLORS)
    ax2.set_title("Waveform Coherence vs Stroke Rower", fontsize=12)
    ax2.set_xlabel("Rower")
    ax2.set_ylabel("Max Cross-Corr")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rowers, rotation=30, ha="right", fontsize=8)
    ax2.grid(True, axis="y", alpha=0.25)

    ax3 = fig.add_axes([0.06, 0.24, 0.88, 0.22])
    ax3.axis("off")
    headers = ["Rower", "Catch Lag Mean (ms)", "Catch Lag SD (ms)", "XCorr Lag (ms)", "XCorr Peak"]
    rows = []
    for i in range(8):
        rows.append([
            _seat_name(i, seat_names),
            f"{metrics['catch_lag_mean'][i]:+.1f}" if np.isfinite(metrics['catch_lag_mean'][i]) else "-",
            f"{metrics['catch_lag_std'][i]:.1f}" if np.isfinite(metrics['catch_lag_std'][i]) else "-",
            f"{metrics['xcorr_lag_ms'][i]:+.1f}" if np.isfinite(metrics['xcorr_lag_ms'][i]) else "-",
            f"{metrics['xcorr_peak'][i]:.3f}" if np.isfinite(metrics['xcorr_peak'][i]) else "-",
        ])
    tbl = ax3.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.4)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1f2d3d")
        tbl[0, j].set_text_props(color="white", weight="bold")

    fig.text(0.06, 0.17,
             f"Catch spread mean: {metrics['catch_spread_mean']:.1f} ms   |   Catch spread stability (SD): {metrics['catch_spread_std']:.1f} ms",
             fontsize=10, color="#444")



def _draw_boat_dynamics(fig, trace, metrics):
    fig.patch.set_facecolor("white")
    ax2 = fig.add_axes([0.07, 0.24, 0.88, 0.64])
    ax2.set_title("Boat Attitude by Stroke Phase (Signed Roll, Pitch, Yaw)", fontsize=14)
    x = metrics["centers"]
    ax2.plot(x, metrics["roll_phase"], color="#8e44ad", linewidth=1.5, label="Roll")
    ax2.plot(x, metrics["pitch_phase"], color="#16a085", linewidth=1.5, label="Pitch")
    ax2.plot(x, metrics["yaw_phase"], color="#f39c12", linewidth=1.5, label="Yaw")
    ax2.axhline(0, color="#333", linewidth=0.8)
    ax2.set_ylabel("Signed angle")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=9)

    fig.text(
        0.07,
        0.11,
        f"Drive accel mean: {metrics['drive_accel_mean']:.3f}   |   Recovery accel mean: {metrics['recovery_accel_mean']:.3f}",
        fontsize=10,
        color="#444",
    )


def _draw_attitude_independent(fig, trace):
    fig.patch.set_facecolor("white")
    t = (trace["time"] - trace["time"][0]) / 1000.0

    ax1 = fig.add_axes([0.07, 0.69, 0.88, 0.22])
    ax1.set_title("Roll Angle Over Time", fontsize=14)
    ax1.plot(t, trace["roll"], color="#8e44ad", linewidth=1.3)
    ax1.axhline(0, color="#333", linewidth=0.8)
    ax1.set_ylabel("Roll")
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_axes([0.07, 0.40, 0.88, 0.22])
    ax2.set_title("Pitch Angle Over Time", fontsize=14)
    ax2.plot(t, trace["pitch"], color="#16a085", linewidth=1.3)
    ax2.axhline(0, color="#333", linewidth=0.8)
    ax2.set_ylabel("Pitch")
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_axes([0.07, 0.11, 0.88, 0.22])
    ax3.set_title("Yaw Angle Over Time", fontsize=14)
    ax3.plot(t, trace["yaw"], color="#f39c12", linewidth=1.3)
    ax3.axhline(0, color="#333", linewidth=0.8)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Yaw")
    ax3.grid(True, alpha=0.25)



def _draw_stability_consistency_fatigue(fig, metrics, seat_names):
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Stability, Shape Consistency, and Fatigue", ha="center",
             fontsize=18, fontweight="bold")

    ax1 = fig.add_axes([0.06, 0.70, 0.42, 0.18])
    ax1.axis("off")
    ax1.text(0.0, 0.95, "Stability / Set Metrics", fontsize=12, fontweight="bold")
    ax1.text(0.0, 0.72, f"Roll RMS: {metrics['roll_rms']:.3f}")
    ax1.text(0.0, 0.54, f"Roll Peak-to-Peak: {metrics['roll_p2p']:.3f}")
    ax1.text(0.0, 0.36, f"Yaw Drift (end-start): {metrics['yaw_drift']:+.3f}")
    ax1.text(0.0, 0.18, f"Yaw Oscillation SD: {metrics['yaw_osc']:.3f}")

    ax2 = fig.add_axes([0.54, 0.70, 0.40, 0.18])
    ax2.axis("off")
    ax2.text(0.0, 0.95, "Force Asymmetry Coupling", fontsize=12, fontweight="bold")
    ax2.text(0.0, 0.62, f"Corr(force asymmetry, |roll|): {metrics['asym_roll_corr']:.3f}")
    ax2.text(0.0, 0.36, f"Corr(force asymmetry, |yaw|):  {metrics['asym_yaw_corr']:.3f}")

    rowers = [_seat_name(i, seat_names) for i in range(8)]
    x = np.arange(8)
    ax3 = fig.add_axes([0.06, 0.40, 0.42, 0.24])
    ax3.bar(x, metrics["seat_repeatability"], color=base.SEAT_COLORS)
    ax3.set_title("Per-Rower Force-Shape Consistency")
    ax3.set_xlabel("Rower")
    ax3.set_xticks(x)
    ax3.set_xticklabels(rowers, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Repeatability score")
    ax3.grid(True, axis="y", alpha=0.25)

    ax4 = fig.add_axes([0.54, 0.40, 0.40, 0.24])
    ax4.bar(x, metrics["seat_shape_cv"], color=base.SEAT_COLORS)
    ax4.set_title("Per-Rower Shape Variability (CV%)")
    ax4.set_xlabel("Rower")
    ax4.set_xticks(x)
    ax4.set_xticklabels(rowers, rotation=30, ha="right", fontsize=8)
    ax4.set_ylabel("CV%")
    ax4.grid(True, axis="y", alpha=0.25)

    ax5 = fig.add_axes([0.06, 0.13, 0.88, 0.20])
    ax5.axis("off")
    headers = ["Rower", "Peak Fade % (Late vs Early)", "Timing SD Change (ms)"]
    rows = []
    for seat in range(8):
        rows.append([
            _seat_name(seat, seat_names),
            f"{metrics['fade_peak_pct'][seat]:+.1f}" if np.isfinite(metrics['fade_peak_pct'][seat]) else "-",
            f"{metrics['fade_timing_sd'][seat]:+.1f}" if np.isfinite(metrics['fade_timing_sd'][seat]) else "-",
        ])
    tbl = ax5.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.7)
    tbl.scale(1.0, 1.4)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1f2d3d")
        tbl[0, j].set_text_props(color="white", weight="bold")

    fig.text(0.06, 0.09, f"Crew shape-coherence score: {metrics['crew_coherence']:.1f}", fontsize=10, color="#444")



def _draw_rederived_validation(fig, metrics, strokes, seat_names):
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.95, "Re-derived Classic Metrics from Trace", ha="center",
             fontsize=18, fontweight="bold")

    # Summary-section baselines.
    drive_summary = np.nanmean(strokes["Drive Time"], axis=0)
    rec_summary = np.nanmean(strokes["Recovery Time"], axis=0)

    ax1 = fig.add_axes([0.06, 0.58, 0.40, 0.30])
    rowers = [_seat_name(i, seat_names) for i in range(8)]
    x = np.arange(8)
    w = 0.35
    ax1.bar(x - w / 2, drive_summary, width=w, color="#3498db", label="Summary Drive")
    ax1.bar(x + w / 2, metrics["drive_time_re"], width=w, color="#2ecc71", label="Trace Drive")
    ax1.set_title("Drive Time Validation")
    ax1.set_xlabel("Rower")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rowers, rotation=30, ha="right", fontsize=8)
    ax1.set_ylabel("Seconds")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(fontsize=8)

    ax2 = fig.add_axes([0.54, 0.58, 0.40, 0.30])
    ax2.bar(x - w / 2, rec_summary, width=w, color="#9b59b6", label="Summary Recovery")
    ax2.bar(x + w / 2, metrics["recovery_time_re"], width=w, color="#f39c12", label="Trace Recovery")
    ax2.set_title("Recovery Time Validation")
    ax2.set_xlabel("Rower")
    ax2.set_xticks(x)
    ax2.set_xticklabels(rowers, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("Seconds")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(fontsize=8)

    ax3 = fig.add_axes([0.06, 0.27, 0.88, 0.24])
    ax3.axis("off")
    headers = ["Rower", "Catch Phase", "Finish Phase", "Drive Δ (trace-summary)", "Recovery Δ (trace-summary)"]
    rows = []
    for seat in range(8):
        d_delta = metrics["drive_time_re"][seat] - drive_summary[seat]
        r_delta = metrics["recovery_time_re"][seat] - rec_summary[seat]
        rows.append([
            _seat_name(seat, seat_names),
            f"{metrics['catch_phase'][seat]:.1f}" if np.isfinite(metrics['catch_phase'][seat]) else "-",
            f"{metrics['finish_phase'][seat]:.1f}" if np.isfinite(metrics['finish_phase'][seat]) else "-",
            f"{d_delta:+.3f}" if np.isfinite(d_delta) else "-",
            f"{r_delta:+.3f}" if np.isfinite(r_delta) else "-",
        ])
    tbl = ax3.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    tbl.scale(1.0, 1.4)
    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1f2d3d")
        tbl[0, j].set_text_props(color="white", weight="bold")



def _generate_trace_pages(trace, seat_names, force_time_pdf, vel_pdf, force_angle_pdf, remaining_pdf):
    """Generate four PDFs: force-time (1p), velocity (1p), force-angle (1p), remaining trace (3p)."""
    metrics = _compute_trace_metrics(trace)

    with PdfPages(force_time_pdf) as pdf:
        fig = plt.figure(figsize=(16.5, 11.7))
        _draw_force_curves_time(fig, trace, metrics, seat_names)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(vel_pdf) as pdf:
        fig = plt.figure(figsize=(16.5, 11.7))
        _draw_angle_kinematics(fig, metrics, seat_names)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(force_angle_pdf) as pdf:
        fig = plt.figure(figsize=(16.5, 11.7))
        _draw_force_curves_angle(fig, trace, metrics, seat_names)
        pdf.savefig(fig)
        plt.close(fig)

    with PdfPages(remaining_pdf) as pdf:
        fig = plt.figure(figsize=(16.5, 11.7))
        _draw_boat_dynamics(fig, trace, metrics)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(16.5, 11.7))
        _draw_attitude_independent(fig, trace)
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(16.5, 11.7))
        _draw_breaker(fig)
        pdf.savefig(fig)
        plt.close(fig)


def _merge_pdfs(segments, output_path):
    """Merge PDFs with optional page ranges. Each segment is either a Path or a
    (Path, page_range_str) tuple where page_range_str uses qpdf syntax (e.g. '1-8', '9-z')."""
    cmd = ["qpdf", "--empty", "--pages"]
    for item in segments:
        if isinstance(item, tuple):
            path, page_range = item
            cmd += [str(path), page_range]
        else:
            cmd += [str(item), "1-z"]
    cmd += ["--", str(output_path)]
    subprocess.run(cmd, check=True)


def generate_combined(csv_path, with_ml=False):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = list(csv.reader(csv_path.open()))
    seat_names = _extract_crew_names(rows)

    print(f"Parsing baseline strokes from: {csv_path.name}")
    strokes = _parse_baseline_strokes(rows)
    strokes["seat_names"] = seat_names

    trace = _parse_trace_section(rows)
    if trace is None:
        raise RuntimeError("Could not parse Gate trace section in CSV")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem
    final_pdf = OUTPUT_DIR / f"{stem}-trace-extended-statsheet.pdf"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        baseline_pdf = tmp / "baseline.pdf"
        force_time_pdf = tmp / "force_time.pdf"
        vel_pdf = tmp / "vel.pdf"
        force_angle_pdf = tmp / "force_angle.pdf"
        remaining_trace_pdf = tmp / "remaining.pdf"

        print("Generating baseline extended statsheet...")
        if (not with_ml) and DISABLE_BASELINE_ML and hasattr(base, "_compute_speed_factors"):
            base._compute_speed_factors = lambda *args, **kwargs: None
        base.generate_pdf(strokes, baseline_pdf, stem)

        print("Generating trace analytics pages...")
        _generate_trace_pages(trace, seat_names, force_time_pdf, vel_pdf, force_angle_pdf, remaining_trace_pdf)

        print("Merging into final PDF...")
        # Layout: baseline 1-15, trace analytics, then baseline heatmaps+ pages.
        _merge_pdfs([
            (baseline_pdf, "1-15"),
            (force_time_pdf, "1-z"),
            (vel_pdf, "1-z"),
            (force_angle_pdf, "1-z"),
            (remaining_trace_pdf, "1-z"),
            (baseline_pdf, "16-z"),
        ], final_pdf)

    print(f"\nCombined report saved to {final_pdf}")
    return final_pdf


def interactive_mode():
    print("=" * 54)
    print("  TRACE + EXTENDED STATSHEET GENERATOR")
    print("=" * 54)
    print()

    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV files found in {DATA_DIR}")

    print("Available CSV files:")
    for i, f in enumerate(csvs, 1):
        print(f"  {i}. {f.name}")
    print("  a. All files")
    print()

    if len(csvs) == 1:
        print(f"Using: {csvs[0].name}")
        return csvs[0]

    choice = input("Enter CSV number (or 'a' for all): ").strip().lower()
    if choice == "a":
        return csvs

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(csvs):
            raise ValueError
        return csvs[idx]
    except ValueError:
        sys.exit("Invalid selection")


def _process_one(csv_path, with_ml=False):
    print(f"\nProcessing {Path(csv_path).name}...")
    return generate_combined(csv_path, with_ml=with_ml)


def main():
    parser = argparse.ArgumentParser(description="Generate combined baseline+trace statsheet")
    parser.add_argument("--csv",
                        help="CSV path (e.g. new-statsheet/18APR-race - trace-work.csv)")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Process all CSV files in statsheet/data")
    parser.add_argument("--with-ml", action="store_true",
                        help="Enable baseline ML speed-factor analysis pages")
    args = parser.parse_args()

    if args.all:
        csvs = sorted(DATA_DIR.glob("*.csv"))
        if not csvs:
            sys.exit(f"No CSV files found in {DATA_DIR}")
        print(f"Processing {len(csvs)} CSV files...")
        done = 0
        for p in csvs:
            try:
                _process_one(p, with_ml=args.with_ml)
                done += 1
            except Exception as exc:
                print(f"  Skipped {p.name}: {exc}")
        print(f"\nDone — generated {done} combined statsheets.")
        return

    if not args.csv:
        result = interactive_mode()
        if isinstance(result, list):
            done = 0
            for p in result:
                try:
                    _process_one(p, with_ml=args.with_ml)
                    done += 1
                except Exception as exc:
                    print(f"  Skipped {p.name}: {exc}")
            print(f"\nDone — generated {done} combined statsheets.")
            return
        _process_one(result, with_ml=args.with_ml)
        return

    csv_input = Path(args.csv)
    csv_path = csv_input if csv_input.exists() else DATA_DIR / args.csv
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    _process_one(csv_path, with_ml=args.with_ml)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.exit(f"Error: {exc}")
