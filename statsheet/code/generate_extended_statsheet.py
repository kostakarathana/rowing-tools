#!/usr/bin/env python3
"""Generate an extended multi-page landscape PDF summary of a rowing session.

Includes everything from the standard statsheet, plus:
  - Crew Timing Sync page
  - Drive:Recovery Ratio page
  - Seat Correlation Heatmap
  - Composite Consistency Score table
  - Force Curve & Angle-at-Max-Force page

Usage:
    python generate_extended_statsheet.py                      # interactive mode
    python generate_extended_statsheet.py --csv "session.csv"  # flags mode
"""

import argparse
import csv
import sys
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore", message=".*empty slice.*")
warnings.filterwarnings("ignore", message=".*Degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

SEAT_COLORS = [
    "#00e640",  # 1 - bright green
    "#e60000",  # 2 - red
    "#00008b",  # 3 - dark blue
    "#000000",  # 4 - black
    "#006400",  # 5 - dark green
    "#800000",  # 6 - maroon
    "#5b9bd5",  # 7 - lighter blue
    "#ff8c00",  # 8 - orange
]

# Green (good) -> Orange (mid) -> Red (bad)
GOOD_CMAP = LinearSegmentedColormap.from_list("good", ["#e74c3c", "#f39c12", "#2ecc71"])
BAD_CMAP = LinearSegmentedColormap.from_list("bad", ["#2ecc71", "#f39c12", "#e74c3c"])


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_csv(csv_path):
    """Parse the CSV and return per-stroke data for all metrics."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_row = None
    for i, row in enumerate(rows):
        if len(row) > 1 and row[1] == "SwivelPower":
            header_row = i
            break

    if header_row is None:
        sys.exit("Could not find stroke data header row in CSV")

    header = rows[header_row]

    def find_cols(name):
        for idx, col in enumerate(header):
            if col == name:
                return idx, idx + 8
        sys.exit(f"Metric '{name}' not found in header")

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
        if all(v == "" for v in row[swivel_start:swivel_end]):
            continue

        for name, (start, end) in metrics.items():
            vals = [float(v) if v else 0.0 for v in row[start:end]]
            strokes[name].append(vals)

        strokes["stroke_num"].append(int(float(row[-4])) if row[-4] else 0)
        strokes["rating"].append(float(row[-6]) if row[-6] else 0.0)
        strokes["boat_speed"].append(float(row[-5]) if row[-5] else 0.0)

    for name in metrics:
        strokes[name] = np.array(strokes[name])

    strokes = _remove_outliers(strokes, list(metrics.keys()))
    strokes["dead_seats"] = _detect_dead_seats(strokes, list(metrics.keys()))

    # Parse rower names from column N (index 13), rows 0-8
    seat_names = []
    for r in range(9):
        if r < len(rows) and len(rows[r]) > 13 and rows[r][13].strip():
            seat_names.append(rows[r][13].strip())
        else:
            seat_names.append(f"Seat {r + 1}" if r < 8 else "Cox")
    strokes["seat_names"] = seat_names  # 0-7 = seats, 8 = cox

    return strokes


def _remove_outliers(strokes, metric_names, iqr_factor=10.0):
    """Replace malfunction-level outlier values with NaN per seat per metric."""
    n_seats = 8
    total_nans = 0

    iqr_floors = {
        "SwivelPower": 30.0,
        "Rower Swivel Power": 30.0,
        "MinAngle": 3.0,
        "MaxAngle": 3.0,
        "CatchSlip": 2.0,
        "FinishSlip": 2.0,
        "Drive Time": 0.1,
        "Recovery Time": 0.1,
    }
    default_floor = 5.0

    for name in metric_names:
        data = strokes[name].astype(float)
        floor = iqr_floors.get(name, default_floor)
        for seat in range(n_seats):
            col = data[:, seat]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = max(q3 - q1, floor)
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            bad = (col < lower) | (col > upper)
            n_bad = bad.sum()
            if n_bad > 0:
                data[bad, seat] = np.nan
                total_nans += n_bad
        strokes[name] = data

    if total_nans > 0:
        print(f"  Masked {total_nans} malfunction-level outlier values as NaN")

    return strokes


def _detect_dead_seats(strokes, metric_names):
    """Detect seats with no real data (all zeros/identical) and NaN them out."""
    dead = set()
    power = strokes["SwivelPower"]
    for seat in range(8):
        col = power[:, seat]
        if np.nanstd(col) == 0 or np.all(col == 0) or np.all(np.isnan(col)):
            dead.add(seat)

    if dead:
        seat_names = ", ".join(f"Seat {s + 1}" for s in sorted(dead))
        print(f"  Dead seats (no data): {seat_names}")
        for name in metric_names:
            for seat in dead:
                strokes[name][:, seat] = np.nan

    return dead


def _seat_label(seat, strokes):
    """Return the rower name for a seat index, falling back to 'Seat N'."""
    names = strokes.get("seat_names", [])
    if seat < len(names):
        return names[seat]
    return f"Seat {seat + 1}"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cell_color(value, all_values, higher_is_better):
    """Return a background color: green for good, red for bad, orange for middle."""
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
    if vmax == vmin:
        return "#ffffff"
    norm = (value - vmin) / (vmax - vmin)
    cmap = GOOD_CMAP if higher_is_better else BAD_CMAP
    r, g, b, _ = cmap(norm)
    r, g, b = 0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b
    return (r, g, b)


def _smooth(data, window=7):
    """Exponentially weighted moving average, NaN-aware."""
    weights = np.exp(np.linspace(-1, 0, window))
    weights /= weights.sum()
    out = np.full_like(data, np.nan)
    for i in range(len(data)):
        start = max(0, i - window + 1)
        chunk = data[start:i + 1]
        w = weights[-(i - start + 1):]
        valid = ~np.isnan(chunk)
        if valid.any():
            out[i] = np.average(chunk[valid], weights=w[valid])
    return out


# ---------------------------------------------------------------------------
# Original statsheet pages
# ---------------------------------------------------------------------------

def _draw_summary_table(fig, strokes, overall_length, effective_length, session_name):
    """Page 1: Clean summary table like the coach's sheet."""
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.78])
    ax.axis("off")

    avg_rate = np.mean(strokes["rating"])
    n_strokes_total = len(strokes["stroke_num"])

    fig.text(0.5, 0.95, session_name, fontsize=18, fontweight="bold",
             ha="center", va="top", family="sans-serif")
    fig.text(0.5, 0.91, f"Rate: {avg_rate:.1f}   |   {n_strokes_total} strokes",
             fontsize=11, ha="center", va="top", color="#555", family="sans-serif")

    metric_defs = [
        ("Avg Watts", strokes["SwivelPower"], True),
        ("Overall Len", overall_length, True),
        ("Effective Len", effective_length, True),
        ("Catch Slip", strokes["CatchSlip"], False),
        ("Finish Slip", strokes["FinishSlip"], False),
        ("Min Angle", strokes["MinAngle"], False),
        ("Max Angle", strokes["MaxAngle"], True),
    ]

    col_labels = ["Name"] + [d[0] for d in metric_defs]
    for d in metric_defs:
        col_labels.append(f"{d[0]} std")

    n_cols = len(col_labels)

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    cell_data = []
    for seat in range(8):
        if seat in dead:
            row = [_seat_label(seat, strokes)] + ["--"] * (len(metric_defs) * 2)
        else:
            row = [_seat_label(seat, strokes)]
            for _, data, _ in metric_defs:
                row.append(f"{np.nanmean(data[:, seat]):.1f}")
            for _, data, _ in metric_defs:
                row.append(f"{np.nanstd(data[:, seat]):.1f}")
        cell_data.append(row)

    boat_row = ["Boat Avg"]
    for _, data, _ in metric_defs:
        boat_row.append(f"{np.nanmean(data[:, live_seats]):.1f}")
    for _, data, _ in metric_defs:
        boat_row.append(f"{np.nanmean([np.nanstd(data[:, s]) for s in live_seats]):.1f}")
    cell_data.append(boat_row)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)

    for i in range(9):
        if i < 8 and i in dead:
            table[i + 1, 0].set_facecolor("#d5d5d5")
            table[i + 1, 0].set_text_props(fontweight="bold", color="#999")
            for j in range(1, n_cols):
                table[i + 1, j].set_facecolor("#d5d5d5")
                table[i + 1, j].set_text_props(color="#999")
            continue

        table[i + 1, 0].set_facecolor("#ecf0f1" if i < 8 else "#bdc3c7")
        table[i + 1, 0].set_text_props(fontweight="bold")

        for j, (_, data, higher_is_better) in enumerate(metric_defs):
            col_idx = j + 1
            all_avgs = np.array([np.nanmean(data[:, s]) for s in live_seats])
            if i < 8:
                val = np.nanmean(data[:, i])
            else:
                val = np.nanmean(data[:, live_seats])
            color = _cell_color(val, all_avgs, higher_is_better)
            table[i + 1, col_idx].set_facecolor(color)

        for j, (_, data, _) in enumerate(metric_defs):
            col_idx = len(metric_defs) + j + 1
            all_stds = np.array([np.nanstd(data[:, s]) for s in live_seats])
            if i < 8:
                val = np.nanstd(data[:, i])
            else:
                val = np.nanmean([np.nanstd(data[:, s]) for s in live_seats])
            color = _cell_color(val, all_stds, higher_is_better=False)
            table[i + 1, col_idx].set_facecolor(color)

        if i == 8:
            for j in range(n_cols):
                table[i + 1, j].set_text_props(fontweight="bold")
                if j > 0:
                    table[i + 1, j].set_edgecolor("#2c3e50")


def _draw_angle_page(fig, strokes):
    """Page 2: Box-and-whisker angle plot per seat."""
    ax = fig.add_axes([0.08, 0.1, 0.84, 0.75])

    fig.text(0.5, 0.95, "Stroke Arc Breakdown",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.91, "Red = slip (wasted angle)  |  Green = effective arc",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    n_seats = 8
    avg_min = np.nanmean(strokes["MinAngle"], axis=0)
    avg_max = np.nanmean(strokes["MaxAngle"], axis=0)
    avg_catch = np.nanmean(strokes["CatchSlip"], axis=0)
    avg_finish = np.nanmean(strokes["FinishSlip"], axis=0)

    bar_height = 0.55
    y = np.arange(n_seats)

    dead = strokes.get("dead_seats", set())

    for i in range(n_seats):
        if i in dead:
            ax.text(0, y[i], "N/A", ha="center", va="center", fontsize=10, color="#ccc")
            continue

        catch_left = avg_min[i]
        catch_right = avg_min[i] + avg_catch[i]
        finish_left = avg_max[i] - avg_finish[i]
        finish_right = avg_max[i]
        eff_left = catch_right
        eff_right = finish_left

        ax.barh(y[i], avg_catch[i], left=catch_left, height=bar_height,
                color="#e74c3c", edgecolor="#c0392b", linewidth=0.8, alpha=0.9)
        ax.text(catch_left + avg_catch[i] / 2, y[i], f"{avg_catch[i]:.1f}",
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        eff_len = eff_right - eff_left
        ax.barh(y[i], eff_len, left=eff_left, height=bar_height,
                color="#2ecc71", edgecolor="#27ae60", linewidth=0.8, alpha=0.9)
        ax.text((eff_left + eff_right) / 2, y[i], f"{eff_len:.1f}",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        ax.barh(y[i], avg_finish[i], left=finish_left, height=bar_height,
                color="#e74c3c", edgecolor="#c0392b", linewidth=0.8, alpha=0.9)
        ax.text(finish_left + avg_finish[i] / 2, y[i], f"{avg_finish[i]:.1f}",
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        min_std = np.nanstd(strokes["MinAngle"][:, i])
        max_std = np.nanstd(strokes["MaxAngle"][:, i])
        ax.errorbar(avg_min[i], y[i], xerr=[[min_std], [0]], fmt="none",
                     ecolor="#555", capsize=5, linewidth=1.2)
        ax.errorbar(avg_max[i], y[i], xerr=[[0], [max_std]], fmt="none",
                     ecolor="#555", capsize=5, linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels([_seat_label(i, strokes) for i in range(n_seats)], fontsize=10)
    ax.set_xlabel("Angle (degrees)", fontsize=11)
    ax.axvline(x=0, color="#aaa", linewidth=0.5, linestyle="--")
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)


def _draw_metric_page(fig, data, title, strokes, higher_is_better):
    """One page per metric: 8 individual subplots with smoothed lines."""
    fig.text(0.5, 0.97, title, fontsize=16, fontweight="bold",
             ha="center", family="sans-serif")

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.2,
                          left=0.07, right=0.95, top=0.92, bottom=0.05)

    n_strokes = data.shape[0]
    x = np.arange(n_strokes)
    boat_smooth = _smooth(np.nanmean(data, axis=1))
    global_min = np.nanmin(data) * (0.95 if np.nanmin(data) > 0 else 1.05)
    global_max = np.nanmax(data) * (1.05 if np.nanmax(data) > 0 else 0.95)

    dead = strokes.get("dead_seats", set())

    for seat in range(8):
        row, col = divmod(seat, 2)
        ax = fig.add_subplot(gs[row, col])

        if seat in dead:
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#ccc", fontweight="bold")
            ax.set_title(_seat_label(seat, strokes), fontsize=9, fontweight="bold", color="#ccc")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
            continue

        seat_data = data[:, seat]
        seat_avg = np.nanmean(seat_data)
        seat_std = np.nanstd(seat_data)
        seat_smooth = _smooth(seat_data)

        ax.plot(x, seat_smooth, color=SEAT_COLORS[seat], linewidth=1.5, alpha=0.9)
        ax.plot(x, boat_smooth, color="#aaa", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(y=seat_avg, color=SEAT_COLORS[seat], linewidth=0.8,
                    linestyle=":", alpha=0.6)

        ax.set_ylim(global_min, global_max)
        ax.set_title(f"{_seat_label(seat, strokes)}   |   avg: {seat_avg:.1f}   std: {seat_std:.1f}",
                     fontsize=9, fontweight="bold", color=SEAT_COLORS[seat])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

        if row == 3:
            ax.set_xlabel("Stroke", fontsize=7)


# ---------------------------------------------------------------------------
# NEW PAGE: Crew Timing Sync
# ---------------------------------------------------------------------------

def _draw_crew_timing_page(fig, strokes):
    """Show each seat's Drive Start T relative to crew average, stroke by stroke.

    A tight band around zero = well-synchronised crew.
    Persistent offsets reveal who is consistently early/late off the catch.
    """
    fig.text(0.5, 0.97, "Crew Timing Sync — Drive Start Offset",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Each seat's Drive Start T minus stroke seat (ms)  |  0 = perfectly in sync",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]
    dst = strokes["Drive Start T"]  # (n_strokes, 8)
    n_strokes = dst.shape[0]
    x = np.arange(n_strokes)

    # Use stroke seat (seat 8, index 7) as reference
    stroke_seat_ref = dst[:, 7]  # (n_strokes,)
    offsets = dst - stroke_seat_ref[:, np.newaxis]  # (n_strokes, 8)

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.2,
                          left=0.07, right=0.95, top=0.90, bottom=0.05)

    # Global y limits across all live seats
    live_offsets = offsets[:, live_seats]
    margin = np.nanmax(np.abs(live_offsets)) * 1.3
    if margin == 0:
        margin = 1.0

    for seat in range(8):
        row, col = divmod(seat, 2)
        ax = fig.add_subplot(gs[row, col])

        if seat in dead:
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#ccc", fontweight="bold")
            ax.set_title(_seat_label(seat, strokes), fontsize=9, fontweight="bold", color="#ccc")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
            continue

        seat_off = offsets[:, seat]
        avg_off = np.nanmean(seat_off)
        std_off = np.nanstd(seat_off)
        smoothed = _smooth(seat_off)

        # Fill positive (late) red, negative (early) blue
        ax.fill_between(x, 0, smoothed, where=smoothed >= 0,
                        color="#e74c3c", alpha=0.3, interpolate=True)
        ax.fill_between(x, 0, smoothed, where=smoothed < 0,
                        color="#3498db", alpha=0.3, interpolate=True)
        ax.plot(x, smoothed, color=SEAT_COLORS[seat], linewidth=1.5)
        ax.axhline(y=0, color="#aaa", linewidth=0.8, linestyle="--")
        ax.axhline(y=avg_off, color=SEAT_COLORS[seat], linewidth=0.8,
                    linestyle=":", alpha=0.6)

        ax.set_ylim(-margin, margin)
        direction = "late" if avg_off > 0 else "early"
        ax.set_title(
            f"{_seat_label(seat, strokes)}   |   avg: {avg_off:+.1f}ms ({direction})   std: {std_off:.1f}ms",
            fontsize=8, fontweight="bold", color=SEAT_COLORS[seat])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

        if row == 3:
            ax.set_xlabel("Stroke", fontsize=7)


# ---------------------------------------------------------------------------
# NEW PAGE: Drive : Recovery Ratio
# ---------------------------------------------------------------------------

def _draw_drive_recovery_page(fig, strokes):
    """Drive Time / Recovery Time plotted per seat over the piece."""
    fig.text(0.5, 0.97, "Drive : Recovery Ratio",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Lower ratio = more time on recovery (controlled slide)  |  "
             "Dashed = boat average",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    drive = strokes["Drive Time"]
    recovery = strokes["Recovery Time"]
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(recovery > 0, drive / recovery, np.nan)

    n_strokes = ratio.shape[0]
    x = np.arange(n_strokes)

    boat_smooth = _smooth(np.nanmean(ratio[:, live_seats], axis=1))

    live_ratios = ratio[:, live_seats]
    global_min = max(0, np.nanmin(live_ratios) * 0.9)
    global_max = np.nanmax(live_ratios) * 1.1

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.2,
                          left=0.07, right=0.95, top=0.90, bottom=0.05)

    for seat in range(8):
        row, col = divmod(seat, 2)
        ax = fig.add_subplot(gs[row, col])

        if seat in dead:
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#ccc", fontweight="bold")
            ax.set_title(_seat_label(seat, strokes), fontsize=9, fontweight="bold", color="#ccc")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
            continue

        seat_data = ratio[:, seat]
        seat_avg = np.nanmean(seat_data)
        seat_std = np.nanstd(seat_data)
        seat_smooth = _smooth(seat_data)

        ax.plot(x, seat_smooth, color=SEAT_COLORS[seat], linewidth=1.5, alpha=0.9)
        ax.plot(x, boat_smooth, color="#aaa", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(y=seat_avg, color=SEAT_COLORS[seat], linewidth=0.8,
                    linestyle=":", alpha=0.6)
        # Ideal 1:2 line
        ax.axhline(y=0.5, color="#2ecc71", linewidth=0.6, linestyle="--", alpha=0.4)

        ax.set_ylim(global_min, global_max)
        ax.set_title(
            f"{_seat_label(seat, strokes)}   |   avg: {seat_avg:.2f}   std: {seat_std:.2f}",
            fontsize=9, fontweight="bold", color=SEAT_COLORS[seat])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

        if row == 3:
            ax.set_xlabel("Stroke", fontsize=7)


# ---------------------------------------------------------------------------
# NEW PAGE: Seat Correlation Heatmap
# ---------------------------------------------------------------------------

def _compute_corr_matrix(data, dead):
    """Compute 8x8 Pearson correlation matrix between seats for a metric."""
    n_seats = 8
    corr = np.full((n_seats, n_seats), np.nan)
    for i in range(n_seats):
        for j in range(n_seats):
            if i in dead or j in dead:
                continue
            a, b = data[:, i], data[:, j]
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 3:
                continue
            corr[i, j] = np.corrcoef(a[valid], b[valid])[0, 1]
    return corr


def _draw_correlation_page(fig, strokes, metric_key=None, metric_label=None,
                           data=None, corr_matrix=None, full_matrix=False):
    """Pearson correlation matrix of a metric between all seat pairs.

    Can be called with:
      - metric_key (looks up strokes[metric_key]) OR
      - data (raw ndarray) OR
      - corr_matrix (pre-computed 8x8 matrix, used for the combined page)
    """
    if metric_label is None:
        metric_label = metric_key or "Combined"

    title = f"Seat Correlation — {metric_label}"
    fig.text(0.5, 0.97, title,
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             f"Pearson r of {metric_label} between each seat pair  |  "
             "1.0 = perfectly correlated",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    n_seats = 8

    if corr_matrix is not None:
        corr = corr_matrix
    else:
        if data is None:
            data = strokes[metric_key]
        corr = _compute_corr_matrix(data, dead)

    ax = fig.add_axes([0.15, 0.10, 0.65, 0.75])

    cmap = LinearSegmentedColormap.from_list(
        "corr", ["#e74c3c", "#27ae60"])

    # Optionally exclude lower triangle for display; always exclude diagonal
    corr_display = corr.copy()
    for i in range(n_seats):
        corr_display[i, i] = np.nan  # always hide diagonal
    if not full_matrix:
        for i in range(n_seats):
            for j in range(n_seats):
                if j < i:  # lower triangle
                    corr_display[i, j] = np.nan

    masked = np.ma.masked_invalid(corr_display)
    valid_vals = corr_display[~np.isnan(corr_display)]
    vmin = np.min(valid_vals) if len(valid_vals) > 0 else 0
    vmax = np.max(valid_vals) if len(valid_vals) > 0 else 1
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    labels = [_seat_label(i, strokes) for i in range(n_seats)]
    ax.set_xticks(range(n_seats))
    ax.set_yticks(range(n_seats))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n_seats):
        for j in range(n_seats):
            if i == j:
                continue
            if not full_matrix and j < i:
                continue
            val = corr[i, j]
            if np.isnan(val):
                ax.text(j, i, "--", ha="center", va="center",
                        fontsize=10, color="#bbb")
            else:
                text_color = "white" if val < 0.35 or val > 0.85 else "#2c3e50"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")


# ---------------------------------------------------------------------------
# NEW PAGE: Composite Consistency Score
# ---------------------------------------------------------------------------

def _draw_consistency_page(fig, strokes, overall_length, effective_length):
    """Composite consistency score: 0-100 per seat based on normalised std devs.

    Components (equal weight):
      - SwivelPower std
      - Effective Length std
      - Catch Slip std
      - Finish Slip std
    Lower std -> higher score.
    """
    fig.text(0.5, 0.95, "Composite Consistency Score",
             fontsize=18, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.91,
             "0 = least consistent  |  100 = most consistent  |  "
             "Based on std devs of Watts, Eff. Length, Catch Slip, Finish Slip",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    components = [
        ("Watts", strokes["SwivelPower"]),
        ("Eff. Length", effective_length),
        ("Catch Slip", strokes["CatchSlip"]),
        ("Finish Slip", strokes["FinishSlip"]),
    ]

    # Compute raw std per seat per component
    raw_stds = {}  # component_name -> array(8)
    for name, data in components:
        stds = np.array([np.nanstd(data[:, s]) for s in range(8)])
        raw_stds[name] = stds

    # For each component, normalise so that among live seats:
    #   best (lowest std) = 1.0, worst = 0.0
    normed = {}
    for name, stds in raw_stds.items():
        live_vals = stds[live_seats]
        vmin, vmax = np.nanmin(live_vals), np.nanmax(live_vals)
        if vmax == vmin:
            normed[name] = np.ones(8) * 100.0
        else:
            # Invert: lower std = higher score
            normed[name] = (1.0 - (stds - vmin) / (vmax - vmin)) * 100.0
        # Dead seats -> NaN
        for s in dead:
            normed[name][s] = np.nan

    # Composite = average of normalised scores
    composite = np.nanmean([normed[n] for n in normed], axis=0)

    # ---- Draw table ----
    ax = fig.add_axes([0.10, 0.15, 0.80, 0.65])
    ax.axis("off")

    col_labels = ["Name"] + [n for n in raw_stds] + \
                 [f"{n} Score" for n in raw_stds] + ["COMPOSITE"]
    n_cols = len(col_labels)

    cell_data = []
    for seat in range(8):
        if seat in dead:
            row = [_seat_label(seat, strokes)] + ["--"] * (n_cols - 1)
        else:
            row = [_seat_label(seat, strokes)]
            for name in raw_stds:
                row.append(f"{raw_stds[name][seat]:.2f}")
            for name in normed:
                row.append(f"{normed[name][seat]:.0f}")
            row.append(f"{composite[seat]:.0f}")
        cell_data.append(row)

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)

    # Color composite column and score columns
    n_components = len(raw_stds)
    for i in range(8):
        if i in dead:
            for j in range(n_cols):
                table[i + 1, j].set_facecolor("#d5d5d5")
                table[i + 1, j].set_text_props(color="#999")
            continue

        table[i + 1, 0].set_facecolor("#ecf0f1")
        table[i + 1, 0].set_text_props(fontweight="bold")

        # Raw std columns — lower is better
        for j, name in enumerate(raw_stds):
            col_idx = j + 1
            all_stds = np.array([raw_stds[name][s] for s in live_seats])
            color = _cell_color(raw_stds[name][i], all_stds, higher_is_better=False)
            table[i + 1, col_idx].set_facecolor(color)

        # Score columns — higher is better
        for j, name in enumerate(normed):
            col_idx = n_components + j + 1
            all_scores = np.array([normed[name][s] for s in live_seats])
            color = _cell_color(normed[name][i], all_scores, higher_is_better=True)
            table[i + 1, col_idx].set_facecolor(color)

        # Composite column
        comp_col = n_cols - 1
        all_comp = np.array([composite[s] for s in live_seats])
        color = _cell_color(composite[i], all_comp, higher_is_better=True)
        table[i + 1, comp_col].set_facecolor(color)
        table[i + 1, comp_col].set_text_props(fontweight="bold", fontsize=11)

    # ---- Bar chart underneath ----
    ax2 = fig.add_axes([0.15, 0.04, 0.70, 0.12])
    bars_x = []
    bars_h = []
    bars_c = []
    for s in range(8):
        if s in dead:
            continue
        bars_x.append(s)
        bars_h.append(composite[s])
        bars_c.append(SEAT_COLORS[s])

    ax2.bar([_seat_label(x, strokes) for x in bars_x], bars_h, color=bars_c, edgecolor="white", width=0.6)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Score", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(axis="y", alpha=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# NEW PAGE: Power Efficiency (Watts / Effective Length)
# ---------------------------------------------------------------------------

def _draw_power_efficiency_page(fig, strokes, effective_length):
    """Watts per degree of effective length — who gets the most from their arc."""
    fig.text(0.5, 0.97, "Power Efficiency (Watts per Degree of Effective Length)",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "SwivelPower / Effective Length  |  Higher = more power per degree of arc",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.where(effective_length > 0,
                              strokes["SwivelPower"] / effective_length, np.nan)

    n_strokes = efficiency.shape[0]
    x = np.arange(n_strokes)

    boat_smooth = _smooth(np.nanmean(efficiency[:, live_seats], axis=1))
    live_eff = efficiency[:, live_seats]
    global_min = np.nanmin(live_eff) * 0.9
    global_max = np.nanmax(live_eff) * 1.1

    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.2,
                          left=0.07, right=0.95, top=0.90, bottom=0.05)

    for seat in range(8):
        row, col = divmod(seat, 2)
        ax = fig.add_subplot(gs[row, col])

        if seat in dead:
            ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="#ccc", fontweight="bold")
            ax.set_title(_seat_label(seat, strokes), fontsize=9, fontweight="bold", color="#ccc")
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
            continue

        seat_data = efficiency[:, seat]
        seat_avg = np.nanmean(seat_data)
        seat_std = np.nanstd(seat_data)
        seat_smooth = _smooth(seat_data)

        ax.plot(x, seat_smooth, color=SEAT_COLORS[seat], linewidth=1.5, alpha=0.9)
        ax.plot(x, boat_smooth, color="#aaa", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(y=seat_avg, color=SEAT_COLORS[seat], linewidth=0.8,
                    linestyle=":", alpha=0.6)

        ax.set_ylim(global_min, global_max)
        ax.set_title(f"{_seat_label(seat, strokes)}   |   avg: {seat_avg:.2f}   std: {seat_std:.2f}",
                     fontsize=9, fontweight="bold", color=SEAT_COLORS[seat])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

        if row == 3:
            ax.set_xlabel("Stroke", fontsize=7)


# ---------------------------------------------------------------------------
# NEW PAGE: Boat Watts vs Boat Speed
# ---------------------------------------------------------------------------

def _draw_watts_vs_speed_page(fig, strokes):
    """Scatter + dual-axis time series of total boat watts vs GPS boat speed."""
    fig.text(0.5, 0.97, "Boat Watts vs Boat Speed",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "How does crew power translate to boat speed?",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    power = strokes["SwivelPower"]           # (n_strokes, 8)
    speed = np.array(strokes["boat_speed"])  # (n_strokes,)

    # Total boat watts = sum of live seats
    boat_watts = np.nansum(power[:, live_seats], axis=1)

    n_strokes = len(speed)
    x = np.arange(n_strokes)

    # Filter valid (both speed > 0 and watts > 0), exclude first 20 strokes
    valid = (speed > 0) & (~np.isnan(boat_watts)) & (boat_watts > 0)
    valid[:20] = False

    # ---- Top: Scatter plot of boat watts vs speed ----
    ax1 = fig.add_axes([0.08, 0.55, 0.84, 0.33])

    if valid.sum() >= 3:
        sv, wv = speed[valid], boat_watts[valid]
        r = np.corrcoef(sv, wv)[0, 1]

        ax1.scatter(wv, sv, s=20, alpha=0.5, color="#2c3e50", edgecolors="none")

        # Trend line
        z = np.polyfit(wv, sv, 1)
        p = np.poly1d(z)
        x_line = np.linspace(np.min(wv), np.max(wv), 50)
        ax1.plot(x_line, p(x_line), color="#e74c3c", linewidth=2, alpha=0.8)

        ax1.set_title(f"r = {r:.3f}   |   {valid.sum()} strokes",
                      fontsize=11, fontweight="bold")
    else:
        ax1.text(0.5, 0.5, "INSUFFICIENT DATA", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=14, color="#ccc",
                 fontweight="bold")

    ax1.set_xlabel("Boat Watts (sum of crew)", fontsize=10)
    ax1.set_ylabel("Boat Speed (m/s)", fontsize=10)
    ax1.grid(alpha=0.2)
    ax1.tick_params(labelsize=9)

    # ---- Bottom: Dual-axis time series (excluding first 20 strokes) ----
    ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.38])
    x_plot = x[20:]
    watts_smooth = _smooth(boat_watts)[20:]
    speed_smooth = _smooth(speed)[20:]

    color_watts = "#2c3e50"
    color_speed = "#e74c3c"

    ax2.plot(x_plot, watts_smooth, color=color_watts, linewidth=1.8, label="Boat Watts")
    ax2.set_xlabel("Stroke", fontsize=10)
    ax2.set_ylabel("Boat Watts", fontsize=10, color=color_watts)
    ax2.tick_params(axis="y", labelcolor=color_watts, labelsize=8)
    ax2.tick_params(axis="x", labelsize=8)
    ax2.grid(alpha=0.2)

    ax3 = ax2.twinx()
    ax3.plot(x_plot, speed_smooth, color=color_speed, linewidth=1.8, label="Boat Speed")
    ax3.set_ylabel("Boat Speed (m/s)", fontsize=10, color=color_speed)
    ax3.tick_params(axis="y", labelcolor=color_speed, labelsize=8)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    ax2.set_title("Watts and Speed over the piece (smoothed)",
                  fontsize=10, fontweight="bold")


# ---------------------------------------------------------------------------
# NEW PAGE: Catch vs Finish Spread
# ---------------------------------------------------------------------------

def _draw_catch_finish_spread_page(fig, strokes):
    """Compare crew spread at catch vs finish.

    Catch = Drive Start T offset from stroke seat.
    Finish = (Drive Start T + Drive Time) offset from stroke seat's finish.
    If the crew is tight at catch but loose at finish (or vice versa), this
    shows which end of the stroke needs work.
    """
    fig.text(0.5, 0.97, "Catch vs Finish Timing Spread",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Crew spread (std of offsets from stroke seat) at catch and finish  |  "
             "Lower = tighter",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    dst = strokes["Drive Start T"]       # (n_strokes, 8)
    drive = strokes["Drive Time"]        # (n_strokes, 8)

    # Catch offset from stroke seat
    catch_ref = dst[:, 7]
    catch_offsets = dst - catch_ref[:, np.newaxis]  # (n_strokes, 8)

    # Finish time = Drive Start T + Drive Time; offset from stroke seat's finish
    finish_times = dst + drive
    finish_ref = finish_times[:, 7]
    finish_offsets = finish_times - finish_ref[:, np.newaxis]  # (n_strokes, 8)

    n_strokes = dst.shape[0]
    x = np.arange(n_strokes)

    # Per-stroke crew spread (std across live seats)
    catch_spread = np.nanstd(catch_offsets[:, live_seats], axis=1)
    finish_spread = np.nanstd(finish_offsets[:, live_seats], axis=1)

    # ---- Top plot: crew spread over the piece ----
    ax1 = fig.add_axes([0.08, 0.55, 0.84, 0.32])
    catch_smooth = _smooth(catch_spread)
    finish_smooth = _smooth(finish_spread)

    ax1.plot(x, catch_smooth, color="#3498db", linewidth=2, label="Catch spread")
    ax1.plot(x, finish_smooth, color="#e74c3c", linewidth=2, label="Finish spread")
    ax1.fill_between(x, catch_smooth, finish_smooth,
                     where=finish_smooth > catch_smooth,
                     color="#e74c3c", alpha=0.1, interpolate=True)
    ax1.fill_between(x, catch_smooth, finish_smooth,
                     where=catch_smooth > finish_smooth,
                     color="#3498db", alpha=0.1, interpolate=True)
    ax1.set_ylabel("Crew spread (ms std)", fontsize=10)
    ax1.set_xlabel("Stroke", fontsize=9)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.2)
    ax1.tick_params(labelsize=8)

    avg_catch_spread = np.nanmean(catch_spread)
    avg_finish_spread = np.nanmean(finish_spread)
    worse_end = "FINISH" if avg_finish_spread > avg_catch_spread else "CATCH"
    ax1.set_title(
        f"Avg catch spread: {avg_catch_spread:.1f}ms   |   "
        f"Avg finish spread: {avg_finish_spread:.1f}ms   |   "
        f"Looser end: {worse_end}",
        fontsize=10, fontweight="bold")

    # ---- Bottom plot: per-rower catch vs finish avg offset ----
    ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.38])

    bar_width = 0.35
    seats_to_plot = [s for s in live_seats]
    x_bars = np.arange(len(seats_to_plot))

    catch_avgs = [np.nanmean(catch_offsets[:, s]) for s in seats_to_plot]
    finish_avgs = [np.nanmean(finish_offsets[:, s]) for s in seats_to_plot]
    catch_stds = [np.nanstd(catch_offsets[:, s]) for s in seats_to_plot]
    finish_stds = [np.nanstd(finish_offsets[:, s]) for s in seats_to_plot]

    ax2.bar(x_bars - bar_width / 2, catch_avgs, bar_width,
            yerr=catch_stds, capsize=3,
            color="#3498db", alpha=0.8, label="Catch offset", edgecolor="white")
    ax2.bar(x_bars + bar_width / 2, finish_avgs, bar_width,
            yerr=finish_stds, capsize=3,
            color="#e74c3c", alpha=0.8, label="Finish offset", edgecolor="white")

    ax2.set_xticks(x_bars)
    ax2.set_xticklabels([_seat_label(s, strokes) for s in seats_to_plot],
                        fontsize=9)
    ax2.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Avg offset from stroke seat (ms)", fontsize=9)
    ax2.set_title("Per-rower: avg catch & finish offset  |  "
                  "Positive = late, Negative = early",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.2)
    ax2.tick_params(labelsize=8)


# ---------------------------------------------------------------------------
# NEW PAGE: Stroke Heatmap
# ---------------------------------------------------------------------------

def _draw_stroke_heatmap_page(fig, strokes):
    """2D heatmap (strokes x seats) of SwivelPower — shows temporal patterns."""
    fig.text(0.5, 0.97, "Stroke Power Heatmap",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "SwivelPower by stroke and seat  |  Blue = low, Green = high",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    power = strokes["SwivelPower"].copy()  # (n_strokes, 8)

    ax = fig.add_axes([0.08, 0.10, 0.82, 0.78])

    cmap = LinearSegmentedColormap.from_list(
        "heat", ["#e74c3c", "#27ae60"])

    # Transpose so seats are rows, strokes are columns
    data = power.T  # (8, n_strokes)

    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=vmin, vmax=vmax)

    ax.set_yticks(range(8))
    ylabels = []
    for i in range(8):
        if i in dead:
            ylabels.append(f"{_seat_label(i, strokes)} (N/A)")
        else:
            ylabels.append(_seat_label(i, strokes))
    ax.set_yticklabels(ylabels, fontsize=10)

    # X axis: show every Nth stroke label
    n_strokes = power.shape[0]
    step = max(1, n_strokes // 15)
    xticks = list(range(0, n_strokes, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(strokes["stroke_num"][i]) for i in xticks], fontsize=7)
    ax.set_xlabel("Stroke #", fontsize=10)

    fig.colorbar(im, ax=ax, shrink=0.6, label="SwivelPower (watts)", pad=0.02)


# ---------------------------------------------------------------------------
# NEW PAGE: Technique Radar Chart
# ---------------------------------------------------------------------------

def _draw_radar_page(fig, strokes, effective_length):
    """Spider/radar chart per seat with normalised technique dimensions.

    Axes (all normalised 0-1 among live seats, higher = better):
      - Watts
      - Effective Length
      - Catch Slip (inverted — lower slip is better)
      - Finish Slip (inverted)
    """
    fig.text(0.5, 0.97, "Technique Radar",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Normalised across crew  |  Outer edge = best in crew",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    # Raw per-seat averages
    raw = {
        "Watts":       np.array([np.nanmean(strokes["SwivelPower"][:, s]) for s in range(8)]),
        "Eff. Length":  np.array([np.nanmean(effective_length[:, s]) for s in range(8)]),
        "Catch Slip":  np.array([np.nanmean(strokes["CatchSlip"][:, s]) for s in range(8)]),
        "Finish Slip": np.array([np.nanmean(strokes["FinishSlip"][:, s]) for s in range(8)]),
    }

    # higher_is_better flags
    hib = {
        "Watts": True, "Eff. Length": True,
        "Catch Slip": False, "Finish Slip": False,
    }

    # Normalise each dimension 0-1 (best = 1)
    normed = {}
    for key, vals in raw.items():
        live_vals = vals[live_seats]
        vmin, vmax = np.nanmin(live_vals), np.nanmax(live_vals)
        if vmax == vmin:
            normed[key] = np.ones(8) * 0.5
        else:
            n = (vals - vmin) / (vmax - vmin)
            if not hib[key]:
                n = 1.0 - n
            normed[key] = n
        for s in dead:
            normed[key][s] = np.nan

    categories = list(normed.keys())
    n_cat = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25,
                          left=0.04, right=0.96, top=0.89, bottom=0.04)

    for seat in range(8):
        row, col = divmod(seat, 4)
        ax = fig.add_subplot(gs[row, col], polar=True)

        if seat in dead:
            ax.set_title(_seat_label(seat, strokes), fontsize=9, fontweight="bold",
                         color="#ccc", pad=10)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            continue

        values = [normed[c][seat] for c in categories]
        values += values[:1]  # close

        ax.fill(angles, values, color=SEAT_COLORS[seat], alpha=0.2)
        ax.plot(angles, values, color=SEAT_COLORS[seat], linewidth=1.8)
        ax.scatter(angles[:-1], values[:-1], color=SEAT_COLORS[seat], s=20, zorder=5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=5, color="#999")
        ax.set_title(_seat_label(seat, strokes), fontsize=10, fontweight="bold",
                     color=SEAT_COLORS[seat], pad=12)
        ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# NEW PAGE: Work Distribution Profile
# ---------------------------------------------------------------------------

def _draw_work_distribution_page(fig, strokes):
    """Stacked bar chart of Work PC Q1–Q4 per seat showing force curve shape."""
    fig.text(0.5, 0.97, "Work Distribution Profile",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Force curve shape: Q1 (catch) → Q4 (finish)  |  "
             "Ideal ≈ even or front-loaded",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())

    q_keys = ["Work PC Q1", "Work PC Q2", "Work PC Q3", "Work PC Q4"]
    q_labels = ["Q1 (catch)", "Q2", "Q3", "Q4 (finish)"]
    q_colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    avgs = {}
    for k in q_keys:
        avgs[k] = np.array([np.nanmean(strokes[k][:, s]) if s not in dead else 0
                            for s in range(8)])

    ax = fig.add_axes([0.08, 0.10, 0.84, 0.78])

    x = np.arange(8)
    bottoms = np.zeros(8)

    for k, label, color in zip(q_keys, q_labels, q_colors):
        vals = avgs[k]
        bars = ax.bar(x, vals, bottom=bottoms, color=color, edgecolor="white",
                       linewidth=0.5, label=label, width=0.65)
        # Annotate inside each segment
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if i in dead or v < 1:
                continue
            ax.text(i, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([_seat_label(s, strokes) for s in range(8)], fontsize=10)
    ax.set_ylabel("Cumulative Work %", fontsize=11)
    ax.set_ylim(0, max(bottoms) * 1.08)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for s in dead:
        ax.text(s, 2, "NO DATA", ha="center", va="bottom", fontsize=8,
                color="#bbb", fontweight="bold")


# ---------------------------------------------------------------------------
# NEW PAGE: Force Application Window
# ---------------------------------------------------------------------------

def _draw_force_application_page(fig, strokes):
    """Horizontal range bars: Angle at Max Force and Angle at 0.7F per seat.

    Shows where in the arc each rower applies peak force.
    """
    fig.text(0.5, 0.97, "Force Application Window",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Where in the arc each seat applies peak force  |  "
             "Diamond = Angle at Max Force, Bar = 0.7× Force zone",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())

    angle_max_f = strokes["Angle Max F"]   # (n, 8)
    angle_07f = strokes["Angle 0.7 F"]     # (n, 8)

    ax = fig.add_axes([0.12, 0.08, 0.80, 0.82])

    y_positions = np.arange(8)

    for seat in range(8):
        if seat in dead:
            ax.text(0, seat, "NO DATA", ha="center", va="center",
                    fontsize=9, color="#bbb", fontweight="bold")
            continue

        max_f_avg = np.nanmean(angle_max_f[:, seat])
        max_f_std = np.nanstd(angle_max_f[:, seat])
        f07_avg = np.nanmean(angle_07f[:, seat])
        f07_std = np.nanstd(angle_07f[:, seat])

        # The 0.7F zone spans from the earlier angle to the later one
        left = min(max_f_avg, f07_avg)
        right = max(max_f_avg, f07_avg)
        width = right - left

        # Draw the 0.7F zone bar
        ax.barh(seat, width, left=left, height=0.5,
                color=SEAT_COLORS[seat], alpha=0.35, edgecolor=SEAT_COLORS[seat],
                linewidth=1.2)

        # Error bars for variability
        ax.errorbar(f07_avg, seat, xerr=f07_std, fmt="o", color=SEAT_COLORS[seat],
                     markersize=5, capsize=4, capthick=1, alpha=0.6)

        # Diamond for angle at max force
        ax.scatter(max_f_avg, seat, marker="D", s=80, color=SEAT_COLORS[seat],
                   zorder=5, edgecolors="white", linewidths=0.8)

        # Annotate
        ax.text(right + 0.8, seat, f"MaxF: {max_f_avg:.1f}°  0.7F: {f07_avg:.1f}°",
                va="center", fontsize=8, color=SEAT_COLORS[seat])

    ax.set_yticks(y_positions)
    ax.set_yticklabels([_seat_label(s, strokes) for s in range(8)], fontsize=10)
    ax.set_xlabel("Angle (degrees)", fontsize=11)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0, color="#ccc", linewidth=0.8, linestyle="--")


# ---------------------------------------------------------------------------
# NEW PAGE: Rate Response Curves
# ---------------------------------------------------------------------------

def _draw_rate_response_page(fig, strokes, effective_length):
    """Scatter of rating vs power and effective length with per-seat trend lines."""
    fig.text(0.5, 0.97, "Rate Response Curves",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "How each seat responds to rate changes  |  "
             "Steeper slope = better rate response",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    rating = np.array(strokes["rating"])

    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.25,
                          left=0.07, right=0.95, top=0.89, bottom=0.10)

    panels = [
        ("SwivelPower", "Power (watts)", strokes["SwivelPower"]),
        ("Eff. Length", "Effective Length (deg)", effective_length),
    ]

    for idx, (name, ylabel, data) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_xlabel("Rating (spm)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.2)

        for seat in range(8):
            if seat in dead:
                continue

            y = data[:, seat]
            valid = ~np.isnan(y) & ~np.isnan(rating) & (rating > 0)
            if valid.sum() < 5:
                continue

            r_valid = rating[valid]
            y_valid = y[valid]

            ax.scatter(r_valid, y_valid, color=SEAT_COLORS[seat], alpha=0.12,
                       s=10, rasterized=True)

            # Linear trend line
            coeffs = np.polyfit(r_valid, y_valid, 1)
            r_range = np.linspace(r_valid.min(), r_valid.max(), 50)
            ax.plot(r_range, np.polyval(coeffs, r_range),
                    color=SEAT_COLORS[seat], linewidth=2,
                    label=f"{_seat_label(seat, strokes)} ({coeffs[0]:+.1f}/spm)")

        ax.legend(fontsize=7, loc="best", ncol=2)


# ---------------------------------------------------------------------------
# NEW PAGE: Rolling Power Dashboard
# ---------------------------------------------------------------------------

def _draw_rolling_power_page(fig, strokes, effective_length):
    """10-stroke rolling average of power, effective length, and catch slip."""
    fig.text(0.5, 0.97, "Rolling Power Dashboard",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "10-stroke rolling average  |  Tracks drift and fatigue in real time",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    stroke_nums = np.array(strokes["stroke_num"])
    window = 10

    def rolling_avg(arr, w):
        """Simple rolling mean, NaN-aware."""
        out = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            start = max(0, i - w + 1)
            chunk = arr[start:i + 1]
            valid = ~np.isnan(chunk)
            if valid.sum() > 0:
                out[i] = np.nanmean(chunk)
        return out

    panels = [
        ("Power (watts)", strokes["SwivelPower"]),
        ("Effective Length (deg)", effective_length),
        ("Catch Slip (deg)", strokes["CatchSlip"]),
    ]

    gs = fig.add_gridspec(3, 1, hspace=0.35,
                          left=0.07, right=0.92, top=0.89, bottom=0.06)

    for idx, (ylabel, data) in enumerate(panels):
        ax = fig.add_subplot(gs[idx, 0])
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.2)

        if idx == 2:
            ax.set_xlabel("Stroke #", fontsize=9)

        for seat in range(8):
            if seat in dead:
                continue
            y = rolling_avg(data[:, seat], window)
            ax.plot(stroke_nums, y, color=SEAT_COLORS[seat], linewidth=1.2,
                    alpha=0.8, label=_seat_label(seat, strokes))

        if idx == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=8,
                      framealpha=0.7, handlelength=1)


# ---------------------------------------------------------------------------
# NEW PAGE: Power Quartile Fingerprint
# ---------------------------------------------------------------------------

def _draw_quartile_fingerprint_page(fig, strokes):
    """Radar/polar chart of Q1–Q4 work distribution per seat.

    Each seat gets a 4-spoke radar showing the shape of their force curve.
    """
    fig.text(0.5, 0.97, "Power Quartile Fingerprint",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Force curve shape per seat  |  Q1 (catch) → Q4 (finish)  |  "
             "Even = circular, Front-loaded = top-heavy",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    q_keys = ["Work PC Q1", "Work PC Q2", "Work PC Q3", "Work PC Q4"]
    q_labels = ["Q1\n(catch)", "Q2", "Q3", "Q4\n(finish)"]

    # Compute per-seat averages
    avgs = np.zeros((8, 4))
    for seat in range(8):
        if seat in dead:
            continue
        for qi, k in enumerate(q_keys):
            avgs[seat, qi] = np.nanmean(strokes[k][:, seat])

    # Normalise: find global min/max across all live seats for radar scaling
    live_vals = avgs[[s for s in range(8) if s not in dead], :]
    vmin = np.nanmin(live_vals)
    vmax = np.nanmax(live_vals)
    if vmax == vmin:
        vmax = vmin + 1

    n_cat = 4
    angles = np.linspace(0, 2 * np.pi, n_cat, endpoint=False).tolist()
    angles += angles[:1]

    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.25,
                          left=0.04, right=0.96, top=0.89, bottom=0.04)

    for seat in range(8):
        row, col = divmod(seat, 4)
        ax = fig.add_subplot(gs[row, col], polar=True)

        if seat in dead:
            ax.set_title(_seat_label(seat, strokes), fontsize=9, fontweight="bold",
                         color="#ccc", pad=10)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            continue

        # Normalise to 0-1 for display
        vals_raw = avgs[seat, :]
        vals_norm = (vals_raw - vmin) / (vmax - vmin)
        vals_plot = vals_norm.tolist() + [vals_norm[0]]

        ax.fill(angles, vals_plot, color=SEAT_COLORS[seat], alpha=0.2)
        ax.plot(angles, vals_plot, color=SEAT_COLORS[seat], linewidth=1.8)
        ax.scatter(angles[:-1], vals_plot[:-1], color=SEAT_COLORS[seat],
                   s=25, zorder=5)

        # Annotate with raw values
        for a, v_raw, v_plot in zip(angles[:-1], vals_raw, vals_plot[:-1]):
            ax.annotate(f"{v_raw:.1f}%", xy=(a, v_plot),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=6, color=SEAT_COLORS[seat], fontweight="bold")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(q_labels, fontsize=7)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "", "", ""], fontsize=5)
        ax.set_title(_seat_label(seat, strokes), fontsize=10, fontweight="bold",
                     color=SEAT_COLORS[seat], pad=12)
        ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# Anomaly Detection & Report
# ---------------------------------------------------------------------------

def _detect_anomalies(strokes, overall_length, effective_length):
    """Scan session data for negative anomalies a coach should know about.

    Returns a list of (severity, seat_index, seat_label, description) tuples.
    severity: 'HIGH', 'MED', or 'LOW'.
    """
    anomalies = []
    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]
    if len(live_seats) < 2:
        return anomalies

    # --- Helper ---
    def _add(severity, seat, desc):
        label = _seat_label(seat, strokes) if seat is not None else "Crew"
        anomalies.append((severity, seat, label, desc))

    power = strokes["SwivelPower"]
    n = power.shape[0]
    q1_end = n // 4
    q4_start = n - n // 4

    # 1. High variance: seat std > 1.8× crew-average std for key metrics
    variance_checks = [
        ("Avg Watts", strokes["SwivelPower"]),
        ("Effective Length", effective_length),
        ("Catch Slip", strokes["CatchSlip"]),
        ("Finish Slip", strokes["FinishSlip"]),
    ]
    for name, data in variance_checks:
        stds = np.array([np.nanstd(data[:, s]) for s in range(8)])
        crew_avg_std = np.nanmean(stds[live_seats])
        for s in live_seats:
            if crew_avg_std > 0 and stds[s] > 1.8 * crew_avg_std:
                ratio = stds[s] / crew_avg_std
                _add("HIGH", s,
                     f"{name} variance {ratio:.1f}× crew avg "
                     f"(std {stds[s]:.1f} vs {crew_avg_std:.1f})")

    # 2. Power fade: last-quarter avg power < first-quarter by >40%
    if q1_end > 2 and (n - q4_start) > 2:
        for s in live_seats:
            first_q = np.nanmean(power[:q1_end, s])
            last_q = np.nanmean(power[q4_start:, s])
            if first_q > 0:
                drop_pct = (first_q - last_q) / first_q * 100
                if drop_pct > 40:
                    sev = "HIGH" if drop_pct > 50 else "MED"
                    _add(sev, s,
                         f"Power faded {drop_pct:.0f}% "
                         f"({first_q:.0f}W → {last_q:.0f}W)")

    # 3. Excessive slip: seat avg catch/finish slip > 1.5× crew average
    for slip_name, slip_key in [("Catch Slip", "CatchSlip"),
                                 ("Finish Slip", "FinishSlip")]:
        data = strokes[slip_key]
        avgs = np.array([np.nanmean(data[:, s]) for s in range(8)])
        crew_avg = np.nanmean(avgs[live_seats])
        for s in live_seats:
            if crew_avg > 0 and avgs[s] > 1.5 * crew_avg:
                _add("MED", s,
                     f"{slip_name} {avgs[s]:.1f}° is {avgs[s]/crew_avg:.1f}× "
                     f"crew avg ({crew_avg:.1f}°)")

    # 4. Low power outlier: seat avg watts > 1.5 std devs below crew mean
    power_avgs = np.array([np.nanmean(power[:, s]) for s in range(8)])
    crew_mean = np.nanmean(power_avgs[live_seats])
    crew_std = np.nanstd(power_avgs[live_seats])
    if crew_std > 0:
        for s in live_seats:
            z = (power_avgs[s] - crew_mean) / crew_std
            if z < -1.5:
                _add("MED", s,
                     f"Avg power {power_avgs[s]:.0f}W is {abs(z):.1f}σ "
                     f"below crew mean ({crew_mean:.0f}W)")

    # 5. Timing consistency: drive start time std much higher than crew avg
    dst = strokes["Drive Start T"]
    dst_stds = np.array([np.nanstd(dst[:, s]) for s in range(8)])
    crew_dst_std = np.nanmean(dst_stds[live_seats])
    if crew_dst_std > 0:
        for s in live_seats:
            if dst_stds[s] > 1.8 * crew_dst_std:
                _add("MED", s,
                     f"Timing variability {dst_stds[s]/crew_dst_std:.1f}× "
                     f"crew avg (inconsistent catch)")

    # 6. Effective length fade
    if q1_end > 2 and (n - q4_start) > 2:
        for s in live_seats:
            first_q = np.nanmean(effective_length[:q1_end, s])
            last_q = np.nanmean(effective_length[q4_start:, s])
            if first_q > 0:
                drop_pct = (first_q - last_q) / first_q * 100
                if drop_pct > 8:
                    _add("LOW", s,
                         f"Eff. length shortened {drop_pct:.0f}% "
                         f"({first_q:.1f}° → {last_q:.1f}°)")

    # 7. Short effective arc: seat avg eff length > 1.5σ below crew mean
    eff_avgs = np.array([np.nanmean(effective_length[:, s]) for s in range(8)])
    eff_crew_mean = np.nanmean(eff_avgs[live_seats])
    eff_crew_std = np.nanstd(eff_avgs[live_seats])
    if eff_crew_std > 0:
        for s in live_seats:
            z = (eff_avgs[s] - eff_crew_mean) / eff_crew_std
            if z < -1.5:
                _add("MED", s,
                     f"Short arc — eff. length {eff_avgs[s]:.1f}° is "
                     f"{abs(z):.1f}σ below crew ({eff_crew_mean:.1f}°)")

    # Sort: HIGH first, then MED, then LOW
    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    anomalies.sort(key=lambda x: order[x[0]])
    return anomalies


def _draw_anomaly_page(fig, anomalies, session_name, strokes):
    """Anomaly Report — clean table with one row per rower, issues listed."""
    fig.text(0.5, 0.97, f"Anomaly Report — {session_name}",
             fontsize=18, fontweight="bold", ha="center", va="top",
             family="sans-serif", color="#2c3e50")

    dead = strokes.get("dead_seats", set())

    SEV_COLORS = {
        "HIGH": "#e74c3c",
        "MED": "#f39c12",
        "LOW": "#3498db",
    }

    # Group anomalies by seat index
    seat_anomalies = {s: [] for s in range(8)}
    for sev, seat_idx, label, desc in anomalies:
        if seat_idx is not None:
            seat_anomalies[seat_idx].append((sev, desc))

    ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
    ax.axis("off")

    # Build table data: Name | Status | Issues
    col_labels = ["Name", "Status", "Issues"]
    cell_data = []
    cell_colors = []

    for seat in range(8):
        name = _seat_label(seat, strokes)
        if seat in dead:
            cell_data.append([name, "NO DATA", "—"])
            cell_colors.append(["#d5d5d5", "#d5d5d5", "#d5d5d5"])
            continue

        items = seat_anomalies.get(seat, [])
        if not items:
            cell_data.append([name, "CLEAN", "No issues detected"])
            cell_colors.append(["#eafaf1", "#eafaf1", "#eafaf1"])
        else:
            # Worst severity determines row color
            if any(s == "HIGH" for s, _ in items):
                status = f"{len(items)} issue{'s' if len(items) > 1 else ''}"
                row_bg = "#fdedec"
            elif any(s == "MED" for s, _ in items):
                status = f"{len(items)} issue{'s' if len(items) > 1 else ''}"
                row_bg = "#fef9e7"
            else:
                status = f"{len(items)} issue{'s' if len(items) > 1 else ''}"
                row_bg = "#ebf5fb"

            # Join issues with severity tags
            issue_lines = []
            for sev, desc in items:
                issue_lines.append(f"[{sev}] {desc}")
            issues_text = "  |  ".join(issue_lines)

            cell_data.append([name, status, issues_text])
            cell_colors.append([row_bg, row_bg, row_bg])

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="upper center",
        cellLoc="left",
        colWidths=[0.10, 0.08, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 3.0)

    # Style header
    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)

    # Style rows
    for i in range(8):
        for j in range(3):
            cell = table[i + 1, j]
            cell.set_facecolor(cell_colors[i][j])
            if j == 0:
                cell.set_text_props(fontweight="bold",
                                    color=SEAT_COLORS[i] if i not in dead else "#999")
            elif j == 1:
                seat_items = seat_anomalies.get(i, [])
                if i in dead:
                    cell.set_text_props(color="#999")
                elif not seat_items:
                    cell.set_text_props(color="#27ae60", fontweight="bold")
                elif any(s == "HIGH" for s, _ in seat_items):
                    cell.set_text_props(color="#e74c3c", fontweight="bold")
                elif any(s == "MED" for s, _ in seat_items):
                    cell.set_text_props(color="#f39c12", fontweight="bold")
                else:
                    cell.set_text_props(color="#3498db", fontweight="bold")
            elif j == 2:
                cell.set_text_props(fontsize=7)

    # Summary count at bottom
    n_high = sum(1 for s, _, _, _ in anomalies if s == "HIGH")
    n_med = sum(1 for s, _, _, _ in anomalies if s == "MED")
    n_low = sum(1 for s, _, _, _ in anomalies if s == "LOW")
    clean_count = sum(1 for s in range(8)
                      if s not in dead and not seat_anomalies.get(s, []))
    fig.text(0.5, 0.935,
             f"{len(anomalies)} total issues  |  "
             f"{n_high} HIGH  •  {n_med} MED  •  {n_low} LOW  |  "
             f"{clean_count} clean rowers",
             fontsize=10, ha="center", va="top", color="#555",
             family="sans-serif")


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

# Readable short labels for metrics used in speed factor analysis
_SPEED_METRIC_CONFIGS = [
    ("Power", "SwivelPower"),
    ("Catch Slip", "CatchSlip"),
    ("Finish Slip", "FinishSlip"),
    ("Timing", "Drive Start T"),
    ("Max Force Angle", "Angle Max F"),
    ("Work Q1", "Work PC Q1"),
    ("Work Q2", "Work PC Q2"),
    ("Work Q3", "Work PC Q3"),
    ("Work Q4", "Work PC Q4"),
]


def _compute_speed_factors(strokes, overall_length, effective_length):
    """Identify which per-rower technique metrics influence boat speed.

    First regresses out stroke rate from boat speed so that rate-dependent
    metrics (drive time, recovery time) don't dominate.  Then trains a
    Random Forest on technique features to predict the *rate-adjusted*
    speed residual, and uses permutation importance to rank features.

    Returns a dict with 'factors', 'model_score', 'rate_r2', 'n_samples',
    or None if there is insufficient data.
    """
    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]
    speed = np.array(strokes["boat_speed"])
    rate = np.array(strokes["rating"])

    # Build feature matrix — technique metrics only -----------------------
    metric_pairs = [(label, strokes[key]) for label, key in _SPEED_METRIC_CONFIGS]
    metric_pairs.append(("Eff. Length", effective_length))
    metric_pairs.append(("Overall Length", overall_length))

    feature_names = []  # list of (seat_index, metric_label)
    feature_cols = []
    for label, data in metric_pairs:
        for s in live_seats:
            feature_names.append((s, label))
            feature_cols.append(data[:, s])

    X = np.column_stack(feature_cols)
    y = speed

    # Filter: skip first 20 strokes, require speed > 0, no NaN rows ------
    valid = np.ones(len(y), dtype=bool)
    valid[:20] = False
    valid &= (y > 0) & (rate > 0)
    valid &= ~np.any(np.isnan(X), axis=1)

    X_valid = X[valid]
    y_valid = y[valid]
    rate_valid = rate[valid]

    if len(y_valid) < 40:
        return None

    # --- Regress out stroke rate from speed -------------------------------
    # Fit a degree-2 polynomial (speed ~ rate + rate²) and use residuals
    rate_coeffs = np.polyfit(rate_valid, y_valid, 2)
    rate_predicted = np.polyval(rate_coeffs, rate_valid)
    y_residual = y_valid - rate_predicted
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    ss_res = np.sum((y_valid - rate_predicted) ** 2)
    rate_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- Random Forest on residuals + permutation importance --------------
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_valid, y_residual)
    perm = permutation_importance(
        rf, X_valid, y_residual, n_repeats=30, random_state=42, n_jobs=-1,
    )
    importances = perm.importances_mean

    # --- Delta correlations (direction of effect) ------------------------
    # Δmetric vs Δresidual, stroke-to-stroke
    delta_corrs = []
    for i in range(X_valid.shape[1]):
        dx = np.diff(X_valid[:, i])
        dy = np.diff(y_residual)
        mask = ~(np.isnan(dx) | np.isnan(dy))
        if mask.sum() > 10:
            r = np.corrcoef(dx[mask], dy[mask])[0, 1]
        else:
            r = 0.0
        delta_corrs.append(r)

    # Partial correlation with residual for overall direction
    raw_corrs = []
    for i in range(X_valid.shape[1]):
        r = np.corrcoef(X_valid[:, i], y_residual)[0, 1]
        raw_corrs.append(r if not np.isnan(r) else 0.0)

    # Assemble results ----------------------------------------------------
    factors = []
    for i, (seat, metric) in enumerate(feature_names):
        factors.append({
            "seat": seat,
            "metric": metric,
            "importance": importances[i],
            "delta_r": delta_corrs[i],
            "raw_r": raw_corrs[i],
        })

    factors.sort(key=lambda x: -x["importance"])

    return {
        "factors": factors,
        "model_score": rf.score(X_valid, y_residual),
        "rate_r2": rate_r2,
        "n_samples": len(y_valid),
    }


def _draw_speed_factors_overview(fig, strokes, speed_data):
    """Page 1: top speed factors bar chart + per-rower summary table."""
    fig.text(0.5, 0.97, "Speed Factor Analysis",
             fontsize=18, fontweight="bold", ha="center", va="top",
             family="sans-serif", color="#2c3e50")
    rate_r2 = speed_data.get("rate_r2", 0)
    fig.text(0.5, 0.94,
             "Rate-adjusted: technique factors that predict speed beyond stroke rate  "
             f"(rate R\u00b2 = {rate_r2:.2f}, "
             f"technique R\u00b2 = {speed_data['model_score']:.2f}, "
             f"n = {speed_data['n_samples']})",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]
    factors = speed_data["factors"]

    # --- Top bar chart: top 15 factors -----------------------------------
    ax = fig.add_axes([0.30, 0.42, 0.65, 0.50])
    top_n = min(15, len(factors))
    top = factors[:top_n][::-1]  # reverse for horizontal bars (top at top)

    bars_y = np.arange(top_n)
    bars_vals = [f["importance"] for f in top]
    bars_colors = [SEAT_COLORS[f["seat"]] for f in top]
    bar_labels = []
    for f in top:
        name = _seat_label(f["seat"], strokes)
        arrow = "\u2191" if f["raw_r"] > 0 else "\u2193"
        bar_labels.append(f"{name} — {f['metric']}  {arrow}")

    ax.barh(bars_y, bars_vals, color=bars_colors, edgecolor="white",
            height=0.7, alpha=0.85)
    ax.set_yticks(bars_y)
    ax.set_yticklabels(bar_labels, fontsize=8, family="sans-serif")
    ax.set_xlabel("Permutation Importance", fontsize=9)
    ax.set_title("Top Technique Factors — Rate-Adjusted (Random Forest)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(labelsize=8)

    # Add delta-r annotation on each bar
    for j, f in enumerate(top):
        x_pos = f["importance"] + max(bars_vals) * 0.01
        dr = f["delta_r"]
        ax.text(x_pos, j, f"Δr={dr:+.2f}", fontsize=6.5, va="center",
                color="#555")

    # --- Bottom: per-rower summary table ---------------------------------
    # For each live seat, find their #1 factor
    ax2 = fig.add_axes([0.05, 0.04, 0.90, 0.33])
    ax2.axis("off")

    col_labels = ["Seat", "Name", "#1 Factor", "Direction", "Importance",
                   "#2 Factor", "Direction", "Importance"]
    table_data = []

    for s in live_seats:
        seat_factors = [f for f in factors if f["seat"] == s]
        name = _seat_label(s, strokes)
        row = [f"Seat {s+1}", name]
        for rank in range(2):
            if rank < len(seat_factors):
                sf = seat_factors[rank]
                arrow = "\u2191 more → faster" if sf["raw_r"] > 0 else "\u2193 less → faster"
                row.extend([sf["metric"], arrow, f"{sf['importance']:.4f}"])
            else:
                row.extend(["—", "—", "—"])
        table_data.append(row)

    table = ax2.table(cellText=table_data, colLabels=col_labels,
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.6)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Color-code rows by seat
    for i, s in enumerate(live_seats):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_facecolor(SEAT_COLORS[s] + "18")  # very light tint


def generate_pdf(strokes, output_path, session_name):
    n_strokes = len(strokes["stroke_num"])

    overall_length = np.abs(strokes["MinAngle"]) + strokes["MaxAngle"]
    effective_length = overall_length - strokes["CatchSlip"] - strokes["FinishSlip"]

    with PdfPages(str(output_path)) as pdf:
        page_num = 1

        # Page 1: Summary table
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_summary_table(fig, strokes, overall_length, effective_length, session_name)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Summary table")
        page_num += 1

        # Page 2: Anomaly Report
        anomalies = _detect_anomalies(strokes, overall_length, effective_length)
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_anomaly_page(fig, anomalies, session_name, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        n_anomalies = len(anomalies)
        print(f"  Page {page_num}: Anomaly Report ({n_anomalies} issues)")
        page_num += 1

        # Angle arc plot
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_angle_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Stroke arc breakdown")
        page_num += 1

        # Core metric pages
        core_pages = [
            ("SwivelPower (watts)", strokes["SwivelPower"], True),
            ("Overall Length (deg)", overall_length, True),
            ("Effective Length (deg)", effective_length, True),
            ("Catch Slip (deg)", strokes["CatchSlip"], False),
            ("Finish Slip (deg)", strokes["FinishSlip"], False),
        ]

        for title, data, hib in core_pages:
            fig = plt.figure(figsize=(16.5, 11.7))
            fig.patch.set_facecolor("white")
            _draw_metric_page(fig, data, title, strokes, hib)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Page {page_num}: {title}")
            page_num += 1

        # Breaker page: Extra Statistics
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.5, "Extra Statistics", fontsize=36, fontweight="bold",
                 ha="center", va="center", color="#2c3e50", family="sans-serif")
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: --- Extra Statistics ---")
        page_num += 1

        extra_pages = [
            ("Min Angle (deg)", strokes["MinAngle"], False),
            ("Max Angle (deg)", strokes["MaxAngle"], True),
            ("Drive Time (s)", strokes["Drive Time"], False),
            ("Recovery Time (s)", strokes["Recovery Time"], False),
        ]

        for title, data, hib in extra_pages:
            fig = plt.figure(figsize=(16.5, 11.7))
            fig.patch.set_facecolor("white")
            _draw_metric_page(fig, data, title, strokes, hib)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Page {page_num}: {title}")
            page_num += 1

        # ---------------------------------------------------------------
        # Breaker page: Extended Analysis
        # ---------------------------------------------------------------
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.5, "Extended Analysis", fontsize=36, fontweight="bold",
                 ha="center", va="center", color="#2c3e50", family="sans-serif")
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: --- Extended Analysis ---")
        page_num += 1

        # Crew Timing Sync
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_crew_timing_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Crew Timing Sync")
        page_num += 1

        # Drive : Recovery Ratio
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_drive_recovery_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Drive:Recovery Ratio")
        page_num += 1

        # Composite Consistency Score
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_consistency_page(fig, strokes, overall_length, effective_length)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Composite Consistency Score")
        page_num += 1

        # Power Efficiency
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_power_efficiency_page(fig, strokes, effective_length)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Power Efficiency")
        page_num += 1

        # Work Distribution Profile
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_work_distribution_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Work Distribution Profile")
        page_num += 1

        # Rate Response Curves
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_rate_response_page(fig, strokes, effective_length)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Rate Response Curves")
        page_num += 1

        # ---------------------------------------------------------------
        # Breaker page: Heatmaps
        # ---------------------------------------------------------------
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.5, "Heatmaps", fontsize=36,
                 fontweight="bold", ha="center", va="center",
                 color="#2c3e50", family="sans-serif")
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: --- Heatmaps ---")
        page_num += 1

        # Watts correlation heatmap
        dead = strokes.get("dead_seats", set())
        watts_corr = _compute_corr_matrix(strokes["SwivelPower"], dead)

        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_correlation_page(fig, strokes,
                               metric_label="Swivel Power (watts)",
                               corr_matrix=watts_corr,
                               full_matrix=True)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Correlation — Watts")
        page_num += 1

        # Timing offset correlation heatmap
        # Correlate raw Drive Start T (Pearson r is shift-invariant, so
        # subtracting stroke seat doesn't change pairwise correlations,
        # but using raw values lets us include stroke seat itself)
        timing_corr = _compute_corr_matrix(strokes["Drive Start T"], dead)

        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_correlation_page(fig, strokes,
                               metric_label="Timing Offset vs Stroke Seat",
                               corr_matrix=timing_corr)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Correlation — Timing Offset")
        page_num += 1

        # Boat Watts vs Boat Speed
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_watts_vs_speed_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Boat Watts vs Boat Speed")
        page_num += 1

        # Speed Factor Analysis (ML)
        speed_data = _compute_speed_factors(strokes, overall_length,
                                            effective_length)
        if speed_data is not None:
            # Breaker page
            fig = plt.figure(figsize=(16.5, 11.7))
            fig.patch.set_facecolor("white")
            fig.text(0.5, 0.5, "Speed Factor Analysis", fontsize=36,
                     fontweight="bold", ha="center", va="center",
                     color="#2c3e50", family="sans-serif")
            fig.text(0.5, 0.42,
                     "ML-identified individual metrics that predict boat speed",
                     fontsize=14, ha="center", color="#555",
                     family="sans-serif")
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Page {page_num}: --- Speed Factor Analysis ---")
            page_num += 1

            # Overview page
            fig = plt.figure(figsize=(16.5, 11.7))
            fig.patch.set_facecolor("white")
            _draw_speed_factors_overview(fig, strokes, speed_data)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Page {page_num}: Speed Factors — Overview")
            page_num += 1
        else:
            print("  Skipped Speed Factor Analysis (insufficient data)")

    print(f"\nPDF saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def interactive_mode():
    print("=" * 50)
    print("  EXTENDED STATSHEET GENERATOR")
    print("=" * 50)
    print()

    csvs = sorted(DATA_DIR.glob("*.csv"))
    if not csvs:
        sys.exit(f"No CSV files found in {DATA_DIR}")

    print("Available CSV files:")
    for i, f in enumerate(csvs, 1):
        print(f"  {i}. {f.name}")
    print(f"  a. All files")
    print()

    if len(csvs) == 1:
        csv_path = csvs[0]
        print(f"Using: {csv_path.name}")
        return csv_path

    choice = input("Enter CSV number (or 'a' for all): ").strip().lower()
    if choice == "a":
        return csvs
    try:
        return csvs[int(choice) - 1]
    except (ValueError, IndexError):
        sys.exit("Invalid selection")


def _process_one(csv_path):
    """Parse a single CSV and generate its extended statsheet PDF."""
    session_name = csv_path.stem
    print(f"\nParsing {csv_path.name}...")
    strokes = parse_csv(csv_path)
    n_strokes = len(strokes["stroke_num"])
    print(f"Found {n_strokes} strokes")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{session_name}-extended-statsheet.pdf"

    print("Generating extended PDF...")
    generate_pdf(strokes, output_path, session_name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate extended PDF statsheet from rowing data")
    parser.add_argument("--csv", help="CSV data file (name or path)")
    parser.add_argument("--all", action="store_true",
                        help="Process all CSV files in the data directory")
    args = parser.parse_args()

    if args.all:
        csvs = sorted(DATA_DIR.glob("*.csv"))
        if not csvs:
            sys.exit(f"No CSV files found in {DATA_DIR}")
        print(f"Processing {len(csvs)} CSV files...")
        for csv_path in csvs:
            _process_one(csv_path)
        print(f"\nDone — generated {len(csvs)} extended statsheets.")
        return

    if not args.csv:
        result = interactive_mode()
        if isinstance(result, list):
            for csv_path in result:
                _process_one(csv_path)
            print(f"\nDone — generated {len(result)} extended statsheets.")
            return
        csv_path = result
    else:
        csv_input = Path(args.csv)
        csv_path = csv_input if csv_input.exists() else DATA_DIR / args.csv
        if not csv_path.exists():
            sys.exit(f"CSV not found: {csv_path}")

    _process_one(csv_path)


if __name__ == "__main__":
    main()
