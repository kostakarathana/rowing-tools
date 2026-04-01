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
warnings.filterwarnings("ignore", message=".*empty slice.*")
warnings.filterwarnings("ignore", message=".*Degrees of freedom.*")
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

    for name in metrics:
        strokes[name] = np.array(strokes[name])

    strokes = _remove_outliers(strokes, list(metrics.keys()))
    strokes["dead_seats"] = _detect_dead_seats(strokes, list(metrics.keys()))
    return strokes


def _remove_outliers(strokes, metric_names, iqr_factor=5.0):
    """Replace malfunction-level outlier values with NaN per seat per metric."""
    n_seats = 8
    total_nans = 0

    for name in metric_names:
        data = strokes[name].astype(float)
        for seat in range(n_seats):
            col = data[:, seat]
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
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

    col_labels = ["Seat"] + [d[0] for d in metric_defs]
    for d in metric_defs:
        col_labels.append(f"{d[0]} std")

    n_cols = len(col_labels)

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    cell_data = []
    for seat in range(8):
        if seat in dead:
            row = [f"Seat {seat + 1}"] + ["--"] * (len(metric_defs) * 2)
        else:
            row = [f"Seat {seat + 1}"]
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
    ax.set_yticklabels([f"Seat {i+1}" for i in range(n_seats)], fontsize=10)
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
            ax.set_title(f"Seat {seat + 1}", fontsize=9, fontweight="bold", color="#ccc")
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
        ax.set_title(f"Seat {seat + 1}   |   avg: {seat_avg:.1f}   std: {seat_std:.1f}",
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
             "Each seat's Drive Start T minus crew average (ms)  |  0 = perfectly in sync",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]
    dst = strokes["Drive Start T"]  # (n_strokes, 8)
    n_strokes = dst.shape[0]
    x = np.arange(n_strokes)

    # Crew average per stroke (only live seats)
    crew_avg = np.nanmean(dst[:, live_seats], axis=1)  # (n_strokes,)
    offsets = dst - crew_avg[:, np.newaxis]  # (n_strokes, 8)

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
            ax.set_title(f"Seat {seat + 1}", fontsize=9, fontweight="bold", color="#ccc")
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
            f"Seat {seat + 1}   |   avg: {avg_off:+.1f}ms ({direction})   std: {std_off:.1f}ms",
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
            ax.set_title(f"Seat {seat + 1}", fontsize=9, fontweight="bold", color="#ccc")
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
            f"Seat {seat + 1}   |   avg: {seat_avg:.2f}   std: {seat_std:.2f}",
            fontsize=9, fontweight="bold", color=SEAT_COLORS[seat])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

        if row == 3:
            ax.set_xlabel("Stroke", fontsize=7)


# ---------------------------------------------------------------------------
# NEW PAGE: Seat Correlation Heatmap
# ---------------------------------------------------------------------------

def _draw_correlation_page(fig, strokes):
    """Pearson correlation matrix of SwivelPower between all seat pairs."""
    fig.text(0.5, 0.97, "Seat Power Correlation Heatmap",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "Pearson r of SwivelPower between each seat pair  |  "
             "1.0 = perfectly correlated",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    power = strokes["SwivelPower"]  # (n_strokes, 8)
    n_seats = 8

    # Compute correlation matrix (NaN-aware)
    corr = np.full((n_seats, n_seats), np.nan)
    for i in range(n_seats):
        for j in range(n_seats):
            if i in dead or j in dead:
                continue
            a, b = power[:, i], power[:, j]
            valid = ~(np.isnan(a) | np.isnan(b))
            if valid.sum() < 3:
                continue
            corr[i, j] = np.corrcoef(a[valid], b[valid])[0, 1]

    ax = fig.add_axes([0.15, 0.10, 0.65, 0.75])

    # Custom diverging colormap: blue (low) -> white (mid) -> dark green (high)
    cmap = LinearSegmentedColormap.from_list(
        "corr", ["#3498db", "#ecf0f1", "#27ae60"])

    # Mask NaN for display
    masked = np.ma.masked_invalid(corr)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    labels = [f"Seat {i+1}" for i in range(n_seats)]
    ax.set_xticks(range(n_seats))
    ax.set_yticks(range(n_seats))
    ax.set_xticklabels(labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate each cell
    for i in range(n_seats):
        for j in range(n_seats):
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

    col_labels = ["Seat"] + [n for n in raw_stds] + \
                 [f"{n} Score" for n in raw_stds] + ["COMPOSITE"]
    n_cols = len(col_labels)

    cell_data = []
    for seat in range(8):
        if seat in dead:
            row = [f"Seat {seat+1}"] + ["--"] * (n_cols - 1)
        else:
            row = [f"Seat {seat+1}"]
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

    ax2.bar([f"S{x+1}" for x in bars_x], bars_h, color=bars_c, edgecolor="white", width=0.6)
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
            ax.set_title(f"Seat {seat + 1}", fontsize=9, fontweight="bold", color="#ccc")
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
        ax.set_title(f"Seat {seat + 1}   |   avg: {seat_avg:.2f}   std: {seat_std:.2f}",
                     fontsize=9, fontweight="bold", color=SEAT_COLORS[seat])
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

        if row == 3:
            ax.set_xlabel("Stroke", fontsize=7)


# ---------------------------------------------------------------------------
# NEW PAGE: Stroke Heatmap
# ---------------------------------------------------------------------------

def _draw_stroke_heatmap_page(fig, strokes):
    """2D heatmap (strokes x seats) of SwivelPower — shows temporal patterns."""
    fig.text(0.5, 0.97, "Stroke Power Heatmap",
             fontsize=16, fontweight="bold", ha="center", family="sans-serif")
    fig.text(0.5, 0.935,
             "SwivelPower by stroke and seat  |  Darker = more power",
             fontsize=10, ha="center", color="#555", family="sans-serif")

    dead = strokes.get("dead_seats", set())
    power = strokes["SwivelPower"].copy()  # (n_strokes, 8)

    ax = fig.add_axes([0.08, 0.10, 0.82, 0.78])

    cmap = LinearSegmentedColormap.from_list(
        "heat", ["#1a1a2e", "#16213e", "#0f3460", "#3498db", "#2ecc71", "#f1c40f", "#e74c3c"])

    # Transpose so seats are rows, strokes are columns
    data = power.T  # (8, n_strokes)

    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")

    ax.set_yticks(range(8))
    ylabels = []
    for i in range(8):
        if i in dead:
            ylabels.append(f"Seat {i+1} (N/A)")
        else:
            ylabels.append(f"Seat {i+1}")
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
            ax.set_title(f"Seat {seat+1}", fontsize=9, fontweight="bold",
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
        ax.set_title(f"Seat {seat+1}", fontsize=10, fontweight="bold",
                     color=SEAT_COLORS[seat], pad=12)
        ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

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

        # Page 2: Angle arc plot
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

        # Seat Correlation Heatmap
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_correlation_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Seat Correlation Heatmap")
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

        # Stroke Heatmap
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_stroke_heatmap_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Stroke Power Heatmap")
        page_num += 1

        # Technique Radar
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_radar_page(fig, strokes, effective_length)
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: Technique Radar")
        page_num += 1

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
    print()

    if len(csvs) == 1:
        csv_path = csvs[0]
        print(f"Using: {csv_path.name}")
    else:
        choice = input("Enter CSV number: ").strip()
        try:
            csv_path = csvs[int(choice) - 1]
        except (ValueError, IndexError):
            sys.exit("Invalid selection")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate extended PDF statsheet from rowing data")
    parser.add_argument("--csv", help="CSV data file (name or path)")
    args = parser.parse_args()

    if not args.csv:
        csv_path = interactive_mode()
    else:
        csv_input = Path(args.csv)
        csv_path = csv_input if csv_input.exists() else DATA_DIR / args.csv
        if not csv_path.exists():
            sys.exit(f"CSV not found: {csv_path}")

    session_name = csv_path.stem
    print(f"\nParsing {csv_path.name}...")
    strokes = parse_csv(csv_path)
    n_strokes = len(strokes["stroke_num"])
    print(f"Found {n_strokes} strokes")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{session_name}-extended-statsheet.pdf"

    print("Generating extended PDF...")
    generate_pdf(strokes, output_path, session_name)


if __name__ == "__main__":
    main()
