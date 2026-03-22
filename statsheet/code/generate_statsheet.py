#!/usr/bin/env python3
"""Generate a multi-page landscape PDF summary of a rowing session.

Page 1: Summary table with conditional coloring (green=good, red=bad)
Page 2: Stroke arc angle plot (box-and-whisker style)
Pages 3+: One page per metric with individual graphs per seat

Usage:
    python generate_statsheet.py                          # interactive mode
    python generate_statsheet.py --csv "session.csv"      # flags mode
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
    """Replace malfunction-level outlier values with NaN per seat per metric.

    Uses a wide IQR threshold (5x) to only catch sensor malfunctions,
    not natural variation. Outlier values are NaN'd for that specific
    seat/stroke — other seats and strokes are kept intact.
    """
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
    # A seat is dead if SwivelPower is all zero
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


def _cell_color(value, all_values, higher_is_better):
    """Return a background color: green for good, red for bad, orange for middle."""
    vmin, vmax = np.nanmin(all_values), np.nanmax(all_values)
    if vmax == vmin:
        return "#ffffff"
    norm = (value - vmin) / (vmax - vmin)
    cmap = GOOD_CMAP if higher_is_better else BAD_CMAP
    r, g, b, _ = cmap(norm)
    # Lighten: blend 55% with white for readability
    r, g, b = 0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b
    return (r, g, b)


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

    # Metric definitions: (display_name, data, higher_is_better)
    metric_defs = [
        ("Avg Watts", strokes["SwivelPower"], True),
        ("Overall Len", overall_length, True),
        ("Effective Len", effective_length, True),
        ("Catch Slip", strokes["CatchSlip"], False),
        ("Finish Slip", strokes["FinishSlip"], False),
        ("Min Angle", strokes["MinAngle"], False),  # more negative = wider catch = better
        ("Max Angle", strokes["MaxAngle"], True),
    ]

    col_labels = ["Seat"] + [d[0] for d in metric_defs]
    # Add std dev columns
    for d in metric_defs:
        col_labels.append(f"{d[0]} std")

    n_cols = len(col_labels)

    dead = strokes.get("dead_seats", set())
    live_seats = [s for s in range(8) if s not in dead]

    # Build row data: seats 1-8 + boat average
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

    # Boat average row (only live seats)
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

    # Style header
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)

    # Color-code data cells
    for i in range(9):  # 8 seats + boat
        # Seat label column
        if i < 8 and i in dead:
            table[i + 1, 0].set_facecolor("#d5d5d5")
            table[i + 1, 0].set_text_props(fontweight="bold", color="#999")
            for j in range(1, n_cols):
                table[i + 1, j].set_facecolor("#d5d5d5")
                table[i + 1, j].set_text_props(color="#999")
            continue

        table[i + 1, 0].set_facecolor("#ecf0f1" if i < 8 else "#bdc3c7")
        table[i + 1, 0].set_text_props(fontweight="bold")

        # Average columns (1 through len(metric_defs))
        for j, (_, data, higher_is_better) in enumerate(metric_defs):
            col_idx = j + 1
            all_avgs = np.array([np.nanmean(data[:, s]) for s in live_seats])
            if i < 8:
                val = np.nanmean(data[:, i])
            else:
                val = np.nanmean(data[:, live_seats])
            color = _cell_color(val, all_avgs, higher_is_better)
            table[i + 1, col_idx].set_facecolor(color)

        # Std dev columns — lower is always better
        for j, (_, data, _) in enumerate(metric_defs):
            col_idx = len(metric_defs) + j + 1
            all_stds = np.array([np.nanstd(data[:, s]) for s in live_seats])
            if i < 8:
                val = np.nanstd(data[:, i])
            else:
                val = np.nanmean([np.nanstd(data[:, s]) for s in live_seats])
            color = _cell_color(val, all_stds, higher_is_better=False)
            table[i + 1, col_idx].set_facecolor(color)

        # Bold the boat row
        if i == 8:
            for j in range(n_cols):
                table[i + 1, j].set_text_props(fontweight="bold")
                if j > 0:
                    existing = table[i + 1, j].get_facecolor()
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

        # Catch slip (red)
        ax.barh(y[i], avg_catch[i], left=catch_left, height=bar_height,
                color="#e74c3c", edgecolor="#c0392b", linewidth=0.8, alpha=0.9)
        ax.text(catch_left + avg_catch[i] / 2, y[i], f"{avg_catch[i]:.1f}",
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        # Effective arc (green)
        eff_len = eff_right - eff_left
        ax.barh(y[i], eff_len, left=eff_left, height=bar_height,
                color="#2ecc71", edgecolor="#27ae60", linewidth=0.8, alpha=0.9)
        ax.text((eff_left + eff_right) / 2, y[i], f"{eff_len:.1f}",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white")

        # Finish slip (red)
        ax.barh(y[i], avg_finish[i], left=finish_left, height=bar_height,
                color="#e74c3c", edgecolor="#c0392b", linewidth=0.8, alpha=0.9)
        ax.text(finish_left + avg_finish[i] / 2, y[i], f"{avg_finish[i]:.1f}",
                ha="center", va="center", fontsize=7, color="white", fontweight="bold")

        # Whiskers for std dev
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


def generate_pdf(strokes, output_path, session_name):
    n_strokes = len(strokes["stroke_num"])

    overall_length = np.abs(strokes["MinAngle"]) + strokes["MaxAngle"]
    effective_length = overall_length - strokes["CatchSlip"] - strokes["FinishSlip"]

    with PdfPages(str(output_path)) as pdf:
        # Page 1: Summary table
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_summary_table(fig, strokes, overall_length, effective_length, session_name)
        pdf.savefig(fig)
        plt.close(fig)
        print("  Page 1: Summary table")

        # Page 2: Angle arc plot
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        _draw_angle_page(fig, strokes)
        pdf.savefig(fig)
        plt.close(fig)
        print("  Page 2: Stroke arc breakdown")

        # Core metric pages
        core_pages = [
            ("SwivelPower (watts)", strokes["SwivelPower"], True),
            ("Overall Length (deg)", overall_length, True),
            ("Effective Length (deg)", effective_length, True),
            ("Catch Slip (deg)", strokes["CatchSlip"], False),
            ("Finish Slip (deg)", strokes["FinishSlip"], False),
        ]

        extra_pages = [
            ("Min Angle (deg)", strokes["MinAngle"], False),
            ("Max Angle (deg)", strokes["MaxAngle"], True),
            ("Drive Time (s)", strokes["Drive Time"], False),
            ("Recovery Time (s)", strokes["Recovery Time"], False),
        ]

        page_num = 3
        for title, data, hib in core_pages:
            fig = plt.figure(figsize=(16.5, 11.7))
            fig.patch.set_facecolor("white")
            _draw_metric_page(fig, data, title, strokes, hib)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Page {page_num}: {title}")
            page_num += 1

        # Breaker page
        fig = plt.figure(figsize=(16.5, 11.7))
        fig.patch.set_facecolor("white")
        fig.text(0.5, 0.5, "Extra Statistics", fontsize=36, fontweight="bold",
                 ha="center", va="center", color="#2c3e50", family="sans-serif")
        pdf.savefig(fig)
        plt.close(fig)
        print(f"  Page {page_num}: --- Extra Statistics ---")
        page_num += 1

        for title, data, hib in extra_pages:
            fig = plt.figure(figsize=(16.5, 11.7))
            fig.patch.set_facecolor("white")
            _draw_metric_page(fig, data, title, strokes, hib)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Page {page_num}: {title}")
            page_num += 1

    print(f"\nPDF saved to {output_path}")


def interactive_mode():
    print("=" * 50)
    print("  STATSHEET GENERATOR")
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
    parser = argparse.ArgumentParser(description="Generate PDF statsheet from rowing data")
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
    output_path = OUTPUT_DIR / f"{session_name}-statsheet.pdf"

    print("Generating PDF...")
    generate_pdf(strokes, output_path, session_name)


if __name__ == "__main__":
    main()
