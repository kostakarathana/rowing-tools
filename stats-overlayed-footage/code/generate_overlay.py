#!/usr/bin/env python3
"""Generate a stats overlay video from rowing stroke data.

Produces a 1000x200 video showing a chosen metric for all 8 seats.
Each seat gets a panel with a running history graph that fills in
stroke by stroke, with the current stroke highlighted.

Usage:
    python generate_overlay.py                                  # interactive mode
    python generate_overlay.py --csv data.csv --metric CatchSlip  # flags mode
    python generate_overlay.py --csv data.csv --metric CatchSlip --video footage.MOV
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
FOOTAGE_DIR = Path(__file__).parent.parent / "footage"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def discover_metrics(csv_path):
    """Read the CSV header and return ordered list of per-seat metrics."""
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for i, row in enumerate(rows):
        if len(row) > 1 and row[1] == "SwivelPower":
            header = rows[i]
            break
    else:
        sys.exit("Could not find stroke data header row in CSV")

    # Collect unique metric names in order (groups of 8)
    seen = set()
    metrics = []
    for col in header[1:]:
        if col and col not in seen and col != "Time":
            # Only include per-seat metrics (appear 8 times)
            count = header.count(col)
            if count == 8:
                metrics.append(col)
                seen.add(col)
    return metrics


def interactive_mode():
    """Prompt the user for inputs when no flags are provided."""
    print("=" * 60)
    print("  STATS OVERLAY VIDEO GENERATOR")
    print("=" * 60)
    print()

    # Find available CSVs
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

    print()

    # Discover and show available metrics
    metrics = discover_metrics(csv_path)
    print("Available metrics:")
    for i, m in enumerate(metrics, 1):
        print(f"  {i}. {m}")
    print()

    choice = input("Select metric number: ").strip()
    try:
        metric = metrics[int(choice) - 1]
    except (ValueError, IndexError):
        sys.exit("Invalid selection")

    print()

    # Optional video for combined output
    videos = sorted(FOOTAGE_DIR.glob("*.*"))
    video_path = None
    if videos:
        print("Available footage files:")
        for i, f in enumerate(videos, 1):
            print(f"  {i}. {f.name}")
        print(f"  0. Skip (overlay only, no combined video)")
        print()
        print("NOTE: Video must START precisely when the FIRST stroke is at the finish.")
        print()

        choice = input("Select video number (0 to skip): ").strip()
        if choice != "0":
            try:
                video_path = videos[int(choice) - 1]
            except (ValueError, IndexError):
                sys.exit("Invalid selection")

    return csv_path, metric, video_path


def parse_stroke_data(csv_path, metric="SwivelPower"):
    """Parse the CSV and extract per-stroke values for the given metric."""
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
    metric_start = None
    for col_idx, col_name in enumerate(header):
        if col_name == metric:
            metric_start = col_idx
            break

    if metric_start is None:
        available = sorted(set(h for h in header if h and h != "Time"))
        sys.exit(f"Metric '{metric}' not found. Available: {', '.join(available)}")

    metric_end = metric_start + 8
    print(f"Using metric '{metric}' from columns {metric_start}-{metric_end - 1}")

    swivel_start = 1
    swivel_end = 9
    data_start = header_row + 2

    strokes = []
    for row in rows[data_start:]:
        if len(row) < metric_end or row[0] == "Time":
            break
        if all(v == "" for v in row[swivel_start:swivel_end]):
            continue

        time_ms = int(row[0])
        values = [float(v) if v else 0.0 for v in row[metric_start:metric_end]]

        stroke_num = int(float(row[-4])) if row[-4] else 0
        rating = float(row[-6]) if row[-6] else 0.0

        strokes.append({
            "time_ms": time_ms,
            "values": values,
            "stroke_num": stroke_num,
            "rating": rating,
        })

    return strokes


def render_frames(strokes, frames_dir, metric):
    """Render one PNG per stroke state and return durations for each."""
    n_seats = 8
    n_strokes = len(strokes)
    all_values = np.array([s["values"] for s in strokes])

    val_min = all_values.min()
    val_max = all_values.max()
    margin = (val_max - val_min) * 0.1 if val_max != val_min else 1.0
    global_min = val_min - margin
    global_max = val_max + margin

    durations = []
    for i in range(n_strokes):
        if i < n_strokes - 1:
            dur = (strokes[i + 1]["time_ms"] - strokes[i]["time_ms"]) / 1000.0
        else:
            dur = 2.0
        durations.append(dur)

    seat_colors = [
        "#00e640",  # 1 - bright green
        "#e60000",  # 2 - red
        "#00008b",  # 3 - dark blue
        "#000000",  # 4 - black
        "#006400",  # 5 - dark green
        "#800000",  # 6 - maroon
        "#5b9bd5",  # 7 - lighter blue
        "#ff8c00",  # 8 - orange
    ]

    fig, axes = plt.subplots(1, n_seats, figsize=(10, 2), dpi=100)
    fig.patch.set_facecolor("#1a1a2e")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.15, wspace=0.35)

    for stroke_idx in range(n_strokes):
        for ax in axes:
            ax.clear()

        visible = stroke_idx + 1

        for seat in range(n_seats):
            ax = axes[seat]
            ax.set_facecolor("#16213e")

            values = all_values[:visible, seat]
            x = np.arange(visible)
            color = seat_colors[seat]

            bar_colors = [color + "80"] * visible
            bar_colors[-1] = color
            ax.bar(x, values, color=bar_colors, width=0.8, edgecolor="none")

            current_val = values[-1]
            fmt = f"{current_val:.1f}" if abs(current_val) < 100 else f"{int(current_val)}"
            ax.text(
                0.5, 0.95, fmt,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold", color="white",
            )

            ax.set_ylim(global_min, global_max)
            ax.set_xlim(-0.5, n_strokes - 0.5)
            ax.set_title(f"Seat {seat + 1}", fontsize=7, color="white", pad=2)
            ax.tick_params(axis="both", which="both", labelsize=5, colors="white", length=2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("#444")
            ax.spines["left"].set_color("#444")
            ax.set_yticks([])
            if seat == 0:
                ax.set_ylabel(metric, fontsize=5, color="white")

        s = strokes[stroke_idx]
        fig.suptitle(
            f"{metric}  |  Stroke #{s['stroke_num']}  |  Rate: {s['rating']}",
            fontsize=8, color="white", y=0.97,
        )

        frame_path = frames_dir / f"frame_{stroke_idx:04d}.png"
        fig.savefig(frame_path, facecolor=fig.get_facecolor(), edgecolor="none")

        if (stroke_idx + 1) % 10 == 0:
            print(f"  Rendered {stroke_idx + 1}/{n_strokes} frames")

    plt.close(fig)
    return durations


def assemble_video(frames_dir, durations, output_path, n_strokes):
    """Use ffmpeg concat demuxer to stitch frames with correct timing."""
    concat_file = frames_dir / "concat.txt"
    with open(concat_file, "w") as f:
        for i in range(n_strokes):
            frame_path = frames_dir / f"frame_{i:04d}.png"
            f.write(f"file '{frame_path}'\n")
            f.write(f"duration {durations[i]:.4f}\n")
        f.write(f"file '{frames_dir / f'frame_{n_strokes - 1:04d}.png'}\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-vf", "fps=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "fast", "-crf", "23",
        str(output_path),
    ]
    print("Assembling overlay video...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error:\n{result.stderr}")
        sys.exit(1)
    print(f"Overlay saved to {output_path}")


def combine_with_footage(overlay_path, video_path, output_path):
    """Stack the overlay underneath the footage video."""
    # Get footage width to scale overlay
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=width",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True,
    )
    width = int(probe.stdout.strip().split("=")[1])

    # Get duration for progress bar
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True,
    )
    total = float(probe.stdout.strip().split("=")[1])

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(overlay_path),
        "-filter_complex", f"[1:v]scale={width}:-1[ovr];[0:v][ovr]vstack",
        "-c:a", "copy", "-c:v", "libx264",
        "-preset", "fast", "-crf", "20",
        "-progress", "pipe:1",
        str(output_path),
    ]

    print(f"Combining with footage ({total:.0f}s)...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_us="):
            val = line.split("=")[1]
            if val == "N/A":
                continue
            pct = min(100, int(val) / (total * 1_000_000) * 100)
            bars = int(pct / 2)
            print(f'\r  [{"#" * bars}{"." * (50 - bars)}] {pct:5.1f}%', end="", flush=True)
    proc.wait()
    print()

    if proc.returncode != 0:
        sys.exit("ffmpeg failed during combine step")
    print(f"Combined video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate stats overlay video from rowing data")
    parser.add_argument("--csv", help="CSV data file (name or path)")
    parser.add_argument("--metric", help="Metric to visualize (e.g. SwivelPower, CatchSlip)")
    parser.add_argument("--video", help="Footage file to combine with (name or path)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # If no flags provided, run interactive mode
    if not args.csv and not args.metric:
        csv_path, metric, video_path = interactive_mode()
    else:
        # Resolve CSV path
        if not args.csv:
            sys.exit("--csv is required in flags mode")
        csv_input = Path(args.csv)
        csv_path = csv_input if csv_input.exists() else DATA_DIR / args.csv
        if not csv_path.exists():
            sys.exit(f"CSV not found: {csv_path}")

        metric = args.metric or "SwivelPower"

        # Resolve video path
        video_path = None
        if args.video:
            vid_input = Path(args.video)
            video_path = vid_input if vid_input.exists() else FOOTAGE_DIR / args.video
            if not video_path.exists():
                sys.exit(f"Video not found: {video_path}")

    print()
    print(f"CSV:    {csv_path.name}")
    print(f"Metric: {metric}")
    if video_path:
        print(f"Video:  {video_path.name}")
    print()

    strokes = parse_stroke_data(csv_path, metric)
    print(f"Found {len(strokes)} strokes")

    if not strokes:
        sys.exit("No stroke data found")

    total_time = (strokes[-1]["time_ms"] - strokes[0]["time_ms"]) / 1000.0
    print(f"Stroke data spans {total_time:.1f}s")

    stem = csv_path.stem
    overlay_file = OUTPUT_DIR / f"{stem}-{metric}-overlay.mp4"

    with tempfile.TemporaryDirectory() as tmpdir:
        frames_dir = Path(tmpdir)
        print("Rendering frames...")
        durations = render_frames(strokes, frames_dir, metric)
        assemble_video(frames_dir, durations, overlay_file, len(strokes))

    if video_path:
        combined_file = OUTPUT_DIR / f"{stem}-{metric}-combined.mp4"
        combine_with_footage(overlay_file, video_path, combined_file)


if __name__ == "__main__":
    main()
