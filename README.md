# rowing-tools

A collection of tools for rowing data analysis and video production.

## Stats Overlayed Footage

Generates a real-time animated stats overlay from rowing telemetry data, showing per-seat metrics (SwivelPower, CatchSlip, etc.) stroke by stroke. Can be combined with footage to produce a single video with stats stacked underneath.

### Requirements

- Python 3 with `numpy` and `matplotlib`
- `ffmpeg` installed and on PATH

### Setup

```
stats-overlayed-footage/
  data/       <- put CSV data files here
  footage/    <- put video files here
  output/     <- generated videos appear here
  code/       <- scripts
```

### What CSV file do I use?

On powerline, select a piece like normal. Then, you can left click on that piece in the session manager on the left, and click 'export profile data'. 
Then, it will copy the data to your clipboard. Paste it into a CSV file (google sheets is fine), save the CSV, then place it in the `data` folder.

### Usage

**Interactive mode** (prompts you for everything):

```bash
python stats-overlayed-footage/code/generate_overlay.py
```

**Flags mode**:

```bash
# Overlay only (no footage)
python stats-overlayed-footage/code/generate_overlay.py --csv addy-piece-1-20March2026.csv --metric SwivelPower

# Overlay + combined with footage
python stats-overlayed-footage/code/generate_overlay.py --csv addy-piece-1-20March2026.csv --metric CatchSlip --video addy-piece-1-20March2026.MOV
```

CSV and video names are resolved from `data/` and `footage/` respectively, or as full paths.

### Available metrics

SwivelPower, MinAngle, CatchSlip, MaxAngle, FinishSlip, Drive Start T, Rower Swivel Power, Drive Time, Recovery Time, Angle Max F, Angle 0.7 F, Work PC Q1-Q4, Max Force PC.

### Important

The footage video **must start precisely when the first stroke is at the finish**. Trim the video beforehand so frame 0 aligns with the first stroke in the data.
