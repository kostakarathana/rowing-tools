# Page-by-Page Stats Glossary

This is a page-indexed explanation of every statistic shown in the combined report from [statsheet/code/generate_trace_extended_statsheet.py](statsheet/code/generate_trace_extended_statsheet.py).

## Page Number Map

### Final PDF layout
- Pages 1–15: Baseline pages 1–15 (up to Rate Response Curves)
- Page 16: **Force Curve — Normalized Time** (trace)
- Page 17: **Gate Angle Velocity** (trace)
- Page 18: **Force Curve — Gate Angle** (trace)
- Page 19: **Boat Attitude by Stroke Phase**
- Page 20: **Roll/Pitch/Yaw Over Time**
- Page 21: **Trace Analytics Breaker**
- Pages 22–25: Baseline pages 16–19 (Heatmaps and final baseline pages)

### Optional ML pages
If `--with-ml` is used, baseline adds two pages after the baseline block:
- Page 26: Speed Factor Analysis breaker
- Page 27: Speed Factors Overview

### Remaining trace pages
- Trace pages are fixed at pages 16–21 in both modes.

For clarity below:
- `Trace P1` means Force Curve — Normalized Time
- `Trace P2` means Gate Angle Velocity
- `Trace P3` means Force Curve — Gate Angle
- `Trace P4` means Boat Attitude by Stroke Phase
- `Trace P5` means Roll/Pitch/Yaw Over Time
- `Trace P6` means Trace Analytics Breaker

## Baseline Pages (1-19)

### Page 1: Summary Table
What is shown:
- Avg Watts
- Overall Len = Max Angle - Min Angle
- Effective Len = Overall Len - Catch Slip - Finish Slip
- Catch Slip
- Finish Slip
- Min Angle
- Max Angle
- Standard deviation for each metric
- Boat-average row for all metrics

What each means:
- Avg Watts: mean rower power output.
- Overall Len: total arc used.
- Effective Len: useful arc after slip losses.
- Catch Slip / Finish Slip: wasted angle at front/back of stroke.
- Min/Max Angle: catch and finish position references.
- Std columns: consistency (lower std = more repeatable).

### Page 2: Anomaly Report
What is shown:
- Severity-tagged anomaly findings by rower.

Anomaly types used:
- High variance outlier (watts/effective length/catch slip/finish slip)
- Power fade (first quarter to last quarter)
- Excessive catch or finish slip vs crew
- Low power outlier vs crew mean
- High timing variability (Drive Start T std)
- Effective length fade
- Short effective arc outlier

What it means:
- Fast triage page for coach priorities and risk flags.

### Page 3: Stroke Arc Breakdown
What is shown:
- Arc decomposition (catch slip, effective arc, finish slip).

What it means:
- Visual split of useful vs wasted angle.

### Page 4: SwivelPower (watts)
What is shown:
- Rower-by-rower power over strokes.

What it means:
- Output trend and rower-level power stability.

### Page 5: Overall Length (deg)
What is shown:
- Overall arc trend by rower.

What it means:
- Arc compression/expansion over piece.

### Page 6: Effective Length (deg)
What is shown:
- Effective arc trend by rower.

What it means:
- Useful stroke-length trend.

### Page 7: Catch Slip (deg)
What is shown:
- Catch slip trend by rower.

What it means:
- Front-end connection quality trend.

### Page 8: Finish Slip (deg)
What is shown:
- Finish slip trend by rower.

What it means:
- Back-end release efficiency trend.

### Page 9: Extra Statistics Breaker
What is shown:
- Section title only.

### Page 10: Min Angle (deg)
What is shown:
- Min angle trend by rower.

What it means:
- Catch-side reach/entry consistency.

### Page 11: Max Angle (deg)
What is shown:
- Max angle trend by rower.

What it means:
- Finish-side extraction consistency.

### Page 12: Extended Analysis Breaker
What is shown:
- Section title only.

### Page 13: Crew Timing Sync
What is shown:
- Drive Start T offsets relative to stroke rower.

What it means:
- Early/late timing relative to crew reference.

### Page 14: Drive:Recovery Ratio
What is shown:
- Drive Time / Recovery Time ratio trends.

What it means:
- Rhythm profile and pressure management.

### Page 15: Rate Response Curves
What is shown vs rate:
- Power
- Effective Length
- Catch Slip
- Finish Slip

What it means:
- Whether output/technique holds at higher rates.

## Trace Pages (fixed at 16-21)

### Page 16 (Trace P1): Force Curve — Normalized Time
What is shown:
- Mean force curve by rower vs normalised time
- Rower table with:
  - Peak Phase
  - Rise Rate
  - Decay Rate
  - Positive Impulse
  - Neg Tail %

Metric meanings:
- Peak Phase: where force peaks in stroke phase.
- Rise Rate: force ramp from catch event to peak.
- Decay Rate: force drop from peak to finish event.
- Positive Impulse: integrated positive force.
- Neg Tail %: negative-force share of total force magnitude.

Note:
- This page is smoothed and gap-filled in the same way as other phase-binned trace curves.

### Page 17 (Trace P2): Gate Angle Velocity
What is shown:
- Mean GateAngleVel curves by rower vs normalised time
- Rower table with:
  - Vel Peak Phase
  - Vel Smoothness
  - Catch Sharpness
  - Finish Release Smoothness

Metric meanings:
- Vel Peak Phase: phase of max angular velocity.
- Vel Smoothness: variability of angular-velocity derivative.
- Catch Sharpness: max force-rise slope near catch.
- Finish Release Smoothness: variability of force slope near release.

### Page 18 (Trace P3): Force Curve — Gate Angle (deg)
What is shown:
- Mean force curve by rower vs gate angle in degrees

What it means:
- Shows stroke loading shape as a function of actual oar position.

Note:
- This page is smoothed and gap-filled in the same way as other phase-binned trace curves.

### Page 19 (Trace P4): Boat Attitude by Stroke Phase (Signed)
What is shown:
- Signed Roll phase curve
- Signed Pitch phase curve
- Signed Yaw phase curve
- Drive accel mean
- Recovery accel mean

Metric meanings:
- Signed curves preserve direction (not absolute value).
- Drive accel mean: mean boat acceleration during drive window.
- Recovery accel mean: mean acceleration during recovery window.

### Page 20 (Trace P5): Roll/Pitch/Yaw Over Time
What is shown:
- Roll over full piece time
- Pitch over full piece time
- Yaw over full piece time

What it means:
- Time-domain set and heading behavior across the entire piece.

### Page 21 (Trace P6): Trace Analytics Breaker
What is shown:
- Section title only.

### Page 22: Heatmaps Breaker
What is shown:
- Section title only.

### Page 23: Correlation - Watts
What is shown:
- Rower-to-rower correlation matrix on watts-related series.

What it means:
- How similarly rowers co-vary in output.

### Page 24: Correlation - Timing Offset
What is shown:
- Rower-to-rower correlation matrix on timing offset series.

What it means:
- Crew timing synchrony pattern.

### Page 25: Boat Watts vs Boat Speed
What is shown:
- Relationship between aggregate watts and boat speed.

What it means:
- Conversion efficiency of power to speed.

### Optional Pages 26-27 when `--with-ml`
Page 26:
- Speed Factor Analysis breaker

Page 27:
- Speed Factors Overview
- Model-ranked predictors of speed in that piece

What it means:
- Which measured metrics are most associated with speed variation.

## Trace Processing Notes

Applied signal handling:
- Gap interpolation for NaN/dropout samples
- Zero-crossing gap repair around normalized-time phase zero
- Low-pass smoothing on rower and boat channels
- Smoothed phase-binned curves (including force curve pages)

Purpose:
- Reduce jagged artifacts while preserving coaching-relevant trends.
