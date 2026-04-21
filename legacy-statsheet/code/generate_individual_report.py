#!/usr/bin/env python3
"""Generate a multi-page PDF report for an individual rower.

Scans all CSVs in archived_data/ to build a longitudinal profile of one rower,
including power trends, arc stats, consistency, timing analysis, seat history,
and personal bests — all broken down by rate band.

Usage:
    python generate_individual_report.py --name "Mershon"
    python generate_individual_report.py                   # interactive
"""

import argparse
import csv
import os
import re
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore", message=".*empty slice.*")
warnings.filterwarnings("ignore", message=".*Degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHIVE_DIR = Path(__file__).parent.parent / "archived_data"
OUTPUT_DIR  = Path(__file__).parent.parent / "output"

PAGE_W, PAGE_H = 16.5, 11.7
BG = "#f5f5f0"

RATE_BANDS = [
    (0,  24, "< r24"),
    (24, 30, "r24–30"),
    (30, 34, "r30–34"),
    (34, 38, "r34–38"),
    (38, 100, "r38+"),
]
RATE_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

PIECE_LEN_TOL = 0.40          # ±40 % of median stroke-count to be "comparable"
MIN_STROKES_BAND = 5          # need ≥5 strokes in a rate band to count

BOAT_COLORS = {"1V": "#e74c3c", "2V": "#2196F3", "3V": "#4CAF50", "4V": "#FF9800"}

# CSV column names that live per-seat (8 columns each)
_CSV_METRICS = [
    "SwivelPower", "MinAngle", "MaxAngle", "CatchSlip", "FinishSlip",
    "Drive Start T", "Drive Time", "Recovery Time",
    "Angle Max F", "Work PC Q1", "Work PC Q2", "Work PC Q3", "Work PC Q4",
]

# Name alias groups — each list is one person.  The FIRST entry is canonical.
_ALIAS_GROUPS = [
    ["Green, T", "GreenT"],
    ["du Croo de Jongh", "Du Croo de Jongh"],
    ["Ramakrishnan, R", "RamakrishnanR"],
    ["Gaensler", "Gaensler/Willott"],
    ["Willott", "Willott/Gaensler"],
    ["Gibbons", "Gibbons/Valt"],
    ["Valt", "Valt/Gibbons"],
    ["Hillicks-Tulip", "Hillicks-Tulip/Abril"],
    ["Cadwallader", "Caddy"],
    ["Mershon", "Mersh"],
    ["Seguin", "Seg"],
]

# Build fast lookup: any variant -> canonical name
_ALIAS_MAP = {}
for _grp in _ALIAS_GROUPS:
    canon = _grp[0]
    for _v in _grp:
        _ALIAS_MAP[_v] = canon
        _ALIAS_MAP[_v.lower()] = canon

# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _parse_date(fname):
    m = re.search(r"(\d+)\s*(?:st|nd|rd|th)?[\s-]*(?:APR|Apr|April)", fname, re.I)
    return datetime(2026, 4, int(m.group(1))) if m else None

def _parse_boat(fname):
    m = re.search(r"(\d)V", fname)
    return m.group(0) if m else None

def _parse_piece(fname):
    m = re.search(r"-(\d+)(?:\s*\(\d+\))?\.csv$", fname)
    return int(m.group(1)) if m else 1

def _list_csvs():
    """List archive CSVs, skipping '(N)' duplicates when a clean version exists."""
    all_f = sorted(f for f in os.listdir(ARCHIVE_DIR) if f.endswith(".csv"))
    clean = {f for f in all_f if not re.search(r"\(\d+\)\.csv$", f)}
    out = []
    for f in all_f:
        base = re.sub(r"\s*\(\d+\)(\.csv)$", r"\1", f)
        if f != base and base in clean:
            continue
        out.append(f)
    return out

# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _norm(name):
    if name is None: return None
    name = name.strip()
    return _ALIAS_MAP.get(name, _ALIAS_MAP.get(name.lower(), name))

def _parse_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    hdr_idx = None
    for i, row in enumerate(rows):
        if len(row) > 1 and row[1] == "SwivelPower":
            hdr_idx = i; break
    if hdr_idx is None:
        return None

    header = rows[hdr_idx]

    def _cols(name):
        for idx, h in enumerate(header):
            if h == name:
                return idx, idx + 8
        return None

    metrics = {}
    for m in _CSV_METRICS:
        c = _cols(m)
        if c: metrics[m] = c

    names = []
    for r in range(min(9, hdr_idx)):
        if len(rows[r]) > 13 and rows[r][13].strip():
            names.append(rows[r][13].strip())
        else:
            names.append(None)
    while len(names) < 9:
        names.append(None)

    if all(n is None for n in names[:8]):
        return None

    sp = metrics.get("SwivelPower")
    if sp is None: return None

    strokes = {m: [] for m in metrics}
    strokes["rating"] = []
    strokes["boat_speed"] = []

    for row in rows[hdr_idx + 2:]:
        if len(row) < 9 or row[0] == "Time":
            break
        if all(v == "" for v in row[sp[0]:sp[1]]):
            continue
        for m, (s, e) in metrics.items():
            vals = []
            for v in row[s:e]:
                try: vals.append(float(v))
                except: vals.append(0.0)
            strokes[m].append(vals)
        try:    strokes["rating"].append(float(row[-6]) if row[-6] else 0.0)
        except: strokes["rating"].append(0.0)
        try:    strokes["boat_speed"].append(float(row[-5]) if row[-5] else 0.0)
        except: strokes["boat_speed"].append(0.0)

    for m in metrics:
        strokes[m] = np.array(strokes[m], dtype=float)

    return {"names": names, "strokes": strokes, "n": len(strokes["rating"])}

# ---------------------------------------------------------------------------
# Collect all sessions for one rower
# ---------------------------------------------------------------------------

def _collect(rower):
    sessions = []
    for fname in _list_csvs():
        p = _parse_csv(str(ARCHIVE_DIR / fname))
        if p is None: continue

        seat = None
        for i in range(min(8, len(p["names"]))):
            if p["names"][i] and _norm(p["names"][i]).lower() == rower.lower():
                seat = i; break
        if seat is None: continue

        st = p["strokes"]
        n  = p["n"]
        if n < 3: continue

        md = {}
        for m in _CSV_METRICS:
            md[m] = st[m][:, seat] if m in st and st[m].shape[1] > seat else np.zeros(n)
        md["OverallLength"]   = md["MaxAngle"] - md["MinAngle"]
        md["EffectiveLength"] = md["OverallLength"] - md["CatchSlip"] - md["FinishSlip"]

        front_name = front_dst = front2_name = front2_dst = None
        if seat < 7:
            fn = p["names"][seat + 1]
            front_name = _norm(fn) if fn else None
            if "Drive Start T" in st and st["Drive Start T"].shape[1] > seat + 1:
                front_dst = st["Drive Start T"][:, seat + 1]
        if seat < 6:
            fn2 = p["names"][seat + 2]
            front2_name = _norm(fn2) if fn2 else None
            if "Drive Start T" in st and st["Drive Start T"].shape[1] > seat + 2:
                front2_dst = st["Drive Start T"][:, seat + 2]

        sessions.append({
            "file": fname, "date": _parse_date(fname), "boat": _parse_boat(fname),
            "piece": _parse_piece(fname), "seat": seat, "n": n,
            "rating": np.array(st["rating"]), "boat_speed": np.array(st["boat_speed"]),
            "m": md,
            "front_name": front_name, "front_dst": front_dst,
            "front2_name": front2_name, "front2_dst": front2_dst,
            "all_names": [_norm(nm) for nm in p["names"][:8]],
        })

    sessions.sort(key=lambda s: (s["date"] or datetime.min, s["piece"]))
    return sessions

# ---------------------------------------------------------------------------
# Rate-band helpers
# ---------------------------------------------------------------------------

def _band_idx(rate):
    for i, (lo, hi, _) in enumerate(RATE_BANDS):
        if lo <= rate < hi: return i
    return len(RATE_BANDS) - 1

def _band_stats(sessions, key):
    """Per-session, per-rate-band: mean / std / cv / n for a metric."""
    out = []
    for sess in sessions:
        r = sess["rating"]
        v = sess["m"][key]
        bands = {}
        for bi, (lo, hi, _) in enumerate(RATE_BANDS):
            mask = (r >= lo) & (r < hi) & np.isfinite(v) & (v != 0)
            c = mask.sum()
            if c >= MIN_STROKES_BAND:
                d = v[mask]
                mu = np.nanmean(d)
                sd = np.nanstd(d)
                bands[bi] = {"mean": mu, "std": sd,
                             "cv": (sd / abs(mu) * 100) if mu else 0, "n": c}
        out.append(bands)
    return out

def _filter_comparable(sessions, bs_list):
    """Remove band entries whose stroke count is not comparable to peers."""
    filt = [dict(bs) for bs in bs_list]
    for bi in range(len(RATE_BANDS)):
        counts = [(si, bs[bi]["n"]) for si, bs in enumerate(bs_list) if bi in bs]
        if len(counts) < 2: continue
        med = np.median([c for _, c in counts])
        lo, hi = med * (1 - PIECE_LEN_TOL), med * (1 + PIECE_LEN_TOL)
        for si, n in counts:
            if n < lo or n > hi:
                filt[si].pop(bi, None)
    return filt

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _label(sess):
    d = sess["date"]
    ds = d.strftime("%b %d") if d else "?"
    b  = sess["boat"] or "?"
    return f"{ds} {b} p{sess['piece']}"

def _labels(sessions):
    return [_label(s) for s in sessions]

def _header(fig, title, sub=""):
    fig.set_facecolor(BG)
    fig.text(0.5, 0.96, title, ha="center", va="top",
             fontsize=22, fontweight="bold", color="#1a1a2e")
    if sub:
        fig.text(0.5, 0.93, sub, ha="center", va="top",
                 fontsize=12, color="#555")

def _style_table(tbl, col_labels, n_rows):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1a1a2e")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(n_rows):
        bg = "#f0f0ea" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)

# ---------------------------------------------------------------------------
# Page 1 — Title & Summary
# ---------------------------------------------------------------------------

def _page_title(fig, name, sessions):
    _header(fig, f"Individual Report — {name}")

    dates = [s["date"] for s in sessions if s["date"]]
    dr = (f'{min(dates).strftime("%b %d")} – {max(dates).strftime("%b %d, %Y")}'
          if dates else "N/A")
    boats   = sorted(set(s["boat"] for s in sessions if s["boat"]))
    seats   = sorted(set(s["seat"] + 1 for s in sessions))
    n_days  = len(set(s["date"] for s in sessions if s["date"]))

    pw  = np.concatenate([s["m"]["SwivelPower"] for s in sessions])
    ef  = np.concatenate([s["m"]["EffectiveLength"] for s in sessions])
    cs  = np.concatenate([s["m"]["CatchSlip"] for s in sessions])
    fs  = np.concatenate([s["m"]["FinishSlip"] for s in sessions])
    dst = np.concatenate([s["m"]["Drive Start T"] for s in sessions])

    lines = [
        f"Pieces analysed   {len(sessions)}  (across {n_days} session days)",
        f"Date range         {dr}",
        f"Boats              {', '.join(boats)}",
        f"Seats occupied     {', '.join(str(s) for s in seats)}",
        "",
        f"Avg Power          {np.nanmean(pw[pw!=0]):.1f} W",
        f"Avg Eff. Length    {np.nanmean(ef[ef!=0]):.1f}°",
        f"Avg Catch Slip     {np.nanmean(cs[cs!=0]):.1f}°",
        f"Avg Finish Slip    {np.nanmean(fs[fs!=0]):.1f}°",
        f"Avg Timing Offset  {np.nanmean(dst[dst!=0]):.1f} ms",
    ]
    y = 0.82
    for ln in lines:
        fig.text(0.08, y, ln, fontsize=14, color="#333", fontfamily="monospace")
        y -= 0.04

# ---------------------------------------------------------------------------
# Page 2 — Power at Rate Bands
# ---------------------------------------------------------------------------

def _page_power(fig, sessions):
    _header(fig, "Power Over Time — by Rate Band",
            "Only comparable pieces shown (similar stroke count within each rate band)")

    bs   = _filter_comparable(sessions, _band_stats(sessions, "SwivelPower"))
    labs = _labels(sessions)
    x    = np.arange(len(sessions))

    ax = fig.add_axes([0.07, 0.08, 0.88, 0.78])
    ax.set_facecolor(BG)
    for bi, (_, _, bl) in enumerate(RATE_BANDS):
        xs, ys = zip(*[(si, bs[si][bi]["mean"])
                       for si in range(len(bs)) if bi in bs[si]]) if any(
            bi in bs[si] for si in range(len(bs))) else ([], [])
        if xs:
            ax.plot(xs, ys, "o-", color=RATE_COLORS[bi], label=bl,
                    markersize=6, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Avg Power (W)", fontsize=12)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Page 3 — Arc / Angle Stats
# ---------------------------------------------------------------------------

_ANGLE_METRICS = [
    ("EffectiveLength", "Effective Length (°)"),
    ("OverallLength",   "Overall Length (°)"),
    ("MinAngle",        "Min Angle (°)"),
    ("MaxAngle",        "Max Angle (°)"),
    ("CatchSlip",       "Catch Slip (°)"),
    ("FinishSlip",      "Finish Slip (°)"),
]

def _page_angles(fig, sessions):
    _header(fig, "Arc & Angle Stats Over Time — by Rate Band")
    labs = _labels(sessions)
    x = np.arange(len(sessions))

    for pi, (key, lbl) in enumerate(_ANGLE_METRICS):
        row, col = divmod(pi, 3)
        ax = fig.add_axes([0.06 + col * 0.32,
                           0.50 - row * 0.43 + 0.06,
                           0.27, 0.34])
        ax.set_facecolor(BG)
        bs = _filter_comparable(sessions, _band_stats(sessions, key))
        for bi, (_, _, bl) in enumerate(RATE_BANDS):
            xs = [si for si in range(len(bs)) if bi in bs[si]]
            ys = [bs[si][bi]["mean"] for si in xs]
            if xs:
                ax.plot(xs, ys, "o-", color=RATE_COLORS[bi],
                        markersize=4, linewidth=1.2, label=bl)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=5)
        ax.grid(True, alpha=0.2)
        if pi == 0:
            ax.legend(fontsize=6, loc="best")

# ---------------------------------------------------------------------------
# Page 4 — Consistency (CV) Over Time
# ---------------------------------------------------------------------------

_CV_METRICS = [
    ("SwivelPower",     "Power CV (%)"),
    ("EffectiveLength", "Eff. Length CV (%)"),
    ("CatchSlip",       "Catch Slip CV (%)"),
    ("FinishSlip",      "Finish Slip CV (%)"),
    ("Angle Max F",     "Max Force Angle CV (%)"),
]

def _page_consistency(fig, sessions):
    _header(fig, "Consistency Over Time — CV at Rate Bands",
            "Lower CV = more consistent stroke-to-stroke")
    labs = _labels(sessions)
    x = np.arange(len(sessions))

    n_cv = len(_CV_METRICS)
    n_cols = 3 if n_cv > 3 else n_cv
    n_rows_g = (n_cv + n_cols - 1) // n_cols
    for pi, (key, lbl) in enumerate(_CV_METRICS):
        row, col = divmod(pi, n_cols)
        ax = fig.add_axes([0.06 + col * 0.32,
                           0.50 - row * 0.43 + 0.06,
                           0.27, 0.34])
        ax.set_facecolor(BG)
        bs = _band_stats(sessions, key)       # no comparability filter for CV
        for bi, (_, _, bl) in enumerate(RATE_BANDS):
            xs = [si for si in range(len(bs)) if bi in bs[si]]
            ys = [bs[si][bi]["cv"] for si in xs]
            if xs:
                ax.plot(xs, ys, "o-", color=RATE_COLORS[bi],
                        markersize=4, linewidth=1.2, label=bl)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=5)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.2)
        if pi == 0:
            ax.legend(fontsize=6, loc="best")

# ---------------------------------------------------------------------------
# Page 5 — Timing: who are they late with?
# ---------------------------------------------------------------------------

def _page_timing(fig, name, sessions):
    _header(fig, f"Timing Analysis — Who Is {name} Late With?",
            "Compares timing with the 1–2 rowers in front (higher seat #, towards stroke)")

    gaps = defaultdict(list)   # front-name -> list of {label, mean_gap, n}

    for sess in sessions:
        my = sess["m"]["Drive Start T"]
        for fn, fd, sdiff in [(sess["front_name"],  sess["front_dst"],  1),
                               (sess["front2_name"], sess["front2_dst"], 2)]:
            if fn is None or fd is None: continue
            g   = my - fd
            ok  = np.isfinite(g) & (my != 0) & (fd != 0)
            if ok.sum() < 5: continue
            gaps[fn].append({"label": _label(sess),
                             "mean": np.nanmean(g[ok]), "n": int(ok.sum()),
                             "diff": sdiff})

    if not gaps:
        fig.text(0.5, 0.5, "No timing data (rower may sit in stroke seat)",
                 ha="center", fontsize=16, color="#999")
        return

    agg = []
    for nm, gs in gaps.items():
        mg = np.mean([g["mean"] for g in gs])
        nl = sum(1 for g in gs if g["mean"] > 0)
        agg.append({"name": nm, "gap": mg, "n_sess": len(gs),
                     "n_late": nl, "pct": nl / len(gs) * 100})
    agg.sort(key=lambda a: a["gap"], reverse=True)

    # ---- bar chart (left) ----
    ax = fig.add_axes([0.07, 0.08, 0.42, 0.78])
    ax.set_facecolor(BG)
    yp = np.arange(len(agg))
    colors = ["#e74c3c" if a["gap"] > 0 else "#2ecc71" for a in agg]
    ax.barh(yp, [a["gap"] for a in agg], color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(yp)
    ax.set_yticklabels([a["name"] for a in agg], fontsize=10)
    ax.set_xlabel("Avg timing gap (ms) — positive = late", fontsize=10)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title("Avg Gap vs Front Rower", fontsize=12, fontweight="bold")

    # ---- table (right) ----
    ax2 = fig.add_axes([0.54, 0.08, 0.42, 0.78])
    ax2.axis("off")
    cols = ["Front Rower", "Sessions", "Avg Gap", "% Late"]
    rows_data = [[a["name"], str(a["n_sess"]),
                  f'{a["gap"]:+.1f} ms', f'{a["pct"]:.0f}%'] for a in agg]
    if rows_data:
        tbl = ax2.table(cellText=rows_data, colLabels=cols,
                        loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#1a1a2e")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i, a in enumerate(agg):
            c = "#ffcccc" if a["gap"] > 0 else "#ccffcc"
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor(c)


# ---------------------------------------------------------------------------
# Page 7 — Personal Bests
# ---------------------------------------------------------------------------

_PB = [
    ("SwivelPower",     "Highest Avg Power (W)",             True),
    ("EffectiveLength", "Longest Effective Length (°)",       True),
    ("OverallLength",   "Longest Overall Arc (°)",           True),
    ("CatchSlip",       "Cleanest Catch Slip (°)",           False),
    ("FinishSlip",      "Cleanest Finish Slip (°)",          False),
    ("Drive Start T",   "Tightest Timing (|ms| from stroke)", False),   # smallest |val|
    ("Angle Max F",     "Best Max Force Angle (°)",          True),
]

def _page_pb(fig, name, sessions):
    _header(fig, f"Personal Bests — {name}")

    ax = fig.add_axes([0.06, 0.08, 0.88, 0.80])
    ax.axis("off")

    cols = ["Metric", "Best Value", "Session", "Boat", "Seat", "Avg Rate"]
    rows_data = []

    for key, lbl, higher in _PB:
        best_v, best_s = None, None
        for sess in sessions:
            v = sess["m"].get(key)
            if v is None: continue
            ok = v[np.isfinite(v) & (v != 0)]
            if len(ok) < 3: continue
            if key == "Drive Start T":
                a = np.nanmean(np.abs(ok))
                if best_v is None or a < best_v:
                    best_v, best_s = a, sess
            elif higher:
                a = np.nanmean(ok)
                if best_v is None or a > best_v:
                    best_v, best_s = a, sess
            else:
                a = np.nanmean(ok)
                if best_v is None or a < best_v:
                    best_v, best_s = a, sess
        if best_s:
            rows_data.append([
                lbl, f"{best_v:.1f}", _label(best_s),
                best_s["boat"] or "?", str(best_s["seat"] + 1),
                f'r{np.nanmean(best_s["rating"]):.0f}',
            ])

    # Peak single-stroke power
    pk, pk_s = 0, None
    for sess in sessions:
        v = sess["m"]["SwivelPower"]
        ok = v[np.isfinite(v)]
        if len(ok) and np.nanmax(ok) > pk:
            pk, pk_s = np.nanmax(ok), sess
    if pk_s:
        rows_data.append([
            "Peak Single-Stroke Power (W)", f"{pk:.1f}", _label(pk_s),
            pk_s["boat"] or "?", str(pk_s["seat"] + 1),
            f'r{np.nanmean(pk_s["rating"]):.0f}',
        ])

    if rows_data:
        tbl = ax.table(cellText=rows_data, colLabels=cols,
                       loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.2)
        for j in range(len(cols)):
            tbl[0, j].set_facecolor("#1a1a2e")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(len(rows_data)):
            bg = "#f0f0ea" if i % 2 == 0 else "white"
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor(bg)

# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------

def _gen(name, sessions):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    path = OUTPUT_DIR / f"Individual_Report_{safe}.pdf"

    pages = [
        ("Title & Summary",       lambda f: _page_title(f, name, sessions)),
        ("Power at Rate Bands",   lambda f: _page_power(f, sessions)),
        ("Arc & Angle Stats",     lambda f: _page_angles(f, sessions)),
        ("Consistency (CV)",      lambda f: _page_consistency(f, sessions)),
        ("Timing — Lateness",     lambda f: _page_timing(f, name, sessions)),
        ("Personal Bests",        lambda f: _page_pb(f, name, sessions)),
    ]

    with PdfPages(str(path)) as pdf:
        for i, (pg_name, draw) in enumerate(pages, 1):
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))
            draw(fig)
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Page {i}: {pg_name}")

    print(f"\n  Saved → {path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Individual rower report")
    ap.add_argument("--name", help="Rower surname (case-insensitive)")
    args = ap.parse_args()

    rower = args.name.strip() if args.name else input("Enter rower name: ").strip()
    if not rower:
        sys.exit("No name provided")

    # Resolve alias before searching (e.g. "Caddy" -> "Cadwallader")
    rower = _norm(rower)

    # Discover all canonical names
    all_names = set()
    for fname in _list_csvs():
        p = _parse_csv(str(ARCHIVE_DIR / fname))
        if not p: continue
        for n in p["names"][:8]:
            if n: all_names.add(_norm(n))

    # Match
    matched = None
    for n in all_names:
        if n.lower() == rower.lower():
            matched = n; break
    if not matched:
        cands = [n for n in all_names if rower.lower() in n.lower()]
        if len(cands) == 1:
            matched = cands[0]
        elif cands:
            print(f"Multiple matches for '{rower}':  {', '.join(sorted(cands))}")
            sys.exit("Be more specific")
        else:
            print(f"'{rower}' not found.  Available rowers:")
            for n in sorted(all_names): print(f"  {n}")
            sys.exit()

    print(f"Generating report for: {matched}")
    sessions = _collect(matched)
    print(f"  Found {len(sessions)} pieces")
    if not sessions: sys.exit("No data found")
    _gen(matched, sessions)


if __name__ == "__main__":
    main()
