"""Near-surface air temperature (NASA SVS 5147, GEOS-FP): decode NASA's hourly frames into degrees Celsius and check them
against GEOS-FP itself (NASA GMAO, OPeNDAP at opendap.nccs.nasa.gov), the model the picture is drawn from.

    uv run --with numpy --with scipy --with pillow python pipeline/air.py offset     # find the GEOS-FP time offset
    uv run --with numpy --with scipy --with pillow python pipeline/air.py validate   # compare decoded frames with GEOS-FP
    uv run --with numpy --with scipy --with pillow python pipeline/air.py notes      # 6-hour app data (mockups/data/air/)

Facts this script relies on (round 65):
    Frames: frames/2048x1024_2x1_12p/MC01_stage3_GMAO_near_surface_temperature_2048x1024_en-hwtool/00000001.png to 00000372.png,
    lossless PNG, one per hour. The time is written on each frame: frame 1 is 30 Oct 2026 05:00 EDT (09:00 UTC) and frame 372 is
    14 Nov 2026 15:00 EST (20:00 UTC), 371 hours later, so frame n (from 0) is 30 Oct 2026 09:00 UTC + n hours.
    The files were written on 14 Nov 2026 about 07:30 UTC, so the last hours are a forecast (see FORECAST_FROM).
    The frame has two labels: the NASA/GMAO logo (top left) and the time (bottom left). Those pixels are not data.
    Colour bar NearSurfaceTemperature_colorbar_plain.png (528 x 165): -40 to 40 degrees Celsius, ticks every 10 degrees at
    x 121.5 (-30) to 414.5 (30), 4.883 px per degree, so -40 at x 73 and 40 at x 463. Rows 80 to 105 are clean.
    NASA's bar prints "-10" under the -30 tick; the ticks themselves are evenly spaced, so we read it as -30.
    The colours do not run in one direction (dark blue, white, purple, blue, cyan, yellow, red, black, grey); the closest
    two colours for temperatures more than 3 degrees apart are 15 RGB steps apart (dark reds near 31 and 35 degrees).
"""
import datetime as dt
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "air"
FRAMES = CACHE / "frames"
OUT = HERE / "out" / "air"
T0 = dt.datetime(2026, 10, 30, 9)  # UTC time of frame 0
N = 372
HOURS_PER_NOTE = 6  # James, round 65
X0, PPC = 268.5, 4.883  # x of 0 degrees, px per degree
X_MIN, X_MAX = 73, 462
MAX_DIST = 20.0  # RGB distance above which a pixel is not on the bar
FILES_WRITTEN = dt.datetime(2026, 11, 14, 7, 30)  # HTTP Last-Modified of the frames
LABELS = [(0, 0, 350, 76), (0, 922, 502, 1024)]  # x0, y0, x1, y1 on the 2048 x 1024 frame: logo, time
GEOS = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/tavg1_2d_slv_Nx"
# frame 0 is near index 78128 if the calendar guess is right; the check needs frames 3 to 291 and shifts up to 30 hours
GEOS_LAST, GEOS_SPAN = 78460, 370


def bar():
    a = np.asarray(Image.open(CACHE / "NearSurfaceTemperature_colorbar_plain.png").convert("RGB")).astype(np.float32)
    band = a[80:106, X_MIN:X_MAX + 1].mean(0)
    values = (np.arange(X_MIN, X_MAX + 1) - X0) / PPC
    return cKDTree(band), values.astype(np.float32), band


def frame(n):
    return np.asarray(Image.open(FRAMES / f"{n + 1:08d}.png").convert("RGB")).astype(np.float32)


def decode(img, tree, values):
    d, j = tree.query(img.reshape(-1, 3))
    v = values[j]
    v[d > MAX_DIST] = np.nan
    v = v.reshape(img.shape[:2])
    for x0, y0, x1, y1 in LABELS:
        v[y0:y1, x0:x1] = np.nan
    return v


def full_lut(tree, values):
    """The frames are 8-bit PNG, so a table for all 256^3 colours decodes exactly (the same nearest colour as decode())."""
    p = OUT / "lut.npy"
    if p.exists():
        return np.load(p)
    lut = np.empty(256 ** 3, np.float32)
    g = np.arange(256, dtype=np.float32)
    for r in range(256):
        pts = np.stack(np.meshgrid([r], g, g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
        d, j = tree.query(pts, workers=-1)
        v = values[j]
        v[d > MAX_DIST] = np.nan
        lut[r * 65536:(r + 1) * 65536] = v
    np.save(p, lut)
    return lut


def decode_fast(img_u8, lut):
    a = img_u8.astype(np.int32)
    v = lut[(a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]].copy()
    for x0, y0, x1, y1 in LABELS:
        v[y0:y1, x0:x1] = np.nan
    return v


def when(n):
    return T0 + dt.timedelta(hours=n)


# GEOS-FP's time axis is "days since 1-1-1" in the GrADS calendar. Index 78506 was the newest hour on 15 Nov 2026; instead of
# trusting a calendar conversion, `offset` finds the hour shift at which the frames and GEOS-FP agree best.
def geos_series(lat, lon, i0, i1):
    """In 24-hour pieces: the server gives no answer within a minute for 480 hours, and answers 24 hours in about 5 s."""
    yi, xi = int(round((lat + 90) / 0.25)), int(round((lon + 180) / 0.3125)) % 1152
    vals = []
    for a in range(i0, i1 + 1, 24):
        b = min(i1, a + 23)
        url = f"{GEOS}.ascii?t2m%5B{a}:{b}%5D%5B{yi}%5D%5B{xi}%5D"
        for _ in range(4):
            out = subprocess.run(["curl", "-s", "-m", "90", url], capture_output=True, text=True).stdout
            rows = [l for l in out.splitlines() if l.startswith("[")]
            if len(rows) == b - a + 1:
                break
        if len(rows) != b - a + 1:
            raise RuntimeError(f"GEOS-FP gave {len(rows)} of {b - a + 1} hours at {lat}, {lon}")
        vals += [float(l.split(",")[-1]) - 273.15 for l in rows]
    return np.array(vals)


POINTS = {
    "Vaasa, Finland": (63.1, 21.6), "Lagos, Nigeria": (6.45, 3.39), "Sahara, Algeria": (25.0, 3.0),
    "Equatorial Pacific": (0.0, -140.0), "North Atlantic": (40.0, -40.0), "Siberia": (62.0, 110.0),
    "Amazon, Brazil": (-5.0, -60.0), "Sydney, Australia": (-33.9, 151.2), "Antarctic coast": (-70.0, 0.0),
    "Tibet": (33.0, 88.0), "Chicago, United States": (41.9, -87.6), "Indian Ocean": (-20.0, 75.0),
    "Cape Town, South Africa": (-33.9, 18.4), "Greenland": (72.0, -40.0), "Delhi, India": (28.6, 77.2),
    "Southern Ocean": (-55.0, 120.0), "Mexico City": (19.4, -99.1), "Tokyo, Japan": (35.7, 139.7),
}
TEST_FRAMES = list(range(3, 300, 12))  # 25 hours spread over the first 12.5 days (before any forecast)


def ours_at(g, lat, lon, r=2):
    y, x = int((90 - lat) / 180 * 1024), int((lon + 180) / 360 * 2048) % 2048
    w = g[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return float(np.nanmedian(w)) if np.isfinite(w).sum() >= 13 else None


def frame_values(tree, values):
    cache = OUT / "frame_values.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    got = {}
    for n in TEST_FRAMES:
        g = decode(frame(n), tree, values)
        got[n] = {k: ours_at(g, la, lo) for k, (la, lo) in POINTS.items()}
    cache.write_text(json.dumps(got))
    return got


def geos_all(last_index, span):
    """All places at once (the server takes about a minute per place), cached in out/air/geos.json."""
    i0 = last_index - span
    cache = OUT / "geos.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c["i0"] == i0 and set(c["series"]) == set(POINTS):
            return i0, {k: np.array(v) for k, v in c["series"].items()}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as ex:
        def safe(kv):  # one place that GEOS-FP does not answer must not stop the check (round 74)
            try:
                return geos_series(kv[1][0], kv[1][1], i0, last_index)
            except RuntimeError as e:
                print(f"   GEOS-FP: {e}; this place is left out", flush=True)
                return np.array([])
        got = dict(zip(POINTS, ex.map(safe, POINTS.items())))
    cache.write_text(json.dumps({"i0": i0, "series": {k: v.tolist() for k, v in got.items()}}))
    return i0, got


def compare(got, i0, geos, idx_of_frame0):
    """tavg1 values are hour means centred on :30, so a frame at hh:00 is the mean of the hours before and after it."""
    diffs = []
    for n, row in got.items():
        i = idx_of_frame0 + n - i0
        for k, ours in row.items():
            s = geos[k]
            if ours is None or i - 1 < 0 or i >= len(s):
                continue
            diffs.append((n, k, ours, (s[i - 1] + s[i]) / 2))
    return diffs


def offset():
    tree, values, _ = bar()
    got = frame_values(tree, values)
    last = GEOS_LAST
    i0, geos = geos_all(last, GEOS_SPAN)
    # if GrADS time = days since 1-1-1 with a 2-day calendar shift, index 78506 (739936.104) is 15 Nov 2026 02:30 UTC
    guess = 78506 - int(round((dt.datetime(2026, 11, 15, 3) - T0).total_seconds() / 3600))
    res = []
    for sh in range(-30, 31):
        d = compare(got, i0, geos, guess + sh)
        e = np.array([o - g for _, _, o, g in d])
        if len(e):
            res.append((float(np.abs(e).mean()), sh, len(e)))
    res.sort()
    print("best hour shifts (mean absolute, shift, pairs):", [(round(a, 2), b, c) for a, b, c in res[:6]])
    best = res[0][1]
    (OUT / "offset.json").write_text(json.dumps({"index_of_frame0": guess + best, "shift_from_guess_h": best,
                                                 "mean_abs_c": res[0][0], "geos_i0": i0}))
    return guess + best


def validate():
    tree, values, _ = bar()
    off = json.loads((OUT / "offset.json").read_text())
    got = frame_values(tree, values)
    i0, geos = geos_all(GEOS_LAST, GEOS_SPAN)
    d = compare(got, i0, geos, off["index_of_frame0"])
    rows = [{"time_utc": when(n).isoformat(), "place": k, "decoded_c": round(o, 2), "geos_fp_c": round(g, 2), "difference_c": round(o - g, 2)} for n, k, o, g in d]
    (OUT / "validate.json").write_text(json.dumps(rows, indent=1))
    e = np.array([r["difference_c"] for r in rows])
    g = np.array([r["geos_fp_c"] for r in rows])
    inb = (g > -39.5) & (g < 39.5)
    print(f"{len(e)} pairs. mean {e.mean():+.2f} C, mean absolute {np.abs(e).mean():.2f} C, 95% within {np.percentile(np.abs(e), 95):.2f} C, largest {np.abs(e).max():.2f} C")
    print(f"inside the bar: {inb.sum()} pairs, mean absolute {np.abs(e[inb]).mean():.2f} C")
    for k in POINTS:
        ek = np.array([r["difference_c"] for r in rows if r["place"] == k])
        if len(ek):
            print(f"   {k:26s} n {len(ek):2d}  mean {ek.mean():+.2f}  mean abs {np.abs(ek).mean():.2f}")


def notes():
    """62 notes of 6 hours on a 1-degree grid. A cell needs data in at least half of its pixels in at least 3 of the 6 hours."""
    tree, values, band = bar()
    lut = full_lut(tree, values)
    rows = (np.arange(1024) * 180 // 1024)
    cols = (np.arange(2048) * 360 // 2048)
    cell = (rows[:, None] * 360 + cols[None, :]).ravel()
    total = np.bincount(cell, None, 64800)
    n_notes = N // HOURS_PER_NOTE
    q = np.full((n_notes, 180, 360), 255, np.uint8)
    for k in range(n_notes):
        hours = []
        for n in range(k * HOURS_PER_NOTE, (k + 1) * HOURS_PER_NOTE):
            v = decode_fast(np.asarray(Image.open(FRAMES / f"{n + 1:08d}.png").convert("RGB")), lut).ravel()
            ok = np.isfinite(v)
            s, c = np.bincount(cell[ok], v[ok], 64800), np.bincount(cell[ok], None, 64800)
            hours.append(np.where(c >= total / 2, s / np.maximum(c, 1), np.nan))
        st = np.stack(hours)
        good = np.isfinite(st).sum(0) >= 3
        mean = np.where(good, np.nanmean(np.where(good[None], st, np.nan), 0), np.nan)
        q[k] = np.where(np.isfinite(mean), np.clip(np.round((mean + 40) / 0.4), 0, 200), 255).astype(np.uint8).reshape(180, 360)
        print(f"note {k + 1}/{n_notes} {when(k * HOURS_PER_NOTE):%d %b %H:%M} UTC", flush=True)
    dest = HERE.parent / "mockups" / "data" / "air"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))
    starts = [when(k * HOURS_PER_NOTE).strftime("%Y-%m-%dT%H:%MZ") for k in range(n_notes)]
    forecast = [when((k + 1) * HOURS_PER_NOTE) > FILES_WRITTEN for k in range(n_notes)]
    colours = [[int(round(c)) for c in band[int(round((t + 40) / 80 * (len(band) - 1)))]] for t in range(-40, 41, 2)]
    meta = {"record": "air", "svs": 5147, "unit": "degrees Celsius", "frames_read": N, "hours_per_note": HOURS_PER_NOTE,
            "note_starts_utc": starts, "week_starts": [s[:10] for s in starts], "forecast": forecast,
            "forecast_rule": f"a note is marked forecast when its last hour is after {FILES_WRITTEN:%d %b %Y %H:%M} UTC, when NASA wrote the frames",
            "grid": {"rows": 180, "cols": 360, "deg": 1, "row0_lat": 89.5, "col0_lon": -179.5},
            "encoding": "uint8 value = round((T + 40) / 0.4); 255 = no data. Values beyond -40 or 40 show as -40 or 40.",
            "colour_bar": {"from_c": -40, "step_c": 2, "source": "NASA SVS 5147 NearSurfaceTemperature_colorbar_plain.png", "rgb": colours},
            "error_vs_geos_fp": "see docs/PROCESS.md (round 65)"}
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n_notes} notes; forecast notes: {sum(forecast)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    {"offset": offset, "validate": validate, "notes": notes, "check": lambda: (offset(), validate())}[sys.argv[1]]()
