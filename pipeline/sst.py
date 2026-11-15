"""Sea surface temperature (NASA SVS 5101): decode NASA's video frames back into degrees Celsius,
then check the decoded values against the original satellite record (MUR SST, NOAA ERDDAP).

Run from the repo root:
    python3 pipeline/sst.py check 0 450 880        # plain decode, compare with MUR at the calibration points
    python3 pipeline/sst.py calibrate              # fit the video colour curve on CAL_FRAMES x POINTS
    python3 pipeline/sst.py validate               # test the fitted curve on TEST_FRAMES x TEST_POINTS (never used in the fit)
    python3 pipeline/sst.py weekly                 # decode all frames into weekly 1-degree app data (mockups/data/sst/)

Inputs (downloaded into pipeline/cache/sst/, not in git):
    nodates.mp4   sst_30_sec_no-dates_4096x2048_2x1_30p.mp4  (901 frames, one per day)
    colorbar.png  sst_mur_colorbar.png  (-5 to 35 degrees Celsius, linear)

Facts this script relies on (checked by eye on the dated video, round 53):
    frame 0 = 27 May 2024, frame 450 = 20 Aug 2025, frame 899 = 12 Nov 2026, frame 900 = 13 Nov 2026.
    So frame n = 27 May 2024 + n days.
Colour bar ticks (pixel x at row 998 of colorbar.png): -5 at 479 ... 35 at 3360, every 5 degrees.
"""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "sst"
OUT = HERE / "out" / "sst"
DAY0 = dt.date(2024, 5, 27)
W, H = 4096, 2048
X_MIN, X_MAX, V_MIN, V_MAX = 479, 3360, -5.0, 35.0
MAX_DIST = 30.0  # colour distance (RGB, 0-255) above which a pixel is not on the colour bar: land, ice, labels


def frame_date(n):
    return DAY0 + dt.timedelta(days=n)


def lookup_table():
    """Colours along the middle of the bar, with the value for each column."""
    a = np.asarray(Image.open(CACHE / "colorbar.png").convert("RGB")).astype(np.float32)
    band = a[1040:1121, X_MIN:X_MAX + 1].mean(axis=0)  # average 81 rows across the bar
    values = V_MIN + (np.arange(X_MIN, X_MAX + 1) - X_MIN) / (X_MAX - X_MIN) * (V_MAX - V_MIN)
    return band, values


def cube(band, values, step=4):
    """Pre-compute the nearest bar colour for a grid of RGB colours, so a 4096 x 2048 frame decodes fast."""
    g = np.arange(0, 256, step, dtype=np.float32)
    rr, gg, bb = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([rr, gg, bb], -1).reshape(-1, 3)
    best_d = np.full(len(pts), 1e9, np.float32)
    best_v = np.zeros(len(pts), np.float32)
    for i in range(0, len(band), 64):
        chunk = band[i:i + 64]
        d = np.sqrt(((pts[:, None, :] - chunk[None, :, :]) ** 2).sum(-1))
        j = d.argmin(1)
        dm = d[np.arange(len(pts)), j]
        better = dm < best_d
        best_d[better] = dm[better]
        best_v[better] = values[i:i + 64][j[better]]
    n = len(g)
    return best_v.reshape(n, n, n), best_d.reshape(n, n, n), step


def read_frame(n):
    # BT.709 matrix, limited range: the video's own tags (round 53). Slightly better than ffmpeg's default BT.601.
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(CACHE / "nodates.mp4"), "-vf",
         f"select=eq(n\\,{n}),scale=in_color_matrix=bt709:in_range=tv:out_range=pc",
         "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W, 3)


def decode(frame, cb):
    vals, dist, step = cb
    idx = (frame.astype(np.int32) + step // 2) // step
    idx = np.clip(idx, 0, vals.shape[0] - 1)
    v = vals[idx[..., 0], idx[..., 1], idx[..., 2]]
    d = dist[idx[..., 0], idx[..., 1], idx[..., 2]]
    v = v.astype(np.float32)
    v[d > MAX_DIST] = np.nan
    return v


def pixel(lat, lon):
    x = int((lon + 180) / 360 * W) % W
    y = min(H - 1, max(0, int((90 - lat) / 180 * H)))
    return y, x


def value_at(grid, lat, lon, r=2):
    y, x = pixel(lat, lon)
    win = grid[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    return float(np.nanmedian(win)) if np.isfinite(win).any() else None


# Test points: open sea, chosen to cover cold, middle and warm water. Vaasa's nearest water is the Gulf of Bothnia.
POINTS = {
    "Gulf of Bothnia, near Vaasa": (63.0, 20.6),
    "Gulf of Guinea": (2.0, 3.0),
    "North Atlantic, mid": (30.0, -40.0),
    "Equatorial Pacific": (0.0, -140.0),
    "Southern Ocean": (-55.0, 0.0),
    "Mediterranean, Ionian Sea": (36.0, 18.0),
    "Arabian Sea": (15.0, 65.0),
    "Coral Sea": (-18.0, 155.0),
    "North Pacific": (40.0, -150.0),
    "Labrador Sea": (57.0, -52.0),
    "Norwegian Sea": (68.0, 5.0),
    "Bering Sea": (58.0, -175.0),
}


# Held-out test points: different places from POINTS, never used to fit the colour curve.
TEST_POINTS = {
    "Caribbean Sea": (15.0, -75.0),
    "Gulf of Mexico": (25.0, -90.0),
    "South Atlantic": (-25.0, -20.0),
    "Benguela, off Namibia": (-23.0, 12.0),
    "Bay of Bengal": (15.0, 88.0),
    "South China Sea": (12.0, 113.0),
    "Tasman Sea": (-38.0, 160.0),
    "East Pacific, off Peru": (-10.0, -85.0),
    "Gulf of Alaska": (56.0, -145.0),
    "Barents Sea": (73.0, 35.0),
    "North Sea": (56.0, 3.0),
    "Black Sea": (43.0, 34.0),
    "Red Sea": (20.0, 38.5),
    "Indian Ocean, south": (-30.0, 80.0),
    "Sea of Japan": (40.0, 135.0),
}
CAL_FRAMES = [100, 300, 600, 800]
TEST_FRAMES = [0, 200, 450, 700, 880]
CAL_FILE = OUT / "calibration.json"


def mur(date, lat, lon):
    """Daily MUR SST (JPL MUR v4.1, the source of NASA's picture) at one point, in degrees Celsius."""
    url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.json?"
           f"analysed_sst%5B({date.isoformat()}T09:00:00Z)%5D%5B({lat})%5D%5B({lon})%5D")
    try:  # curl, because this Python install has no SSL certificate bundle
        out = subprocess.run(["curl", "-s", "-m", "40", url], capture_output=True, text=True, check=True).stdout
        j = json.loads(out)
        v = j["table"]["rows"][0][-1]
        return None if v is None else float(v)
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


def bar_at(band, values, v):
    i = int(np.clip(np.round((v - values[0]) / (values[-1] - values[0]) * (len(values) - 1)), 0, len(band) - 1))
    return band[i]


def calibrate(band, values):
    """Fit pixel = 255 * (bar / 255) ** g, one g per colour channel, from (video pixel, bar colour at the MUR value) pairs."""
    P, B, n_used = [], [], 0
    for n in CAL_FRAMES:
        f = read_frame(n).astype(float)
        date = frame_date(n)
        for name, (lat, lon) in POINTS.items():
            if "Bothnia" in name:
                continue  # open problem, kept out of the fit
            ref = mur(date, lat, lon)
            if not isinstance(ref, float):
                continue
            y, x = pixel(lat, lon)
            P.append(np.median(f[y - 2:y + 3, x - 2:x + 3].reshape(-1, 3), 0))
            B.append(bar_at(band, values, ref))
            n_used += 1
    P, B = np.array(P), np.array(B)
    g = []
    for c in range(3):
        m = (B[:, c] > 8) & (P[:, c] > 4)
        lx, ly = np.log(B[m, c] / 255), np.log(P[m, c] / 255)
        g.append(float((lx * ly).sum() / (lx * lx).sum()))
    CAL_FILE.write_text(json.dumps({"gamma_rgb": g, "pairs": n_used, "frames": CAL_FRAMES,
                                    "points": [k for k in POINTS if "Bothnia" not in k]}, indent=1))
    print(f"calibration: gamma (r, g, b) = {[round(x, 3) for x in g]} from {n_used} pairs")
    return g


def corrected_cube(band, values):
    g = np.array(json.loads(CAL_FILE.read_text())["gamma_rgb"])
    return cube((255 * (band / 255) ** g).astype(np.float32), values)


def weekly():
    """Decode every frame once (one pass through the video), average 7 days per 1-degree cell, and write the app data.
    Output: mockups/data/sst/weekly.bin.gz  (gzip of) uint8, weeks x 180 x 360 (row 0 = 89.5 N, column 0 = 179.5 W),
            value = round((T + 5) / 0.2), 255 = no sea data.  mockups/data/sst/meta.json describes it."""
    band, values = lookup_table()
    cb = corrected_cube(band, values)
    SW, SH = 1440, 720  # nearest-neighbour sample of the 4096 x 2048 frame: no colour mixing
    vf = f"scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24,scale={SW}:{SH}:flags=neighbor"
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(CACHE / "nodates.mp4"), "-vf", vf,
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = SW * SH * 3
    n_weeks = 901 // 7
    acc = np.zeros((n_weeks, 180, 360), np.float64)
    cnt = np.zeros((n_weeks, 180, 360), np.int32)
    n = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        w = n // 7
        if w < n_weeks:
            g = decode(np.frombuffer(buf, np.uint8).reshape(SH, SW, 3), cb).reshape(180, 4, 360, 4)
            ok = np.isfinite(g)
            acc[w] += np.where(ok, g, 0).sum((1, 3))
            cnt[w] += ok.sum((1, 3))
        n += 1
        if n % 100 == 0:
            print(f"frame {n}", flush=True)
    proc.wait()
    mean = np.where(cnt >= 21, acc / np.maximum(cnt, 1), np.nan)  # at least 3 of 16 samples on 7 days
    q = np.where(np.isfinite(mean), np.clip(np.round((mean + 5) / 0.2), 0, 200), 255).astype(np.uint8)
    dest = HERE.parent / "mockups" / "data" / "sst"
    dest.mkdir(parents=True, exist_ok=True)
    import gzip
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))  # the browser unpacks it with DecompressionStream
    (OUT / "weekly.bin").write_bytes(q.tobytes())
    weeks = [(DAY0 + dt.timedelta(days=7 * i)).isoformat() for i in range(n_weeks)]
    cal = json.loads(CAL_FILE.read_text())
    meta = {"record": "sst", "svs": 5101, "unit": "degrees Celsius", "frames_read": n,
            "week_starts": weeks, "days_per_week": 7, "grid": {"rows": 180, "cols": 360, "deg": 1,
            "row0_lat": 89.5, "col0_lon": -179.5}, "encoding": "uint8 value = round((T + 5) / 0.2); 255 = no data",
            "colour_curve_gamma_rgb": cal["gamma_rgb"],
            "colour_bar": {"from_c": -5, "step_c": 1, "source": "NASA SVS 5101 sst_mur_colorbar.png, one colour per degree",
                           "rgb": [[int(round(v)) for v in band[min(len(band) - 1, int(round((t - values[0]) / (values[-1] - values[0]) * (len(band) - 1))))]] for t in range(-5, 36)]},
            "error_vs_mur": "see docs/PROCESS.md (held-out test: mean absolute 0.33 C, 95% within 0.87 C)"}
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n_weeks} weeks from {n} frames; sea cells per week: {int((q[0] != 255).sum())}; {len(q.tobytes())} bytes")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    cmd, frames = sys.argv[1], [int(x) for x in sys.argv[2:]]
    OUT.mkdir(parents=True, exist_ok=True)
    band, values = lookup_table()
    if cmd == "calibrate":
        calibrate(band, values)
        return
    if cmd == "weekly":
        weekly()
        return
    points = POINTS
    if cmd == "validate":
        cb, frames, points = corrected_cube(band, values), TEST_FRAMES, TEST_POINTS
    else:
        cb = cube(band, values)
    rows = []
    for n in frames:
        date = frame_date(n)
        grid = decode(read_frame(n), cb)
        sea = np.isfinite(grid).mean()
        print(f"frame {n} = {date}: decoded {sea:.1%} of pixels as sea, range {np.nanmin(grid):.1f} to {np.nanmax(grid):.1f} C")
        if cmd in ("check", "validate"):
            for name, (lat, lon) in points.items():
                ours = value_at(grid, lat, lon)
                ref = mur(date, lat, lon)
                diff = (ours - ref) if isinstance(ref, float) and ours is not None else None
                rows.append({"frame": n, "date": date.isoformat(), "place": name, "lat": lat, "lon": lon,
                             "decoded_c": None if ours is None else round(ours, 2),
                             "mur_c": ref if not isinstance(ref, float) else round(ref, 2),
                             "difference_c": None if diff is None else round(diff, 2)})
                print(f"   {name:30s} decoded {ours if ours is None else round(ours, 2)!s:>6}  MUR {ref if not isinstance(ref, float) else round(ref, 2)!s:>6}  diff {diff if diff is None else round(diff, 2)}")
    if rows:
        (OUT / f"{cmd}.json").write_text(json.dumps(rows, indent=1))
        d = np.array([r["difference_c"] for r in rows if r["difference_c"] is not None])
        print(f"\n{len(d)} comparisons. Mean difference {d.mean():+.2f} C, mean absolute {np.abs(d).mean():.2f} C, "
              f"95th percentile absolute {np.percentile(np.abs(d), 95):.2f} C, largest {np.abs(d).max():.2f} C")


if __name__ == "__main__":
    main()
