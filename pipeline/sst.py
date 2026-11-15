"""Sea surface temperature (NASA SVS 5101): decode NASA's video frames back into degrees Celsius,
then check the decoded values against the original satellite record (MUR SST, NOAA ERDDAP).

Run from the repo root:
    python3 pipeline/sst.py decode 0 450 900      # decode frames by number
    python3 pipeline/sst.py check 0 450 880        # decode and compare with MUR at test points

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
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(CACHE / "nodates.mp4"), "-vf", f"select=eq(n\\,{n})",
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


def main():
    sys.stdout.reconfigure(line_buffering=True)
    cmd, frames = sys.argv[1], [int(x) for x in sys.argv[2:]]
    OUT.mkdir(parents=True, exist_ok=True)
    band, values = lookup_table()
    cb = cube(band, values)
    rows = []
    for n in frames:
        date = frame_date(n)
        grid = decode(read_frame(n), cb)
        sea = np.isfinite(grid).mean()
        print(f"frame {n} = {date}: decoded {sea:.1%} of pixels as sea, range {np.nanmin(grid):.1f} to {np.nanmax(grid):.1f} C")
        if cmd == "check":
            for name, (lat, lon) in POINTS.items():
                ours = value_at(grid, lat, lon)
                ref = mur(date, lat, lon)
                diff = (ours - ref) if isinstance(ref, float) and ours is not None else None
                rows.append({"frame": n, "date": date.isoformat(), "place": name, "lat": lat, "lon": lon,
                             "decoded_c": None if ours is None else round(ours, 2),
                             "mur_c": ref if not isinstance(ref, float) else round(ref, 2),
                             "difference_c": None if diff is None else round(diff, 2)})
                print(f"   {name:30s} decoded {ours if ours is None else round(ours, 2)!s:>6}  MUR {ref if not isinstance(ref, float) else round(ref, 2)!s:>6}  diff {diff if diff is None else round(diff, 2)}")
    if rows:
        (OUT / "check.json").write_text(json.dumps(rows, indent=1))
        d = np.array([r["difference_c"] for r in rows if r["difference_c"] is not None])
        print(f"\n{len(d)} comparisons. Mean difference {d.mean():+.2f} C, mean absolute {np.abs(d).mean():.2f} C, "
              f"95th percentile absolute {np.percentile(np.abs(d), 95):.2f} C, largest {np.abs(d).max():.2f} C")


if __name__ == "__main__":
    main()
