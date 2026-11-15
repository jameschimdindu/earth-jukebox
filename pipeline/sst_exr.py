"""Sea surface temperature (NASA SVS 5101) from NASA's daily EXR frames (round 62). Replaces the video decode in sst.py:
the EXR frames have exact colours and their dates in the file names, so there is no colour curve to fit.
Same commands and method as sst_anomaly.py (validate, days, days all, weekly); checked against MUR SST (NOAA ERDDAP).
The app data keeps the format of sst.py: uint8 value = round((T + 5) / 0.2), 255 = no data.

    uv run --with openexr --with numpy --with scipy python pipeline/sst_exr.py validate | days all | weekly

Facts (round 62):
    Frames: sst_mur_YYYYMMDD_no-dates.exr, 27 May 2024 to 13 November 2026, 901 days, 4096 x 2048.
    Colour bar sst_mur_colorbar.exr (3840 x 2160): -5 to 35 degrees Celsius, linear. Ticks every 5 degrees at
    x 837.5 (0), 1198.5, 1558.5, 1919.5, 2280.5, 2641, 3001.5 (30) (72.13 px per degree), so -5 at x 477 and 35 at x 3362.
    Rows 1049 to 1109 have no tick marks.
"""
import datetime as dt
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import OpenEXR
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "sst_exr"
OUT = HERE / "out" / "sst_exr"
DAYS = OUT / "days"
BASE = "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005101/"
FRAMES = BASE + "frames/4096x2048_2x1_30p/sst_mur_no-dates/sst_mur_{}_no-dates.exr"
BAR = "sst_mur_colorbar.exr"
DAY0, N_DAYS = dt.date(2024, 5, 27), 901
X0, PX_PER_C = 837.5, 72.13  # x of 0 degrees
X_MIN, X_MAX = 477, 3362  # -5 and 35
V_MIN, V_MAX, STEP = -5.0, 35.0, 0.2  # app encoding, as in sst.py
BAR_ROWS = (1049, 1110)
MAX_DIST = 0.05  # colour distance (float RGB) above which a pixel is not on the bar: land, labels
# The bar jumps at zero, from light blue to cream. NASA's rendered map smooths that jump, so near-normal water has
# in-between colours up to about 0.12 from the bar (found round 60: a strict 0.05 dropped near-normal water as "no data").
# Same palette as the anomaly bar, so the same sharp jump, here at 15 degrees (blue to cream). NASA's map smooths it, so
# water near 15 degrees has in-between colours. One degree here is 72 px of bar (the anomaly's 0.2 degrees), so the window is 1 degree.
JUMP_AT, NEAR_ZERO, NEAR_ZERO_DIST = 15.0, 1.0, 0.13


def on_bar(d, v):
    return (d <= MAX_DIST) | ((np.abs(v - JUMP_AT) < NEAR_ZERO) & (d <= NEAR_ZERO_DIST))


def day(n):
    return DAY0 + dt.timedelta(days=n)


def curl(url, dest):
    # NASA's server sometimes resets a connection (curl error 56), so retry
    subprocess.run(["curl", "-s", "-f", "-m", "300", "--retry", "5", "--retry-all-errors", "--retry-delay", "5",
                    "-o", str(dest), url], check=True)


def rgb(path):
    return OpenEXR.File(str(path)).channels()["RGBA"].pixels[..., :3].astype(np.float32)


def bar_tree():
    p = CACHE / BAR
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        curl(BASE + BAR, p)
    band = rgb(p)[BAR_ROWS[0]:BAR_ROWS[1], X_MIN:X_MAX + 1].mean(0)
    values = (np.arange(X_MIN, X_MAX + 1) - X0) / PX_PER_C
    return cKDTree(band), values.astype(np.float32), band


LUT_N = 160  # colour cube steps per channel: about 0.006 in colour, about 0.02 degrees along the bar


def lut(tree, values):
    """Nearest bar value for every colour of a 160 x 160 x 160 cube, so a whole frame decodes by lookup."""
    g = np.linspace(0, 1, LUT_N, dtype=np.float32)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    d, j = tree.query(pts, workers=-1)
    v = values[j].astype(np.float32)
    v[~on_bar(d, v)] = np.nan
    return v.reshape(LUT_N, LUT_N, LUT_N)


def decode(img, tree, values, cube=None):
    if cube is None:  # exact nearest colour (used by validate)
        d, j = tree.query(img.reshape(-1, 3), workers=-1)
        v = values[j].astype(np.float32)
        v[~on_bar(d, v)] = np.nan
        return v.reshape(img.shape[:2])
    i = np.clip(np.rint(img * (LUT_N - 1)), 0, LUT_N - 1).astype(np.int16)
    return cube[i[..., 0], i[..., 1], i[..., 2]]


def frame(n, folder=CACHE):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{day(n):%Y%m%d}.exr"
    if not p.exists():
        part = p.with_suffix(".part")  # a stopped run leaves only .part files, never a broken frame
        curl(FRAMES.format(f"{day(n):%Y%m%d}"), part)
        part.rename(p)
    return p


def mur_box(date, lat, lon, half=0.2):
    """Median MUR SST over the same small box our 5 x 5 pixel window covers (about 0.4 degrees)."""
    url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.json?"
           f"analysed_sst%5B({date.isoformat()}T09:00:00Z)%5D%5B({lat - half}):4:({lat + half})%5D%5B({lon - half}):4:({lon + half})%5D")
    try:
        out = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True).stdout
        v = [r[-1] for r in json.loads(out)["table"]["rows"] if r[-1] is not None]
        return float(np.median(v)) if len(v) >= 10 else None
    except Exception:  # noqa: BLE001
        return None


# Open-sea points (the same places as sst.py, both sets). Nothing is fitted, so every comparison is a test.
POINTS = {
    "Gulf of Bothnia, near Vaasa": (63.0, 20.6), "Gulf of Guinea": (2.0, 3.0), "North Atlantic, mid": (30.0, -40.0),
    "Equatorial Pacific": (0.0, -140.0), "Southern Ocean": (-55.0, 0.0), "Mediterranean, Ionian Sea": (36.0, 18.0),
    "Arabian Sea": (15.0, 65.0), "Coral Sea": (-18.0, 155.0), "North Pacific": (40.0, -150.0),
    "Labrador Sea": (57.0, -52.0), "Norwegian Sea": (68.0, 5.0), "Bering Sea": (58.0, -175.0),
    "Caribbean Sea": (15.0, -75.0), "Gulf of Mexico": (25.0, -90.0), "South Atlantic": (-25.0, -20.0),
    "Benguela, off Namibia": (-23.0, 12.0), "Bay of Bengal": (15.0, 88.0), "South China Sea": (12.0, 113.0),
    "Tasman Sea": (-38.0, 160.0), "East Pacific, off Peru": (-10.0, -85.0), "Gulf of Alaska": (56.0, -145.0),
    "Barents Sea": (73.0, 35.0), "North Sea": (56.0, 3.0), "Black Sea": (43.0, 34.0), "Red Sea": (20.0, 38.5),
    "Indian Ocean, south": (-30.0, 80.0), "Sea of Japan": (40.0, 135.0),
}
TEST_DAYS = [100, 300, 450, 600, 800, 899]


def validate():
    tree, values, _ = bar_tree()
    old = OUT / "validate.json"  # MUR values from an earlier run, so a re-run does not ask NOAA again
    known = {(r["date"], r["place"]): r["mur_c"] for r in json.loads(old.read_text())} if old.exists() else {}
    rows = []
    for n in TEST_DAYS:
        g = decode(rgb(frame(n, CACHE / "test")), tree, values)  # own folder: days() deletes its frames
        for name, (lat, lon) in POINTS.items():
            y, x = int((90 - lat) / 180 * 2048), int((lon + 180) / 360 * 4096)
            win = g[y - 2:y + 3, x - 2:x + 3]
            ours = float(np.nanmedian(win)) if np.isfinite(win).sum() >= 13 else None
            ref = known.get((day(n).isoformat(), name))
            if ref is None:
                ref = mur_box(day(n), lat, lon)
            diff = None if ours is None or ref is None else ours - ref
            rows.append({"date": day(n).isoformat(), "place": name, "decoded_c": None if ours is None else round(ours, 2),
                         "mur_c": None if ref is None else round(ref, 2), "difference_c": None if diff is None else round(diff, 2)})
            print(f"{day(n)} {name:28s} ours {ours if ours is None else round(ours, 2)!s:>6}  MUR {ref if ref is None else round(ref, 2)!s:>6}")
    (OUT / "validate.json").write_text(json.dumps(rows, indent=1))
    d = np.array([r["difference_c"] for r in rows if r["difference_c"] is not None])
    inside = [r for r in rows if r["difference_c"] is not None and abs(r["mur_c"]) < 3.9]
    di = np.array([r["difference_c"] for r in inside])
    print(f"\n{len(d)} comparisons. Mean difference {d.mean():+.2f} C, mean absolute {np.abs(d).mean():.2f} C, "
          f"95th percentile {np.percentile(np.abs(d), 95):.2f} C, largest {np.abs(d).max():.2f} C")
    print(f"Inside the bar (|MUR| < 3.9): {len(di)}, mean absolute {np.abs(di).mean():.2f} C, 95th {np.percentile(np.abs(di), 95):.2f} C")


# 1-degree cells from a full frame: each pixel row and column belongs to one cell.
ROW_CELL = (np.arange(2048) * 180 // 2048)
COL_CELL = (np.arange(4096) * 360 // 4096)
CELL = (ROW_CELL[:, None] * 360 + COL_CELL[None, :]).ravel()


def one_day(n, tree, values, cube):
    dest = DAYS / f"{n:03d}.npy"
    if dest.exists():
        return n
    p = frame(n)
    v = decode(rgb(p), tree, values, cube).ravel()
    ok = np.isfinite(v)
    s = np.bincount(CELL[ok], v[ok], 180 * 360)
    c = np.bincount(CELL[ok], None, 180 * 360)
    total = np.bincount(CELL, None, 180 * 360)
    mean = np.where(c >= total / 2, s / np.maximum(c, 1), np.nan)  # at least half the cell is sea on that day
    np.save(dest, mean.reshape(180, 360).astype(np.float32))
    p.unlink()  # 21 MB per frame: keep only the 1-degree result
    return n


def days():
    DAYS.mkdir(parents=True, exist_ok=True)
    tree, values, _ = bar_tree()
    cube = lut(tree, values)
    # Downloads run at about 2 MB/s in all, so all 901 days take over 2 hours. Round 60: days 2, 4 and 6 of each week
    # (Monday, Wednesday, Friday). `days all` fetches the missing days later.
    every = len(sys.argv) > 2 and sys.argv[2] == "all"
    todo = [n for n in range(N_DAYS) if not (DAYS / f"{n:03d}.npy").exists() and (every or n % 7 in (1, 3, 5))]
    print(f"{len(todo)} days to decode", flush=True)
    with ThreadPoolExecutor(12) as ex:  # downloads are the slow part (about 0.8 MB/s each)
        for i, n in enumerate(ex.map(lambda k: one_day(k, tree, values, cube), todo)):
            if i % 25 == 0:
                print(f"{i + 1}/{len(todo)} done (day {day(n)})", flush=True)


def weekly():
    import gzip
    _, values, band = bar_tree()
    n_weeks = N_DAYS // 7
    q = np.full((n_weeks, 180, 360), 255, np.uint8)
    used = []
    for w in range(n_weeks):
        have = [n for n in range(7 * w, 7 * w + 7) if (DAYS / f"{n:03d}.npy").exists()]
        used.append(len(have))
        stack = np.stack([np.load(DAYS / f"{n:03d}.npy") for n in have])
        ok = np.isfinite(stack).sum(0) * 2 >= len(have)  # sea on at least half of the decoded days
        mean = np.nanmean(np.where(ok[None], stack, np.nan), 0) if ok.any() else None
        q[w][ok] = np.clip(np.round((mean[ok] - V_MIN) / STEP), 0, 200).astype(np.uint8)
    dest = HERE.parent / "mockups" / "data" / "sst"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))
    weeks = [day(7 * i).isoformat() for i in range(n_weeks)]
    colours = [[int(round(255 * c)) for c in np.clip(band[int(round((t - values[0]) / (values[-1] - values[0]) * (len(band) - 1)))], 0, 1) ** (1 / 2.2)]
               for t in np.arange(-5, 35.01, 1)]
    meta = {"record": "sst", "svs": 5101, "unit": "degrees Celsius", "days_read": N_DAYS, "source": "NASA daily EXR frames (pipeline/sst_exr.py)",
            "week_starts": weeks, "days_per_week": 7, "days_decoded_per_week": used,
            "grid": {"rows": 180, "cols": 360, "deg": 1, "row0_lat": 89.5, "col0_lon": -179.5},
            "encoding": "uint8 value = round((T + 5) / 0.2); 255 = no data",
            "colour_bar": {"from_c": -5, "step_c": 1, "source": "NASA SVS 5101 sst_mur_colorbar.exr (linear light, shown with gamma 2.2)",
                           "rgb": colours},
            "error_vs_mur": "see docs/PROCESS.md (round 62)"}
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n_weeks} weeks; sea cells in week 0: {int((q[0] != 255).sum())}; {q.nbytes} bytes")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"validate": validate, "days": days, "weekly": weekly}[sys.argv[1]]()
