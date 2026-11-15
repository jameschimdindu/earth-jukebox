"""Global biosphere, ocean part (NASA SVS 5075): read ocean chlorophyll (plankton) from NASA's daily frames.

    uv run --with numpy --with scipy --with pillow python pipeline/bio.py fit        # learn colour -> chlorophyll, test on other days
    uv run --with numpy --with scipy --with pillow python pipeline/bio.py weekly     # one frame per week -> mockups/data/bio/

Facts this script relies on (round 69):
    Frames: frames/4096x2048_2x1_30p/Plate_Carree/newbio2k_YYYYMMDD.tif, RGBA, 3 Sep 2022 to 17 Feb 2025 (NASA stopped there).
    Land in these frames has no plant values: it is one flat colour (pink in July, grey in January) everywhere, so only the
    ocean can be read. The ocean colours do not follow NASA's chlorophyll colour bar (newbio_chloro_sci_colorbar, 0 to
    0.7 mg/m3): the frame's darkest purple is (100, 8, 126) where the bar has (112, 67, 127). So we learn the colour -> value
    mapping from an independent satellite record (NOAA VIIRS gap-filled chlorophyll, DINEOF, 9 km, daily) on training days,
    and test it on other days that were never used to learn it.
"""
import datetime as dt
import gzip
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "bio"
OUT = HERE / "out" / "bio"
URL = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005075/frames/4096x2048_2x1_30p/Plate_Carree/newbio2k_{}.tif"
REF = "https://coastwatch.noaa.gov/erddap/griddap/noaacwNPPN20VIIRSSCIDINEOFDaily.csv"
DAY0, DAY_LAST = dt.date(2022, 9, 3), dt.date(2025, 2, 17)
TRAIN = [dt.date(2023, 1, 18), dt.date(2023, 7, 19), dt.date(2024, 4, 17), dt.date(2024, 10, 16)]
TEST = [dt.date(2023, 4, 19), dt.date(2023, 10, 18), dt.date(2024, 1, 17), dt.date(2024, 7, 17)]
K = 25


def frame(d):
    p = CACHE / f"{d:%Y%m%d}.tif"
    if not p.exists():
        part = p.with_suffix(".part")
        subprocess.run(["curl", "-s", "-f", "-m", "300", "--retry", "5", "--retry-all-errors", "-o", str(part), URL.format(f"{d:%Y%m%d}")], check=True)
        part.rename(p)
    return p


def rgb(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32)


def is_sea(px):
    """Land is a flat pink or grey, and ice and clouds are white: sea colours are purple, blue, green or yellow."""
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    grey = (np.max(px, -1) - np.min(px, -1)) < 18
    pink = (r > 190) & (g > 150) & (b > 130) & (r > b)
    return ~grey & ~pink


def ref_grid(d):
    """NOAA chlorophyll on a 1-degree grid (every 12th 9-km cell), only if the server has exactly this day."""
    url = f"{REF}?chlor_a%5B({d.isoformat()}T12:00:00Z)%5D%5B0%5D%5B(-79.96):12:(79.96)%5D%5B(-179.96):12:(179.96)%5D"
    out = subprocess.run(["curl", "-sL", "-m", "300", url], capture_output=True, text=True).stdout
    rows = [l.split(",") for l in out.splitlines()[2:]]
    if not rows or not rows[0][0].startswith(d.isoformat()):
        return None
    a = np.array([[float(r[2]), float(r[3]), float(r[4]) if r[4] not in ("NaN", "") else np.nan] for r in rows])
    return a


def pairs(d):
    img = rgb(frame(d))
    ref = ref_grid(d)
    if ref is None:
        print(f"{d}: NOAA has no data for exactly this day")
        return None
    lat, lon, chl = ref.T
    y = np.clip(((90 - lat) / 180 * 2048).astype(int), 2, 2045)
    x = np.clip(((lon + 180) / 360 * 4096).astype(int), 2, 4093)
    win = np.stack([img[y + dy, x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1)  # 3 x 3 pixels
    col = np.median(win, 1)
    ok = np.isfinite(chl) & (chl > 0) & is_sea(col) & is_sea(win).all(1)
    return col[ok], np.log10(chl[ok]), d


def fit():
    with ThreadPoolExecutor(4) as ex:
        got = [g for g in ex.map(pairs, TRAIN + TEST) if g is not None]
    tr = [g for g in got if g[2] in TRAIN]
    te = [g for g in got if g[2] in TEST]
    X = np.concatenate([g[0] for g in tr])
    Y = np.concatenate([g[1] for g in tr])
    np.savez(OUT / "model.npz", X=X, Y=Y)
    tree = cKDTree(X)
    lines = []
    for name, sets in (("training days", tr), ("test days (never used to learn)", te)):
        Xs = np.concatenate([g[0] for g in sets])
        Ys = np.concatenate([g[1] for g in sets])
        _, j = tree.query(Xs, K + (1 if sets is tr else 0))
        j = j[:, 1:] if sets is tr else j  # on the training days, leave the point itself out
        P = np.median(Y[j], 1)
        e = P - Ys
        f = 10 ** np.abs(e)
        line = (f"{name}: {len(e)} cells on {len(sets)} days. median factor {np.median(f):.2f}, 90% within a factor of "
                f"{np.percentile(f, 90):.2f}, bias {np.median(e):+.3f} in log10 (x{10 ** np.median(e):.2f})")
        print(line)
        lines.append(line)
    (OUT / "fit.txt").write_text("\n".join(lines) + f"\ntraining {[str(d) for d in TRAIN]}\ntest {[str(d) for d in TEST]}\n")


def weekly():
    m = np.load(OUT / "model.npz")
    tree, Y = cKDTree(m["X"]), m["Y"]
    n_weeks = (DAY_LAST - DAY0).days // 7 + 1
    days = [min(DAY_LAST, DAY0 + dt.timedelta(days=7 * w + 3)) for w in range(n_weeks)]
    rows = (np.arange(2048) * 180 // 2048)
    cols = (np.arange(4096) * 360 // 4096)
    cell = (rows[:, None] * 360 + cols[None, :]).ravel()
    total = np.bincount(cell, None, 64800)
    q = np.full((n_weeks, 180, 360), 255, np.uint8)

    def one(w):
        raw = OUT / "weeks" / f"{w:03d}.npy"
        if raw.exists():
            return w, np.load(raw)
        p = frame(days[w])
        img = rgb(p)[::2, ::2]  # 2048 x 1024 is enough for 1-degree cells
        sea = is_sea(img)
        v = np.full(img.shape[:2], np.nan, np.float32)
        cs, inv = np.unique(img[sea].astype(np.int32) @ np.array([65536, 256, 1]), return_inverse=True)  # each colour once
        uc = np.stack([cs >> 16, (cs >> 8) & 255, cs & 255], 1).astype(np.float32)
        _, j = tree.query(uc, K)
        v[sea] = np.median(Y[j], 1)[inv]
        c2 = cell.reshape(2048, 4096)[::2, ::2].ravel()
        t2 = np.bincount(c2, None, 64800)
        vv = v.ravel()
        ok = np.isfinite(vv)
        s, c = np.bincount(c2[ok], vv[ok], 64800), np.bincount(c2[ok], None, 64800)
        g = np.where(c >= t2 / 2, s / np.maximum(c, 1), np.nan).reshape(180, 360)  # mean of log10 chlorophyll
        raw.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw, g)
        p.unlink()
        return w, g
    with ThreadPoolExecutor(6) as ex:
        for w, g in ex.map(one, range(n_weeks)):
            # value = round((log10(chl) + 2) / 0.025): 0.01 to 100 mg/m3 in 160 steps
            q[w] = np.where(np.isfinite(g), np.clip(np.round((g + 2) / 0.025), 0, 160), 255).astype(np.uint8)
            print(f"week {w + 1}/{n_weeks} {days[w]}", flush=True)
    dest = HERE.parent / "mockups" / "data" / "bio"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))
    stops = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    tree_c = cKDTree(Y[:, None])
    colours = []
    for s_ in stops:  # the typical frame colour for each value, from the learned pairs
        _, j = tree_c.query([[np.log10(s_)]], 50)
        colours.append([int(round(c)) for c in np.median(m["X"][j[0]], 0)])
    meta = {"record": "bio", "svs": 5075, "unit": "milligrams of chlorophyll per cubic metre", "part": "ocean only (NASA's frames have no plant values on land)",
            "week_starts": [(DAY0 + dt.timedelta(days=7 * w)).isoformat() for w in range(n_weeks)], "frame_used": [d.isoformat() for d in days],
            "days_per_week": 7, "days_decoded_per_week": 1,
            "grid": {"rows": 180, "cols": 360, "deg": 1, "row0_lat": 89.5, "col0_lon": -179.5},
            "encoding": "uint8 value = round((log10(chlorophyll) + 2) / 0.025); 255 = no data (land, ice, clouds)",
            "colour_stops": {"mg_m3": stops, "rgb": colours, "source": "typical frame colour for each value, from the learned pairs"},
            "error_vs_noaa": "see docs/PROCESS.md and pipeline/out/bio/fit.txt (round 69)"}
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n_weeks} weeks")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    {"fit": fit, "weekly": weekly}[sys.argv[1]]()
