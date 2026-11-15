"""Active fires (NASA SVS 5113, VIIRS fire radiative power): count the fire dots NASA draws in each 1-degree square.

    uv run --with openexr --with numpy python pipeline/fire.py validate   # compare with NASA FIRMS VIIRS detections
    uv run --with openexr --with numpy python pipeline/fire.py weekly     # one frame per week -> mockups/data/fire/

Facts this script relies on (round 69):
    Frames: frames/4096x2048_2x1_30p/EIC/fires_frp_VIIRS.NNN.exr, 900 frames, RGBA float16, Robinson projection.
    Each frame shows a moving 5-day window of VIIRS fire radiative power. The file names have no dates; NASA's video
    (fires_frp_VIIRS_2048p30.mp4, same 900 frames) writes the date on each frame: frame 1 = 14 Nov 2024, frame 900 =
    14 Nov 2026, and every 50th frame in between lies on the same straight line, so frame n = 14 Nov 2024 + (n - 1) * 730 / 899 days.
    The last frames (12 to 14 Nov 2026) have no fires drawn yet: NASA's data had not arrived. Weeks without fire dots in the
    whole world are "no data", not "no fires".
    Map (fitted round 69): Robinson, R = 752 px, centre x 2048.6, y 1024.2 (the sea outline agrees with NASA's sea temperature
    picture in 99% of 1-degree cells). Fire dots are red, orange, yellow or white; land is black, the sea is dark blue.
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
import OpenEXR

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "fire"
OUT = HERE / "out" / "fire"
URL = "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005113/frames/4096x2048_2x1_30p/EIC/fires_frp_VIIRS.{:03d}.exr"
FIRMS = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv"
D1, N = dt.date(2024, 11, 14), 900
R, CX, CY = 752.0, 2048.6, 1024.2
RX = np.array([1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322])
RY = np.array([0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000])


def date_of(n):
    return D1 + dt.timedelta(days=(n - 1) * 730 / 899)


def frame_for(d):
    return int(round((d - D1).days * 899 / 730)) + 1


def frame(n):
    p = CACHE / f"fires_frp_VIIRS.{n:03d}.exr"
    if not p.exists():
        part = p.with_suffix(".part")
        subprocess.run(["curl", "-s", "-f", "-m", "300", "--retry", "5", "--retry-all-errors", "-o", str(part), URL.format(n)], check=True)
        part.rename(p)
    return p


def fire_mask(p):
    a = OpenEXR.File(str(p)).channels()["RGBA"].pixels.astype(np.float32)[..., :3]
    m = (a[..., 0] > a[..., 2] + 0.05) & (a[..., 0] > 0.05)
    m[1700:, :700] = False  # the date and the colour bar
    return m


def robinson(lat, lon):
    a = np.abs(lat)
    return CX + 0.8487 * R * np.interp(a, np.arange(0, 91, 5), RX) * np.radians(lon), CY - np.sign(lat) * 1.3523 * R * np.interp(a, np.arange(0, 91, 5), RY)


# each pixel's 1-degree cell: invert by sampling every cell at 8 x 8 points and marking the pixels they fall on
def pixel_cells():
    p = OUT / "pixel_cells.npy"
    if p.exists():
        return np.load(p)
    cellmap = np.full((2048, 4096), -1, np.int32)
    s = (np.arange(8) + 0.5) / 8
    for r in range(180):
        la = 90 - r - s
        for c in range(360):
            lo = -180 + c + s
            LA, LO = np.meshgrid(la, lo, indexing="ij")
            x, y = robinson(LA.ravel(), LO.ravel())
            xi, yi = np.clip(np.round(x).astype(int), 0, 4095), np.clip(np.round(y).astype(int), 0, 2047)
            cellmap[yi, xi] = r * 360 + c
    np.save(p, cellmap)
    return cellmap


def counts(mask, cellmap):
    ok = mask & (cellmap >= 0)
    return np.bincount(cellmap[ok], None, 64800).reshape(180, 360)


def validate():
    """FIRMS keeps the last 7 days public. Compare the newest frame that has fire dots with FIRMS detections in its days."""
    cellmap = pixel_cells()
    n = 896  # 10 Nov 2026, the newest frame with fire dots (frames 898 to 900 are empty)
    d = date_of(n)
    ours = counts(fire_mask(frame(n)), cellmap)
    raw = subprocess.run(["curl", "-s", "-m", "300", FIRMS], capture_output=True).stdout.decode()
    lines = raw.splitlines()
    head = lines[0].split(",")
    li, lo, da = head.index("latitude"), head.index("longitude"), head.index("acq_date")
    days = sorted({l.split(",")[da] for l in lines[1:]})
    keep = {(d - dt.timedelta(days=k)).isoformat() for k in range(5)}
    used = sorted(keep & set(days))
    ref = np.zeros((180, 360))
    for l in lines[1:]:
        f = l.split(",")
        if f[da] in keep:
            r = min(179, int(90 - float(f[li])))
            c = min(359, int(float(f[lo]) + 180))
            ref[r, c] += 1
    land = cellmap.reshape(-1)
    a, b = ours.ravel(), ref.ravel()
    seen = (a > 0) | (b > 0)
    both = (a > 0) & (b > 0)
    rank = lambda x: np.argsort(np.argsort(x))
    rho = np.corrcoef(rank(a[seen]), rank(b[seen]))[0, 1]
    res = {"frame": n, "frame_date": d.isoformat(), "firms_days_in_file": days, "firms_days_compared": used,
           "cells_with_fire_ours": int((a > 0).sum()), "cells_with_fire_firms": int((b > 0).sum()), "cells_both": int(both.sum()),
           "share_of_our_fire_cells_with_firms_fire": round(float(both.sum() / max(1, (a > 0).sum())), 3),
           "share_of_firms_fire_cells_we_show": round(float(both.sum() / max(1, (b > 0).sum())), 3),
           "rank_correlation_of_counts": round(float(rho), 3)}
    (OUT / "validate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def weekly():
    cellmap = pixel_cells()
    n_weeks = 730 // 7
    frames = [min(N, frame_for(D1 + dt.timedelta(days=7 * w + 3))) for w in range(n_weeks)]
    q = np.full((n_weeks, 180, 360), 255, np.uint8)
    land = np.zeros(64800, bool)  # cells that are ever on the map
    land[np.unique(cellmap[cellmap >= 0])] = True

    def one(w):
        p = frame(frames[w])
        m = fire_mask(p)
        return w, counts(m, cellmap), int(m.sum())
    empty = []
    with ThreadPoolExecutor(6) as ex:
        for w, c, total in ex.map(one, range(n_weeks)):
            if total == 0:
                empty.append(w)  # NASA had no fire data for this frame yet: no data, not "no fires"
            else:
                q[w] = np.where(land.reshape(180, 360), np.clip(c, 0, 250), 255).astype(np.uint8)
            print(f"week {w + 1}/{n_weeks} frame {frames[w]} {date_of(frames[w])} fire dots {total}", flush=True)
    dest = HERE.parent / "mockups" / "data" / "fire"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))
    meta = {"record": "fire", "svs": 5113, "unit": "fire dots in NASA's picture of this 1-degree square, over 5 days",
            "week_starts": [(D1 + dt.timedelta(days=7 * w)).isoformat() for w in range(n_weeks)],
            "frame_used": frames, "frame_dates": [date_of(f).isoformat() for f in frames], "empty_weeks": empty,
            "days_per_week": 7, "days_decoded_per_week": 1,
            "grid": {"rows": 180, "cols": 360, "deg": 1, "row0_lat": 89.5, "col0_lon": -179.5},
            "encoding": "uint8 value = number of fire pixels in the cell (a cell has about 130 pixels near the equator); 255 = no data",
            "colour_bar": {"from_c": 0, "step_c": 10, "source": "NASA's fire colours from dark red to white (fire radiative power 0 to 8 MW)",
                           "rgb": [[40, 30, 30], [140, 40, 20], [190, 80, 25], [215, 130, 40], [230, 180, 60], [240, 215, 90], [248, 235, 150], [252, 245, 200], [255, 252, 235], [255, 255, 255], [255, 255, 255]]},
            "error_vs_firms": "see docs/PROCESS.md (round 69)"}
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n_weeks} weeks, {len(empty)} without fire data")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    {"validate": validate, "weekly": weekly}[sys.argv[1]]()
