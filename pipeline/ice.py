"""Polar sea ice (NASA SVS 5099): read percent ice cover from NASA's rendered polar globes, and check it against the sea ice
fraction in JPL MUR SST v4.1 (NOAA ERDDAP), which is drawn from the same kind of satellite record.

    uv run --with openexr --with numpy --with scipy python pipeline/ice.py validate
    uv run --with openexr --with numpy --with scipy python pipeline/ice.py weekly

Facts this script relies on (round 69):
    Frames: frames/4320x2160_2x1_30p/sxs/no_dates/polar_sea_ice_sxs_no_date.YYYYDDD.exr (year and day of the year), 730 days,
    2024-315 (10 Nov 2024) to 2026-317 (13 Nov 2026), float16 RGB. Two rendered globes side by side: the Arctic on the left
    (centre x 1079.5), the Antarctic on the right (centre x 3239.5), both y 1080. There is no colour bar: sea ice is drawn white
    over a blue sea, and land is a satellite picture of the ground (snow is white too).
    Camera (fitted round 69, see docs/PROCESS.md): a perspective view from k = 1.749 globe radii, the visible edge at 1008 px from
    the centre, the pole at the centre. Arctic: longitude 89.95 points down; Antarctic: longitude -0.18 points up.
    A point at colatitude c (angle from the pole) is at radius r = f sin(c) / (k - cos(c)), f = RL (k - 1/k) / sqrt(1 - 1/k^2).
"""
import datetime as dt
import gzip
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import OpenEXR

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "ice"
OUT = HERE / "out" / "ice"
URL = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005099/frames/4320x2160_2x1_30p/sxs/no_dates/polar_sea_ice_sxs_no_date.{}.exr"
DAY0, N_DAYS = dt.date(2024, 11, 10), 730
K, RL = 1.749, 1008.3
GEO = {"N": dict(cx=1079.5 - 0.74, cy=1080 + 2.76, l0=89.95, s=1), "S": dict(cx=3239.5 - 0.67, cy=1080 + 1.18, l0=-0.18, s=-1)}
MAX_COLAT = 40.0  # degrees from the pole: further out, the globe's bright edge (atmosphere) changes the colours
F = RL * (K - 1 / K) / np.sqrt(1 - 1 / K ** 2)


def day(n):
    return DAY0 + dt.timedelta(days=n)


def fname(d):
    return f"{d.year}{d.timetuple().tm_yday:03d}"


def frame(d):
    p = CACHE / f"{fname(d)}.exr"
    if not p.exists():
        part = p.with_suffix(".part")
        subprocess.run(["curl", "-s", "-f", "-m", "300", "--retry", "5", "--retry-all-errors", "-o", str(part), URL.format(fname(d))], check=True)
        part.rename(p)
    return p


def rgb(p):
    return OpenEXR.File(str(p)).channels()["RGB"].pixels.astype(np.float32)


def xy(lat, lon):
    """Pixel position of points (degrees); also whether the point is inside the usable part of a globe."""
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    north = lat >= 0
    c = np.radians(90 - np.abs(lat))
    r = F * np.sin(c) / (K - np.cos(c))
    x = np.empty_like(lat)
    y = np.empty_like(lat)
    for h, m in (("N", north), ("S", ~north)):
        g = GEO[h]
        lo = np.radians(lon[m] - g["l0"])
        x[m] = g["cx"] + r[m] * np.sin(lo)
        y[m] = g["cy"] + g["s"] * r[m] * np.cos(lo)
    ok = np.degrees(c) <= MAX_COLAT
    return x, y, ok


# Reference colours (linear light), measured round 69 from the frames: open water and full sea ice.
WATER = np.array([0.075, 0.180, 0.320], np.float32)
ICE = np.array([0.900, 0.925, 0.985], np.float32)


def ice_fraction(px):
    """Where the colour sits on the line from open water to ice, 0 to 1."""
    d = ICE - WATER
    return np.clip(((px - WATER) @ d) / (d @ d), 0, 1)


def sample(img, lat, lon):
    x, y, ok = xy(lat, lon)
    xi, yi = np.clip(np.round(x).astype(int), 0, 4319), np.clip(np.round(y).astype(int), 0, 2159)
    f = ice_fraction(img[yi, xi])
    f[~ok] = np.nan
    return f


def sea_cells():
    """1-degree cells that are sea: NASA's sea temperature picture has a value there in some week (it colours the sea under
    the ice too). Land, ice sheets and lakes outside that picture are not sea."""
    q = np.frombuffer(gzip.decompress((HERE.parent / "mockups/data/sst/weekly.bin.gz").read_bytes()), np.uint8).reshape(-1, 180, 360)
    return (q != 255).any(0)


SUB = (np.arange(4) + 0.5) / 4 - 0.5  # 4 x 4 samples inside each 1-degree cell


def grid(img, sea):
    """Percent ice for every sea cell within MAX_COLAT of a pole, NaN elsewhere."""
    out = np.full((180, 360), np.nan, np.float32)
    rows = [r for r in range(180) if abs(89.5 - r) >= 90 - MAX_COLAT]
    for r in rows:
        lat = 89.5 - r
        cols = np.where(sea[r])[0]
        if not len(cols):
            continue
        la = (lat + SUB[:, None, None]).repeat(4, 1).repeat(len(cols), 2)
        lo = (-179.5 + cols[None, None, :] + SUB[None, :, None]).repeat(4, 0)
        f = sample(img, la.ravel(), lo.ravel()).reshape(4, 4, len(cols))
        out[r, cols] = np.nanmean(f.reshape(16, -1), 0) * 100
    return out


POINTS = {  # (lat, lon): sea ice areas in both hemispheres
    "Bothnian Bay, near Oulu": (65.0, 23.5), "Kvarken, near Vaasa": (63.3, 20.8), "Gulf of Finland": (60.0, 26.0),
    "Hudson Bay": (60.0, -85.0), "Beaufort Sea": (73.0, -140.0), "Barents Sea": (76.0, 40.0), "Kara Sea": (75.0, 70.0),
    "Laptev Sea": (76.0, 125.0), "Chukchi Sea": (70.0, -170.0), "Baffin Bay": (72.0, -65.0), "Greenland Sea": (75.0, -10.0),
    "Labrador Sea": (57.0, -55.0), "Sea of Okhotsk": (56.0, 145.0), "Central Arctic": (86.0, 0.0), "Bering Sea": (60.0, -175.0),
    "Weddell Sea": (-70.0, -40.0), "Ross Sea": (-75.0, -175.0), "Southern Ocean, south of Africa": (-60.0, 20.0),
    "Bellingshausen Sea": (-70.0, -85.0), "Amundsen Sea": (-72.0, -115.0), "East Antarctica coast": (-65.0, 110.0),
    "Southern Ocean, Indian sector": (-60.0, 80.0),
}
TEST_DAYS = [dt.date(2025, 1, 15), dt.date(2025, 3, 1), dt.date(2025, 5, 15), dt.date(2025, 8, 1), dt.date(2025, 9, 15), dt.date(2025, 12, 1),
             dt.date(2026, 3, 1), dt.date(2026, 9, 15)]


def mur_ice(d, lat, lon, half=0.2):
    url = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.json?"
           f"sea_ice_fraction%5B({d.isoformat()}T09:00:00Z)%5D%5B({lat - half}):4:({lat + half})%5D%5B({lon - half}):4:({lon + half})%5D")
    try:
        out = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True).stdout
        rows = json.loads(out)["table"]["rows"]
        v = [r[-1] for r in rows if r[-1] is not None]
        return float(np.mean(v)) * 100 if len(v) >= 5 else (0.0 if rows else None)  # MUR leaves open water empty: no ice
    except Exception:  # noqa: BLE001
        return None


def validate():
    rows = []
    for d in TEST_DAYS:
        img = rgb(frame(d))
        for name, (la, lo) in POINTS.items():
            sl = la + np.repeat(np.linspace(-0.2, 0.2, 5), 5)
            sn = lo + np.tile(np.linspace(-0.2, 0.2, 5), 5)
            v = sample(img, sl, sn)
            ours = float(np.nanmean(v) * 100) if np.isfinite(v).any() else None
            ref = mur_ice(d, la, lo)
            rows.append({"date": d.isoformat(), "place": name, "decoded_pct": None if ours is None else round(ours, 1),
                         "mur_pct": None if ref is None else round(ref, 1),
                         "difference_pct": None if ours is None or ref is None else round(ours - ref, 1)})
            print(f"{d} {name:32s} ours {rows[-1]['decoded_pct']!s:>6}  MUR {rows[-1]['mur_pct']!s:>6}", flush=True)
    (OUT / "validate.json").write_text(json.dumps(rows, indent=1))
    e = np.array([r["difference_pct"] for r in rows if r["difference_pct"] is not None])
    print(f"\n{len(e)} pairs. mean {e.mean():+.1f} points, mean absolute {np.abs(e).mean():.1f} points, 95% within {np.percentile(np.abs(e), 95):.1f}, largest {np.abs(e).max():.1f}")
    agree = [(r["decoded_pct"] >= 15) == (r["mur_pct"] >= 15) for r in rows if r["difference_pct"] is not None]
    print(f"ice or no ice (15% cover, the usual edge of the ice): same answer in {np.mean(agree):.0%} of pairs")


def weekly():
    """One frame per week (the middle day, Wednesday), because each frame is 56 MB."""
    sea = sea_cells()
    n_weeks = N_DAYS // 7
    days = [day(7 * w + 3) for w in range(n_weeks)]
    q = np.full((n_weeks, 180, 360), 255, np.uint8)
    used = [d.isoformat() for d in days]

    def one(w):
        raw = OUT / "weeks" / f"{w:03d}.npy"
        if raw.exists():
            return w, np.load(raw)
        p = None
        for k in (0, -1, 1, -2, 2, -3, 3):  # NASA's server is missing a few days (for example 14 Oct 2026)
            try:
                p = frame(days[w] + dt.timedelta(days=k))
                used[w] = (days[w] + dt.timedelta(days=k)).isoformat()
                break
            except subprocess.CalledProcessError:
                continue
        if p is None:
            return w, np.full((180, 360), np.nan, np.float32)
        g = grid(rgb(p), sea)
        raw.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw, g)  # kept, so a correction can be applied later without downloading again
        p.unlink()
        return w, g
    with ThreadPoolExecutor(6) as ex:
        for w, g in ex.map(one, range(n_weeks)):
            q[w] = np.where(np.isfinite(g), np.clip(np.round(g), 0, 100), 255).astype(np.uint8)
            print(f"week {w + 1}/{n_weeks} {days[w]}", flush=True)
    dest = HERE.parent / "mockups" / "data" / "ice"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))
    meta = {"record": "ice", "svs": 5099, "unit": "percent ice cover", "week_starts": [day(7 * w).isoformat() for w in range(n_weeks)],
            "frame_used": used, "days_per_week": 7, "days_decoded_per_week": 1,
            "grid": {"rows": 180, "cols": 360, "deg": 1, "row0_lat": 89.5, "col0_lon": -179.5},
            "encoding": "uint8 value = percent ice cover (0 to 100); 255 = no data (land, or more than 40 degrees from a pole)",
            "colour_bar": {"from_c": 0, "step_c": 10, "source": "no colour bar: NASA draws ice white over a blue sea; these are the two reference colours mixed",
                           "rgb": [[int(round(255 * c)) for c in np.clip(WATER + (ICE - WATER) * t / 100, 0, 1) ** (1 / 2.2)] for t in range(0, 101, 10)]},
            "error_vs_mur": "see docs/PROCESS.md (round 69)"}
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {n_weeks} weeks")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    OUT.mkdir(parents=True, exist_ok=True)
    {"validate": validate, "weekly": weekly}[sys.argv[1]]()
