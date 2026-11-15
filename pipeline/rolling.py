"""Daily updates for the records whose NASA frames carry dates in their names (round 74).

NASA's Scientific Visualization Studio keeps a moving window of daily frames (about 2.5 years) and adds a day every day.
This module keeps one small file per week (Monday to Sunday) in pipeline/state/<record>/weeks/YYYY-MM-DD.npz: the running
mean of the decoded days (1-degree grid, after the centre-band correction), how many days each cell had sea, and which
days are counted. It downloads only the days that are new, drops weeks that left NASA's window, and rebuilds the app data.

    uv run --with openexr --with numpy --with scipy python pipeline/rolling.py sst          # sea surface temperature
    uv run --with openexr --with numpy --with scipy python pipeline/rolling.py sst_anomaly  # anomaly
    uv run --with openexr --with numpy --with scipy python pipeline/rolling.py migrate      # once: old day files -> state
"""
import datetime as dt
import gzip
import io
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATE = HERE / "state"
sys.path.insert(0, str(HERE))

RECORDS = {
    "sst": {"module": "sst_exr", "list": "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005101/frames/4096x2048_2x1_30p/sst_mur_no-dates/",
            "pattern": r"sst_mur_(\d{8})_no-dates\.exr"},
    "sst_anomaly": {"module": "sst_anomaly", "list": "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005176/frames/4096x2048_2x1_30p/sst_mur_anomaly_nodates/",
                    "pattern": r"sst_mur_anomaly_nodates_(\d{8})\.exr"},
}


def listing(rec):
    cfg = RECORDS[rec]
    html = subprocess.run(["curl", "-s", "-f", "-m", "120", "--retry", "5", cfg["list"]], capture_output=True, text=True, check=True).stdout
    days = sorted({dt.datetime.strptime(m, "%Y%m%d").date() for m in re.findall(cfg["pattern"], html)})
    if len(days) < 300:
        raise RuntimeError(f"{rec}: NASA's frame list has only {len(days)} days; not updating")
    return days


def save_day(path, grid):
    buf = io.BytesIO()
    np.save(buf, grid.astype(np.float16))
    path.write_bytes(gzip.compress(buf.getvalue(), 6))


def load_day(path):
    return np.load(io.BytesIO(gzip.decompress(path.read_bytes()))).astype(np.float32)


def migrate():
    """The days decoded in rounds 60 to 63 were named by their number from the first day; name them by date."""
    import sst_anomaly
    import sst_exr
    for rec, mod in (("sst", sst_exr), ("sst_anomaly", sst_anomaly)):
        dest = STATE / rec
        dest.mkdir(parents=True, exist_ok=True)
        k = 0
        for p in sorted(mod.DAYS.glob("*.npy")):
            d = mod.DAY0 + dt.timedelta(days=int(p.stem))
            q = dest / f"{d.isoformat()}.npy.gz"
            if not q.exists():
                save_day(q, np.load(p))
                k += 1
        print(f"{rec}: {k} days moved to {dest}")


ENC = {"sst": (-5.0, 0.2), "sst_anomaly": (-4.0, 0.04)}  # app encoding: value = round((x - lo) / step), 0..200


def week_of(d):
    return d - dt.timedelta(days=d.weekday())  # weeks run Monday to Sunday


def load_week(p):
    z = np.load(p)
    return z["mean"].astype(np.float32), z["count"].astype(np.int32), [str(x) for x in z["days"]]


def save_week(p, mean, count, days):
    np.savez_compressed(p, mean=mean.astype(np.float16), count=count.astype(np.uint8), days=np.array(sorted(days)))


def add_day(wdir, d, grid):
    """Add one decoded day (after the centre-band correction) to its week's running mean."""
    p = wdir / f"{week_of(d).isoformat()}.npz"
    if p.exists():
        mean, count, days = load_week(p)
        if d.isoformat() in days:
            return
    else:
        mean, count, days = np.zeros((180, 360), np.float32), np.zeros((180, 360), np.int32), []
    ok = np.isfinite(grid)
    total = mean * count
    total[ok] += grid[ok]
    count = count + ok
    mean = np.where(count > 0, total / np.maximum(count, 1), 0)
    save_week(p, mean, count, days + [d.isoformat()])


def to_weeks(rec):
    """Once: turn the day files into week files, then remove the day files (110 MB -> about 9 MB per record)."""
    mod = __import__(RECORDS[rec]["module"])
    dest = STATE / rec
    wdir = dest / "weeks"
    wdir.mkdir(parents=True, exist_ok=True)
    files = sorted(dest.glob("*.npy.gz"))
    for p in files:
        add_day(wdir, dt.date.fromisoformat(p.name[:10]), mod.centre_fix(load_day(p)))
    for p in files:
        p.unlink()
    print(f"{rec}: {len(files)} days -> {len(list(wdir.glob('*.npz')))} weeks")


def update(rec):
    import json
    mod = __import__(RECORDS[rec]["module"])
    wdir = STATE / rec / "weeks"
    wdir.mkdir(parents=True, exist_ok=True)
    days = listing(rec)
    counted = set()
    for p in wdir.glob("*.npz"):
        counted |= set(load_week(p)[2])
    missing = [d for d in days if d.isoformat() not in counted]
    print(f"{rec}: NASA has {days[0]} to {days[-1]} ({len(days)} days); {len(missing)} new", flush=True)
    work = Path(tempfile.mkdtemp(prefix=f"{rec}-"))
    try:
        if missing:
            mod.DAY0, mod.N_DAYS, mod.DAYS = days[0], (days[-1] - days[0]).days + 1, work
            tree, values, _ = mod.bar_tree()
            cube = mod.lut(tree, values)

            def one(d):
                n = (d - days[0]).days
                mod.one_day(n, tree, values, cube)
                return d, np.load(work / f"{n:03d}.npy")
            with ThreadPoolExecutor(6) as ex:
                for d, g in ex.map(one, missing):
                    add_day(wdir, d, mod.centre_fix(g))
                    print(f"   decoded {d}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    # weeks that have fully left NASA's window are dropped; a week needs at least 3 decoded days to be played
    first = week_of(days[0])
    weeks = []
    for p in sorted(wdir.glob("*.npz")):
        w = dt.date.fromisoformat(p.stem)
        if w + dt.timedelta(days=6) < days[0]:
            p.unlink()
            continue
        mean, count, ds = load_week(p)
        if len(ds) >= 3 and w >= first:
            weeks.append((w, mean, count, ds))
    lo, step = ENC[rec]
    q = np.full((len(weeks), 180, 360), 255, np.uint8)
    for i, (w, mean, count, ds) in enumerate(weeks):
        ok = count * 2 >= len(ds)  # sea on at least half of the decoded days, as before
        q[i][ok] = np.clip(np.round((mean[ok] - lo) / step), 0, 200).astype(np.uint8)
    dest = HERE.parent / "mockups" / "data" / ("sst" if rec == "sst" else "sst_anomaly")
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))
    meta = json.loads((dest / "meta.json").read_text())
    meta.update({"week_starts": [w.isoformat() for w, *_ in weeks], "days_decoded_per_week": [len(ds) for *_, ds in weeks],
                 "days_read": sum(len(ds) for *_, ds in weeks), "weeks_start_on": "Monday",
                 "nasa_window": [days[0].isoformat(), days[-1].isoformat()], "updated": dt.date.today().isoformat(),
                 "update": "pipeline/rolling.py, daily (.github/workflows/update-data.yml)"})
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"{rec}: wrote {len(weeks)} weeks, {weeks[0][0]} to {weeks[-1][0]}")


# ---------------- sea ice: one frame per week, frames named by year and day of the year ----------------
ICE_LIST = "https://svs.gsfc.nasa.gov/vis/a000000/a005000/a005099/frames/4320x2160_2x1_30p/sxs/no_dates/"


def update_ice():
    import json
    import ice
    html = subprocess.run(["curl", "-s", "-f", "-m", "120", "--retry", "5", ICE_LIST], capture_output=True, text=True, check=True).stdout
    days = sorted({dt.datetime.strptime(m, "%Y%j").date() for m in re.findall(r"polar_sea_ice_sxs_no_date\.(\d{7})\.exr", html)})
    if len(days) < 300:
        raise RuntimeError(f"ice: NASA's frame list has only {len(days)} days; not updating")
    wdir = STATE / "ice" / "weeks"
    wdir.mkdir(parents=True, exist_ok=True)
    have = {dt.date.fromisoformat(p.name[:10]): p for p in wdir.glob("*.npy.gz")}
    avail = set(days)
    weeks = []
    w = week_of(days[0])
    if w < days[0]:
        w += dt.timedelta(days=7)  # only weeks that are fully inside NASA's window
    sea = None
    while w + dt.timedelta(days=6) <= days[-1]:
        in_week = [w + dt.timedelta(days=k) for k in range(7)]
        done = [d for d in have if d in in_week]
        if done:
            used = done[0]
        else:
            wed = w + dt.timedelta(days=2)
            cands = sorted((d for d in in_week if d in avail), key=lambda d: abs((d - wed).days))
            if not cands:
                w += dt.timedelta(days=7)
                continue
            used = cands[0]
            if sea is None:
                sea = ice.sea_cells()
            p = ice.frame(used)
            g = ice.grid(ice.rgb(p), sea)
            p.unlink()
            save_day(wdir / f"{used.isoformat()}.npy.gz", g)
            have[used] = wdir / f"{used.isoformat()}.npy.gz"
            print(f"   ice: decoded {used}", flush=True)
        weeks.append((w, used))
        w += dt.timedelta(days=7)
    for d, p in list(have.items()):
        if d < days[0] - dt.timedelta(days=7):
            p.unlink()
    q = np.full((len(weeks), 180, 360), 255, np.uint8)
    for i, (w, used) in enumerate(weeks):
        g = load_day(have[used])
        q[i] = np.where(np.isfinite(g), np.clip(np.round(g), 0, 100), 255).astype(np.uint8)
    dest = HERE.parent / "mockups" / "data" / "ice"
    (dest / "weekly.bin.gz").write_bytes(gzip.compress(q.tobytes(), 9))
    meta = json.loads((dest / "meta.json").read_text())
    meta.update({"week_starts": [w.isoformat() for w, _ in weeks], "frame_used": [u.isoformat() for _, u in weeks], "weeks_start_on": "Monday",
                 "nasa_window": [days[0].isoformat(), days[-1].isoformat()], "updated": dt.date.today().isoformat(),
                 "update": "pipeline/rolling.py, daily (.github/workflows/update-data.yml)"})
    (dest / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"ice: wrote {len(weeks)} weeks, {weeks[0][0]} to {weeks[-1][0]}")


def migrate_ice():
    """The 104 weeks decoded in round 69 (pipeline/out/ice/weeks/NNN.npy), named by the frame day they used."""
    import json
    wdir = STATE / "ice" / "weeks"
    wdir.mkdir(parents=True, exist_ok=True)
    used = json.loads((HERE.parent / "mockups/data/ice/meta.json").read_text())["frame_used"]
    k = 0
    for i, d in enumerate(used):
        p = HERE / "out" / "ice" / "weeks" / f"{i:03d}.npy"
        if p.exists():
            save_day(wdir / f"{d}.npy.gz", np.load(p))
            k += 1
    print(f"ice: {k} weeks moved")


# ---------------- fires: 900 frames without dates, a moving 2-year window ----------------
def update_fire():
    import fire
    html = subprocess.run(["curl", "-s", "-f", "-m", "120", "--retry", "5", fire.URL.rsplit("/", 1)[0] + "/"], capture_output=True, text=True, check=True).stdout
    nums = sorted({int(m) for m in re.findall(r"fires_frp_VIIRS\.(\d{3})\.exr", html)})
    if len(nums) < 300:
        raise RuntimeError(f"fire: NASA's frame list has only {len(nums)} frames; not updating")
    n = nums[-1]
    head = subprocess.run(["curl", "-sI", "-m", "60", fire.URL.format(n)], capture_output=True, text=True).stdout
    lm = re.search(r"(?im)^last-modified:\s*(.+)$", head).group(1).strip()
    written = dt.datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S GMT").date()
    # checked round 69 on NASA's video: the last frame is the day before NASA wrote the frames, the first is 2 years earlier
    last = written - dt.timedelta(days=1)
    fire.N, fire.D1 = n, last - dt.timedelta(days=730)
    for p in fire.CACHE.glob("fires_frp_VIIRS.*.exr"):
        p.unlink()  # the frames move every day: never reuse an old download
    fire.weekly()




# ---------------- air: 372 hourly frames without dates, rewritten every day ----------------
AIR_LIST = "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005147/frames/2048x1024_2x1_12p/MC01_stage3_GMAO_near_surface_temperature_2048x1024_en-hwtool/"
# GEOS-FP's hourly index for a time (found in round 66 by matching frames to GEOS-FP; index 78507 is 15 Nov 2026 03:00 UTC)
GEOS_ANCHOR_I, GEOS_ANCHOR_T = 78507, dt.datetime(2026, 11, 15, 3)


def geos_index(t):
    return GEOS_ANCHOR_I + int(round((t - GEOS_ANCHOR_T).total_seconds() / 3600))


def update_air():
    import air
    html = subprocess.run(["curl", "-s", "-f", "-m", "120", "--retry", "5", AIR_LIST], capture_output=True, text=True, check=True).stdout
    nums = sorted({int(m) for m in re.findall(r"(\d{8})\.png", html)})
    if len(nums) < 200:
        raise RuntimeError(f"air: NASA's frame list has only {len(nums)} frames; not updating")
    n = nums[-1]
    head = subprocess.run(["curl", "-sI", "-m", "60", AIR_LIST + "00000001.png"], capture_output=True, text=True).stdout
    lm = re.search(r"(?im)^last-modified:\s*(.+)$", head).group(1).strip()
    written = dt.datetime.strptime(lm, "%a, %d %b %Y %H:%M:%S GMT")
    # round 65: written 14 Nov 07:30 UTC, frame 1 read on the frame as 30 Oct 09:00 UTC, 358.5 hours earlier
    guess = (written - dt.timedelta(hours=358.5)).replace(minute=0, second=0, microsecond=0)
    air.N = n
    air.FILES_WRITTEN = written
    for p in air.FRAMES.glob("*.png"):
        p.unlink()
    air.FRAMES.mkdir(parents=True, exist_ok=True)

    def get(k):
        dest = air.FRAMES / f"{k:08d}.png"
        subprocess.run(["curl", "-s", "-f", "-m", "120", "--retry", "5", "--retry-all-errors", "-o", str(dest), AIR_LIST + f"{k:08d}.png"], check=True)
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, range(1, n + 1)))
    print(f"air: {n} frames, written {written} UTC; first frame guessed {guess} UTC", flush=True)
    # confirm the hour against GEOS-FP: decode test frames, compare at every shift from -6 to +6 hours
    for c in ("frame_values.json", "geos.json"):
        (air.OUT / c).unlink(missing_ok=True)
    tree, values, _ = air.bar()
    got = air.frame_values(tree, values)
    i_first = geos_index(guess) - 8
    span = max(air.TEST_FRAMES) + 20
    i0, geos = air.geos_all(i_first + span, span)
    res = []
    for sh in range(-6, 7):
        d = air.compare(got, i0, geos, geos_index(guess) + sh)
        e = np.array([o - g for _, _, o, g in d])
        if len(e) > 100:
            res.append((float(np.abs(e).mean()), sh, len(e)))
    res.sort()
    if not res or res[0][0] > 1.5:
        raise RuntimeError(f"air: the frames do not match GEOS-FP at any hour near {guess} ({res[:3]}); not updating")
    t0 = guess + dt.timedelta(hours=res[0][1])
    print(f"air: GEOS-FP check: best shift {res[0][1]} h, mean absolute {res[0][0]:.2f} C over {res[0][2]} pairs; first frame {t0} UTC", flush=True)
    air.T0 = t0
    air.notes()
    import json
    dest = HERE.parent / "mockups" / "data" / "air" / "meta.json"
    meta = json.loads(dest.read_text())
    meta.update({"first_frame_utc": t0.isoformat(), "time_check_vs_geos_fp_c": round(res[0][0], 2), "updated": dt.date.today().isoformat(),
                 "update": "pipeline/rolling.py, daily (.github/workflows/update-data.yml)"})
    dest.write_text(json.dumps(meta, indent=1))




def ensure(arg):
    """A fresh machine (GitHub Actions) has no cache folders and no colour bar files."""
    mods = {"sst": ["sst_exr"], "sst_anomaly": ["sst_anomaly"], "ice": ["ice"], "fire": ["fire"], "air": ["air"]}.get(arg, [])
    for m in mods:
        mod = __import__(m)
        for d in ("CACHE", "OUT"):
            if hasattr(mod, d):
                getattr(mod, d).mkdir(parents=True, exist_ok=True)
    if arg == "air":
        import air
        bar = air.CACHE / "NearSurfaceTemperature_colorbar_plain.png"
        if not bar.exists():
            subprocess.run(["curl", "-s", "-f", "-m", "120", "--retry", "5", "-o", str(bar),
                            "https://svs.gsfc.nasa.gov/vis/a000000/a005100/a005147/NearSurfaceTemperature_colorbar_plain.png"], check=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    arg = sys.argv[1]
    ensure(arg)
    if arg == "migrate":
        migrate()
    elif arg == "ice":
        update_ice()
    elif arg == "migrate_ice":
        migrate_ice()
    elif arg == "air":
        update_air()
    elif arg == "fire":
        update_fire()
    elif arg == "to_weeks":
        to_weeks("sst")
        to_weeks("sst_anomaly")
    else:
        update(arg)
