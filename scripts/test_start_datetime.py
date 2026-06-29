"""
test_start_datetime.py – sensitivity study over different trajectory start times.
Usage: python scripts/test_start_datetime.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from trajsolver import LagrangianTrajectories, save_cf_compliant, read_falcon

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"

BASE_PATH  = Path("/home/deterministic-nonperiodic/IAP/Experiments/falcon")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
filename = BASE_PATH / "UA-ICON_NWP_atm_DOM01_falcon2_80-120_20250219T03-20T01.nc"
wind_data = xr.open_dataset(filename)

falcon_orbit = read_falcon(BASE_PATH / "orbgen#12.dat")
falcon_orbit = falcon_orbit[falcon_orbit["GAlt"] < 110]

target_orbit = falcon_orbit.set_index("timestamp").to_xarray()
target_orbit = target_orbit[["GLon", "GLat", "GAlt"]].rename(
    {"timestamp": "time", "GLon": "lon", "GLat": "lat", "GAlt": "z"}
)

# ---------------------------------------------------------------------------
# Initial positions
# ---------------------------------------------------------------------------
lidar_lon = 11.771847
lidar_lat  = 54.116714
altitudes  = np.arange(95.3, 97.3, 0.2)

initial_positions = [(lidar_lon, lidar_lat, 1e3 * alt) for alt in altitudes]

time_step     = "10 min"
solver_method = "RK45"
interp_method = "linear"
noise_type    = "lognormal"
ensemble_size = 1000

out_dir = DATA_DIR / "start_datetime"
out_dir.mkdir(parents=True, exist_ok=True)

end_date = "2025-02-19T03:40:00"

start_dates = pd.date_range(
    start=pd.Timestamp("2025-02-20T00:21:00"),
    end=pd.Timestamp("2025-02-20T00:48:00"),
    freq="1min",
).strftime("%Y-%m-%dT%H:%M:%S")

horizontal_tolerance = 50e3  # metres

for start_date in tqdm(start_dates[22:],
                       desc=f"Calculating trajectories for {altitudes.size} particles"):
    print(f"   - Starting from {start_date} ...")
    out_file = out_dir / f"trajectories_{start_date}--{end_date}_fulltime.nc"

    solver = LagrangianTrajectories(
        wind_data,
        timestep=time_step,
        start_time=start_date,
        integration_method=solver_method,
        interpolation_method=interp_method,
        noise_type=noise_type,
        verbose_level=1,
    )

    trajectories_dataset = solver.advect_particles(
        initial_positions,
        end_date=end_date,
        ensemble_size=ensemble_size,
        target=target_orbit,
        distance_tolerance=horizontal_tolerance,
    )

    events = int(trajectories_dataset.attrs["intersection_events"])
    if events:
        print(f"Found {events} intersections between trajectories and Falcon orbit.")
        save_cf_compliant(trajectories_dataset, str(out_file))
    else:
        print("No intersections found.")

    trajectories_dataset.close()

print("Done.")
