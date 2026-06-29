"""
run_falcon.py – main Falcon re-entry trajectory calculation script.
Usage: python scripts/run_falcon.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from trajsolver import LagrangianTrajectories, save_cf_compliant, read_falcon
from trajsolver.visualization import plot_orbit_and_ensemble_3d, visualize_trajectories_percentile_kde

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR   = REPO_ROOT / "data"
FIG_DIR    = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

BASE_PATH  = Path("/home/deterministic-nonperiodic/IAP/Experiments/falcon")

# ---------------------------------------------------------------------------
# Load wind data
# ---------------------------------------------------------------------------
filename = BASE_PATH / "UA-ICON_NWP_atm_DOM01_falcon2_80-120_20250219T03-20T01.nc"
wind_data = xr.open_dataset(filename)

# ---------------------------------------------------------------------------
# Load Falcon orbit
# ---------------------------------------------------------------------------
falcon_orbit = read_falcon(BASE_PATH / "Trajectory_2025-02-19/orbgen#12.dat")
falcon_orbit = falcon_orbit[falcon_orbit["GAlt"] < 110]

# Single-point target (the re-entry observation)
target_time = np.array(["2025-02-20T00:21:00"], dtype="datetime64[ns]")
target_orbit = xr.Dataset(
    coords={"time": target_time},
    data_vars={
        "lon": ("time", np.array([11.46])),
        "lat": ("time", np.array([54.07])),
        "z":   ("time", np.array([97.1])),
    },
)

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
start_date = "2025-02-19T03:42"
end_date   = "2025-02-20T00:21:45"

intersection_lon = [-12.292, -11.06844946, -5.891]
intersection_lat = [52.257, 52.7158398, 52.953]
intersection_altitudes = [100.8, 100.6, 100.2, 100.0, 99.8, 99.6]

initial_positions = [
    (lon, lat, 1e3 * alt)
    for lon, lat in zip(intersection_lon, intersection_lat)
    for alt in intersection_altitudes
]

time_step     = "10 min"
time_lag      = "0 min"
solver_method = "RK23"
interp_method = "linear"
noise_type    = "lognormal"
ensemble_size = 200

# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------
out_filename = DATA_DIR / (
    f"trajectories_{solver_method}_{interp_method}_{noise_type}"
    f"_{start_date}--{end_date}"
    f"_particles:{len(initial_positions)}_members:{ensemble_size}.nc"
)

# ---------------------------------------------------------------------------
# Solve (or load from cache)
# ---------------------------------------------------------------------------
if not out_filename.exists():
    solver = LagrangianTrajectories(
        wind_data,
        timestep=time_step,
        start_time=start_date,
        integration_method=solver_method,
        interpolation_method=interp_method,
        noise_type=noise_type,
        verbose_level=1,
        time_lag=time_lag,
    )
    trajectories_dataset = solver.advect_particles(
        initial_positions,
        end_date=end_date,
        ensemble_size=ensemble_size,
        target=target_orbit,
        distance_tolerance=25e3,
    )
    save_cf_compliant(trajectories_dataset, str(out_filename))
else:
    trajectories_dataset = xr.open_dataset(out_filename)

# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
fig_name = str(FIG_DIR / f"forward_trajectories_{solver_method}_{interp_method}_{start_date}--{end_date}")

visualize_trajectories_percentile_kde(
    trajectories_dataset,
    wind=wind_data,
    orbit=falcon_orbit,
    calculate_intersections=False,
    figure_name=fig_name,
)

plot_orbit_and_ensemble_3d(
    trajectories_dataset, falcon_orbit,
    max_ensemble=trajectories_dataset.ensemble.size,
    particle_subset=None,
    horiz_tol_km=50, vert_tol_km=10,
    figsize=(8, 7), elev=20, azim=-60,
    figure_name=str(FIG_DIR / f"forward_orbit_vs_ensemble_3d_{end_date}.png"),
)
