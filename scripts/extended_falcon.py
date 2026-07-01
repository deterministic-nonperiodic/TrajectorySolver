"""
extended_falcon.py – extended-duration forward trajectory calculation.
Usage: python scripts/extended_falcon.py
"""
from pathlib import Path

import numpy as np
import xarray as xr

from trajsolver import LagrangianTrajectories, save_cf_compliant, read_falcon, sample_orbit_positions
from trajsolver.visualization import plot_orbit_and_ensemble_3d, visualize_trajectories_percentile_kde

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

BASE_PATH = Path("/home/deterministic-nonperiodic/IAP/Experiments/falcon")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
wind_data = xr.open_dataset(DATA_DIR / "jawara_winds_HL_02-2025.nc")

falcon_orbit = read_falcon(BASE_PATH / "Trajectory_2025-02-19/orbgen#12.dat")
falcon_orbit = falcon_orbit[falcon_orbit["GAlt"] < 110]

target_time = np.array(["2025-02-22T15:35:00"], dtype="datetime64[ns]")

target_point = xr.Dataset(
    coords={"time": target_time},
    data_vars={
        "lon": ("time", np.array([37.0])),
        "lat": ("time", np.array([58.0])),
        "z": ("time", np.array([94.0])),
    },
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
start_date = "2025-02-19T03:42:00"
end_date = "2025-02-22T15:32:00"

n_particles = 10  # starting locations drawn from the re-entry arc
orbit_seed = 42   # change for a different draw; None → non-reproducible

initial_positions = sample_orbit_positions(
    falcon_orbit, n=n_particles, alt_min=70.0, lon_min=-10.0, seed=orbit_seed
)

time_step = "10 min"
time_lag = "0 min"
solver_method = "RK45"
interp_method = "linear"
noise_type = "lognormal"
ensemble_size = 100

out_filename = DATA_DIR / (
    f"trajectories_{solver_method}_{interp_method}_{noise_type}"
    f"_{start_date}--{end_date}"
    f"_particles:{len(initial_positions)}_members:{ensemble_size}.nc"
)

# ---------------------------------------------------------------------------
# Solve
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
        target=target_point,
        distance_tolerance=25e3,
    )
    save_cf_compliant(trajectories_dataset, str(out_filename))

# Always load the dataset for analysis
print(f"Loading trajectories from {out_filename}...")
trajectories_dataset = xr.open_dataset(out_filename)

# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
fig_stem = f"extended_forward_trajectories_{solver_method}_{interp_method}_{start_date}--{end_date}"
visualize_trajectories_percentile_kde(
    trajectories_dataset,
    wind=wind_data,
    orbit=falcon_orbit,
    calculate_intersections=True,
    figure_name=str(FIG_DIR / fig_stem),
    map_extent=[-30, 50, 40, 85],
    target_point=target_point,
    target_label="OSIRIS",
    max_dist_km=500,
)

plot_orbit_and_ensemble_3d(
    trajectories_dataset, falcon_orbit, target_orbit=target_point,
    max_ensemble=trajectories_dataset.ensemble.size,
    particle_subset=None,
    horiz_tol_km=120, vert_tol_km=15,
    figsize=(8, 7), elev=20, azim=-75,
    figure_name=str(FIG_DIR / f"extended_forward_orbit_vs_ensemble_3d_{end_date}.png"),
)
