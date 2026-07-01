import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib.pyplot as plt

from trajsolver.visualization import (
    visualize_trajectories_percentile_kde,
    generate_representative_particles,
    _target_point_to_orbit_df,
)


@pytest.fixture
def dummy_data():
    times = pd.date_range("2025-02-19T04:00:00", periods=5, freq="10min")
    particles = np.array([0])
    ensembles = np.array([0, 1, 2])

    shape = (len(times), len(particles), len(ensembles))

    trajectories = xr.Dataset(
        data_vars={
            "lon": (["time", "particle", "ensemble"], np.full(shape, 10.0)),
            "lat": (["time", "particle", "ensemble"], np.full(shape, 50.0)),
            "z": (["time", "particle", "ensemble"], np.full(shape, 90.0)),
        },
        coords={
            "time": times,
            "particle": particles,
            "ensemble": ensembles,
        }
    )

    wind_times = pd.date_range("2025-02-19T03:00:00", periods=10, freq="10min")
    z_mc = np.array([80000.0, 90000.0, 100000.0])
    lats = np.array([48.0, 50.0, 52.0])
    lons = np.array([8.0, 10.0, 12.0])

    wind_shape = (len(wind_times), len(z_mc), len(lats), len(lons))

    wind = xr.Dataset(
        data_vars={
            "u": (["time", "z_mc", "lat", "lon"], np.ones(wind_shape)),
            "v": (["time", "z_mc", "lat", "lon"], np.ones(wind_shape)),
        },
        coords={
            "time": wind_times,
            "z_mc": z_mc,
            "lat": lats,
            "lon": lons,
        }
    )

    return trajectories, wind


def test_visualize_trajectories_with_wind(dummy_data):
    trajectories, wind = dummy_data
    fig = visualize_trajectories_percentile_kde(
        trajectories,
        wind=wind,
        show_wind=True,
        calculate_intersections=False,
    )
    assert fig is not None
    plt.close(fig)


def test_visualize_trajectories_without_wind(dummy_data):
    trajectories, wind = dummy_data
    fig = visualize_trajectories_percentile_kde(
        trajectories,
        wind=wind,
        show_wind=False,
        calculate_intersections=False,
    )
    assert fig is not None
    plt.close(fig)


def test_visualize_trajectories_no_wind_dataset(dummy_data):
    trajectories, _ = dummy_data
    fig = visualize_trajectories_percentile_kde(
        trajectories,
        wind=None,
        calculate_intersections=False,
    )
    assert fig is not None
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tests for generate_representative_particles
# ---------------------------------------------------------------------------

@pytest.fixture
def traj_with_hit():
    """
    Two particles, three ensemble members.
    Particle 0, ensemble 1 passes exactly through lon=20, lat=55, z=90 at t=2 (06:00 UTC).
    Particle 1 never hits – tests the fallback path.
    """
    times = pd.date_range("2025-02-19T04:00:00", periods=5, freq="1h")
    particles = np.array([0, 1])
    ensembles = np.array([0, 1, 2])

    shape = (len(times), len(particles), len(ensembles))
    lon = np.full(shape, 0.0)
    lat = np.full(shape, 0.0)
    z   = np.full(shape, 50.0)

    # Particle 0, ensemble 1, time index 2 → right on target
    lon[2, 0, 1] = 20.0
    lat[2, 0, 1] = 55.0
    z  [2, 0, 1] = 90.0

    return xr.Dataset(
        {"lon": (["time", "particle", "ensemble"], lon),
         "lat": (["time", "particle", "ensemble"], lat),
         "z":   (["time", "particle", "ensemble"], z)},
        coords={"time": times, "particle": particles, "ensemble": ensembles},
    )


def test_target_point_conversion_xr_dataset():
    """_target_point_to_orbit_df converts xr.Dataset correctly."""
    tp = xr.Dataset(
        {"lon": ("time", [20.0]), "lat": ("time", [55.0]), "z": ("time", [90.0])},
        coords={"time": pd.to_datetime(["2025-02-19T06:00:00"])},
    )
    df = _target_point_to_orbit_df(tp)
    assert df is not None
    assert list(df.columns) == ["timestamp", "GLon", "GLat", "GAlt"]
    assert df["GLon"].iloc[0] == 20.0
    assert df["GAlt"].iloc[0] == 90.0


def test_target_point_conversion_none():
    assert _target_point_to_orbit_df(None) is None


def test_representative_target_point_hit(traj_with_hit):
    """Particle 0 gets the ensemble member that hits the target_point."""
    tp = xr.Dataset(
        {"lon": ("time", [20.0]), "lat": ("time", [55.0]), "z": ("time", [90.0])},
        coords={"time": pd.to_datetime(["2025-02-19T06:00:00"])},
    )
    result = generate_representative_particles(
        traj_with_hit,
        orbit_df=None,
        target_point=tp,
        horiz_tol_km=10,
        vert_tol_km=5,
    )
    # Particle 0: ensemble 1 is the closest member at t_idx=2
    assert result[0][0] == 1, f"Expected ensemble 1, got {result[0][0]}"
    # Particle 1: target_point selection is tolerance-free so it also gets
    # a target-based result (nearest member at the snapped time step).
    # We just check the time index matches the snapped step (t_idx=2).
    assert result[1][1] == 2, f"Expected time_idx=2 (snapped), got {result[1][1]}"


def test_representative_orbit_fallback(traj_with_hit):
    """Without a target_point but with orbit_df, the orbit intersection is used."""
    orbit_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-02-19T06:00:00"]),
        "GLon": [20.0],
        "GLat": [55.0],
        "GAlt": [90.0],
    })
    result = generate_representative_particles(
        traj_with_hit,
        orbit_df=orbit_df,
        target_point=None,
        horiz_tol_km=10,
        vert_tol_km=5,
    )
    assert result[0][0] == 1


def test_representative_target_beats_orbit(traj_with_hit):
    """When both target_point and orbit_df match different members,
    target_point takes priority."""
    # Make ensemble 2 of particle 0 hit the orbit (at a completely different position)
    traj = traj_with_hit.copy(deep=True)
    # Orbit point at a different location
    orbit_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-02-19T08:00:00"]),
        "GLon": [1.0],
        "GLat": [1.0],
        "GAlt": [50.0],
    })
    traj["lon"].values[4, 0, 2] = 1.0
    traj["lat"].values[4, 0, 2] = 1.0
    traj["z"].values[4, 0, 2]   = 50.0

    tp = xr.Dataset(
        {"lon": ("time", [20.0]), "lat": ("time", [55.0]), "z": ("time", [90.0])},
        coords={"time": pd.to_datetime(["2025-02-19T06:00:00"])},
    )
    result = generate_representative_particles(
        traj,
        orbit_df=orbit_df,
        target_point=tp,
        horiz_tol_km=10,
        vert_tol_km=5,
    )
    # Target hit (ensemble 1) must win over orbit hit (ensemble 2)
    assert result[0][0] == 1, f"Expected ensemble 1 (target_point), got {result[0][0]}"
