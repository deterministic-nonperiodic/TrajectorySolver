"""
tests/test_numerics.py
=======================
Numeric integration tests using purely synthetic wind fields (no real data).

All tests build a small xr.Dataset on the fly so they run offline and fast.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from trajsolver import LagrangianTrajectories


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALT_M = np.array([80e3, 85e3, 90e3, 95e3, 100e3], dtype=np.float64)
_LAT   = np.array([40., 50., 60., 70.], dtype=np.float64)
_LON   = np.array([-20., -10., 0., 10., 20.], dtype=np.float64)
_TIMES = pd.date_range("2025-01-01", periods=5, freq="1h")


def _make_wind(u_val=0.0, v_val=0.0, w_val=0.0) -> xr.Dataset:
    """
    Uniform (constant) wind field on a small regular grid.
    Units: m, degrees, m/s.
    """
    nt, nz, nlat, nlon = len(_TIMES), len(_ALT_M), len(_LAT), len(_LON)
    shape = (nt, nz, nlat, nlon)

    return xr.Dataset(
        {
            "u": (["time", "z_mc", "lat", "lon"], np.full(shape, u_val, dtype=np.float32)),
            "v": (["time", "z_mc", "lat", "lon"], np.full(shape, v_val, dtype=np.float32)),
            "w": (["time", "z_mc", "lat", "lon"], np.full(shape, w_val, dtype=np.float32)),
        },
        coords={
            "time":  _TIMES,
            "z_mc": xr.DataArray(_ALT_M, dims="z_mc", attrs={"units": "m", "standard_name": "altitude"}),
            "lat":  xr.DataArray(_LAT, dims="lat", attrs={"units": "degrees_north"}),
            "lon":  xr.DataArray(_LON, dims="lon", attrs={"units": "degrees_east"}),
        },
    )


def _solver(ds: xr.Dataset, noise_type=None, **kwargs) -> LagrangianTrajectories:
    """Build a solver with noise disabled and tight tolerances."""
    return LagrangianTrajectories(
        ds,
        timestep="10 min",
        integration_method="RK45",
        noise_type=noise_type,
        rtol=1e-8,
        atol=1e-3,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test 1 – Zero wind: particles must not move
# ---------------------------------------------------------------------------

class TestZeroWind:
    def test_no_displacement(self):
        ds = _make_wind(u_val=0.0, v_val=0.0, w_val=0.0)
        solver = _solver(ds, start_time="2025-01-01T00:00:00")

        start_pos = [(0.0, 55.0, 90e3)]  # (lon, lat, z_m)
        traj = solver.advect_particles(start_pos, duration="2h", ensemble_size=1, n_jobs=1)

        # Convert Mercator back – the solver returns lon/lat
        final_lon = traj.lon.isel(time=-1, particle=0, ensemble=0).item()
        final_lat = traj.lat.isel(time=-1, particle=0, ensemble=0).item()
        final_z   = traj.z.isel(time=-1, particle=0, ensemble=0).item()  # km

        assert abs(final_lon - 0.0) < 0.01, f"lon displaced: {final_lon}"
        assert abs(final_lat - 55.0) < 0.01, f"lat displaced: {final_lat}"
        assert abs(final_z - 90.0) < 0.1,    f"z displaced: {final_z} km"


# ---------------------------------------------------------------------------
# Test 2 – Constant vertical wind: check altitude advances at correct rate
# ---------------------------------------------------------------------------

class TestConstantVerticalWind:
    def test_altitude_rate(self):
        w_ms = 10.0   # 10 m/s upward
        ds = _make_wind(w_val=w_ms)
        solver = _solver(ds, start_time="2025-01-01T00:00:00")

        start_pos = [(0.0, 55.0, 85e3)]
        duration_s = 600  # 10 minutes → expected Δz = 6 000 m
        traj = solver.advect_particles(start_pos, duration=f"{duration_s} s", ensemble_size=1, n_jobs=1)

        z0_km = 85.0
        z1_km = traj.z.isel(time=-1, particle=0, ensemble=0).item()
        dz_m  = (z1_km - z0_km) * 1e3
        expected_dz = w_ms * duration_s

        assert abs(dz_m - expected_dz) < 100, \
            f"Expected Δz ≈ {expected_dz} m, got {dz_m:.1f} m"


# ---------------------------------------------------------------------------
# Test 3 – Ensemble mean converges to deterministic with lognormal noise
# ---------------------------------------------------------------------------

class TestEnsembleConvergence:
    def test_mean_converges_to_deterministic(self):
        """
        With lognormal noise, the ensemble mean should converge toward the
        noise-free trajectory as ensemble size grows.  We use small ensemble
        sizes and a short duration to keep runtime < 30 s.
        """
        ds = _make_wind(u_val=20.0, v_val=10.0, w_val=2.0)
        start_pos = [(0.0, 55.0, 90e3)]
        common_kw = dict(
            timestep="10 min",
            integration_method="RK23",
            rtol=1e-6,
            atol=1.0,
            start_time="2025-01-01T00:00:00",
        )

        # Deterministic baseline (noise_type=None skips mean-wind computation)
        det = LagrangianTrajectories(ds, **{**common_kw, "noise_type": None})
        traj_det = det.advect_particles(start_pos, duration="15 min", ensemble_size=1, n_jobs=1)
        z_det = traj_det.z.isel(particle=0).mean("ensemble").values

        # Noisy ensemble with small (n=5) and larger (n=20) sizes
        errors = {}
        for n in (5, 20):
            solver = LagrangianTrajectories(ds, **{**common_kw, "noise_type": "lognormal"})
            traj = solver.advect_particles(start_pos, duration="15 min", ensemble_size=n, n_jobs=1)
            z_mean = traj.z.isel(particle=0).mean("ensemble").values
            errors[n] = float(np.mean(np.abs(z_mean - z_det)))

        assert errors[20] < errors[5], (
            f"Larger ensemble did not reduce error: err_5={errors[5]:.4f}, err_20={errors[20]:.4f}"
        )



# ---------------------------------------------------------------------------
# Test 4 – Forward/backward symmetry (no noise)
# ---------------------------------------------------------------------------

class TestForwardBackwardSymmetry:
    def test_roundtrip_position(self):
        """
        A deterministic forward run followed by a backward run of the same
        duration should return the particle close to its starting point.
        Uses only horizontal wind (w=0) to avoid z drifting outside the grid.
        """
        # w=0 so z stays fixed; u and v move the particle horizontally
        ds = _make_wind(u_val=15.0, v_val=-5.0, w_val=0.0)
        start_pos = [(0.0, 55.0, 90e3)]
        common_kw = dict(
            noise_type=None,
            integration_method="RK45",
            rtol=1e-9,
            atol=1e-3,
        )

        # Forward 30 min
        fwd = LagrangianTrajectories(
            ds, timestep="5 min",
            start_time="2025-01-01T00:00:00", **common_kw
        )
        traj_fwd = fwd.advect_particles(start_pos, duration="30 min", ensemble_size=1, n_jobs=1)

        final_lon = traj_fwd.lon.isel(time=-1, particle=0, ensemble=0).item()
        final_lat = traj_fwd.lat.isel(time=-1, particle=0, ensemble=0).item()
        final_z   = traj_fwd.z.isel(time=-1, particle=0, ensemble=0).item() * 1e3  # m

        # Backward from the final position – start at the forward end-time
        bwd = LagrangianTrajectories(
            ds, timestep="-5 min",
            start_time="2025-01-01T00:30:00", **common_kw
        )
        traj_bwd = bwd.advect_particles(
            [(final_lon, final_lat, final_z)],
            duration="-30 min",
            ensemble_size=1,
            n_jobs=1,
        )

        return_lon = traj_bwd.lon.isel(time=-1, particle=0, ensemble=0).item()
        return_lat = traj_bwd.lat.isel(time=-1, particle=0, ensemble=0).item()
        return_z   = traj_bwd.z.isel(time=-1, particle=0, ensemble=0).item()

        # With w=0, z should be exactly unchanged; lon/lat within 0.5 degrees
        assert abs(return_lon - 0.0)  < 0.5,  f"lon roundtrip error: {return_lon}"
        assert abs(return_lat - 55.0) < 0.5,  f"lat roundtrip error: {return_lat}"
        assert abs(return_z   - 90.0) < 0.1,  f"z roundtrip error: {return_z} km"


# ---------------------------------------------------------------------------
# Test 5 – RegularGridInterpolator vs known exact value at a grid node
# ---------------------------------------------------------------------------

class TestInterpolatorExact:
    def test_grid_node_exact(self):
        """
        At a grid-node position, the interpolated wind must equal the stored value.
        """
        u0, v0, w0 = 7.3, -3.1, 0.8
        ds = _make_wind(u_val=u0, v_val=v0, w_val=w0)
        solver = _solver(ds, start_time="2025-01-01T00:00:00")

        # Query exactly at t=0, z=90 km, lat=55°, lon=0°
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, _ = tr.transform(0.0, 0.0)
        _, y = tr.transform(0.0, 55.0)

        state = np.array([x, y, 90e3])
        vel = solver.velocity_vectorized(0.0, state, noise_scale=False)

        np.testing.assert_allclose(vel[0], u0, rtol=1e-4, err_msg="u mismatch")
        np.testing.assert_allclose(vel[1], v0, rtol=1e-4, err_msg="v mismatch")
        np.testing.assert_allclose(vel[2], w0, rtol=1e-4, err_msg="w mismatch")


# ---------------------------------------------------------------------------
# Test 6 – Pressure velocity to m/s conversion
# ---------------------------------------------------------------------------

class TestPressureVelocityConversion:
    def test_hpa_to_mps(self):
        """
        Verify that vertical velocity w in hPa/s is converted to m/s.
        """
        w_hpa = -1.8457e-5
        ds = _make_wind(u_val=0.0, v_val=0.0, w_val=w_hpa)
        # Set the units attribute to hPa/s
        ds.w.attrs["units"] = "hPa/s"
        
        solver = _solver(ds, start_time="2025-01-01T00:00:00")
        
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, _ = tr.transform(0.0, 0.0)
        _, y = tr.transform(0.0, 55.0)
        
        state = np.array([x, y, 80e3])
        vel = solver.velocity_vectorized(0.0, state, noise_scale=False)
        
        expected_w_mps = -w_hpa * 100.0 / (1.8457e-5 * 9.80665)
        np.testing.assert_allclose(vel[2], expected_w_mps, rtol=1e-4)

    def test_hpa_to_mps_with_density(self):
        """
        Verify that vertical velocity w in hPa/s is converted to m/s
        using the actual density variable 'rho' present in the dataset.
        """
        w_hpa = -0.5
        ds = _make_wind(u_val=0.0, v_val=0.0, w_val=w_hpa)
        ds.w.attrs["units"] = "hPa/s"
        
        # Add a density variable 'rho' with constant value 2.0
        ds["rho"] = (ds.w.dims, np.full_like(ds.w.values, 2.0))
        
        solver = _solver(ds, start_time="2025-01-01T00:00:00")
        
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, _ = tr.transform(0.0, 0.0)
        _, y = tr.transform(0.0, 55.0)
        
        state = np.array([x, y, 80e3])
        vel = solver.velocity_vectorized(0.0, state, noise_scale=False)
        
        # expected w = -(-0.5) * 100.0 / (2.0 * 9.80665)
        expected_w_mps = 50.0 / (2.0 * 9.80665)
        np.testing.assert_allclose(vel[2], expected_w_mps, rtol=1e-4)
