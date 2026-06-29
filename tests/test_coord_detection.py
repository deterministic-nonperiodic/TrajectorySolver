"""
tests/test_coord_detection.py
==============================
Unit tests for trajsolver.coord_detection – all pure-Python, no real data files.
"""
import numpy as np
import pytest
import xarray as xr

from trajsolver.coord_detection import (
    CoordNames,
    curate_coords,
    detect_and_curate,
    detect_coords,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ds(lon_vals, lat_vals, z_vals, z_units="m", z_std="altitude",
             lon_name="lon", lat_name="lat", z_name="z_mc", time_name="time"):
    """Build a minimal synthetic dataset for detection tests."""
    nt, nz, nlat, nlon = 2, len(z_vals), len(lat_vals), len(lon_vals)
    data = np.ones((nt, nz, nlat, nlon), dtype=np.float32)
    coords = {
        time_name: np.array(["2025-01-01", "2025-01-02"], dtype="datetime64"),
        z_name:    xr.DataArray(z_vals, dims=z_name,
                                attrs={"units": z_units, "standard_name": z_std}),
        lat_name:  xr.DataArray(lat_vals, dims=lat_name,
                                attrs={"units": "degrees_north"}),
        lon_name:  xr.DataArray(lon_vals, dims=lon_name,
                                attrs={"units": "degrees_east"}),
    }
    return xr.Dataset(
        {"u": (([time_name, z_name, lat_name, lon_name]), data),
         "v": (([time_name, z_name, lat_name, lon_name]), data),
         "w": (([time_name, z_name, lat_name, lon_name]), data)},
        coords=coords,
    )


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestDetectCoords:
    def test_standard_icon_names(self):
        ds = _make_ds([0., 5., 10.], [50., 55.], [80e3, 85e3, 90e3])
        cn = detect_coords(ds)
        assert cn.time == "time"
        assert cn.lat == "lat"
        assert cn.lon == "lon"
        assert cn.z == "z_mc"

    def test_verbose_names(self):
        ds = _make_ds(
            [0., 5.], [50., 55.], [80., 85., 90.],
            z_units="km", z_std="altitude",
            lon_name="longitude", lat_name="latitude", z_name="altitude",
        )
        cn = detect_coords(ds)
        assert cn.lon == "longitude"
        assert cn.lat == "latitude"
        assert cn.z == "altitude"

    def test_pressure_raises_by_default(self):
        ds = _make_ds([0., 5.], [50., 55.], [100., 200., 500.],
                      z_units="hPa", z_std="air_pressure", z_name="plev")
        with pytest.raises(ValueError, match="pressure"):
            detect_coords(ds, require_z=True)

    def test_pressure_accepted_when_allowed(self):
        ds = _make_ds([0., 5.], [50., 55.], [100., 200., 500.],
                      z_units="hPa", z_std="air_pressure", z_name="plev")
        cn = detect_coords(ds, require_z=False)
        assert cn.z == "plev"


# ---------------------------------------------------------------------------
# Curation tests
# ---------------------------------------------------------------------------

class TestCurateCoords:
    def test_lon_wrap_and_sort(self):
        """[0, 360] longitude must become [-180, 180] and be sorted."""
        lon_in = np.array([0., 90., 180., 270., 355.])
        ds = _make_ds(lon_in, [50., 55.], [80e3, 85e3])
        cn = detect_coords(ds)
        ds_out = curate_coords(ds, cn)
        # Rename to canonical so we can use 'lon'
        ds_out = ds_out.rename({cn.lon: "lon"}) if cn.lon != "lon" else ds_out
        lon_out = ds_out.lon.values
        assert np.all(lon_out >= -180.0) and np.all(lon_out <= 180.0), \
            f"Longitude out of [-180, 180]: {lon_out}"
        assert np.all(np.diff(lon_out) >= 0), \
            f"Longitude not sorted: {lon_out}"

    def test_already_normalised_lon_unchanged(self):
        lon_in = np.array([-10., 0., 10., 20.])
        ds = _make_ds(lon_in, [50., 55.], [80e3, 85e3])
        cn = detect_coords(ds)
        ds_out = curate_coords(ds, cn)
        np.testing.assert_array_equal(ds_out[cn.lon].values, lon_in)

    def test_km_to_metres(self):
        """Altitude in km must be converted to metres."""
        z_km = np.array([80., 85., 90.])
        ds = _make_ds([0., 5.], [50., 55.], z_km, z_units="km")
        # detect_and_curate both converts and renames → z coord is 'z'
        ds_out, cn = detect_and_curate(ds)
        z_out = ds_out["z"].values
        np.testing.assert_allclose(z_out, z_km * 1e3, rtol=1e-6)
        assert ds_out["z"].attrs["units"] == "m"
        assert cn.z == "z"

    def test_metres_unchanged(self):
        """Altitude already in metres must not be scaled."""
        z_m = np.array([80e3, 85e3, 90e3])
        ds = _make_ds([0., 5.], [50., 55.], z_m, z_units="m")
        ds_out, _ = detect_and_curate(ds)
        np.testing.assert_array_equal(ds_out["z"].values, z_m)

    def test_canonical_rename(self):
        """After curate_coords + detect_and_curate, dims must be time/lat/lon/z."""
        ds = _make_ds([0., 5.], [50., 55.], [80., 85., 90.],
                      z_units="km",
                      lon_name="longitude", lat_name="latitude", z_name="altitude")
        ds_out, cn = detect_and_curate(ds)
        assert "lon" in ds_out.dims
        assert "lat" in ds_out.dims
        assert "z"   in ds_out.dims
        assert cn.lon == "lon"
        assert cn.lat == "lat"
        assert cn.z   == "z"
        assert cn.z_units == "m"

    def test_non_monotonic_lon_after_wrap(self):
        """Irregular lons that scramble after wrap must still be sorted."""
        lon_in = np.array([10., 200., 350.])
        ds = _make_ds(lon_in, [50., 55.], [80e3, 85e3])
        ds_out, _ = detect_and_curate(ds)
        lon_out = ds_out.lon.values
        assert np.all(np.diff(lon_out) >= 0), f"Not sorted: {lon_out}"
        assert np.all(lon_out >= -180) and np.all(lon_out <= 180)
