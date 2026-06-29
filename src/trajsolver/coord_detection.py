"""
coord_detection.py
==================
CF-aware coordinate detection and curation for TrajectorySolver.

Detection
---------
Finds the canonical lon / lat / vertical / time dimension names inside an
arbitrary xr.Dataset by querying CF attributes (axis, standard_name, units)
and common name patterns – adapted from HealICON's cf_coords.py.

Curation
--------
After detection the dataset is normalised so that downstream code can always
rely on four canonical dimension names: ``time``, ``lat``, ``lon``, and
``z`` (the chosen vertical coordinate).

  * Metric (height / altitude) coordinates are converted to **metres**.
    Recognised non-metre units: km, ft/feet, gpm (already metres – no-op).
  * Longitude is normalised to **[−180, 180]** (wrapping values > 180).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal lookup tables  (kept minimal – borrow spirit from HealICON)
# ---------------------------------------------------------------------------

# Candidate names for each logical axis, ordered by preference
_LON_NAMES = ("lon", "long", "longitude", "clon", "rlon", "x_lon")
_LAT_NAMES = ("lat", "latitude", "clat", "rlat", "y_lat")
_TIME_NAMES = ("time", "t", "date", "datetime", "valid_time")

# Height/altitude – positive upward, metric
_Z_HEIGHT_NAMES = ("z", "z_mc", "z_ifc", "height", "altitude", "alt",
                   "zlev", "height_above_geopotential_datum",
                   "geometric_height", "level", "lev")
_Z_HEIGHT_STD = frozenset({
    "altitude", "height", "geometric_height",
    "height_above_geopotential_datum",
    "height_above_mean_sea_level",
    "height_above_reference_ellipsoid",
    "atmosphere_hybrid_height_coordinate",
    "atmosphere_sigma_coordinate",
    "atmosphere_sleve_coordinate",
})
_Z_PRESSURE_NAMES = ("plev", "pressure", "pres", "isobaric", "p")
_Z_PRESSURE_UNITS = frozenset({"pa", "hpa", "mbar", "millibar", "bar", "mb"})
_Z_METER_UNITS = frozenset({"m", "meter", "meters", "metre", "metres", "gpm"})
_Z_KM_UNITS = frozenset({"km", "kilometer", "kilometers", "kilometre", "kilometres"})
_Z_FT_UNITS = frozenset({"ft", "feet", "foot"})

# Degree / radian axis units
_DEG_UNITS = frozenset({"degree", "degrees", "degrees_north", "degrees_east",
                        "degrees_west", "degrees_south", "deg"})


# ---------------------------------------------------------------------------
# Low-level attribute helpers
# ---------------------------------------------------------------------------

def _attr(da: xr.DataArray, key: str) -> str:
    """Return a lower-stripped attribute value, or ''."""
    return str(da.attrs.get(key, "")).strip().lower()


def _name(da: xr.DataArray) -> str:
    """Lower-stripped DataArray name."""
    return str(da.name or "").strip().lower()


# ---------------------------------------------------------------------------
# Coordinate type tests
# ---------------------------------------------------------------------------

def _is_lon(da: xr.DataArray) -> bool:
    """Heuristic: is *da* a longitude coordinate?"""
    nm = _name(da)
    std = _attr(da, "standard_name")
    axis = _attr(da, "axis")
    units = _attr(da, "units")

    if axis == "x" and any(h in units for h in ("degree", "deg", "rad")):
        return True
    if "longitude" in std:
        return True
    if nm in _LON_NAMES:
        # Extra guard: latitude values fit in [-90, 90], longitude values
        # often exceed that range *or* the name is unambiguously lon.
        vals = da.values[np.isfinite(da.values)] if da.size else np.array([])
        if nm in ("lon", "long", "longitude", "clon"):
            return True
        if vals.size and float(np.abs(vals).max()) > 90.0:
            return True
    return False


def _is_lat(da: xr.DataArray) -> bool:
    """Heuristic: is *da* a latitude coordinate?"""
    nm = _name(da)
    std = _attr(da, "standard_name")
    axis = _attr(da, "axis")
    units = _attr(da, "units")

    if axis == "y" and any(h in units for h in ("degree", "deg", "rad")):
        return True
    if "latitude" in std:
        return True
    if nm in _LAT_NAMES:
        vals = da.values[np.isfinite(da.values)] if da.size else np.array([])
        if nm in ("lat", "latitude", "clat"):
            return True
        if vals.size and float(np.abs(vals).max()) <= 90.0:
            return True
    return False


def _is_time(da: xr.DataArray) -> bool:
    """Heuristic: is *da* a time coordinate?"""
    nm = _name(da)
    axis = _attr(da, "axis")
    std = _attr(da, "standard_name")

    if axis == "t":
        return True
    if nm in _TIME_NAMES:
        return True
    if "time" in std:
        return True
    # Check dtype
    if np.issubdtype(da.dtype, np.datetime64) or np.issubdtype(da.dtype, np.timedelta64):
        return True
    return False


def _is_pressure_z(da: xr.DataArray) -> bool:
    """True if this looks like a pressure (isobaric) vertical coordinate."""
    nm = _name(da)
    units = _attr(da, "units")
    std = _attr(da, "standard_name")

    if any(pu in units for pu in _Z_PRESSURE_UNITS):
        return True
    if any(pn in nm for pn in _Z_PRESSURE_NAMES):
        return True
    if std in ("air_pressure", "atmosphere_ln_pressure_coordinate"):
        return True
    return False


def _is_height_z(da: xr.DataArray) -> bool:
    """
    True if *da* is a height/altitude vertical coordinate (metric, positive-up).
    Pressure coordinates return False.
    """
    if _is_pressure_z(da):
        return False

    nm = _name(da)
    units = _attr(da, "units")
    std = _attr(da, "standard_name")
    axis = _attr(da, "axis")

    # CF axis='Z' with metric units → definitive
    all_z_units = _Z_METER_UNITS | _Z_KM_UNITS | _Z_FT_UNITS | {"gpm"}
    if axis == "z" and any(u in units for u in all_z_units):
        return True

    # CF standard_name (height family)
    if std in _Z_HEIGHT_STD:
        return True

    # Name pattern + any metric unit
    if any(pat in nm for pat in _Z_HEIGHT_NAMES):
        if any(u in units for u in all_z_units):
            return True
        # If units are absent but name is clearly height-like, accept with warning
        if not units and any(pat in nm for pat in ("z", "height", "altitude", "alt")):
            logger.debug("Accepting '%s' as vertical coord without units (name heuristic).", nm)
            return True

    # 'lev' / 'level' with metre-like units
    if ("lev" in nm or "level" in nm) and any(u in units for u in all_z_units):
        return True

    return False


# ---------------------------------------------------------------------------
# Public detection API
# ---------------------------------------------------------------------------

class CoordNames:
    """Container for resolved canonical coordinate names."""

    __slots__ = ("time", "lat", "lon", "z", "z_units")

    def __init__(self, time: str, lat: str, lon: str,
                 z: str, z_units: str):
        self.time = time
        self.lat = lat
        self.lon = lon
        self.z = z
        self.z_units = z_units  # original units string of the vertical coord

    def __repr__(self) -> str:
        return (f"CoordNames(time={self.time!r}, lat={self.lat!r}, "
                f"lon={self.lon!r}, z={self.z!r}, z_units={self.z_units!r})")


def detect_coords(ds: xr.Dataset,
                  require_z: bool = True) -> CoordNames:
    """
    Detect canonical coordinate names from an xr.Dataset using CF conventions
    and common naming heuristics.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (must contain wind data with at least time, horizontal,
        and vertical dimensions).
    require_z : bool
        If True, raise ValueError when no height-like vertical coordinate is
        found.  Set False to accept pressure-coordinate datasets (z will be
        the pressure dim name, z_units will be the pressure unit).

    Returns
    -------
    CoordNames
        Struct with resolved names for time / lat / lon / z.

    Raises
    ------
    ValueError
        When a required coordinate cannot be identified.
    """
    all_vars = {name: ds[name] for name in ds.coords}
    # Also check dimension-only coords and variables that happen to be dims
    for name in ds.dims:
        if name not in all_vars and name in ds:
            all_vars[name] = ds[name]

    time_name = _find_coord(all_vars, _is_time, "time")
    lat_name = _find_coord(all_vars, _is_lat, "lat")
    lon_name = _find_coord(all_vars, _is_lon, "lon")
    z_name, z_units = _find_z_coord(ds, all_vars, require_z)

    cn = CoordNames(time=time_name, lat=lat_name, lon=lon_name,
                    z=z_name, z_units=z_units)
    logger.info("Detected coordinates: %s", cn)
    return cn


def _find_coord(all_vars: dict[str, xr.DataArray],
                predicate,
                role: str) -> str:
    """Find a single coordinate matching *predicate*; raise if ambiguous or missing."""
    hits = [name for name, da in all_vars.items() if predicate(da)]
    if not hits:
        raise ValueError(
            f"Cannot find '{role}' coordinate in dataset. "
            f"Available variables/coords: {list(all_vars)}"
        )
    if len(hits) > 1:
        # Prefer the first hit from the ordered preference lists
        preference = {"time": _TIME_NAMES,
                      "lat": _LAT_NAMES,
                      "lon": _LON_NAMES}.get(role, ())
        for pref in preference:
            if pref in hits:
                logger.debug("Ambiguous '%s' candidates %s; choosing '%s' by preference.",
                             role, hits, pref)
                return pref
        logger.warning("Ambiguous '%s' candidates %s; using first: '%s'.", role, hits, hits[0])
    return hits[0]


def _find_z_coord(ds: xr.Dataset,
                  all_vars: dict[str, xr.DataArray],
                  require_z: bool) -> tuple[str, str]:
    """
    Return (name, units_string) of the vertical coordinate.

    Prefers height-type coordinates; falls back to pressure when require_z=False.
    """
    # Priority 1: height / altitude type
    height_hits = [name for name, da in all_vars.items() if _is_height_z(da)]
    if height_hits:
        if len(height_hits) > 1:
            # Prefer dims over aux coords, then name-order preference
            dim_hits = [n for n in height_hits if n in ds.dims]
            preferred = dim_hits if dim_hits else height_hits
            # Prefer 'z_mc' (cell-centre) over 'z_ifc' (interface), then alphabetic
            for pref in _Z_HEIGHT_NAMES:
                if pref in preferred:
                    chosen = pref
                    break
            else:
                chosen = preferred[0]
            logger.warning(
                "Multiple height coords found %s; using '%s'.", height_hits, chosen)
        else:
            chosen = height_hits[0]
        units = _attr(all_vars[chosen], "units")
        return chosen, units

    # Priority 2: pressure coordinate (only if require_z=False)
    pressure_hits = [name for name, da in all_vars.items() if _is_pressure_z(da)]
    if pressure_hits:
        if require_z:
            raise ValueError(
                f"Only pressure-type vertical coordinates found {pressure_hits}, "
                "but require_z=True.  Pass require_z=False to allow pressure coords."
            )
        chosen = pressure_hits[0]
        units = _attr(all_vars[chosen], "units")
        warnings.warn(
            f"Using pressure coordinate '{chosen}' as vertical axis. "
            "Trajectory integration will operate in pressure space.",
            UserWarning, stacklevel=3,
        )
        return chosen, units

    if require_z:
        raise ValueError(
            "No vertical coordinate found in dataset. "
            f"Available variables/coords: {list(all_vars)}"
        )
    return "", ""


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------

_KM_TO_M = 1_000.0
_FT_TO_M = 0.3048


def curate_coords(ds: xr.Dataset, cn: CoordNames) -> xr.Dataset:
    """
    Normalise coordinates in *ds* so that downstream code receives a dataset
    with predictable units and ranges:

    * **Altitude / height** → converted to **metres** (from km, ft, gpm).
    * **Longitude** → normalised to **[−180, 180]** (wrapping values > 180).
    * Canonical dimension names are renamed to ``time``, ``lat``, ``lon``, ``z``
      so that ``LagrangianTrajectories`` can always use the same names.

    Parameters
    ----------
    ds : xr.Dataset
        Raw input dataset.
    cn : CoordNames
        Result of :func:`detect_coords`.

    Returns
    -------
    xr.Dataset
        Dataset with normalised coordinates and (possibly) renamed dimensions.
    """
    ds = ds.copy()

    # ------------------------------------------------------------------
    # 1. Normalise longitude to [-180, 180]
    # ------------------------------------------------------------------
    if cn.lon in ds.coords:
        lon_vals = ds[cn.lon].values.copy()
        needs_wrap = np.any(lon_vals > 180.0) or np.any(lon_vals < -180.0)
        if needs_wrap:
            wrapped = (lon_vals + 180.0) % 360.0 - 180.0
            n_changed = int(np.sum(lon_vals != wrapped))
            logger.info(
                "Longitude '%s': wrapping %d value(s) from [0, 360] to [-180, 180].",
                cn.lon, n_changed,
            )
            # Rebuild coord preserving attributes
            old_attrs = ds[cn.lon].attrs.copy()
            ds = ds.assign_coords({cn.lon: (ds[cn.lon].dims, wrapped)})
            ds[cn.lon].attrs.update(old_attrs)
            ds[cn.lon].attrs["valid_min"] = -180.0
            ds[cn.lon].attrs["valid_max"] = 180.0

        # Always sort along the longitude dimension after any wrapping so that
        # label-based sel(slice(...)) works (xarray requires a monotonic index).
        if cn.lon in ds.dims:
            ds = ds.sortby(cn.lon)

    # ------------------------------------------------------------------
    # 2. Convert vertical coordinate to metres
    # ------------------------------------------------------------------
    if cn.z and cn.z in ds.coords:
        z_units_lower = cn.z_units.lower().strip()
        z_vals = ds[cn.z].values

        scale: Optional[float] = None
        if z_units_lower in _Z_KM_UNITS:
            scale = _KM_TO_M
        elif z_units_lower in _Z_FT_UNITS:
            scale = _FT_TO_M
        elif z_units_lower in _Z_METER_UNITS or z_units_lower in ("gpm", ""):
            scale = None  # already metres (or unknown – leave as-is)
        else:
            logger.warning(
                "Vertical coordinate '%s' has unrecognised units '%s'. "
                "No unit conversion applied.",
                cn.z, cn.z_units,
            )

        if scale is not None:
            logger.info(
                "Vertical coordinate '%s': converting from '%s' to metres (×%.4g).",
                cn.z, cn.z_units, scale,
            )
            old_attrs = ds[cn.z].attrs.copy()
            ds = ds.assign_coords({cn.z: (ds[cn.z].dims, z_vals * scale)})
            ds[cn.z].attrs.update(old_attrs)
            ds[cn.z].attrs["units"] = "m"

        # If the z-variable also appears as a data variable (e.g. z_mc / z_ifc
        # stored as full 3-D fields), convert those too.
        for var in list(ds.data_vars):
            if var == cn.z or str(var).lower() in (_Z_HEIGHT_NAMES + ("z_ifc",)):
                vunits = _attr(ds[var], "units")
                if vunits.lower() in _Z_KM_UNITS and scale is None:
                    # scale wasn't set above (coord was already m) but this var is in km
                    ds[var] = ds[var] * _KM_TO_M
                    ds[var].attrs["units"] = "m"
                elif scale is not None and vunits.lower() in (cn.z_units.lower(),):
                    ds[var] = ds[var] * scale
                    ds[var].attrs["units"] = "m"

    # ------------------------------------------------------------------
    # 3. Rename dimensions / coords to canonical names
    # ------------------------------------------------------------------
    rename_map: dict[str, str] = {}
    if cn.time and cn.time != "time":
        rename_map[cn.time] = "time"
    if cn.lat and cn.lat != "lat":
        rename_map[cn.lat] = "lat"
    if cn.lon and cn.lon != "lon":
        rename_map[cn.lon] = "lon"
    if cn.z and cn.z != "z":
        rename_map[cn.z] = "z"

    if rename_map:
        logger.info("Renaming coordinates: %s", rename_map)
        ds = ds.rename(rename_map)

    return ds


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def detect_and_curate(ds: xr.Dataset,
                      require_z: bool = True) -> tuple[xr.Dataset, CoordNames]:
    """
    Detect coordinate names and return a curated dataset together with the
    resolved (post-rename) CoordNames (all canonical: time/lat/lon/z).

    Parameters
    ----------
    ds : xr.Dataset
    require_z : bool

    Returns
    -------
    curated_ds : xr.Dataset
    canonical_names : CoordNames
        Always has time='time', lat='lat', lon='lon', z='z' after curation.
    """
    cn = detect_coords(ds, require_z=require_z)
    ds_curated = curate_coords(ds, cn)
    canonical = CoordNames(
        time="time", lat="lat", lon="lon", z="z",
        z_units="m" if cn.z else cn.z_units,
    )
    return ds_curated, canonical
