"""
core.py
=======
Lagrangian ensemble trajectory solver.

The ``LagrangianTrajectories`` class integrates particle positions forward (or
backward) in time through a 3-D wind field, using scipy's ``solve_ivp``
adaptive ODE solvers with optional stochastic noise.

Key optimisations vs. the original flat-file version
-----------------------------------------------------
* **Fast interpolation**: the wind field is loaded into memory once and wrapped
  in a ``scipy.interpolate.RegularGridInterpolator`` (tri-linear by default).
  This avoids the heavy xarray label-lookup overhead at every ODE step.
* **Pre-computed noise profiles**: sigma and mean-wind vertical profiles are
  evaluated on the vertical grid at startup; subsequent noise calls use cheap
  ``np.interp`` rather than xarray interp.
* **Exposed tolerances**: ``rtol`` / ``atol`` are constructor arguments so the
  caller can trade accuracy for speed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import dask
import numpy as np
import pandas as pd
import xarray as xr
from dateutil import parser
from pyproj import Geod, Transformer
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from .coord_detection import detect_and_curate
from .tools import (
    convert_to_seconds,
    generate_eval_time,
    generate_mean_wind,
    insert_event_times,
    sigma_components,
)

logger = logging.getLogger(__name__)

# Mercator projection: lon/lat (degrees) ↔ x/y (metres)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
reference_geode = Geod(ellps="WGS84")

dask.config.set({"array.chunk-size": "256 MiB"})


def _convert_pressure_velocity(
        w_vals: np.ndarray,
        w_units: str,
        z_axis: np.ndarray,
        z_idx: int,
        dims: tuple,
        full_ds: xr.Dataset = None,
) -> np.ndarray:
    """
    Convert vertical velocity from pressure velocity (hPa/s, Pa/s, etc.) to physical velocity (m/s).
    Uses the density variable if present in full_ds, otherwise falls back to US Standard Atmosphere 1976.
    """
    w_units = str(w_units).strip().lower()
    if not any(p_unit in w_units for p_unit in ["hpa/s", "pa/s", "mb/s", "mbar/s"]):
        return w_vals

    g = 9.80665  # standard gravity m/s^2
    scale_factor = 100.0 if any(u in w_units for u in ["hpa", "mb", "mbar"]) else 1.0

    # Scan full_ds for a density variable
    density_var = None
    if full_ds is not None:
        candidates = ["rho", "density", "dens", "air_density"]
        for var_name in full_ds.data_vars:
            if var_name.lower() in candidates:
                density_var = var_name
                break

    if density_var is not None:
        print(f"Using actual density variable '{density_var}' from dataset for conversion.")
        rho_ds = full_ds[density_var]

        # If density has same dimensions as w, we can align and use it directly
        if list(rho_ds.dims) == list(dims):
            rho_vals = rho_ds.values.astype(w_vals.dtype)
            conv_factors = -scale_factor / (rho_vals * g)
            return w_vals * conv_factors

        # If density is 1D along the z axis
        z_dim_name = dims[z_idx]
        if list(rho_ds.dims) == [z_dim_name]:
            rho_z = rho_ds.values.astype(np.float64)
            conv_factors = -scale_factor / (rho_z * g)
            broadcast_shape = [1] * w_vals.ndim
            broadcast_shape[z_idx] = len(z_axis)
            return w_vals * conv_factors.reshape(broadcast_shape)

        print(
            f"Density variable '{density_var}' found but shape {rho_ds.shape} is incompatible. Falling back to US Standard Atmosphere.")

    # Fallback: US Standard Atmosphere 1976 density profile from 0 to 150 km
    print("Detected vertical velocity units (pressure velocity).")
    print(
        "Converting vertical velocity to m/s using US Standard Atmosphere 1976 density profile...")
    z_ref = np.arange(0, 160000, 10000, dtype=np.float64)
    rho_ref = np.array([
        1.22500,  # 0 km
        4.12707e-1,  # 10 km
        8.89100e-2,  # 20 km
        1.84100e-2,  # 30 km
        3.99570e-3,  # 40 km
        1.02690e-3,  # 50 km
        3.09680e-4,  # 60 km
        8.28300e-5,  # 70 km
        1.84570e-5,  # 80 km
        3.25100e-6,  # 90 km
        5.60400e-7,  # 100 km
        9.70800e-8,  # 110 km
        2.22200e-8,  # 120 km
        8.15200e-9,  # 130 km
        3.85200e-9,  # 140 km
        2.07000e-9,  # 150 km
    ], dtype=np.float64)

    log_rho_ref = np.log(rho_ref)
    log_rho_z = np.interp(z_axis, z_ref, log_rho_ref, left=log_rho_ref[0], right=log_rho_ref[-1])
    rho_z = np.exp(log_rho_z)

    conv_factors = -scale_factor / (rho_z * g)
    broadcast_shape = [1] * w_vals.ndim
    broadcast_shape[z_idx] = len(z_axis)
    return w_vals * conv_factors.reshape(broadcast_shape)


class LagrangianTrajectories:
    """
    Lagrangian ensemble trajectory calculator.

    Parameters
    ----------
    data : xr.Dataset
        Wind dataset containing ``u``, ``v``, ``w`` components.
        Coordinate names and units are detected automatically via CF conventions
        and normalised before use.
    timestep : str or float, optional
        Time step string (e.g. ``"10 min"``) or seconds. Inferred from data
        if omitted.
    integration_method : str
        scipy ``solve_ivp`` method: ``"RK23"`` (default), ``"RK45"``,
        ``"DOP853"``, ``"LSODA"``.
    start_time : str, optional
        ISO-8601 start time (``"YYYY-MM-DDTHH:MM:SS"``). Defaults to first
        timestamp in the dataset.
    interpolation_method : str
        Interpolation method for ``RegularGridInterpolator``:
        ``"linear"`` (default) or ``"nearest"``.
    noise_type : str or None
        Stochastic noise model: ``"lognormal"`` (default), ``"gaussian"``,
        or ``None``.
    verbose_level : int
        Verbosity (0 = silent).
    time_lag : str or None
        Time offset applied to the wind-data time axis (e.g. ``"50 min"``).
    rtol : float
        Relative tolerance for the ODE solver (default ``2e-5``).
    atol : float
        Absolute tolerance in metres for the ODE solver (default ``5``).
    """

    def __init__(
            self,
            data: xr.Dataset,
            timestep=None,
            integration_method: str = "RK23",
            start_time: str | None = None,
            interpolation_method: str = "linear",
            noise_type: str | None = None,
            verbose_level: int | None = None,
            time_lag: str | None = None,
            rtol: float = 2e-5,
            atol: float = 5.0,
    ):
        self.verbose = verbose_level or 0

        # ------------------------------------------------------------------
        # 1. Detect and curate coordinates
        # ------------------------------------------------------------------
        data, self._cn = detect_and_curate(data)

        self.components = ["u", "v", "w"]
        full_ds = data
        wind_ds = data[self.components]
        del data

        self.time_lag = time_lag
        if time_lag is not None:
            wind_ds = wind_ds.assign_coords(
                time=wind_ds.time - pd.Timedelta(self.time_lag)
            )

        time_vals = wind_ds.time.values

        if timestep is None:
            timestep = convert_to_seconds(np.median(np.diff(time_vals)))

        self.timestep = timestep
        self.method = integration_method
        self.interp_method = interpolation_method
        self.rtol = rtol
        self.atol = atol

        if start_time is None:
            self.start_time = time_vals[0]
        else:
            self.start_time = pd.to_datetime(parser.parse(start_time)).to_numpy()

        self.rel_time = convert_to_seconds(time_vals - self.start_time)

        # ------------------------------------------------------------------
        # 2. Build fast RegularGridInterpolator from the wind dataset
        # ------------------------------------------------------------------
        # Load & sort the wind data along each axis so the grid is regular.
        wind_ds = wind_ds.sortby(["time", "lat", "lon", "z"])

        # Convert time to float seconds (relative to start_time) for the
        # interpolator axis.
        t_axis = self.rel_time  # already sorted

        lat_axis = wind_ds.lat.values.astype(np.float64)
        lon_axis = wind_ds.lon.values.astype(np.float64)
        z_axis = wind_ds.z.values.astype(np.float64)

        # Convert lat/lon axes to Mercator metres for the interpolator so that
        # during integration we work entirely in metres.
        x_axis, _ = transformer.transform(lon_axis, np.zeros_like(lon_axis))
        _, y_axis = transformer.transform(np.zeros_like(lat_axis), lat_axis)

        # Load wind values into memory: shape (nt, nz, nlat, nlon) → reorder to (nt, nlat, nlon, nz)
        # scipy RGI expects axes in the same order as the points argument.
        # We use axis order: (t, y, x, z)
        u_vals = wind_ds.u.values.astype(np.float32)  # (nt, nz, nlat, nlon)
        v_vals = wind_ds.v.values.astype(np.float32)
        w_vals = wind_ds.w.values.astype(np.float32)

        # Determine dimension order from the dataset
        dims = wind_ds.u.dims  # e.g. ('time', 'z', 'lat', 'lon')
        t_idx = dims.index("time")
        z_idx = dims.index("z")
        lat_idx = dims.index("lat")
        lon_idx = dims.index("lon")

        # Convert vertical velocity from pressure velocity (hPa/s or Pa/s) to m/s
        w_units = str(wind_ds.w.attrs.get("units", "m/s")).strip().lower()
        w_vals = _convert_pressure_velocity(
            w_vals, w_units, z_axis, z_idx, dims, full_ds=full_ds
        )
        del full_ds

        # Transpose to (time, lat, lon, z) for the interpolator
        order = [t_idx, lat_idx, lon_idx, z_idx]
        u_vals = np.transpose(u_vals, order)
        v_vals = np.transpose(v_vals, order)
        w_vals = np.transpose(w_vals, order)

        rgi_kw = dict(
            points=(t_axis, y_axis, x_axis, z_axis),
            method=interpolation_method,
            bounds_error=False,
            fill_value=np.nan,  # returns NaN for out-of-domain points
        )
        self._rgi_u = RegularGridInterpolator(values=u_vals, **rgi_kw)
        self._rgi_v = RegularGridInterpolator(values=v_vals, **rgi_kw)
        self._rgi_w = RegularGridInterpolator(values=w_vals, **rgi_kw)

        # Store the original (geographic degree) axes for reference
        self._lat_axis = lat_axis
        self._lon_axis = lon_axis
        self._z_axis = z_axis

        # Store the Mercator axes for external use
        self._x_axis = x_axis
        self._y_axis = y_axis

        print(f"Wind grid: nt={len(t_axis)}, ny={len(y_axis)}, nx={len(x_axis)}, nz={len(z_axis)}")

        # ------------------------------------------------------------------
        # 3. Noise model – pre-compute profiles on the vertical grid
        # ------------------------------------------------------------------
        self.noise_type = noise_type or "lognormal"
        self.state_size = len(self.components)

        z_coord = self._cn.z  # always 'z' after curation

        # sigma_components returns an xr.Dataset indexed by z_coord
        sigma_ds = sigma_components(z_coord=z_coord)
        # Interpolate onto the actual vertical grid (metres)
        sigma_on_grid = sigma_ds.interp(
            {z_coord: z_axis}, method="linear",
            kwargs={"fill_value": "extrapolate"}
        )
        # Shape (3, nz) – rows: u, v, w
        self._sigma_arr = np.stack([
            sigma_on_grid["u"].values,
            sigma_on_grid["v"].values,
            sigma_on_grid["w"].values,
        ]).astype(np.float64)  # (3, nz)

        if self.noise_type == "lognormal":
            wind_subset = xr.Dataset(
                {
                    "u": xr.DataArray(u_vals, dims=("time", "lat", "lon", "z"),
                                      coords={"z": z_axis, "lat": lat_axis, "lon": lon_axis}),
                    "v": xr.DataArray(v_vals, dims=("time", "lat", "lon", "z"),
                                      coords={"z": z_axis, "lat": lat_axis, "lon": lon_axis}),
                    "w": xr.DataArray(w_vals, dims=("time", "lat", "lon", "z"),
                                      coords={"z": z_axis, "lat": lat_axis, "lon": lon_axis}),
                }
            )
            mean_speed_da = generate_mean_wind(wind_subset, z_coord="z")
            # Pre-interpolate mean speed onto z_axis for fast noise calls
            self._mean_speed_arr = np.interp(
                z_axis,
                mean_speed_da.z.values,
                mean_speed_da.values,
            ).clip(1e-6)  # (nz,) – avoid division by zero

    # ------------------------------------------------------------------
    # Noise generators (now use pre-computed numpy arrays)
    # ------------------------------------------------------------------

    def _sigma_at_z(self, z: np.ndarray) -> np.ndarray:
        """
        Interpolate the (3, nz) sigma array to arbitrary altitudes.

        Parameters
        ----------
        z : (N,) altitudes in metres

        Returns
        -------
        (3, N) sigma values
        """
        return np.stack([
            np.interp(z, self._z_axis, self._sigma_arr[i],
                      left=self._sigma_arr[i, 0], right=self._sigma_arr[i, -1])
            for i in range(3)
        ])

    def lognormal_noise_generator(self, z: np.ndarray,
                                  seed: int | None = None,
                                  max_sigma: float = 0.5) -> np.ndarray:
        """
        Altitude-dependent lognormal noise for (u, v, w).

        Returns
        -------
        (3, len(z)) multiplicative noise factors (mean ≈ 1 in linear space).
        """
        rng = np.random.default_rng(seed)

        sigma_vals = self._sigma_at_z(z)  # (3, N)
        mean_speed = np.interp(z, self._z_axis, self._mean_speed_arr,
                               left=self._mean_speed_arr[0],
                               right=self._mean_speed_arr[-1])

        sigma_rel = (sigma_vals / mean_speed[np.newaxis, :]).clip(0, max_sigma)
        mu = -0.5 * sigma_rel ** 2
        return rng.lognormal(mean=mu, sigma=sigma_rel, size=sigma_rel.shape)

    def gaussian_noise_generator(self, z: np.ndarray,
                                 seed: int | None = None,
                                 max_sigma: float = 25.0) -> np.ndarray:
        """
        Altitude-dependent Gaussian noise for (u, v, w).

        Returns
        -------
        (3, len(z)) additive noise in wind units.
        """
        rng = np.random.default_rng(seed)
        sigma_arr = self._sigma_at_z(z).clip(0, max_sigma)
        return rng.normal(loc=0.0, scale=sigma_arr, size=sigma_arr.shape)

    def _apply_noise(self, wind_vector: np.ndarray,
                     z: np.ndarray, seed: int) -> np.ndarray:
        """Apply stochastic perturbation to the wind vector."""
        if seed == 0 or seed is False:
            return wind_vector
        if self.noise_type == "lognormal":
            return wind_vector / self.lognormal_noise_generator(z, seed=seed, max_sigma=1.5)
        if self.noise_type == "gaussian":
            return wind_vector + self.gaussian_noise_generator(z, seed=seed, max_sigma=25.0)
        raise ValueError(f"Unknown noise type: {self.noise_type!r}")

    # ------------------------------------------------------------------
    # ODE right-hand side (hot path)
    # ------------------------------------------------------------------

    def velocity_vectorized(self, time: float, state: np.ndarray,
                            noise_scale=False) -> np.ndarray:
        """
        Wind velocity at all particle positions (called by ``solve_ivp``).

        Parameters
        ----------
        time : float
            Seconds since ``self.start_time``.
        state : (num_particles * 3,) ndarray
            Flat array of [x0, y0, z0, x1, y1, z1, …] in metres.
        noise_scale : int or False
            Non-zero → use as RNG seed for the noise generator.

        Returns
        -------
        (num_particles * 3,) ndarray
        """
        x, y, z = np.asarray(state).reshape(self.state_size, -1)

        # Query points for the RGI: shape (N, 4) – axes are (t, y, x, z)
        pts = np.column_stack([
            np.full(x.size, time),
            y,
            x,
            z,
        ])

        u = self._rgi_u(pts)
        v = self._rgi_v(pts)
        w = self._rgi_w(pts)

        # fill_value=None means out-of-domain points return NaN via extrapolation;
        # clamp to zero so solve_ivp never sees NaN and particles freeze at the boundary.
        wind_vector = np.nan_to_num(np.stack([u, v, w]), nan=0.0)  # (3, N)
        wind_vector = self._apply_noise(wind_vector, z, seed=noise_scale)

        return wind_vector.reshape(-1)

    # ------------------------------------------------------------------
    # Intersection event
    # ------------------------------------------------------------------

    def intersection_event(self, time: float, state: np.ndarray,
                           target=None,
                           horizontal_tolerance: float = 1e3,
                           vertical_tolerance: float = 1e3,
                           event_log: list | None = None,
                           transformer=None,
                           geode=None) -> float:
        """
        Scalar event function for ``solve_ivp``: negative while outside tolerance
        ellipsoid, zero/positive at intersection.

        Parameters
        ----------
        event_log : list, optional
            Per-member list to which intersection info dicts are appended.
            Using a list argument (rather than ``self._last_event_info``) makes
            this method thread-safe when multiple ensemble members run concurrently.
        """
        if horizontal_tolerance is None:
            horizontal_tolerance = 5e3
        if target is None:
            return horizontal_tolerance + 1.0

        timestamp = self.start_time + pd.to_timedelta(time, unit="s")
        x, y, z = np.asarray(state).reshape(self.state_size, -1)

        if isinstance(target, dict):
            t_target = target["time"]
            t_query = np.datetime64(timestamp)
            idx = np.argmin(np.abs(t_target - t_query))
            lon0 = target["lon"][idx]
            lat0 = target["lat"][idx]
            z_target_km = target["z"][idx]
        else:
            target_point = target.sel(time=timestamp, method="nearest")
            lon0, lat0 = target_point.lon.values, target_point.lat.values
            z_target_km = target_point.z.values

        # Use thread-local/passed pyproj objects if available, else fall back to globals
        tx = transformer if transformer is not None else globals().get("transformer")
        gd = geode if geode is not None else globals().get("reference_geode")

        lons, lats = tx.transform(x.flatten(), y.flatten(), direction="INVERSE")
        _, _, horiz_distances = gd.inv(
            lons, lats,
            np.full_like(lons, lon0),
            np.full_like(lats, lat0),
        )

        vert_distances = np.abs(z.flatten() - z_target_km * 1e3)

        combined_metric = (
                (horiz_distances / horizontal_tolerance) ** 2
                + (vert_distances / vertical_tolerance) ** 2
        )
        particle_id = int(np.argmin(combined_metric))
        event_value = combined_metric[particle_id] - 1.0

        if event_value <= 0.0:
            info = {
                "particle": particle_id,
                "time": timestamp.isoformat(),
                "horizontal_distance_km": float(horiz_distances[particle_id]) / 1e3,
                "vertical_distance_km": float(vert_distances[particle_id]) / 1e3,
            }
            if event_log is not None:
                event_log.append(info)
            logger.info(
                "Intersection event: %s",
                ", ".join(f"{k}={v}" for k, v in info.items()),
            )
            if self.verbose:
                print("Intersection event: " +
                      ", ".join(f"{k}={v}" for k, v in info.items()))

        return event_value

    # ------------------------------------------------------------------
    # Main integration
    # ------------------------------------------------------------------

    def advect_particles(
            self,
            start_positions,
            duration=None,
            end_date: str | None = None,
            ensemble_size: int | None = None,
            target=None,
            distance_tolerance: float | None = None,
            n_jobs: int = -1,
    ) -> xr.Dataset:
        """
        Compute trajectories for a set of particles.

        Parameters
        ----------
        start_positions : list of (lon, lat, z_m) tuples
        duration : str or float, optional
        end_date : str, optional
        ensemble_size : int, optional
        target : xr.Dataset, optional
        distance_tolerance : float, optional
        n_jobs : int
            Number of parallel workers (default ``-1`` = all CPUs).
            Set to ``1`` for sequential execution (useful in tests or when
            the overhead of spawning workers exceeds the integration cost).
        """
        if end_date is not None:
            if self.verbose:
                print(f"Using end_date={end_date}. Ignoring duration.")
            end_dt = pd.to_datetime(parser.parse(end_date))
            duration = end_dt - pd.to_datetime(self.start_time)

        times = generate_eval_time(duration, self.timestep)
        rel_time = self.rel_time.astype(float)
        times = times[(times >= np.min(rel_time)) & (times <= np.max(rel_time))]
        time_span = (times[0], times[-1])

        initial_positions_m = np.array([
            (*transformer.transform(lon, lat), alt)
            for lon, lat, alt in start_positions
        ]).T.reshape(-1)

        if ensemble_size is None:
            ensemble_size = 1

        num_particles, state_size = np.shape(start_positions)[:2]

        # Pre-extract target dataset coordinates and variables into a dict of
        # numpy arrays to avoid thread-safety and GIL issues in the thread pool.
        target_data = None
        if target is not None:
            target_data = {
                "time": target.time.values,
                "lon": target.lon.values,
                "lat": target.lat.values,
                "z": target.z.values,
            }

        def integrate_member(member: int):
            import time as pytime
            t_start = pytime.time()
            if self.verbose >= 2:
                print(
                    f"Integrating {num_particles} particles "
                    f"for member {member} over {duration}"
                )

            # Per-member accumulator – no shared mutable state between threads.
            event_log: list[dict] = []

            # Create thread-local pyproj objects to avoid contention and lock wait times
            local_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            local_geode = Geod(ellps="WGS84")

            def _event(t, y):
                return self.intersection_event(
                    t, y, target=target_data,
                    horizontal_tolerance=distance_tolerance or 5e3,
                    vertical_tolerance=2e3,
                    event_log=event_log,
                    transformer=local_transformer,
                    geode=local_geode,
                )

            _event.terminal = False
            _event.direction = -1.0

            velocity_func = partial(self.velocity_vectorized, noise_scale=member)

            result = solve_ivp(
                velocity_func,
                t_span=time_span,
                y0=initial_positions_m,
                t_eval=None,
                method=self.method,
                events=_event,
                rtol=self.rtol,
                atol=self.atol,
                dense_output=True,
            )

            valid_times = insert_event_times(times, result)
            traj = result.sol(valid_times)
            traj = traj.reshape(state_size, num_particles, -1).transpose(0, 2, 1)

            event_info = None
            if event_log:
                info = event_log[-1]
                event_info = (
                    f"Intersection event: particle={info['particle']}, member={member}, "
                    f"time={info['time']}, "
                    f"dx={info['horizontal_distance_km']:.2f} km, "
                    f"dz={info['vertical_distance_km']:.2f} km"
                )
                if self.verbose >= 2:
                    print("  " + event_info)

            if self.verbose >= 2:
                print(
                    f"Member {member} finished in {pytime.time() - t_start:.2f} seconds. nfev={result.nfev}")

            return member, valid_times, traj, event_info

        # Show a tqdm progress bar if verbose >= 1 and we have multiple ensemble members
        show_progress = (self.verbose >= 1) and (ensemble_size > 1)

        if n_jobs == 1:
            if show_progress:
                from tqdm import tqdm
                results = [integrate_member(m) for m in
                           tqdm(range(ensemble_size), desc="Integrating ensemble")]
            else:
                results = [integrate_member(m) for m in range(ensemble_size)]
        else:
            # ThreadPoolExecutor: lightweight, same GIL semantics as Joblib threads.
            # NumPy / SciPy release the GIL for the hot-path RGI and ODE work,
            # so threads genuinely overlap without pickling the large RGI objects.
            workers = n_jobs if n_jobs > 0 else None
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(integrate_member, m): m
                           for m in range(ensemble_size)}
                if show_progress:
                    from tqdm import tqdm
                    results = []
                    for f in tqdm(as_completed(futures), total=len(futures),
                                  desc="Integrating ensemble"):
                        results.append(f.result())
                else:
                    results = [f.result() for f in as_completed(futures)]
            # Restore order (as_completed yields in completion order)
            results.sort(key=lambda r: r[0])

        valid_times = max((r[1] for r in results), key=len)
        trajectories_arr = np.full(
            (state_size, len(valid_times), num_particles, ensemble_size), np.nan
        )
        event_metadata = []

        for j, member_times, traj, info in results:
            n_steps = member_times.size
            pad_len = len(valid_times) - n_steps
            if pad_len > 0:
                pad = np.repeat(traj[:, -1:, :], pad_len, axis=1)
                traj = np.concatenate([traj, pad], axis=1)
            trajectories_arr[..., :, j] = traj
            if info:
                event_metadata.append(info)

        lon, lat = transformer.transform(*trajectories_arr[:2], direction="INVERSE")
        z_km = 1e-3 * trajectories_arr[2]

        valid_times_ts = pd.to_datetime(valid_times, origin=self.start_time, unit="s")

        ds = xr.Dataset(
            {
                "lon": (("time", "particle", "ensemble"), lon),
                "lat": (("time", "particle", "ensemble"), lat),
                "z": (("time", "particle", "ensemble"), z_km),
            },
            coords={
                "particle": np.arange(num_particles),
                "ensemble": np.arange(ensemble_size),
                "time": valid_times_ts,
            },
        )
        ds["z"].attrs.update({"standard_name": "geometric_height", "units": "km"})
        ds.attrs.update({
            "description": "Lagrangian ensemble trajectories.",
            "start_time": str(self.start_time),
            "end_time": str(valid_times_ts[-1]),
            "duration": str(duration),
            "initial_longitude": [round(p[0], 2) for p in start_positions],
            "initial_latitude": [round(p[1], 2) for p in start_positions],
            "initial_altitudes": [round(p[2], 2) for p in start_positions],
            "time_step": self.timestep or "60s",
            "time_lag": self.time_lag or "0h",
            "solver_method": self.method,
            "interpolation_method": self.interp_method,
            "noise_type": self.noise_type,
            "rtol": self.rtol,
            "atol": self.atol,
            "intersection_events": len(event_metadata),
        })
        if event_metadata:
            ds.attrs["intersection_details"] = ";\n".join(event_metadata)

        return ds
