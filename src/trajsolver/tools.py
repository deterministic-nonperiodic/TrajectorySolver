import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod
from scipy.interpolate import CubicSpline, interp1d


def save_cf_compliant(ds: xr.Dataset, path: str):
    """Save with CF-compliant time encoding compatible with CDO."""
    time_origin = pd.to_datetime(ds.time.values[0])
    origin_str = time_origin.strftime("%Y-%m-%d %H:%M:%S")

    encoding = {var: {"zlib": True} for var in ds.data_vars}
    encoding["time"] = {
        "units": f"seconds since {origin_str}",
        "calendar": "proleptic_gregorian",
        "dtype": "float64"
    }

    print(f"Saving to {path}...")
    ds.to_netcdf(path, encoding=encoding)
    print("Done.")


def sigma_component(component="u"):
    """
    Create a function to vertically interpolate the standard deviation of wind components
    :param component: string, one of ['u', 'v', 'w']
    :return: interpolation functions for the specified wind component
    """

    # Define altitude and sigma values for wind components
    altitude = 1e3 * np.array([80.496086, 81.70819, 82.933266, 84.171394, 85.42264, 86.68708,
                               87.96479, 89.255844, 90.56032, 91.87829, 93.20984, 94.55504,
                               95.91399, 97.28675, 98.673416, 100.07407])

    sigma = {
        "u": np.array([21.072638, 19.778282, 18.733944, 17.160027, 15.1502495, 13.419383,
                       13.337099, 15.494116, 18.56262, 22.313557, 26.949926, 31.411144,
                       35.637962, 38.088654, 39.60116, 40.94975]),
        "v": np.array([19.0583, 19.531872, 19.139727, 17.517744, 15.989676, 16.746874,
                       19.3816, 22.937468, 27.505886, 31.952532, 35.201454, 37.258953,
                       39.073856, 40.34578, 39.15983, 37.215282]),
        "w": np.array([0.28650776, 0.29934645, 0.3305828, 0.3605683, 0.37941748, 0.3868968,
                       0.39157304, 0.39810547, 0.41244808, 0.44748694, 0.4843474, 0.511466,
                       0.5366538, 0.5540057, 0.555168, 0.52239084])
    }
    # create interpolation functions
    sigma_interp = interp1d(
        altitude, sigma[component], kind='linear',
        fill_value=(sigma[component][0], sigma[component][-1]), bounds_error=False
    )

    return sigma_interp


def sigma_components(z_coord: str = 'z_mc'):
    """
    Create a dataset with vertical profiles of standard deviation for u, v, and w wind components.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions:
            - 'z_mc': altitude in meters
        Variables:
            - 'u', 'v', 'w': standard deviations (m/s)
    """
    # Altitudes in meters
    altitude = 1e3 * np.array([
        80.496086, 81.70819, 82.933266, 84.171394, 85.42264, 86.68708,
        87.96479, 89.255844, 90.56032, 91.87829, 93.20984, 94.55504,
        95.91399, 97.28675, 98.673416, 100.07407
    ])

    sigma_data = {
        "u": np.array([21.072638, 19.778282, 18.733944, 17.160027, 15.1502495, 13.419383,
                       13.337099, 15.494116, 18.56262, 22.313557, 26.949926, 31.411144,
                       35.637962, 38.088654, 39.60116, 40.94975]),
        "v": np.array([19.0583, 19.531872, 19.139727, 17.517744, 15.989676, 16.746874,
                       19.3816, 22.937468, 27.505886, 31.952532, 35.201454, 37.258953,
                       39.073856, 40.34578, 39.15983, 37.215282]),
        "w": np.array([0.28650776, 0.29934645, 0.3305828, 0.3605683, 0.37941748, 0.3868968,
                       0.39157304, 0.39810547, 0.41244808, 0.44748694, 0.4843474, 0.511466,
                       0.5366538, 0.5540057, 0.555168, 0.52239084])
    }

    sigma_data = xr.Dataset({
        comp: xr.DataArray(data, dims=[z_coord], coords={z_coord: altitude})
        for comp, data in sigma_data.items()
    })

    return sigma_data


def generate_ou_noise_field(z_vals, times, tau=600, sigma_fn=sigma_component, seed=None):
    """
    Generate a temporally correlated OU noise field on a (time, altitude) grid.

    Parameters:
        z_vals : 1D np.ndarray
            Altitudes (in meters).
        times : 1D np.ndarray
            Time values (in seconds).
        tau : float
            Relaxation time-scale (in seconds).
        sigma_fn : callable
            Function that returns standard deviation given altitude.
        seed : int
            RNG seed.

    Returns:
        xr.Dataset with variables 'u_noise', 'v_noise', 'w_noise' on dimensions (time, z_mc)
    """
    rng = np.random.default_rng(seed)
    dt = np.diff(times).min()
    n_time = len(times)
    n_z = len(z_vals)

    # Shape: (3, n_time, n_z)
    noise = np.zeros((3, n_time, n_z))

    if sigma_fn is None:
        raise ValueError("sigma_fn must be provided.")

    sigma_vals = np.stack([sigma_fn(c)(z_vals) for c in ['u', 'v', 'w']])  # shape: (3, n_z)

    # Initialize with zero
    for i in range(1, n_time):
        decay = np.exp(-dt / tau)
        noise[:, i] = decay * noise[:, i - 1] + \
                      np.sqrt(1 - decay ** 2) * rng.normal(loc=0, scale=sigma_vals, size=(3, n_z))

    # Package into xarray
    return xr.Dataset({
        'u': (['time', 'z_mc'], noise[0]),
        'v': (['time', 'z_mc'], noise[1]),
        'w': (['time', 'z_mc'], noise[2]),
    }, coords={'time': times, 'z_mc': z_vals})


def generate_mean_wind(wind: xr.Dataset, z_coord: str = 'z_mc') -> xr.DataArray:
    """
    Compute the vertical mean wind speed profile, weighted by cosine(latitude),
    and averaged over time and horizontal space.

    Parameters
    ----------
    wind : xarray.Dataset
        Dataset with variables 'u', 'v', 'w' and coordinates including 'lat', 'lon',
        'time', and the vertical coordinate given by *z_coord*.
    z_coord : str
        Name of the vertical coordinate dimension (default: 'z_mc').

    Returns
    -------
    xr.DataArray
        1D array of mean wind speed vs. vertical level (*z_coord*).
    """
    print("Generating mean wind speed profile for noise scaling...")

    # Determine horizontal dim names (they are always 'lat' / 'lon' after curation)
    lat_dim = 'lat' if 'lat' in wind.dims else None
    lon_dim = 'lon' if 'lon' in wind.dims else None

    # Efficient chunking: keep horizontal dims contiguous
    chunk_map = {}
    for dim in wind.dims:
        if dim in (lat_dim, lon_dim):
            chunk_map[dim] = -1
        else:
            chunk_map[dim] = "auto"
    wind = wind.chunk(chunk_map)

    if lon_dim and lat_dim:
        # Use .where() instead of .sel(slice(...)) so we are robust to
        # non-monotonic indices (e.g. after longitude wrapping) and to datasets
        # that only partially overlap the target region.
        lon_mask = (wind[lon_dim] >= -12) & (wind[lon_dim] <= 12)
        lat_mask = (wind[lat_dim] >= 45) & (wind[lat_dim] <= 65)
        wind = wind.where(lon_mask & lat_mask, drop=True)

    # Compute magnitude of wind vector: √(u² + v² + w²)
    wind_speed = np.sqrt(wind.u ** 2 + wind.v ** 2 + wind.w ** 2)

    # Weight by cosine latitude (broadcast safely)
    if lat_dim:
        weights = np.cos(np.deg2rad(wind[lat_dim]))
        weighted_speed = wind_speed.weighted(weights)
        mean_dims = [d for d in ['time', lat_dim, lon_dim] if d and d in wind.dims]
    else:
        weighted_speed = wind_speed
        mean_dims = [d for d in ['time'] if d in wind.dims]

    # Mean over time and horizontal dims, preserving the vertical profile
    mean_profile = weighted_speed.mean(dim=mean_dims)

    return mean_profile.compute()


def insert_event_times(times, result, rtol=1e-9, atol=0.0):
    """
    Filters `times` to lie within the integration window of `result.t` and inserts
    unique event times from `result.t_events` if not already present.

    Returns
    -------
    valid_times : ndarray
        Times within the integration bounds of `result`, including unique event times.
        The original order of `times` is preserved (ascending or descending).
    """
    times = np.asarray(times)
    t_start, t_stop = result.t[0], result.t[-1]
    t_min, t_max = min(t_start, t_stop), max(t_start, t_stop)

    # Filter times within the integration window
    valid_times = times[(times >= t_min) & (times <= t_max)]

    is_descending = times[0] > times[-1]

    if hasattr(result, "t_events") and result.t_events:
        event_lists = [ev for ev in result.t_events if ev is not None and ev.size > 0]
        if event_lists:
            all_event_times = np.concatenate(event_lists)
            event_times_in_range = all_event_times[
                (all_event_times >= t_min) & (all_event_times <= t_max)
                ]

            for t_event in np.unique(event_times_in_range):
                if not np.any(np.isclose(valid_times, t_event, rtol=rtol, atol=atol)):
                    # Compute insert index depending on time order
                    if is_descending:
                        insert_idx = np.searchsorted(-valid_times, -t_event, side="left")
                    else:
                        insert_idx = np.searchsorted(valid_times, t_event, side="left")

                    valid_times = np.insert(valid_times, insert_idx, t_event)

    return valid_times


def read_falcon(filename):
    orbit_df = pd.read_csv(
        filename,
        comment="#",  # skip all header lines starting with #
        sep='\s+',  # auto-split by whitespace
        header=None,  # no header in the data row,
        names=[
            "timestamp", "dt_min", "IFA", "F10.7", "FB10.7", "Ap", "IDW", "Rho",
            "Vn", "Ve", "MMWT", "Tloc", "Texo", "LST", "GLat", "GLon", "GAlt",
            "Va", "gamma", "gload", "qdot", "SRat", "TRat", "KnInf", "MaInf",
            "CD", "CD_CD0", "Orb", "ULat", "dS", "Hpe", "Hap", "H", "Torb",
            "mjd1950"
        ]
    )

    orbit_df["timestamp"] = pd.to_datetime(orbit_df["timestamp"])
    orbit_df = orbit_df.drop_duplicates(subset="timestamp")

    return orbit_df


def sample_orbit_positions(
        orbit_df: pd.DataFrame,
        n: int,
        alt_min: float = 70.0,
        lon_min: float = -180.0,
        lon_max: float = 180.0,
        seed=None,
) -> list[tuple[float, float, float]]:
    """Sample n starting positions uniformly along a filtered orbit segment.

    Fits a cubic spline through the filtered knots, parameterised by cumulative
    3-D arc-length, then draws n random positions uniformly along the arc.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Orbit data from read_falcon.  Requires columns GLon, GLat, GAlt (km).
    n : int
        Number of positions to generate.
    alt_min : float
        Lower altitude bound [km].
    lon_min, lon_max : float
        Longitude window [degrees].
    seed : int or None
        RNG seed.

    Returns list of (lon, lat, z_m) tuples where z_m is altitude in metres.
    Raises ValueError if fewer than 2 orbit rows survive the filter.
    """
    seg = orbit_df[
        (orbit_df["GAlt"] >= alt_min)
        & (orbit_df["GLon"] >= lon_min)
        & (orbit_df["GLon"] <= lon_max)
    ].sort_values("GAlt", ascending=False).reset_index(drop=True)

    if len(seg) < 2:
        raise ValueError(
            f"sample_orbit_positions: only {len(seg)} orbit row(s) survive the "
            f"filter (alt_min={alt_min}, lon_min={lon_min}, lon_max={lon_max})."
        )

    dlat = np.diff(seg["GLat"].values)
    dlon = np.diff(seg["GLon"].values)
    dalt = np.diff(seg["GAlt"].values)
    ds = np.sqrt(dlat ** 2 + dlon ** 2 + dalt ** 2)
    t_knots = np.concatenate([[0.0], np.cumsum(ds)])
    t_knots /= t_knots[-1]

    cs_lon = CubicSpline(t_knots, seg["GLon"].values)
    cs_lat = CubicSpline(t_knots, seg["GLat"].values)
    cs_alt = CubicSpline(t_knots, seg["GAlt"].values)

    rng = np.random.default_rng(seed)
    t_samples = np.sort(rng.uniform(0.0, 1.0, n))

    return [
        (float(cs_lon(t)), float(cs_lat(t)), 1e3 * float(cs_alt(t)))
        for t in t_samples
    ]


def convert_to_seconds(value, unit='s'):
    if isinstance(value, (int, float)):
        return pd.Timedelta(value, unit=unit).total_seconds()
    elif isinstance(value, str):
        return pd.Timedelta(value).total_seconds()
    elif isinstance(value, pd.Timedelta):
        return value.total_seconds()
    elif np.issubdtype(value.dtype, np.timedelta64):
        return value / np.timedelta64(1, 's')
    else:
        raise ValueError(f"Unknown format for argument '{value}'")


def generate_eval_time(duration, timestep):
    """
    Generate an array of evaluation times starting at 0 and incrementing by a specified timestep.

    :param duration: Total duration (can be a string, int, or float).
    :param timestep: Time step (can be a string, int, or float). It can be negative for backward time progression.
    :return: A numpy array of evaluation times in seconds.
    :raises ValueError: If 'duration' or 'timestep' are in an unknown format.
    """

    # Return empty array if duration is None
    if duration is None:
        return np.zeros(1)

    # Convert both duration and timestep to seconds
    duration = convert_to_seconds(duration)
    timestep = convert_to_seconds(timestep, 's')

    # Compute the number of steps
    num_steps = int(abs(duration) / abs(timestep)) + 1  # Include zero

    direction = np.sign(timestep) * np.sign(duration)

    return np.linspace(0, direction * abs(duration), num_steps)


def compute_intersections(
        trajectories: xr.Dataset,
        orbit_df: pd.DataFrame,
        horiz_tol_km=50,
        vert_tol_km=5,
        time_tol_m=30
):
    """
    Find where particle trajectories intersect with a re-entry trajectory in space and time.

    Parameters:
        trajectories : xarray.Dataset
            Must include 'lon', 'lat', and 'z' with dims (time, particle, ensemble).
        orbit_df : pd.DataFrame
            Must include ['timestamp', 'GLon', 'GLat', 'GAlt'].
        horiz_tol_km : float
            Horizontal tolerance for intersection [km].
        vert_tol_km : float
            Vertical tolerance for intersection [km].
        time_tol_m : float
            Time tolerance in minutes for considering same-time intersections.

    Returns:
        intersect_mask : xarray.DataArray (bool)
            Mask with shape (time, particle, ensemble) showing intersection flags.
        intersect_indices : np.ndarray
            Indices of shape (N, 3) for (time, particle, ensemble) of intersecting points.
    """

    # Normalise orbit: 'timestamp' must be a plain column (plot_orbit_and_ensemble_3d
    # passes the DataFrame after .set_index("timestamp"), making it the index).
    if orbit_df.index.name == 'timestamp':
        orbit_df = orbit_df.reset_index()

    geode = Geod(ellps="WGS84")

    # Flatten coordinates and time
    traj_time = trajectories.time.values
    flat_time = np.repeat(traj_time[:, None, None],
                          repeats=trajectories.particle.size * trajectories.ensemble.size,
                          axis=1).reshape(-1)

    flat_coords = pd.DataFrame({
        'timestamp': pd.to_datetime(flat_time).astype('datetime64[us]'),
        'lon': trajectories.lon.values.ravel(),
        'lat': trajectories.lat.values.ravel(),
        'z': trajectories.z.values.ravel()
    })

    # Align orbit in time
    # Track original index before sorting
    flat_coords["orig_index"] = flat_coords.index
    flat_coords_sorted = flat_coords.sort_values("timestamp").reset_index(drop=True)

    # Merge with orbit – both sides must share the same datetime resolution (pandas ≥ 2.0)
    orbit_df_sorted = orbit_df.sort_values('timestamp').copy()
    orbit_df_sorted['timestamp'] = orbit_df_sorted['timestamp'].astype('datetime64[us]')
    aligned = pd.merge_asof(flat_coords_sorted, orbit_df_sorted,
                            on='timestamp', tolerance=pd.Timedelta(minutes=time_tol_m),
                            direction='nearest')

    # Drop unmatched rows (e.g. where orbit was NaN)
    aligned = aligned.dropna(subset=["GLon", "GLat", "GAlt"])

    # Compute distances
    _, _, dist_m = geode.inv(aligned["lon"].values, aligned["lat"].values,
                             aligned["GLon"].values, aligned["GLat"].values)
    horiz_km = dist_m / 1000.0
    vert_km = np.abs(aligned["z"].values - aligned["GAlt"].values)

    intersect = (horiz_km < horiz_tol_km) & (vert_km < vert_tol_km)

    # Use original indices to construct the mask
    intersect_flat_indices = aligned.loc[intersect, "orig_index"].to_numpy()

    # Build mask
    shape = trajectories.lon.shape
    mask = np.zeros(np.prod(shape), dtype=bool)
    mask[intersect_flat_indices] = True
    intersect_mask = mask.reshape(shape)

    # Final output
    intersect_indices = np.argwhere(intersect_mask)
    intersect_mask = xr.DataArray(intersect_mask,
                                  dims=trajectories.lon.dims,
                                  coords=trajectories.lon.coords)

    # # Get intersecting coordinates instead of indices
    # intersect_coords = intersect_mask.stack(all_idx=("time", "particle", "ensemble"))
    # intersect_coords = intersect_coords[intersect_coords]  # Filter where mask is True

    # # Convert to (N, 3) array of coordinate values
    # intersect_values = np.vstack([
    #     intersect_coords["time"].values,
    #     intersect_coords["particle"].values,
    #     intersect_coords["ensemble"].values
    # ]).T

    return intersect_mask, intersect_indices
