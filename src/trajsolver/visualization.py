import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.mixture import GaussianMixture

from .tools import compute_intersections

# Apply Nature-style theme using seaborn and matplotlib after kernel reset
sns.set_theme(style="whitegrid", font_scale=1.1)

# Update Matplotlib rcParams for Nature-quality styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.6,
    'xtick.major.size': 3,
    'xtick.major.width': 0.6,
    'ytick.major.size': 3,
    'ytick.major.width': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.edgecolor': 'black',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

from matplotlib.collections import LineCollection


def plot_faded_trajectory(ax, lon, lat, times, color='gray', linewidth=0.8,
                          transform=None, alpha_start=0.3, alpha_end=0.01):
    """
    Plot a trajectory with fading transparency using LineCollection.
    """
    from matplotlib.colors import to_rgba

    # Create segments
    points = np.array([lon, lat]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Alpha fades with time
    n = len(segments)
    alphas = np.linspace(alpha_start, alpha_end, n)

    rgba = np.tile(to_rgba(color), (n, 1))
    rgba[:, 3] = alphas

    lc = LineCollection(segments, colors=rgba, linewidths=linewidth, transform=transform)
    ax.add_collection(lc)


def custom_viridis():
    # Get the viridis colormap
    viridis = plt.cm.get_cmap('viridis', 256)

    # Convert to the array of colors
    viridis_colors = viridis(np.linspace(0, 1, 256))

    # Create a new colormap that fades in from white
    white_to_start = 100
    white_to_viridis = np.vstack([
        np.linspace([1, 1, 1, 1], viridis_colors[white_to_start], white_to_start),
        viridis_colors[white_to_start:]  # rest of viridis
    ])

    return mcolors.LinearSegmentedColormap.from_list("custom_viridis", white_to_viridis)


def _target_point_to_orbit_df(target_point):
    """Convert target_point to an orbit-style DataFrame with columns
    [timestamp, GLon, GLat, GAlt].  Returns None for unsupported types.
    """
    if target_point is None:
        return None

    if isinstance(target_point, xr.Dataset):
        times = pd.to_datetime(target_point["time"].values)
        lons = target_point["lon"].values.ravel()
        lats = target_point["lat"].values.ravel()
        alts = target_point["z"].values.ravel()  # km
        return pd.DataFrame({"timestamp": times, "GLon": lons, "GLat": lats, "GAlt": alts})

    if isinstance(target_point, dict):
        row = {
            "timestamp": pd.to_datetime(target_point.get("time", pd.NaT)),
            "GLon": float(target_point.get("lon", target_point.get("GLon", np.nan))),
            "GLat": float(target_point.get("lat", target_point.get("GLat", np.nan))),
            "GAlt": float(target_point.get("z", target_point.get("GAlt", np.nan))),
        }
        return pd.DataFrame([row])

    return None


def _select_nearest_to_target(trajectories, target_point, max_dist_km=None):
    """For each particle, pick the ensemble member closest to target_point
    at the trajectory time step nearest to the target timestamp.

    If max_dist_km is given, particles whose closest member exceeds that
    horizontal distance are excluded from the result.

    Returns dict: particle → (ensemble_index, time_index)
    """
    from pyproj import Geod
    geod = Geod(ellps="WGS84")

    tp_time = pd.to_datetime(target_point["time"].values.ravel()[0])
    tp_lon  = float(target_point["lon"].values.ravel()[0])
    tp_lat  = float(target_point["lat"].values.ravel()[0])

    traj_times_pd = pd.to_datetime(trajectories.time.values)
    t_idx = int(np.abs(traj_times_pd - tp_time).argmin())

    lon_at_t = trajectories.lon.isel(time=t_idx).values   # (particle, ensemble)
    lat_at_t = trajectories.lat.isel(time=t_idx).values
    n_particles, n_ens = lon_at_t.shape

    result = {}
    for pi, p in enumerate(trajectories.particle.values):
        tp_lons = np.full(n_ens, tp_lon)
        tp_lats = np.full(n_ens, tp_lat)
        _, _, dist_m = geod.inv(lon_at_t[pi], lat_at_t[pi], tp_lons, tp_lats)
        best_ens = int(np.nanargmin(dist_m))
        min_dist_km = float(dist_m[best_ens]) / 1000.0
        if max_dist_km is None or min_dist_km <= max_dist_km:
            result[p] = (best_ens, t_idx)

    return result


def generate_representative_particles(trajectories, orbit_df=None, target_point=None,
                                      horiz_tol_km=100, vert_tol_km=15,
                                      max_dist_km=None):
    """Select one representative ensemble member per particle.

    Preference order: target_point proximity > orbit intersection.
    Particles satisfying neither criterion are omitted from the result.

    Parameters
    ----------
    orbit_df : pd.DataFrame or None
        Re-entry orbit [timestamp, GLon, GLat, GAlt].
    target_point : xr.Dataset or None
        Single observation point [time, lon, lat, z].
    horiz_tol_km, vert_tol_km : float
        Thresholds for orbit intersection.
    max_dist_km : float or None
        Maximum distance from target_point [km].  None = no filter.

    Returns dict: particle → (ensemble_index, time_index)
    """
    particles = trajectories.particle.values

    target_hits = {}
    if isinstance(target_point, xr.Dataset):
        target_hits = _select_nearest_to_target(
            trajectories, target_point, max_dist_km=max_dist_km
        )

    orbit_hits = {}
    if orbit_df is not None:
        _, orb_indices = compute_intersections(
            trajectories, orbit_df,
            horiz_tol_km=horiz_tol_km, vert_tol_km=vert_tol_km
        )
        for p in particles:
            mask = trajectories.particle.values[orb_indices[:, 1]] == p
            ens_hits = orb_indices[mask][:, 2]
            time_hits = orb_indices[mask][:, 0]
            if ens_hits.size > 0:
                orbit_hits[p] = (int(ens_hits[0]), int(time_hits[0]))

    selected = {}
    for p in particles:
        if p in target_hits:
            selected[p] = target_hits[p]
        elif p in orbit_hits:
            selected[p] = orbit_hits[p]

    return dict(sorted(selected.items()))


def _marker_stride(lon, lat, ax, min_sep_px=50):
    """Return a time-step stride so markers along a trajectory are separated
    by roughly min_sep_px display pixels.

    Projects the path into display coordinates, computes arc-length, and
    picks the integer stride that places total_px / min_sep_px markers.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    n = len(lon)
    if n < 2:
        return 1

    try:
        data_to_display = ax.transData
        import cartopy.crs as _ccrs
        lonlat_to_disp = _ccrs.PlateCarree()._as_mpl_transform(ax) + data_to_display
        pts_display = lonlat_to_disp.transform(np.column_stack([lon, lat]))
    except Exception:
        pts_display = np.column_stack([lon, lat])

    diffs = np.diff(pts_display, axis=0)
    total_px = np.hypot(diffs[:, 0], diffs[:, 1]).sum()

    if total_px <= 0:
        return max(1, n)

    desired_markers = max(2, total_px / min_sep_px)
    return max(1, int(round((n - 1) / desired_markers)))


def annotate_times_dedup_display(
        ax,
        lons, lats, times,
        min_sep_px=24,
        text_fmt="%Y-%m-%d %H:%M UTC",
        text_offset_pts=(-10, -10),
        fontsize=10,
):
    # Ensure datatypes
    times = pd.to_datetime(times)
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)

    # Transform from PlateCarree data to display (pixels)
    data_to_disp = ccrs.PlateCarree()._as_mpl_transform(ax).transform

    # Track placed positions per label (in display pixels)
    placed = {}  # label -> list of (x_px, y_px)

    for lon, lat, t in zip(lons, lats, times):
        label = pd.to_datetime(t).strftime(text_fmt)
        x_px, y_px = data_to_disp((lon, lat))

        # If we've already placed this label near here, skip
        pts = placed.setdefault(label, [])
        if any((x_px - xp) ** 2 + (y_px - yp) ** 2 < (min_sep_px ** 2) for xp, yp in pts):
            continue

        # Place the annotation (offset in points; arrow to the data point)
        ax.annotate(
            label,
            xy=(lon, lat), xycoords=ccrs.PlateCarree(),
            xytext=text_offset_pts, textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='black', linewidth=1.5),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8),
            fontsize=fontsize, ha='left'
        )

        pts.append((x_px, y_px))


def visualize_trajectories_percentile_kde(trajectories, wind=None, particle_subset=None,
                                          orbit=None, figure_name=None,
                                          representative_members=None,
                                          calculate_intersections=True,
                                          map_extent=None,
                                          target_point=None,
                                          target_label="OSIRIS",
                                          show_wind=True,
                                          max_dist_km=None):
    # Visualizes ensemble back-trajectories with wind streamlines and KDE of endpoints.
    fig, ax = plt.subplots(figsize=(8.42, 5.5), constrained_layout=False,
                           subplot_kw={"projection": ccrs.PlateCarree()})

    mayor_fontsize = 12
    minor_fontsize = 11

    # Parse target point early if provided to include in map bounds
    tp_lon, tp_lat = None, None
    if target_point is not None:
        if hasattr(target_point, "lon"):
            tp_lon = float(target_point["lon"].values[0])
            tp_lat = float(target_point["lat"].values[0])
        elif isinstance(target_point, (tuple, list, np.ndarray)) and len(target_point) >= 2:
            tp_lon = float(target_point[0])
            tp_lat = float(target_point[1])
        elif isinstance(target_point, dict):
            tp_lon = float(target_point.get("lon", target_point.get("GLon", 0)))
            tp_lat = float(target_point.get("lat", target_point.get("GLat", 0)))

    # Determine dynamic map extent based on trajectory and orbit coordinates
    if map_extent is None:
        lons = []
        lats = []

        if tp_lon is not None and tp_lat is not None:
            lons.append(tp_lon)
            lats.append(tp_lat)

        traj_lon = trajectories.lon.values
        traj_lat = trajectories.lat.values
        if np.any(~np.isnan(traj_lon)):
            traj_lon_min = float(np.nanpercentile(traj_lon, 2.5))
            traj_lon_max = float(np.nanpercentile(traj_lon, 97.5))
            lons.extend([traj_lon_min, traj_lon_max])
        else:
            traj_lon_min, traj_lon_max = -30.01, 20.01

        if np.any(~np.isnan(traj_lat)):
            traj_lat_min = float(np.nanpercentile(traj_lat, 2.5))
            traj_lat_max = float(np.nanpercentile(traj_lat, 97.5))
            lats.extend([traj_lat_min, traj_lat_max])
        else:
            traj_lat_min, traj_lat_max = 41.0, 69.0

        if orbit is not None:
            orbit_lon = orbit['GLon'].values
            orbit_lat = orbit['GLat'].values

            # Filter orbit coordinates to a bounding box around the trajectories
            valid_orbit = (
                    (orbit_lon >= traj_lon_min - 20.0) & (orbit_lon <= traj_lon_max + 20.0) &
                    (orbit_lat >= traj_lat_min - 10.0) & (orbit_lat <= traj_lat_max + 10.0)
            )
            filtered_lon = orbit_lon[valid_orbit]
            filtered_lat = orbit_lat[valid_orbit]

            if np.any(~np.isnan(filtered_lon)) and filtered_lon.size > 0:
                lons.extend([np.nanmin(filtered_lon), np.nanmax(filtered_lon)])
            if np.any(~np.isnan(filtered_lat)) and filtered_lat.size > 0:
                lats.extend([np.nanmin(filtered_lat), np.nanmax(filtered_lat)])

        # Default fallback values if no valid coords found
        lon_min = min(lons) if lons else traj_lon_min
        lon_max = max(lons) if lons else traj_lon_max
        lat_min = min(lats) if lats else traj_lat_min
        lat_max = max(lats) if lats else traj_lat_max

        # Add padding to make the plot look professional and unconstrained
        lon_span = lon_max - lon_min
        lat_span = lat_max - lat_min

        lon_padding = max(0.05 * lon_span, 2.0)
        lat_padding = max(0.05 * lat_span, 2.0)

        map_extent = [
            max(lon_min - lon_padding, -180.0),
            min(lon_max + lon_padding, 180.0),
            max(lat_min - lat_padding, -90.0),
            min(lat_max + lat_padding, 90.0)
        ]
    ax.set_extent(map_extent)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='gray', alpha=0.25)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)

    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {'size': minor_fontsize, 'color': '#555555'}

    # Particle subset
    particles = particle_subset or trajectories.particle.values

    # Wind streamlines (if provided and show_wind is True)
    if wind is not None and show_wind:
        altitude_km = trajectories.isel(particle=particles[0]).z.isel(time=0).mean(
            'ensemble').item()
        if 'time' in wind.dims or 'time' in wind.coords:
            t_min = trajectories.time.min().values
            t_max = trajectories.time.max().values
            wind_slice = wind.sel(time=slice(t_min, t_max))
            if wind_slice.time.size == 0:
                wind_slice = wind.sel(time=slice(t_max, t_min))
            wind_avg = (
                wind_slice.sel(z_mc=1e3 * altitude_km, method='nearest')
                .mean(dim='time').squeeze()
            )
        else:
            wind_avg = wind.sel(z_mc=1e3 * altitude_km, method='nearest').squeeze()

        speed = np.sqrt(wind_avg.u ** 2 + wind_avg.v ** 2)
        speed_max = np.maximum(speed.max(), 1e-6)
        lw = 0.2 * (speed / speed_max).values  # Normalize and convert to NumPy

        wind_avg.plot.streamplot(
            x="lon", y="lat", u="u", v="v", ax=ax,
            linewidth=lw, density=0.8, color='gray',
            transform=ccrs.PlateCarree()
        )

    # Colormap for altitude
    cmap_altitude = sns.color_palette("Spectral_r", as_cmap=True)
    norm_altitude = mcolors.Normalize(vmin=96, vmax=104)

    # Orbit plot (if provided)
    if orbit is not None:
        ax.plot(orbit['GLon'], orbit['GLat'], color='black', linewidth=1.6,
                transform=ccrs.PlateCarree())

    # Select particles to plot
    if representative_members is not None:
        representative_ensemble = {
            p: (m, trajectories.time.size - 1) for p, m in representative_members
        }
    else:
        has_orbit = orbit is not None
        has_target = target_point is not None
        if (has_orbit or has_target) and calculate_intersections:
            representative_ensemble = generate_representative_particles(
                trajectories,
                orbit_df=orbit if has_orbit else None,
                target_point=target_point if has_target else None,
                max_dist_km=max_dist_km,
            )
            # Narrow the particles list to those with a qualifying hit.
            # Particles absent from representative_ensemble had no ensemble
            # member close enough to the reference and are excluded from
            # the trajectory plot (the KDE still uses all particles).
            particles = [p for p in particles if p in representative_ensemble]
        else:
            representative_ensemble = {p: (0, trajectories.time.size - 1) for p in particles}

    print(representative_ensemble)

    # Plot each particle's trajectory
    particle_patches = []
    markers = ['o', 's', 'd', 'o', 'v', '<', '>', 'p', 'h', 'D', 'P', 'X']
    all_markers = plt.Line2D.markers
    markers += [m for m in all_markers if
                m not in (None, ' ', '', 'None', 'none') and m not in markers]

    pc_transform = ccrs.PlateCarree()

    for i in particles:
        particle = trajectories.sel(particle=i)
        t_end_idx = representative_ensemble[i][1] + 1
        rep_ens = representative_ensemble[i][0]

        # Extract numpy arrays once — avoids repeated xarray label lookups
        # particle.lon has shape (time, ensemble); transpose → (ensemble, time)
        ens_lon = particle.lon.values[:t_end_idx, :].T  # (ensemble, time)
        ens_lat = particle.lat.values[:t_end_idx, :].T

        # --- Ensemble cloud: one LineCollection per particle (replaces n_ensemble ax.plot calls) ---
        n_ens, n_t = ens_lon.shape
        # Build (n_ens, n_t, 2) array of (lon, lat) path vertices
        coords = np.stack([ens_lon, ens_lat], axis=-1)  # (n_ens, n_t, 2)
        # LineCollection expects a list/array of (n_t, 2) paths
        lc_ens = LineCollection(
            coords,
            colors='#A3B3CC',
            alpha=0.04,
            linewidths=0.8,
            zorder=2,
            transform=pc_transform,
        )
        ax.add_collection(lc_ens)
        # LineCollection bypasses Cartopy's line-clipping pipeline; clip explicitly
        # to the axes patch so lines outside the map extent don't escape the frame.
        lc_ens.set_clip_path(ax.patch)

        # --- Representative trajectory (single member) ---
        rep_lon = ens_lon[rep_ens]
        rep_lat = ens_lat[rep_ens]
        ax.plot(rep_lon, rep_lat, color='black', alpha=0.6,
                linewidth=1.0, zorder=3, transform=pc_transform)

        # Sampled altitude markers – spacing controlled by a pixel-arc heuristic
        # so markers don't cluster on slow/dense portions of the track.
        representative = particle.isel(ensemble=rep_ens,
                                       time=slice(None, t_end_idx))
        stride = _marker_stride(rep_lon, rep_lat, ax)
        sampled = representative.isel(time=slice(None, None, stride))
        sampled.plot.scatter(x="lon", y="lat", hue='z', cmap=cmap_altitude,
                             norm=norm_altitude, marker=markers[i], s=50,
                             ax=ax, zorder=2, add_colorbar=False,
                             transform=pc_transform, add_labels=False)

        # Legend entry
        start_z = representative.z.isel(time=0).item()
        end_z = representative.z.isel(time=-1).item()

        patch = mlines.Line2D([], [], color='k',
                              markerfacecolor=cmap_altitude(norm_altitude(end_z)),
                              marker=markers[i], linestyle='-', markersize=8,
                              markeredgecolor='w',
                              label=f"{start_z:.1f}–{end_z:.1f} km")
        particle_patches.append(patch)

        # Annotate last representative point (only once to avoid clutter/overlapping date boxes)
        if i == particles[0]:
            rep_time_str = pd.to_datetime(representative.time[-1].item()).strftime(
                '%Y-%m-%d %H:%M UTC')
            ann_lon = rep_lon[-1]
            ann_lat = rep_lat[-1]

            ax.annotate(rep_time_str, xy=(ann_lon, ann_lat), xytext=(ann_lon - 7, ann_lat - 6),
                        textcoords=pc_transform._as_mpl_transform(ax),
                        arrowprops=dict(arrowstyle="->", color='black', linewidth=1.5),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black",
                                  alpha=0.8), fontsize=minor_fontsize - 3,
                        ha='left', transform=pc_transform)

    # Add orbit legend
    if orbit is not None:
        particle_patches.append(mlines.Line2D([], [], color='black',
                                              linestyle='-', linewidth=1.6, label="Falcon 9"))

    # Add target point if provided
    if target_point is not None and tp_lon is not None and tp_lat is not None:
        ax.scatter(tp_lon, tp_lat, color='gold', marker='*', s=140, edgecolor='black',
                   linewidth=0.8, zorder=12, transform=ccrs.PlateCarree())

        # ax.text(tp_lon, tp_lat - 0.6, target_label, color='black',
        #         fontsize=minor_fontsize - 1, fontweight='bold', va='top', ha='center',
        #         transform=ccrs.PlateCarree(),
        #         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.85))

        particle_patches.append(mlines.Line2D([], [], color='gold', marker='*', linestyle='None',
                                              markersize=10, markeredgecolor='black',
                                              label=target_label))

    # KDE plot of endpoint density
    time_endpoints = trajectories.time[-1]
    # t_slice = slice(t_end + pd.Timedelta(hours=.005), t_end)

    print(f"Computing KDE of endpoints for time slice {time_endpoints.values} ...")
    points = np.vstack([
        trajectories.sel(time=time_endpoints).lon.values.flatten(),
        trajectories.sel(time=time_endpoints).lat.values.flatten()
    ]).T

    if points.shape[0] == 0:
        raise ValueError("No trajectory data found in the specified time range.")

    # Fit Gaussian Mixture Model to the endpoint distribution
    gmm = GaussianMixture(
        n_components=3,  # Number of Gaussian components
        covariance_type='full',
        n_init=10,  # Run the EM algorithm 10 times with different initializations
        random_state=42  # Ensures these 10 initializations are the same each time
    )
    gmm.fit(points)

    # Convert min/max bounds to python floats to avoid Xarray wrapping errors in np.linspace
    traj_lon_min = float(trajectories.lon.min().item())
    traj_lon_max = float(trajectories.lon.max().item())
    traj_lat_min = float(trajectories.lat.min().item())
    traj_lat_max = float(trajectories.lat.max().item())

    lon_grid, lat_grid = np.meshgrid(
        np.linspace(max(traj_lon_min, map_extent[0]),
                    min(traj_lon_max, map_extent[1]), 800),
        np.linspace(max(traj_lat_min, map_extent[2]),
                    min(traj_lat_max, map_extent[3]), 800)
    )
    grid_points = np.c_[lon_grid.ravel(), lat_grid.ravel()]
    density = np.exp(gmm.score_samples(grid_points)).reshape(lon_grid.shape)

    # Normalize to %
    dx = np.abs(lon_grid[0, 1] - lon_grid[0, 0])
    dy = np.abs(lat_grid[1, 0] - lat_grid[0, 0])
    area_element = dx * dy

    density *= 100.0 / np.nansum(density) / area_element  # Normalize to percentage

    print("Probability density total = ", area_element * density.sum(), "%")

    lower_bound = np.percentile(density, 90)
    density_masked = np.ma.masked_less(density, lower_bound)

    white_to_blue = LinearSegmentedColormap.from_list(
        "white_to_blue",
        ["#ffffff", "#e7f2f9", "#bcdff0", "#7fbbd6", "#4989b5", "#2a5583"], N=256
    )

    cs = ax.contourf(lon_grid, lat_grid, density_masked, levels=21,
                     cmap=white_to_blue, alpha=0.9,
                     transform=ccrs.PlateCarree(), extend='max')
    # Suppress hairline gaps between contour bands (API changed in mpl ≥ 3.8)
    cs.set_edgecolor("face")

    # Legend
    leg = ax.legend(handles=particle_patches, loc='upper right',
                    ncol=2, frameon=True, fontsize=mayor_fontsize)
    leg.get_frame().set_alpha(0.85)
    leg.get_frame().set_linewidth(0.4)

    # Title
    draw_title = False
    if draw_title:
        t_start_str = pd.to_datetime(trajectories.time[0].item()).strftime('%Y-%m-%d %H:%M:%S UTC')
        ax.set_title(f"Lithium back-trajectories initialized on {t_start_str}",
                     fontsize=mayor_fontsize, pad=10)
    else:
        ax.set_title("")

    # Adjust subplot to leave room at the bottom for colorbars
    fig.subplots_adjust(bottom=0.22, top=0.98, left=0.08, right=0.92)

    # Colorbar for endpoint density
    upper_bound = density.max()

    # ticks_pdf = np.linspace(0.22, 0.42, 5)
    def nice_ticks(lower, upper, n=5):
        # Round bounds to the nearest hundredth
        lower = np.floor(lower * 100) / 100
        upper = np.ceil(upper * 100) / 100
        ticks = np.linspace(lower, upper, n)
        return np.round(ticks, 2)

    cax_pdf = fig.add_axes([0.1326, 0.09, 0.36, 0.042])
    cbar_pdf = fig.colorbar(cs, cax=cax_pdf, orientation='horizontal',
                            ticks=nice_ticks(lower_bound, upper_bound, n=5),
                            extend='max')
    cbar_pdf.set_label('Endpoint Probability (%)', fontsize=mayor_fontsize)
    cbar_pdf.ax.tick_params(labelsize=minor_fontsize - 1, direction='out', length=3, width=0.5)

    # Colorbar for altitude
    cax_alt = fig.add_axes([0.53, 0.09, 0.36, 0.042])  # left, bottom, width, height
    sm_alt = plt.cm.ScalarMappable(cmap=cmap_altitude, norm=norm_altitude)
    sm_alt.set_array([])
    cbar_alt = fig.colorbar(sm_alt, cax=cax_alt, orientation='horizontal',
                            ticks=np.linspace(96, 104, 5), extend='both')
    cbar_alt.set_label('Altitude (km)', fontsize=mayor_fontsize)
    cbar_alt.ax.tick_params(labelsize=minor_fontsize - 1, direction='out', length=3, width=0.5)

    if figure_name:
        fig.canvas.draw()
        fig.savefig(figure_name, dpi=300, bbox_inches='tight')
    else:
        return fig


def plot_orbit_and_ensemble_3d(trajectories, orbit, target_orbit=None,
                               max_ensemble=100, particle_subset=None,
                               horiz_tol_km=5, vert_tol_km=1,
                               figsize=(9, 7), elev=20, azim=45, figure_name=None):
    """
    Plots the 3D position of the orbital path and the ensemble trajectories over a time range.
    Highlights intersecting trajectories and annotates them.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    orbit = orbit.set_index("timestamp")
    traj = trajectories.sortby("time")

    times = traj.time
    if len(times) == 0:
        raise ValueError("No trajectory data found in the specified time range.")

    particles = particle_subset or traj.particle.values
    ensemble = min(traj.ensemble.size, max_ensemble)
    traj = traj.sel(particle=particles, ensemble=slice(0, ensemble))

    # Compute intersections with re-entry orbit
    intersect_mask, intersect_indices = compute_intersections(
        traj, orbit, horiz_tol_km, vert_tol_km
    )

    # Compute intersections with target orbit if provided
    intersect_mask_target = None
    intersect_indices_target = np.empty((0, 3), dtype=int)
    if target_orbit is not None:
        if isinstance(target_orbit, xr.Dataset):
            target_df = pd.DataFrame({
                "timestamp": pd.to_datetime(target_orbit.time.values),
                "GLon": target_orbit.lon.values,
                "GLat": target_orbit.lat.values,
                "GAlt": target_orbit.z.values
            })
        else:
            target_df = target_orbit
        intersect_mask_target, intersect_indices_target = compute_intersections(
            traj, target_df, horiz_tol_km, vert_tol_km
        )

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Set relative aspect ratio to make horizontal path more elongated
    ax.set_box_aspect((2.5, 1.0, 0.7))

    # === Plot orbit path ===
    ax.plot(orbit["GLon"], orbit["GLat"], orbit["GAlt"],
            label="Re-entry Trajectory", color="black", linewidth=2, zorder=5)

    ax.plot(orbit["GLon"], orbit["GLat"], zs=0, zdir='z',
            color='gray', linewidth=1, linestyle='--', label="Ground Track", zorder=3)

    # === Plot target orbit ===
    if target_orbit is not None:
        if isinstance(target_orbit, xr.Dataset):
            target_lons = np.atleast_1d(target_orbit["lon"].values)
            target_lats = np.atleast_1d(target_orbit["lat"].values)
            target_alts = np.atleast_1d(target_orbit["z"].values)
        else:
            target_lons = np.atleast_1d(target_orbit["GLon"].values)
            target_lats = np.atleast_1d(target_orbit["GLat"].values)
            target_alts = np.atleast_1d(target_orbit["GAlt"].values)
        ax.scatter(target_lons, target_lats, zs=target_alts,
                   color='gold', marker='*', s=120, edgecolor='black',
                   linewidth=0.8, label="Target Orbit Point", zorder=6)
        ax.scatter(target_lons, target_lats, zs=0, zdir='z',
                   color='orange', marker='*', s=60, alpha=0.6, zorder=3)

    # === Plot trajectories ===
    for p in particles:
        for e in range(ensemble):
            lon = traj.lon.sel(particle=p, ensemble=e)
            lat = traj.lat.sel(particle=p, ensemble=e)
            alt = traj.z.sel(particle=p, ensemble=e)

            # Isolate intersection points
            intersect = intersect_mask.sel(particle=p, ensemble=e)

            # Highlight starting intersections (with orbit)
            has_intersect = False
            if intersect.any():
                ax.plot(lon, lat, alt, color='blue', alpha=0.15, linewidth=0.5, zorder=1)
                ax.scatter(lon.where(intersect, drop=True),
                           lat.where(intersect, drop=True),
                           zs=alt.where(intersect, drop=True),
                           color='red', s=15,
                           label='Intersection (Start)' if (p == particles[0] and e == 0) else "",
                           zorder=10)
                has_intersect = True

            # Highlight target intersections (with target_orbit)
            if intersect_mask_target is not None:
                intersect_t = intersect_mask_target.sel(particle=p, ensemble=e)
                if intersect_t.any():
                    if not has_intersect:
                        ax.plot(lon, lat, alt, color='blue', alpha=0.15, linewidth=0.5, zorder=1)
                    ax.scatter(lon.where(intersect_t, drop=True),
                               lat.where(intersect_t, drop=True),
                               zs=alt.where(intersect_t, drop=True),
                               color='green', s=25, marker='o', edgecolor='black', linewidth=0.5,
                               label='Intersection (Target)' if (
                                           p == particles[0] and e == 0) else "",
                               zorder=11)
                    has_intersect = True

            if not has_intersect:
                ax.plot(lon, lat, alt, color='blue',
                        alpha=0.04, linewidth=0.6, zorder=1)

    # === Compute axes limits using percentiles to ignore long-drifting outliers ===
    traj_lon = traj.lon.values
    traj_lat = traj.lat.values

    lons_to_bound = []
    lats_to_bound = []

    if np.any(~np.isnan(traj_lon)):
        lons_to_bound.extend(
            [float(np.nanpercentile(traj_lon, 2.5)), float(np.nanpercentile(traj_lon, 97.5))])
    if np.any(~np.isnan(traj_lat)):
        lats_to_bound.extend(
            [float(np.nanpercentile(traj_lat, 2.5)), float(np.nanpercentile(traj_lat, 97.5))])

    lons_to_bound.extend([orbit["GLon"].min(), orbit["GLon"].max()])
    lats_to_bound.extend([orbit["GLat"].min(), orbit["GLat"].max()])

    if target_orbit is not None:
        if isinstance(target_orbit, xr.Dataset):
            target_lons = np.atleast_1d(target_orbit["lon"].values)
            target_lats = np.atleast_1d(target_orbit["lat"].values)
        else:
            target_lons = np.atleast_1d(target_orbit["GLon"].values)
            target_lats = np.atleast_1d(target_orbit["GLat"].values)
        lons_to_bound.extend([target_lons.min(), target_lons.max()])
        lats_to_bound.extend([target_lats.min(), target_lats.max()])

    lon_min, lon_max = min(lons_to_bound), max(lons_to_bound)
    lat_min, lat_max = min(lats_to_bound), max(lats_to_bound)

    # Add small padding (5% of span or at least 2 degrees)
    lon_span = lon_max - lon_min
    lat_span = lat_max - lat_min
    lon_pad = max(0.05 * lon_span, 2.0)
    lat_pad = max(0.05 * lat_span, 2.0)

    ax.set_xlim(lon_min - lon_pad, lon_max + lon_pad)
    ax.set_ylim(lat_min - lat_pad, lat_max + lat_pad)

    # === Labels and title ===
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_zlabel("Altitude (km)")
    ax.set_zlim(0, 120)

    time_start = np.datetime_as_string(times[0].values, unit='s')
    time_end = np.datetime_as_string(times[-1].values, unit='s')
    ax.set_title(f"Ensemble Trajectories and Re-entry Track\n{time_start} to {time_end}",
                 fontsize=11)

    # === Annotate intersections ===
    total_intersects = len(intersect_indices) + len(intersect_indices_target)
    if total_intersects > 0:
        preview_lines = []
        if intersect_indices.size > 0:
            intersect_df_start = pd.DataFrame({
                "intersect_time": trajectories.time.values[intersect_indices[:, 0]],
                "particle": intersect_indices[:, 1],
                "ensemble": intersect_indices[:, 2],
            }).sort_values(by="intersect_time")

            preview_lines.append("Start Intersections:")
            preview_lines.extend([
                f"  P{row.particle}, E{row.ensemble}: "
                f"{pd.to_datetime(row.intersect_time).strftime('%H:%M:%S')}"
                for _, row in intersect_df_start.head(3).iterrows()
            ])
            if len(intersect_df_start) > 3:
                preview_lines.append("  ...")

        if intersect_indices_target.size > 0:
            intersect_df_target = pd.DataFrame({
                "intersect_time": trajectories.time.values[intersect_indices_target[:, 0]],
                "particle": intersect_indices_target[:, 1],
                "ensemble": intersect_indices_target[:, 2],
            }).sort_values(by="intersect_time")

            preview_lines.append("Target Intersections:")
            preview_lines.extend([
                f"  P{row.particle}, E{row.ensemble}: "
                f"{pd.to_datetime(row.intersect_time).strftime('%H:%M:%S')}"
                for _, row in intersect_df_target.head(3).iterrows()
            ])
            if len(intersect_df_target) > 3:
                preview_lines.append("  ...")

        preview = "\n".join(preview_lines)
        ax.text2D(0.015, 0.95,
                  f"Intersections ({total_intersects} total):\n" + preview,
                  transform=ax.transAxes, color='red', fontsize=9, va='top')

    else:
        ax.text2D(0.015, 0.95, "No intersections detected", transform=ax.transAxes,
                  color='gray', fontsize=9, va='top')

    # === Legend and layout ===
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    # plt.tight_layout()
    if figure_name:
        fig.savefig(figure_name, dpi=300)


# Re-define the animation function after state reset
def animate_trajectories_over_time(
        trajectories, wind, orbit,
        output_dir: str,
        time_step_minutes: int = 10
):
    """
    Animate ensemble trajectories and re-entry path over time by generating individual frames.

    Parameters:
        trajectories : xr.Dataset
            Ensemble back-trajectories with dimensions (time, particle, ensemble)
        wind : xr.Dataset
            Wind field with (time, z, lat, lon) or similar dims
        orbit : pd.DataFrame
            Re-entry trajectory with 'timestamp', 'GLon', 'GLat', 'GAlt'
        output_dir : str
            Directory to save the output frames (e.g., for ffmpeg)
        time_step_minutes : int
            Time step (in minutes) between animation frames
    """
    os.makedirs(output_dir, exist_ok=True)

    start_time = pd.to_datetime(trajectories.time.min().values)
    end_time = pd.to_datetime(trajectories.time.max().values)

    times = pd.date_range(start=start_time, end=end_time, freq=f"{time_step_minutes}min")

    for i, t in enumerate(times):
        print(f"Generating frame {i + 1}/{len(times)} at {t}")
        # Slice backwards in time (since these are back-trajectories)
        traj_slice = trajectories.sel(time=slice(trajectories.time.max().values, t))
        orbit_slice = orbit[orbit["timestamp"] <= t]

        # Skip if the slice is empty
        if traj_slice.time.size == 0:
            print(f"  Skipped frame at {t} — no data in time slice.")
            continue

        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")

        try:
            fig = visualize_trajectories_percentile_kde(
                trajectories=traj_slice,
                wind=wind,
                orbit=orbit_slice,
                figure_name=None
            )

            fig.canvas.draw()
            fig.savefig(frame_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"  Error generating frame at {t}: {e}")
