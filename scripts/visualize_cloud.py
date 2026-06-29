"""
visualize_cloud.py – 3-D PyVista point-cloud visualisation of ensemble trajectories.
Usage: python scripts/visualize_cloud.py
"""
import os
from pathlib import Path

import numpy as np
import xarray as xr
import pyvista as pv
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from io import BytesIO
from PIL import Image
from pyproj import Geod

from trajsolver import read_falcon

geod = Geod(ellps="WGS84")

BASE_PATH = Path("/home/deterministic-nonperiodic/IAP/Experiments")
REPO_ROOT = Path(__file__).resolve().parent.parent

trajectories = xr.open_dataset(
    BASE_PATH / "vortex/data/FALCON/trajectories_RK23_linear_2025-02-20T00:30"
               ":00_lag0minutes_members1000_dev.nc"
)
falcon_orbit = read_falcon(BASE_PATH / "falcon/Trajectory_2025-02-19/orbgen#12-cut.dat")

# ---- Build map texture ----
extent = [-20, 30, 40, 70]
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.OCEAN, facecolor="white")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3)
ax.axis("off")
buf = BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
plt.close(fig)
buf.seek(0)
map_array = np.array(Image.open(buf))

texture = pv.numpy_to_texture(map_array)
nx, ny = map_array.shape[1], map_array.shape[0]
plane = pv.Plane(i_size=extent[1] - extent[0], j_size=extent[3] - extent[2],
                 i_resolution=nx - 1, j_resolution=ny - 1)
plane.points[:, 0] += extent[0]
plane.points[:, 1] += extent[2]
plane.texture_map_to_plane(inplace=True)

plotter = pv.Plotter(off_screen=True)
plotter.set_background("white")
plotter.add_mesh(plane, texture=texture)

orb_lon = falcon_orbit["GLon"].values
orb_lat = falcon_orbit["GLat"].values
orb_alt = falcon_orbit["GAlt"].values
orbit_points = np.column_stack([orb_lon, orb_lat, orb_alt])
plotter.add_mesh(pv.lines_from_points(orbit_points), color="black", line_width=3)

time_index = -1
max_ensemble = 500
horizontal_thresh_km = 50
vertical_thresh_km = 5

time_slice = trajectories.isel(time=time_index)
particles = time_slice.particle.values
ensemble = min(time_slice.ensemble.size, max_ensemble)

points = []
intersecting_members = []
for p in particles:
    for e in range(ensemble):
        member = time_slice.isel(particle=p, ensemble=e)
        lon, lat, alt = member.lon.item(), member.lat.item(), member.z.item()
        _, _, dist_m = geod.inv(orb_lon[0], orb_lat[0], lon, lat)
        horiz_km = dist_m / 1000.0
        vert_km = abs(alt - orb_alt[0])
        if horiz_km <= horizontal_thresh_km and vert_km <= vertical_thresh_km:
            intersecting_members.append((p, e))
        points.append([lon, lat, alt])

plotter.add_mesh(pv.PolyData(np.array(points)), color="blue",
                 point_size=5, render_points_as_spheres=True, opacity=0.5)

if intersecting_members:
    label_text = "Intersecting:\n" + "\n".join(f"P{p}, E{e}" for p, e in intersecting_members)
else:
    label_text = "No intersections detected"

plotter.add_text(label_text, position="upper_left", font_size=10, color="red")
plotter.add_axes()
plotter.view_vector((1, 1, 0.4), (0, 0, 1))
plotter.camera.zoom(1.0)
plotter.set_scale(1, 1, 0.5)
plotter.show_bounds(grid="back", location="outer", all_edges=True,
                    xtitle="Longitude (°E)", ytitle="Latitude (°N)", ztitle="Altitude (km)")
plotter.show(title="3D Orbital Path and Trajectories (PyVista)",
             screenshot="orbit_ensemble_3d.png")
