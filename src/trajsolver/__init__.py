"""
trajsolver – Lagrangian ensemble trajectory solver for atmospheric data.

Public API
----------
LagrangianTrajectories : core solver class
detect_and_curate      : CF-aware coordinate detection and normalisation
save_cf_compliant      : CF-compliant NetCDF writer
read_falcon            : reader for ESA Falcon orbit dat files
"""

from .core import LagrangianTrajectories
from .coord_detection import detect_and_curate, detect_coords, curate_coords
from .tools import save_cf_compliant, read_falcon, sample_orbit_positions

__all__ = [
    "LagrangianTrajectories",
    "detect_and_curate",
    "detect_coords",
    "curate_coords",
    "save_cf_compliant",
    "read_falcon",
    "sample_orbit_positions",
]
