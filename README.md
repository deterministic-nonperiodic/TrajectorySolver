# Lagrangian Trajectory Model with Ensemble Support

This repository contains a flexible and efficient Python implementation for computing Lagrangian trajectories using 3D wind fields from reanalyses or models, such as ICON. The tool supports ensemble simulations with noise perturbations, automatic coordinate and unit normalization (including pressure velocity to physical velocity conversion), and early stopping when particles approach a specified target region.

---

## 🚀 Features

- ✅ **Vectorized Integration**: Fast integration using `scipy.solve_ivp` with `dense_output=True` and multi-threaded parallel execution.
- ✅ **Automatic Coordinate Detection**: Seamlessly detects and curates standard and non-standard dimension names and coordinate conventions.
- ✅ **Stochastic Ensemble Spread**: Support for lognormal or Gaussian wind perturbations.
- ✅ **Target Proximity Event Detection**: Early termination when trajectories approach a moving target trajectory/orbit.
- ✅ **CF-compliant Output**: Results are returned as a standard self-describing `xarray.Dataset` containing coordinates and metadata.

---

## 📦 Installation

This package can be installed via Conda (recommended for managing binary geospatial dependencies like PROJ and Cartopy) or using Pip (from a local directory or directly from a Git repository).

### Option 1: Conda (Recommended)

To create a new environment and install all dependencies:

```bash
conda env create -f environment.yml
conda activate trajsolver
pip install -e .
```

### Option 2: Pip (From Git Repository)

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/deterministic-nonperiodic/TrajectorySolver.git
```

### Option 3: Pip (Local Development)

To install in editable mode for local development:

```bash
# Standard installation
pip install -e .

# Development installation (includes pytest, coverage, etc.)
pip install -e ".[dev]"
```

---

## 🚀 Basic Usage

```python
import xarray as xr
from trajsolver import LagrangianTrajectories

# Load your wind dataset (containing u, v, w)
wind_data = xr.open_dataset("data/jawara_winds_HL_02-2025.nc")

# Initialize the solver
model = LagrangianTrajectories(
    data=wind_data,
    timestep="10 minutes",  # Any pandas-compatible timedelta or seconds
    integration_method="RK23",
    interpolation_method="linear",
    noise_type="lognormal",
    verbose_level=1,
)

# Run the simulation
trajectories = model.advect_particles(
    start_positions=[(11.4, 54.1, 96000)],  # List of tuples: (lon [deg], lat [deg], z [meters])
    duration="3 days",
    ensemble_size=100,
    target=target_ds,  # Optional target orbit xarray.Dataset
    distance_tolerance=10e3,  # Target tolerance in meters
    n_jobs=-1,  # Use all available CPU threads
)
```
