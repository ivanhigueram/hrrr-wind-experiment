# HRRR Wind Experiment

Tools for fetching HRRR (High-Resolution Rapid Refresh) wind data and downscaling with WindNinja for wildfire applications.

## Overview

This project provides an end-to-end workflow for:

1. **HRRR Data Acquisition**: Fetch 3km operational wind data using the Herbie library
2. **WindNinja Downscaling**: Terrain-adjust winds to 30-100m resolution
3. **HPC Integration**: Run on clusters via SLURM + Apptainer/Singularity
4. **Visualization**: Wind barbs, streamlines, quiver plots, animations

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# For visualization (optional)
pip install jupyter cartopy
```

### 2. Fetch HRRR Data

```python
from datetime import datetime, timedelta
from fetch_hrrr import HRRRWindFetcher

fetcher = HRRRWindFetcher()

# Fetch surface winds for California
ds = fetcher.fetch_surface_winds(
    date=datetime.utcnow() - timedelta(hours=2),
    forecast_hour=0,
    bounds=(-122, 39.5, -121, 40),  # Paradise, CA region
)

print(ds)
```

### 3. Run WindNinja (with container)

```bash
# Pull WindNinja container
apptainer pull windninja.sif docker://firelab/windninja:latest

# Run with domain-average initialization
apptainer exec windninja.sif WindNinja_cli \
    --elevation_file=dem.tif \
    --initialization_method=domainAverageInitialization \
    --input_speed=15 \
    --input_speed_units=mph \
    --input_direction=225 \
    --mesh_resolution=100 \
    --output_path=./output
```

### 4. Submit HPC Job

```bash
# Edit configuration in submit_windninja_job.slurm, then:
sbatch submit_windninja_job.slurm
```

## Files

| File | Description |
|------|-------------|
| `fetch_hrrr.py` | HRRR data fetching using Herbie library |
| `windninja_runner.py` | Python wrapper for running WindNinja |
| `visualization.py` | Wind field plotting utilities |
| `windninja.def` | Apptainer/Singularity container definition |
| `submit_windninja_job.slurm` | HPC batch job script |
| `requirements.txt` | Python dependencies |
| `notebooks/` | Jupyter notebooks with examples |

## Key Concepts

### HRRR (High-Resolution Rapid Refresh)

- **Resolution**: 3km horizontal
- **Updates**: Hourly (most frequent operational NWP)
- **Coverage**: CONUS
- **Forecasts**: 0-48 hours
- **Best for**: Fire weather, convective storms, aviation

### WindNinja

- **Purpose**: Downscale NWP winds to terrain-adjusted high-resolution
- **Method**: Mass-conserving CFD solver
- **Output**: 30-500m resolution
- **Accounts for**: Ridge acceleration, valley channeling, flow separation

### Initialization Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `domainAverageInitialization` | Single wind input | Quick runs, uniform conditions |
| `pointInitialization` | Weather station data | When obs are available |
| `wxModelInitialization` | NWP model (HRRR, NAM) | Best for spatial variability |

## Example Regions

```python
# Paradise / Camp Fire (2018)
PARADISE_BOUNDS = (-122.0, 39.5, -121.0, 40.0)

# Wine Country / Tubbs Fire (2017)
WINE_COUNTRY_BOUNDS = (-123.0, 38.2, -122.2, 38.8)

# LA Basin / Santa Ana winds
LA_BOUNDS = (-118.5, 33.8, -117.5, 34.5)

# Diablo wind corridor
DIABLO_BOUNDS = (-122.3, 37.7, -121.8, 37.95)
```

## Fire Weather Thresholds

Red Flag Warning criteria vary by region, but common thresholds:

| Condition | Threshold |
|-----------|-----------|
| Sustained winds | ≥25 mph |
| Wind gusts | ≥35 mph |
| Relative humidity | ≤15% |
| Fuel moisture | Low |

## Building the Container

If you need to build from source (requires fakeroot or root):

```bash
# Build from definition file
apptainer build --fakeroot windninja.sif windninja.def

# Or pull pre-built from Docker Hub
apptainer pull windninja.sif docker://firelab/windninja:latest
```

## Resources

- [Herbie Documentation](https://herbie.readthedocs.io/)
- [WindNinja GitHub](https://github.com/firelab/windninja)
- [WindNinja User Guide](https://weather.firelab.org/windninja/)
- [HRRR on AWS](https://registry.opendata.aws/noaa-hrrr-pds/)
- [3DEP Elevation Data](https://www.usgs.gov/3d-elevation-program)

## License

MIT
