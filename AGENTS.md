# AGENTS.md - Project Context for AI Assistants

## Project Overview

**hrrr-wind-experiment** provides tools for fetching HRRR (High-Resolution Rapid Refresh) wind data and downscaling it with WindNinja for wildfire applications.

### Purpose
- Fetch 3km operational wind data from NOAA's HRRR model
- Downscale to 30-100m resolution using WindNinja's terrain-aware CFD
- Support HPC batch processing via SLURM + Apptainer containers
- Visualize wind fields for fire weather analysis

### Target Users
- Fire weather forecasters
- Wildfire researchers
- Atmospheric scientists working with NWP data

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HRRR (3km)     │────▶│  WindNinja       │────▶│  High-res winds │
│  via Herbie     │     │  (CFD downscale) │     │  (30-100m)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │                       │
        ▼                        ▼                       ▼
   fetch_hrrr.py          windninja_runner.py     visualization.py
```

## Key Files

| File | Purpose |
|------|---------|
| `fetch_hrrr.py` | HRRR data fetching using Herbie library |
| `windninja_runner.py` | Python wrapper for WindNinja CLI |
| `visualization.py` | Wind plotting (barbs, streamlines, quiver) |
| `windninja.def` | Apptainer/Singularity container definition |
| `submit_windninja_job.slurm` | HPC batch job script |
| `notebooks/01_hrrr_windninja_workflow.ipynb` | Tutorial notebook |

## Technical Notes

### HRRR Coordinate System
- Uses **Lambert Conformal** projection with 2D lat/lon arrays
- Longitude is in **0-360°** convention (not -180 to 180)
- When subsetting by bounds, convert negative longitudes: `lon = lon + 360 if lon < 0`

### Herbie Library
- Import from submodules: `from herbie.core import Herbie` (not `from herbie import Herbie`)
- Expects **timezone-naive** datetime objects for the `date` parameter
- Data cached in `hrrr_cache/` directory

### Visualization with 2D Coordinates
- HRRR's curvilinear grid has 2D lat/lon arrays
- Don't use `np.meshgrid()` - coordinates are already 2D
- Check `lon.ndim == 2` before plotting

### WindNinja
- Container available at `docker://firelab/windninja:latest`
- Initialization methods: `domainAverageInitialization`, `pointInitialization`, `wxModelInitialization`
- For HRRR input, use `wx_model_type="HRRR"` with the GRIB2 file path

## Development Setup

```bash
cd ~/fun/hrrr_wind_experiment
uv sync                          # Install dependencies
uv run python fetch_hrrr.py      # Test HRRR fetch
uv run jupyter notebook          # Open notebooks
```

## Common Tasks

### Fetch wind data for a region
```python
from datetime import datetime, timedelta
from fetch_hrrr import HRRRWindFetcher

fetcher = HRRRWindFetcher()
ds = fetcher.fetch_surface_winds(
    date=datetime.utcnow() - timedelta(hours=2),
    forecast_hour=0,
    bounds=(-122, 39.5, -121, 40),  # (west, south, east, north)
)
```

### Create visualization
```python
from visualization import plot_wind_speed, plot_quiver
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
plot_wind_speed(ds, ax=ax, use_cartopy=False)
plt.savefig('output/wind.png')
```

### Run WindNinja on HPC
```bash
# Pull container first
apptainer pull windninja.sif docker://firelab/windninja:latest

# Submit job
sbatch submit_windninja_job.slurm
```

## Fire-Prone Regions (Preset Bounds)

| Region | Bounds (W, S, E, N) |
|--------|---------------------|
| Paradise/Camp Fire | (-122.0, 39.5, -121.0, 40.0) |
| Wine Country | (-123.0, 38.2, -122.2, 38.8) |
| LA Basin/Santa Ana | (-118.5, 33.8, -117.5, 34.5) |
| Diablo Wind Corridor | (-122.3, 37.7, -121.8, 37.95) |

## Dependencies

Key packages (managed via `uv`):
- `herbie-data` - HRRR data access
- `xarray` + `cfgrib` - GRIB2 file handling
- `matplotlib` + `cartopy` - Visualization
- `rioxarray` - Geospatial raster operations

## Known Issues

1. **LSP type errors in visualization.py**: These are false positives from optional cartopy imports - code works at runtime
2. **Herbie deprecation warnings**: `datetime.utcnow()` warnings can be ignored (Herbie requires naive datetimes)
3. **xarray FutureWarning**: `merge` compat warnings from cfgrib - harmless

## Future Enhancements

- [ ] Add DEM auto-download via py3dep
- [ ] Support for other NWP models (NAM, GFS, RAP)
- [ ] Animated wind field visualizations
- [ ] Integration with fire behavior models (FARSITE, FlamMap)
- [ ] Real-time Red Flag Warning detection
