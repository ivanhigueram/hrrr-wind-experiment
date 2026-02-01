"""
HRRR Wind Data Fetcher

Fetches High-Resolution Rapid Refresh (HRRR) wind data using the Herbie library.
HRRR provides 3km resolution wind fields, updated hourly.

This module handles:
- Fetching U/V wind components at 10m and various pressure levels
- Computing wind speed and direction
- Subsetting to regions of interest
- Exporting to formats compatible with WindNinja
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr
from herbie.core import Herbie
from herbie.fast import FastHerbie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HRRRWindFetcher:
    """Fetch and process HRRR wind data for wildfire applications."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./hrrr_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_surface_winds(
        self,
        date: datetime,
        forecast_hour: int = 0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> xr.Dataset:
        """
        Fetch 10m wind components from HRRR.

        Parameters
        ----------
        date : datetime
            Model initialization time
        forecast_hour : int
            Forecast hour (0 for analysis, up to 48 for forecast)
        bounds : tuple, optional
            (west, south, east, north) bounding box to subset

        Returns
        -------
        xr.Dataset
            Dataset with u10, v10, wind_speed, wind_direction
        """
        logger.info(f"Fetching HRRR surface winds for {date} F{forecast_hour:02d}")

        H = Herbie(
            date,
            model="hrrr",
            product="sfc",
            fxx=forecast_hour,
        )

        ds = H.xarray("(?:UGRD|VGRD):10 m above ground")

        ds = ds.herbie.with_wind()

        if bounds:
            west, south, east, north = bounds
            # HRRR uses 0-360 longitude convention
            if west < 0:
                west = west + 360
            if east < 0:
                east = east + 360

            if hasattr(ds, "longitude"):
                mask = (
                    (ds.longitude >= west)
                    & (ds.longitude <= east)
                    & (ds.latitude >= south)
                    & (ds.latitude <= north)
                )
                y_idx = mask.any(dim="x").values
                x_idx = mask.any(dim="y").values
                ds = ds.isel(y=y_idx, x=x_idx)

        return ds

    def fetch_multilevel_winds(
        self,
        date: datetime,
        forecast_hour: int = 0,
        levels: List[int] = [1000, 925, 850, 700, 500],
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> xr.Dataset:
        """
        Fetch wind components at multiple pressure levels.

        Parameters
        ----------
        date : datetime
            Model initialization time
        forecast_hour : int
            Forecast hour
        levels : list
            Pressure levels in hPa
        bounds : tuple, optional
            Bounding box (west, south, east, north)

        Returns
        -------
        xr.Dataset
            Dataset with u, v, wind_speed at each pressure level
        """
        logger.info(
            f"Fetching HRRR pressure level winds for {date} F{forecast_hour:02d}"
        )

        H = Herbie(
            date,
            model="hrrr",
            product="prs",
            fxx=forecast_hour,
        )

        level_str = "|".join([f"{l} mb" for l in levels])
        ds = H.xarray(f"(?:UGRD|VGRD):({level_str})")

        u = ds["u"]
        v = ds["v"]
        ds["wind_speed"] = np.sqrt(u**2 + v**2)
        ds["wind_direction"] = (270 - np.degrees(np.arctan2(v, u))) % 360

        if bounds:
            west, south, east, north = bounds
            # HRRR uses 0-360 longitude convention
            if west < 0:
                west = west + 360
            if east < 0:
                east = east + 360

            if hasattr(ds, "longitude"):
                mask = (
                    (ds.longitude >= west)
                    & (ds.longitude <= east)
                    & (ds.latitude >= south)
                    & (ds.latitude <= north)
                )
                y_idx = mask.any(dim="x").values
                x_idx = mask.any(dim="y").values
                ds = ds.isel(y=y_idx, x=x_idx)

        return ds

    def fetch_time_series(
        self,
        start_date: datetime,
        end_date: datetime,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        hours_interval: int = 1,
    ) -> xr.Dataset:
        """
        Fetch a time series of surface winds using FastHerbie for parallelization.

        Parameters
        ----------
        start_date : datetime
            Start of time range
        end_date : datetime
            End of time range
        bounds : tuple, optional
            Bounding box
        hours_interval : int
            Hours between fetches

        Returns
        -------
        xr.Dataset
            Concatenated dataset with time dimension
        """
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(hours=hours_interval)

        logger.info(f"Fetching {len(dates)} HRRR timestamps with FastHerbie")

        FH = FastHerbie(
            dates,
            model="hrrr",
            product="sfc",
            fxx=0,
        )

        ds = FH.xarray("(?:UGRD|VGRD):10 m above ground")

        if bounds:
            west, south, east, north = bounds
            if hasattr(ds, "longitude") and hasattr(ds, "latitude"):
                mask = (
                    (ds.longitude >= west)
                    & (ds.longitude <= east)
                    & (ds.latitude >= south)
                    & (ds.latitude <= north)
                )
                if ds.longitude.ndim == 2:
                    y_idx = mask.any(dim="x").values
                    x_idx = mask.any(dim="y").values
                    ds = ds.isel(y=y_idx, x=x_idx)
                else:
                    ds = ds.where(mask, drop=True)

        return ds

    def to_windninja_ascii(
        self,
        ds: xr.Dataset,
        output_dir: Path,
        filename_prefix: str = "wind",
    ) -> Tuple[Path, Path]:
        """
        Export wind data to ASCII grid format for WindNinja input.

        WindNinja accepts ASCII grid files for wind speed and direction.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset with wind_speed and wind_direction
        output_dir : Path
            Output directory
        filename_prefix : str
            Prefix for output files

        Returns
        -------
        tuple
            Paths to (speed_file, direction_file)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        speed = ds["si10"] if "si10" in ds else ds["wind_speed"]
        direction = ds["wdir10"] if "wdir10" in ds else ds["wind_direction"]

        speed_file = output_dir / f"{filename_prefix}_speed.asc"
        dir_file = output_dir / f"{filename_prefix}_direction.asc"

        self._write_ascii_grid(speed, speed_file)
        self._write_ascii_grid(direction, dir_file)

        logger.info(f"Wrote WindNinja input files: {speed_file}, {dir_file}")
        return speed_file, dir_file

    def to_netcdf(
        self,
        ds: xr.Dataset,
        output_path: Path,
    ) -> Path:
        """Export dataset to NetCDF for archival or further processing."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ds.to_netcdf(output_path)
        logger.info(f"Wrote NetCDF: {output_path}")
        return output_path

    def _write_ascii_grid(self, data: xr.DataArray, filepath: Path):
        """Write xarray DataArray to ESRI ASCII grid format."""
        if "x" in data.dims:
            lon_dim, lat_dim = "x", "y"
        else:
            lon_dim, lat_dim = "longitude", "latitude"

        lons = data[lon_dim].values
        lats = data[lat_dim].values

        cellsize_x = abs(lons[1] - lons[0]) if len(lons) > 1 else 0.03
        cellsize_y = abs(lats[1] - lats[0]) if len(lats) > 1 else 0.03

        header = f"""ncols {len(lons)}
nrows {len(lats)}
xllcorner {lons.min()}
yllcorner {lats.min()}
cellsize {cellsize_x}
NODATA_value -9999
"""

        values = data.values
        if np.isnan(values).any():
            values = np.nan_to_num(values, nan=-9999)

        values = np.flipud(values)

        with open(filepath, "w") as f:
            f.write(header)
            for row in values:
                f.write(" ".join([f"{v:.2f}" for v in row]) + "\n")


def example_california_fire_region():
    """Example: Fetch wind data for a California fire-prone region."""
    fetcher = HRRRWindFetcher()

    socal_bounds = (-121.0, 33.5, -116.0, 37.0)

    ds = fetcher.fetch_surface_winds(
        date=datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        - timedelta(hours=2),
        forecast_hour=0,
        bounds=socal_bounds,
    )

    print(ds)
    return ds


if __name__ == "__main__":
    ds = example_california_fire_region()
