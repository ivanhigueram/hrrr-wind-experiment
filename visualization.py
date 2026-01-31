"""
Wind Field Visualization Utilities

Provides functions for visualizing wind data from HRRR and WindNinja:
- Wind barbs
- Streamlines
- Quiver plots
- Animated time series
- Side-by-side comparisons
"""

from pathlib import Path
from typing import Optional, Tuple, Union, List
import logging

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# Optional imports for cartographic projections
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Fire weather color scales
WIND_SPEED_CMAP = "YlOrRd"
WIND_SPEED_LEVELS = np.arange(0, 45, 2.5)  # mph

# Red Flag Warning thresholds (varies by region)
RED_FLAG_SPEED_MPH = 25  # Sustained winds
RED_FLAG_GUST_MPH = 35  # Gusts


def ms_to_mph(speed_ms: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert meters per second to miles per hour."""
    return speed_ms * 2.237


def mph_to_ms(speed_mph: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Convert miles per hour to meters per second."""
    return speed_mph / 2.237


def wind_components_from_speed_dir(
    speed: np.ndarray, direction: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate U/V wind components from speed and meteorological direction.

    Parameters
    ----------
    speed : array
        Wind speed (any units)
    direction : array
        Wind direction in degrees (meteorological convention: direction wind comes FROM)

    Returns
    -------
    u, v : tuple of arrays
        U (eastward) and V (northward) wind components
    """
    # Meteorological direction: 0=N, 90=E, 180=S, 270=W
    # Convert to math angle (counterclockwise from east)
    angle_rad = np.radians(270 - direction)
    u = speed * np.cos(angle_rad)
    v = speed * np.sin(angle_rad)
    return u, v


def load_ascii_grid(filepath: Path) -> xr.DataArray:
    """
    Load ESRI ASCII grid file into xarray DataArray.

    Parameters
    ----------
    filepath : Path
        Path to .asc file

    Returns
    -------
    xr.DataArray
        Data with x, y coordinates
    """
    filepath = Path(filepath)

    with open(filepath) as f:
        header = {}
        for _ in range(6):
            line = f.readline().split()
            header[line[0].lower()] = float(line[1])

        data = np.loadtxt(f)

    ncols = int(header["ncols"])
    nrows = int(header["nrows"])
    xll = header["xllcorner"]
    yll = header["yllcorner"]
    cellsize = header["cellsize"]
    nodata = header.get("nodata_value", -9999)

    data[data == nodata] = np.nan
    data = np.flipud(data)

    x = np.arange(xll, xll + ncols * cellsize, cellsize)[:ncols]
    y = np.arange(yll, yll + nrows * cellsize, cellsize)[:nrows]

    da = xr.DataArray(
        data, dims=["y", "x"], coords={"y": y, "x": x}, attrs={"source": str(filepath)}
    )
    return da


def plot_wind_speed(
    ds: xr.Dataset,
    ax: Optional[plt.Axes] = None,
    title: str = "Wind Speed",
    units: str = "mph",
    cmap: str = WIND_SPEED_CMAP,
    levels: Optional[np.ndarray] = None,
    add_colorbar: bool = True,
    use_cartopy: bool = True,
) -> plt.Axes:
    """
    Plot wind speed as filled contours.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with wind_speed, si10, or u/v components
    ax : plt.Axes, optional
        Matplotlib axes (created if None)
    title : str
        Plot title
    units : str
        'mph' or 'm/s'
    cmap : str
        Colormap name
    levels : array, optional
        Contour levels
    add_colorbar : bool
        Whether to add colorbar
    use_cartopy : bool
        Use cartopy for map projection

    Returns
    -------
    plt.Axes
    """
    # Extract speed
    if "si10" in ds:
        speed = ds["si10"].values
    elif "wind_speed" in ds:
        speed = ds["wind_speed"].values
    elif "u10" in ds and "v10" in ds:
        speed = np.sqrt(ds["u10"].values ** 2 + ds["v10"].values ** 2)
    elif "u" in ds and "v" in ds:
        speed = np.sqrt(ds["u"].values ** 2 + ds["v"].values ** 2)
    else:
        raise ValueError("Cannot find wind speed in dataset")

    # Convert units
    if units == "mph":
        speed = ms_to_mph(speed)
        unit_label = "mph"
    else:
        unit_label = "m/s"

    # Get coordinates
    if "longitude" in ds.coords:
        lon, lat = ds["longitude"].values, ds["latitude"].values
    elif "x" in ds.coords:
        lon, lat = ds["x"].values, ds["y"].values
    else:
        raise ValueError("Cannot find coordinates in dataset")

    # Create axes if needed
    if ax is None:
        if use_cartopy and HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 8))

    # Set levels
    if levels is None:
        levels = WIND_SPEED_LEVELS if units == "mph" else WIND_SPEED_LEVELS / 2.237

    # Plot
    if use_cartopy and HAS_CARTOPY and hasattr(ax, "projection"):
        cf = ax.contourf(
            lon,
            lat,
            speed,
            levels=levels,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            extend="max",
        )
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        cf = ax.contourf(lon, lat, speed, levels=levels, cmap=cmap, extend="max")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    if add_colorbar:
        plt.colorbar(cf, ax=ax, label=f"Wind Speed ({unit_label})", shrink=0.8)

    ax.set_title(title)
    return ax


def plot_wind_barbs(
    ds: xr.Dataset,
    ax: Optional[plt.Axes] = None,
    title: str = "Wind Barbs",
    skip: int = 3,
    barb_length: float = 6,
    use_cartopy: bool = True,
    add_speed_background: bool = True,
) -> plt.Axes:
    """
    Plot wind barbs (meteorological convention).

    Wind barbs show:
    - Half barb = 5 knots
    - Full barb = 10 knots
    - Flag = 50 knots

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with u/v wind components
    ax : plt.Axes, optional
        Matplotlib axes
    title : str
        Plot title
    skip : int
        Plot every Nth barb
    barb_length : float
        Length of barbs
    use_cartopy : bool
        Use cartopy projection
    add_speed_background : bool
        Add wind speed contour fill

    Returns
    -------
    plt.Axes
    """
    # Extract components
    if "u10" in ds:
        u, v = ds["u10"].values, ds["v10"].values
    elif "u" in ds:
        u, v = ds["u"].values, ds["v"].values
    else:
        raise ValueError("Cannot find u/v components in dataset")

    # Get coordinates
    if "longitude" in ds.coords:
        lon, lat = ds["longitude"].values, ds["latitude"].values
    else:
        lon, lat = ds["x"].values, ds["y"].values

    # Convert m/s to knots for barbs
    u_knots = u * 1.944
    v_knots = v * 1.944

    # Create axes
    if ax is None:
        if use_cartopy and HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 8))

    # Background speed
    if add_speed_background:
        speed_mph = ms_to_mph(np.sqrt(u**2 + v**2))
        if use_cartopy and HAS_CARTOPY and hasattr(ax, "projection"):
            ax.contourf(
                lon,
                lat,
                speed_mph,
                levels=WIND_SPEED_LEVELS,
                cmap=WIND_SPEED_CMAP,
                alpha=0.5,
                transform=ccrs.PlateCarree(),
            )
        else:
            ax.contourf(
                lon,
                lat,
                speed_mph,
                levels=WIND_SPEED_LEVELS,
                cmap=WIND_SPEED_CMAP,
                alpha=0.5,
            )

    # Subsample for barbs - handle 2D coordinate arrays
    if lon.ndim == 2:
        lon_grid = lon[::skip, ::skip]
        lat_grid = lat[::skip, ::skip]
    else:
        lon_grid, lat_grid = np.meshgrid(lon[::skip], lat[::skip])
    u_sub = u_knots[::skip, ::skip]
    v_sub = v_knots[::skip, ::skip]

    # Plot barbs
    if use_cartopy and HAS_CARTOPY and hasattr(ax, "projection"):
        ax.barbs(
            lon_grid,
            lat_grid,
            u_sub,
            v_sub,
            length=barb_length,
            transform=ccrs.PlateCarree(),
            barbcolor="black",
            flagcolor="black",
        )
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        ax.barbs(
            lon_grid,
            lat_grid,
            u_sub,
            v_sub,
            length=barb_length,
            barbcolor="black",
            flagcolor="black",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax.set_title(title)
    return ax


def plot_streamlines(
    ds: xr.Dataset,
    ax: Optional[plt.Axes] = None,
    title: str = "Wind Streamlines",
    density: float = 1.5,
    linewidth: float = 1.0,
    color_by_speed: bool = True,
    use_cartopy: bool = True,
) -> plt.Axes:
    """
    Plot wind streamlines.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with u/v wind components
    ax : plt.Axes, optional
        Matplotlib axes
    title : str
        Plot title
    density : float
        Streamline density
    linewidth : float
        Line width
    color_by_speed : bool
        Color streamlines by wind speed
    use_cartopy : bool
        Use cartopy projection

    Returns
    -------
    plt.Axes
    """
    # Extract components
    if "u10" in ds:
        u, v = ds["u10"].values, ds["v10"].values
    elif "u" in ds:
        u, v = ds["u"].values, ds["v"].values
    else:
        raise ValueError("Cannot find u/v components in dataset")

    # Get coordinates
    if "longitude" in ds.coords:
        lon, lat = ds["longitude"].values, ds["latitude"].values
    else:
        lon, lat = ds["x"].values, ds["y"].values

    speed = np.sqrt(u**2 + v**2)

    # Create axes
    if ax is None:
        if use_cartopy and HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 8))

    # Streamlines require 1D coordinate arrays and 2D data
    # Note: Cartopy streamplot has some quirks with projections
    if use_cartopy and HAS_CARTOPY and hasattr(ax, "projection"):
        if color_by_speed:
            strm = ax.streamplot(
                lon,
                lat,
                u,
                v,
                density=density,
                linewidth=linewidth,
                color=ms_to_mph(speed),
                cmap=WIND_SPEED_CMAP,
                transform=ccrs.PlateCarree(),
            )
            plt.colorbar(strm.lines, ax=ax, label="Wind Speed (mph)", shrink=0.8)
        else:
            ax.streamplot(
                lon,
                lat,
                u,
                v,
                density=density,
                linewidth=linewidth,
                color="black",
                transform=ccrs.PlateCarree(),
            )

        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        if color_by_speed:
            strm = ax.streamplot(
                lon,
                lat,
                u,
                v,
                density=density,
                linewidth=linewidth,
                color=ms_to_mph(speed),
                cmap=WIND_SPEED_CMAP,
            )
            plt.colorbar(strm.lines, ax=ax, label="Wind Speed (mph)", shrink=0.8)
        else:
            ax.streamplot(
                lon, lat, u, v, density=density, linewidth=linewidth, color="black"
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax.set_title(title)
    return ax


def plot_quiver(
    ds: xr.Dataset,
    ax: Optional[plt.Axes] = None,
    title: str = "Wind Vectors",
    skip: int = 2,
    scale: float = 150,
    width: float = 0.003,
    add_speed_background: bool = True,
    use_cartopy: bool = True,
) -> plt.Axes:
    """
    Plot wind quiver (arrow) plot.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with u/v wind components
    ax : plt.Axes, optional
        Matplotlib axes
    title : str
        Plot title
    skip : int
        Plot every Nth arrow
    scale : float
        Arrow scale factor
    width : float
        Arrow width
    add_speed_background : bool
        Add wind speed contour fill
    use_cartopy : bool
        Use cartopy projection

    Returns
    -------
    plt.Axes
    """
    # Extract components
    if "u10" in ds:
        u, v = ds["u10"].values, ds["v10"].values
    elif "u" in ds:
        u, v = ds["u"].values, ds["v"].values
    else:
        raise ValueError("Cannot find u/v components in dataset")

    # Get coordinates
    if "longitude" in ds.coords:
        lon, lat = ds["longitude"].values, ds["latitude"].values
    else:
        lon, lat = ds["x"].values, ds["y"].values

    # Create axes
    if ax is None:
        if use_cartopy and HAS_CARTOPY:
            fig, ax = plt.subplots(
                figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()}
            )
        else:
            fig, ax = plt.subplots(figsize=(12, 8))

    # Background speed
    if add_speed_background:
        speed_mph = ms_to_mph(np.sqrt(u**2 + v**2))
        if use_cartopy and HAS_CARTOPY and hasattr(ax, "projection"):
            cf = ax.contourf(
                lon,
                lat,
                speed_mph,
                levels=WIND_SPEED_LEVELS,
                cmap=WIND_SPEED_CMAP,
                alpha=0.6,
                transform=ccrs.PlateCarree(),
            )
        else:
            cf = ax.contourf(
                lon,
                lat,
                speed_mph,
                levels=WIND_SPEED_LEVELS,
                cmap=WIND_SPEED_CMAP,
                alpha=0.6,
            )
        plt.colorbar(cf, ax=ax, label="Wind Speed (mph)", shrink=0.8)

    # Subsample - handle both 1D and 2D coordinate arrays
    if lon.ndim == 2:
        lon_grid = lon[::skip, ::skip]
        lat_grid = lat[::skip, ::skip]
    else:
        lon_grid, lat_grid = np.meshgrid(lon[::skip], lat[::skip])
    u_sub = u[::skip, ::skip]
    v_sub = v[::skip, ::skip]

    # Quiver
    if use_cartopy and HAS_CARTOPY and hasattr(ax, "projection"):
        q = ax.quiver(
            lon_grid,
            lat_grid,
            u_sub,
            v_sub,
            scale=scale,
            width=width,
            headwidth=4,
            transform=ccrs.PlateCarree(),
        )
        ax.quiverkey(q, 0.85, 0.02, 10, "10 m/s", labelpos="W")
        ax.add_feature(cfeature.STATES, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        q = ax.quiver(
            lon_grid, lat_grid, u_sub, v_sub, scale=scale, width=width, headwidth=4
        )
        ax.quiverkey(q, 0.85, 0.02, 10, "10 m/s", labelpos="W")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    ax.set_title(title)
    return ax


def compare_resolutions(
    coarse_ds: xr.Dataset,
    fine_ds: xr.Dataset,
    coarse_label: str = "HRRR (3km)",
    fine_label: str = "WindNinja (100m)",
    title: str = "Resolution Comparison",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Side-by-side comparison of coarse and fine resolution wind fields.

    Parameters
    ----------
    coarse_ds : xr.Dataset
        Coarse resolution dataset (e.g., HRRR)
    fine_ds : xr.Dataset
        Fine resolution dataset (e.g., WindNinja output)
    coarse_label : str
        Label for coarse data
    fine_label : str
        Label for fine data
    title : str
        Overall title
    save_path : Path, optional
        Save figure to this path

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Coarse
    plot_wind_speed(
        coarse_ds, ax=axes[0], title=coarse_label, use_cartopy=False, add_colorbar=True
    )

    # Fine
    plot_wind_speed(
        fine_ds, ax=axes[1], title=fine_label, use_cartopy=False, add_colorbar=True
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved comparison figure to {save_path}")

    return fig


def animate_time_series(
    ds: xr.Dataset,
    time_dim: str = "time",
    interval: int = 500,
    save_path: Optional[Path] = None,
    title_template: str = "Wind Speed - {time}",
) -> FuncAnimation:
    """
    Create animated GIF/MP4 of wind field time series.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with time dimension
    time_dim : str
        Name of time dimension
    interval : int
        Milliseconds between frames
    save_path : Path, optional
        Save animation to this path (.gif or .mp4)
    title_template : str
        Title template with {time} placeholder

    Returns
    -------
    FuncAnimation
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    times = ds[time_dim].values
    n_frames = len(times)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get data ranges for consistent colorbar
    if "si10" in ds:
        speed_data = ds["si10"].values
    elif "wind_speed" in ds:
        speed_data = ds["wind_speed"].values
    else:
        u = ds.get("u10", ds.get("u")).values
        v = ds.get("v10", ds.get("v")).values
        speed_data = np.sqrt(u**2 + v**2)

    speed_mph = ms_to_mph(speed_data)
    vmin, vmax = 0, np.nanmax(speed_mph) * 1.1

    # Get coordinates
    if "longitude" in ds.coords:
        lon, lat = ds["longitude"].values, ds["latitude"].values
    else:
        lon, lat = ds["x"].values, ds["y"].values

    # Initial plot
    cf = ax.contourf(
        lon,
        lat,
        speed_mph[0],
        levels=WIND_SPEED_LEVELS,
        cmap=WIND_SPEED_CMAP,
        extend="max",
    )
    plt.colorbar(cf, ax=ax, label="Wind Speed (mph)")
    title = ax.set_title(title_template.format(time=str(times[0])[:19]))

    def update(frame):
        ax.clear()
        ax.contourf(
            lon,
            lat,
            speed_mph[frame],
            levels=WIND_SPEED_LEVELS,
            cmap=WIND_SPEED_CMAP,
            extend="max",
        )
        ax.set_title(title_template.format(time=str(times[frame])[:19]))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        return ax

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        save_path = Path(save_path)
        if save_path.suffix == ".gif":
            anim.save(save_path, writer="pillow", fps=1000 // interval)
        else:
            anim.save(save_path, writer="ffmpeg", fps=1000 // interval)
        logger.info(f"Saved animation to {save_path}")

    return anim


def plot_fire_weather_summary(
    ds: xr.Dataset,
    red_flag_speed: float = RED_FLAG_SPEED_MPH,
    title: str = "Fire Weather Summary",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a fire weather summary plot with Red Flag Warning thresholds.

    Parameters
    ----------
    ds : xr.Dataset
        Wind dataset
    red_flag_speed : float
        Red Flag Warning threshold (mph)
    title : str
        Plot title
    save_path : Path, optional
        Save path

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Extract speed
    if "si10" in ds:
        speed_ms = ds["si10"].values
    elif "wind_speed" in ds:
        speed_ms = ds["wind_speed"].values
    else:
        u = ds.get("u10", ds.get("u")).values
        v = ds.get("v10", ds.get("v")).values
        speed_ms = np.sqrt(u**2 + v**2)

    speed_mph = ms_to_mph(speed_ms)

    # Get coordinates
    if "longitude" in ds.coords:
        lon, lat = ds["longitude"].values, ds["latitude"].values
    else:
        lon, lat = ds["x"].values, ds["y"].values

    # 1. Wind speed
    ax1 = axes[0, 0]
    cf1 = ax1.contourf(
        lon,
        lat,
        speed_mph,
        levels=WIND_SPEED_LEVELS,
        cmap=WIND_SPEED_CMAP,
        extend="max",
    )
    plt.colorbar(cf1, ax=ax1, label="Wind Speed (mph)")
    ax1.set_title("Wind Speed")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")

    # 2. Red Flag areas
    ax2 = axes[0, 1]
    red_flag_mask = speed_mph >= red_flag_speed
    cmap_rf = mcolors.ListedColormap(["lightgray", "red"])
    ax2.contourf(lon, lat, red_flag_mask.astype(int), levels=[0, 0.5, 1], cmap=cmap_rf)
    ax2.set_title(f"Red Flag Warning Areas (≥{red_flag_speed} mph)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor="lightgray", label="Below threshold"),
        mpatches.Patch(facecolor="red", label="Red Flag Warning"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right")

    # 3. Wind direction (if available)
    ax3 = axes[1, 0]
    if "wdir10" in ds:
        direction = ds["wdir10"].values
    elif "wind_direction" in ds:
        direction = ds["wind_direction"].values
    else:
        u = ds.get("u10", ds.get("u")).values
        v = ds.get("v10", ds.get("v")).values
        direction = (270 - np.degrees(np.arctan2(v, u))) % 360

    cf3 = ax3.contourf(lon, lat, direction, levels=np.arange(0, 361, 22.5), cmap="hsv")
    plt.colorbar(cf3, ax=ax3, label="Wind Direction (°)")
    ax3.set_title("Wind Direction")
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")

    # 4. Histogram of wind speeds
    ax4 = axes[1, 1]
    ax4.hist(speed_mph.flatten(), bins=30, edgecolor="black", alpha=0.7)
    ax4.axvline(
        red_flag_speed,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Red Flag ({red_flag_speed} mph)",
    )
    ax4.set_xlabel("Wind Speed (mph)")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Wind Speed Distribution")
    ax4.legend()

    # Statistics
    pct_red_flag = 100 * np.sum(red_flag_mask) / red_flag_mask.size
    stats_text = (
        f"Max: {np.nanmax(speed_mph):.1f} mph\n"
        f"Mean: {np.nanmean(speed_mph):.1f} mph\n"
        f"Area ≥{red_flag_speed} mph: {pct_red_flag:.1f}%"
    )
    ax4.text(
        0.98,
        0.98,
        stats_text,
        transform=ax4.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved fire weather summary to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    print("Wind Visualization Utilities")
    print("=" * 40)
    print("\nAvailable functions:")
    print("  - plot_wind_speed(): Filled contour plot")
    print("  - plot_wind_barbs(): Meteorological wind barbs")
    print("  - plot_streamlines(): Flow visualization")
    print("  - plot_quiver(): Arrow/vector plot")
    print("  - compare_resolutions(): Side-by-side comparison")
    print("  - animate_time_series(): Create animations")
    print("  - plot_fire_weather_summary(): Fire weather analysis")
    print("\nHelpers:")
    print("  - load_ascii_grid(): Load WindNinja ASCII output")
    print("  - ms_to_mph() / mph_to_ms(): Unit conversion")
    print("  - wind_components_from_speed_dir(): U/V from speed/direction")
