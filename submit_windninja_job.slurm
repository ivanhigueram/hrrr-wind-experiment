#!/bin/bash
#===============================================================================
# HRRR Wind + WindNinja Batch Processing Script
# 
# This script demonstrates a complete HPC workflow for:
# 1. Fetching HRRR wind data for a region
# 2. Running WindNinja downscaling
# 3. Post-processing results
#
# Usage:
#   sbatch submit_windninja_job.slurm
#
# Or run sections interactively:
#   ./submit_windninja_job.slurm --interactive
#===============================================================================

#SBATCH --job-name=hrrr_windninja
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/windninja_%j.out
#SBATCH --error=logs/windninja_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@example.com

#===============================================================================
# Configuration - Edit these for your run
#===============================================================================

# Region of interest (west, south, east, north)
REGION_NAME="paradise_ca"
WEST=-122.0
SOUTH=39.5
EAST=-121.0
NORTH=40.0

# Time settings (UTC)
# Leave empty to use most recent available
FETCH_DATE=""  # Format: YYYYMMDD
FETCH_HOUR=""  # Format: HH (00-23)
FORECAST_HOURS="0 1 2 3 6 12"  # Which forecast hours to process

# WindNinja settings
MESH_RESOLUTION=100  # Output resolution in meters (30-500)
VEGETATION="brush"   # grass, brush, or trees
NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Paths
WORK_DIR="${SCRATCH:-/tmp}/windninja_${SLURM_JOB_ID:-$$}"
OUTPUT_DIR="./output/${REGION_NAME}_$(date +%Y%m%d)"
CONTAINER_PATH="./windninja.sif"
DEM_PATH="./data/dem_${REGION_NAME}.tif"

# Python environment
PYTHON_ENV="./venv"
CONDA_ENV=""  # Set if using conda instead

#===============================================================================
# Setup
#===============================================================================

set -euo pipefail

# Create directories
mkdir -p "${WORK_DIR}"/{hrrr,dem,windninja_input,windninja_output}
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=============================================="
echo "HRRR + WindNinja Processing Job"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Region: ${REGION_NAME} (${WEST}, ${SOUTH}, ${EAST}, ${NORTH})"
echo "Working directory: ${WORK_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=============================================="

# Load modules (adjust for your HPC system)
module purge 2>/dev/null || true
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
module load python/3.10 2>/dev/null || true
module load gdal 2>/dev/null || true

# Activate Python environment
if [[ -n "${CONDA_ENV}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
elif [[ -d "${PYTHON_ENV}" ]]; then
    source "${PYTHON_ENV}/bin/activate"
fi

# Set OpenMP threads
export OMP_NUM_THREADS=${NUM_THREADS}

#===============================================================================
# Step 1: Fetch HRRR Data
#===============================================================================

echo ""
echo "[Step 1/4] Fetching HRRR wind data..."
echo "----------------------------------------------"

# Determine fetch time
if [[ -z "${FETCH_DATE}" ]]; then
    # Use 2 hours ago (HRRR data availability lag)
    FETCH_DATE=$(date -u -d '2 hours ago' +%Y%m%d)
    FETCH_HOUR=$(date -u -d '2 hours ago' +%H)
fi

echo "Fetching HRRR for: ${FETCH_DATE} ${FETCH_HOUR}Z"

# Create Python script for HRRR fetch
cat > "${WORK_DIR}/fetch_hrrr.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Fetch HRRR data for WindNinja processing."""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fetch_hrrr import HRRRWindFetcher

def main():
    # Get parameters from environment
    date_str = os.environ.get('FETCH_DATE')
    hour_str = os.environ.get('FETCH_HOUR', '00')
    forecast_hours = [int(h) for h in os.environ.get('FORECAST_HOURS', '0').split()]
    
    west = float(os.environ.get('WEST', '-122'))
    south = float(os.environ.get('SOUTH', '39'))
    east = float(os.environ.get('EAST', '-121'))
    north = float(os.environ.get('NORTH', '40'))
    
    work_dir = Path(os.environ.get('WORK_DIR', './work'))
    
    # Parse date
    date = datetime.strptime(f"{date_str}{hour_str}", "%Y%m%d%H")
    bounds = (west, south, east, north)
    
    print(f"Fetching HRRR for {date} UTC")
    print(f"Bounds: {bounds}")
    print(f"Forecast hours: {forecast_hours}")
    
    fetcher = HRRRWindFetcher(cache_dir=work_dir / "hrrr")
    
    for fxx in forecast_hours:
        print(f"\n--- Forecast hour F{fxx:02d} ---")
        
        try:
            ds = fetcher.fetch_surface_winds(
                date=date,
                forecast_hour=fxx,
                bounds=bounds,
            )
            
            # Export to NetCDF and ASCII
            output_prefix = f"hrrr_{date_str}_{hour_str}z_f{fxx:02d}"
            
            nc_path = work_dir / "hrrr" / f"{output_prefix}.nc"
            fetcher.to_netcdf(ds, nc_path)
            
            speed_path, dir_path = fetcher.to_windninja_ascii(
                ds,
                output_dir=work_dir / "windninja_input",
                filename_prefix=output_prefix,
            )
            
            print(f"  NetCDF: {nc_path}")
            print(f"  Speed ASCII: {speed_path}")
            print(f"  Direction ASCII: {dir_path}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    print("\nHRRR fetch complete!")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

# Run HRRR fetch
export FETCH_DATE FETCH_HOUR FORECAST_HOURS WEST SOUTH EAST NORTH WORK_DIR
python3 "${WORK_DIR}/fetch_hrrr.py"

#===============================================================================
# Step 2: Prepare DEM
#===============================================================================

echo ""
echo "[Step 2/4] Preparing DEM..."
echo "----------------------------------------------"

if [[ -f "${DEM_PATH}" ]]; then
    echo "Using existing DEM: ${DEM_PATH}"
    cp "${DEM_PATH}" "${WORK_DIR}/dem/dem.tif"
else
    echo "DEM not found at ${DEM_PATH}"
    echo "Attempting to download from 3DEP..."
    
    # Create Python script for DEM download
    cat > "${WORK_DIR}/download_dem.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""Download DEM from USGS 3DEP."""

import os
from pathlib import Path

try:
    import py3dep
    
    west = float(os.environ.get('WEST', '-122'))
    south = float(os.environ.get('SOUTH', '39'))
    east = float(os.environ.get('EAST', '-121'))
    north = float(os.environ.get('NORTH', '40'))
    work_dir = Path(os.environ.get('WORK_DIR', './work'))
    
    bounds = (west, south, east, north)
    print(f"Downloading 30m DEM for bounds: {bounds}")
    
    dem = py3dep.get_dem(bounds, resolution=30)
    dem_path = work_dir / "dem" / "dem.tif"
    dem.rio.to_raster(dem_path)
    
    print(f"DEM saved to: {dem_path}")
    
except ImportError:
    print("py3dep not installed. Please provide DEM manually.")
    print("Install with: pip install py3dep")
    exit(1)
except Exception as e:
    print(f"DEM download failed: {e}")
    exit(1)
PYTHON_SCRIPT

    python3 "${WORK_DIR}/download_dem.py" || {
        echo "ERROR: Could not obtain DEM. Please provide one at ${DEM_PATH}"
        exit 1
    }
fi

DEM_FILE="${WORK_DIR}/dem/dem.tif"

# Verify DEM
if command -v gdalinfo &> /dev/null; then
    echo "DEM info:"
    gdalinfo "${DEM_FILE}" | head -20
fi

#===============================================================================
# Step 3: Run WindNinja
#===============================================================================

echo ""
echo "[Step 3/4] Running WindNinja..."
echo "----------------------------------------------"

# Check for container
if [[ ! -f "${CONTAINER_PATH}" ]]; then
    echo "WindNinja container not found at ${CONTAINER_PATH}"
    echo "Pulling from Docker Hub..."
    apptainer pull "${CONTAINER_PATH}" docker://firelab/windninja:latest || {
        echo "ERROR: Failed to pull WindNinja container"
        exit 1
    }
fi

# Process each forecast hour
for fxx in ${FORECAST_HOURS}; do
    echo ""
    echo "--- Processing forecast hour F${fxx} ---"
    
    input_prefix="hrrr_${FETCH_DATE}_${FETCH_HOUR}z_f$(printf '%02d' ${fxx})"
    output_subdir="${WORK_DIR}/windninja_output/f$(printf '%02d' ${fxx})"
    mkdir -p "${output_subdir}"
    
    # Check if input exists
    speed_file="${WORK_DIR}/windninja_input/${input_prefix}_speed.asc"
    if [[ ! -f "${speed_file}" ]]; then
        echo "  Skipping - input not found: ${speed_file}"
        continue
    fi
    
    # Calculate domain-average wind from ASCII file
    # (In production, you might use wxModelInitialization with the GRIB2 file)
    avg_speed=$(awk 'NR>6 {for(i=1;i<=NF;i++) {sum+=$i; n++}} END {print sum/n}' "${speed_file}")
    avg_dir=$(awk 'NR>6 {for(i=1;i<=NF;i++) {sum+=$i; n++}} END {print sum/n}' \
              "${WORK_DIR}/windninja_input/${input_prefix}_direction.asc")
    
    echo "  Input wind: ${avg_speed} m/s from ${avg_dir}°"
    
    # Run WindNinja
    apptainer exec \
        --bind "${WORK_DIR}:${WORK_DIR}" \
        "${CONTAINER_PATH}" \
        WindNinja_cli \
        --elevation_file="${DEM_FILE}" \
        --initialization_method=domainAverageInitialization \
        --input_speed="${avg_speed}" \
        --input_speed_units=mps \
        --input_direction="${avg_dir}" \
        --mesh_resolution="${MESH_RESOLUTION}" \
        --units_mesh_resolution=m \
        --output_speed_units=mph \
        --output_wind_height=10.0 \
        --units_output_wind_height=m \
        --vegetation="${VEGETATION}" \
        --num_threads="${NUM_THREADS}" \
        --output_path="${output_subdir}" \
        --write_ascii_output=true \
        --write_shapefile_output=true
    
    echo "  Output: ${output_subdir}"
    ls -la "${output_subdir}"
done

#===============================================================================
# Step 4: Post-processing and Cleanup
#===============================================================================

echo ""
echo "[Step 4/4] Post-processing..."
echo "----------------------------------------------"

# Copy results to output directory
cp -r "${WORK_DIR}/windninja_output"/* "${OUTPUT_DIR}/"
cp -r "${WORK_DIR}/hrrr"/*.nc "${OUTPUT_DIR}/" 2>/dev/null || true

# Create summary
cat > "${OUTPUT_DIR}/run_summary.txt" << EOF
HRRR + WindNinja Processing Summary
====================================
Job ID: ${SLURM_JOB_ID:-interactive}
Date: $(date)
Region: ${REGION_NAME}
Bounds: (${WEST}, ${SOUTH}, ${EAST}, ${NORTH})

HRRR Settings:
  Date: ${FETCH_DATE} ${FETCH_HOUR}Z
  Forecast hours: ${FORECAST_HOURS}

WindNinja Settings:
  Mesh resolution: ${MESH_RESOLUTION}m
  Vegetation: ${VEGETATION}
  Threads: ${NUM_THREADS}

Output Files:
$(find "${OUTPUT_DIR}" -type f -name "*.asc" -o -name "*.shp" -o -name "*.nc" | head -20)
EOF

echo "Summary written to: ${OUTPUT_DIR}/run_summary.txt"

# Cleanup work directory
if [[ "${SLURM_JOB_ID:-}" ]]; then
    echo "Cleaning up work directory..."
    rm -rf "${WORK_DIR}"
fi

#===============================================================================
# Done
#===============================================================================

echo ""
echo "=============================================="
echo "Job completed successfully!"
echo "End time: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="
