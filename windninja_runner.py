"""
WindNinja Runner for HPC

Provides utilities to run WindNinja on HPC systems using Apptainer/Singularity.
WindNinja downscales coarse NWP winds (like HRRR at 3km) to high-resolution
terrain-adjusted winds (30-100m).

Usage on HPC:
    1. Build or pull the WindNinja container
    2. Prepare DEM and wind initialization files
    3. Run WindNinja via Apptainer
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WindNinjaConfig:
    """Configuration for a WindNinja run."""

    elevation_file: Path
    output_path: Path

    initialization_method: str = "domainAverageInitialization"
    input_speed: float = 10.0
    input_speed_units: str = "mph"
    input_direction: float = 270.0
    input_wind_height: float = (
        10.0  # Height at which input wind is measured (typically 10m for HRRR)
    )
    units_input_wind_height: str = "m"

    wx_model_type: Optional[str] = None
    forecast_filename: Optional[Path] = None

    mesh_resolution: float = 100.0
    units_mesh_resolution: str = "m"

    output_speed_units: str = "mph"
    output_wind_height: float = 10.0
    units_output_wind_height: str = "m"

    write_goog_output: bool = False
    write_shapefile_output: bool = False
    write_ascii_output: bool = True
    write_vtk_output: bool = False

    num_threads: int = 4

    vegetation: str = "grass"

    def to_cli_args(self) -> List[str]:
        """
        Convert config to WindNinja CLI arguments.

        Note: All file paths are converted to absolute paths to ensure
        they work correctly inside containers regardless of working directory.
        """
        # Convert paths to absolute to avoid issues with container working directory
        elevation_file_abs = Path(self.elevation_file).resolve()
        output_path_abs = Path(self.output_path).resolve()

        args = [
            f"--elevation_file={elevation_file_abs}",
            f"--initialization_method={self.initialization_method}",
            f"--mesh_resolution={self.mesh_resolution}",
            f"--units_mesh_resolution={self.units_mesh_resolution}",
            f"--output_speed_units={self.output_speed_units}",
            f"--output_wind_height={self.output_wind_height}",
            f"--units_output_wind_height={self.units_output_wind_height}",
            f"--vegetation={self.vegetation}",
            f"--num_threads={self.num_threads}",
            f"--output_path={output_path_abs}",
        ]

        if self.initialization_method == "domainAverageInitialization":
            args.extend(
                [
                    f"--input_speed={self.input_speed}",
                    f"--input_speed_units={self.input_speed_units}",
                    f"--input_direction={self.input_direction}",
                    f"--input_wind_height={self.input_wind_height}",
                    f"--units_input_wind_height={self.units_input_wind_height}",
                ]
            )
        elif self.initialization_method == "wxModelInitialization":
            if self.wx_model_type:
                args.append(f"--wx_model_type={self.wx_model_type}")
            if self.forecast_filename:
                forecast_file_abs = Path(self.forecast_filename).resolve()
                args.append(f"--forecast_filename={forecast_file_abs}")

        if self.write_goog_output:
            args.append("--write_goog_output=true")
        if self.write_shapefile_output:
            args.append("--write_shapefile_output=true")
        if self.write_ascii_output:
            args.append("--write_ascii_output=true")
        if self.write_vtk_output:
            args.append("--write_vtk_output=true")

        return args


class WindNinjaRunner:
    """Run WindNinja using Apptainer/Singularity on HPC."""

    CONTAINER_DOCKER_URI = "docker://firelab/windninja:latest"

    def __init__(
        self,
        container_path: Optional[Path] = None,
        use_apptainer: bool = True,
    ):
        """
        Initialize WindNinja runner.

        Parameters
        ----------
        container_path : Path, optional
            Path to existing .sif container. If None, will pull from Docker Hub.
        use_apptainer : bool
            Use 'apptainer' command (True) or 'singularity' (False)
        """
        self.container_path = container_path
        self.container_cmd = "apptainer" if use_apptainer else "singularity"
        self._check_container_runtime()

    def _check_container_runtime(self):
        """Verify Apptainer/Singularity is available."""
        try:
            result = subprocess.run(
                [self.container_cmd, "--version"],
                capture_output=True,
                text=True,
            )
            logger.info(f"Found: {result.stdout.strip()}")
        except FileNotFoundError:
            alt_cmd = (
                "singularity" if self.container_cmd == "apptainer" else "apptainer"
            )
            try:
                result = subprocess.run(
                    [alt_cmd, "--version"],
                    capture_output=True,
                    text=True,
                )
                self.container_cmd = alt_cmd
                logger.info(f"Using fallback: {result.stdout.strip()}")
            except FileNotFoundError:
                logger.warning(
                    "Neither apptainer nor singularity found. "
                    "Container operations will fail until installed."
                )

    def pull_container(self, output_path: Path) -> Path:
        """
        Pull WindNinja container from Docker Hub.

        Parameters
        ----------
        output_path : Path
            Where to save the .sif file

        Returns
        -------
        Path
            Path to the pulled container
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Pulling WindNinja container to {output_path}")

        cmd = [
            self.container_cmd,
            "pull",
            str(output_path),
            self.CONTAINER_DOCKER_URI,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to pull container: {result.stderr}")

        self.container_path = output_path
        logger.info(f"Successfully pulled container to {output_path}")
        return output_path

    def build_from_definition(
        self,
        definition_file: Path,
        output_path: Path,
    ) -> Path:
        """
        Build container from Singularity/Apptainer definition file.

        Note: Building typically requires root or fakeroot on HPC.

        Parameters
        ----------
        definition_file : Path
            Path to .def file
        output_path : Path
            Where to save the .sif file

        Returns
        -------
        Path
            Path to built container
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.container_cmd,
            "build",
            "--fakeroot",
            str(output_path),
            str(definition_file),
        ]

        logger.info(f"Building container from {definition_file}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to build container: {result.stderr}")

        self.container_path = output_path
        return output_path

    def run(
        self,
        config: WindNinjaConfig,
        bind_paths: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run WindNinja with the given configuration.

        Parameters
        ----------
        config : WindNinjaConfig
            WindNinja run configuration
        bind_paths : list, optional
            Additional paths to bind into container
        verbose : bool
            If True, print all stdout/stderr to console during execution

        Returns
        -------
        dict
            Run results including output file paths
        """
        if not self.container_path or not self.container_path.exists():
            raise RuntimeError("No container available. Call pull_container() first.")

        config.output_path.mkdir(parents=True, exist_ok=True)

        # Verify required files exist
        if not config.elevation_file.exists():
            raise RuntimeError(f"Elevation file not found: {config.elevation_file}")

        bind_dirs = set()
        bind_dirs.add(str(config.elevation_file.parent.resolve()))
        bind_dirs.add(str(config.output_path.resolve()))
        if config.forecast_filename:
            if not config.forecast_filename.exists():
                raise RuntimeError(
                    f"Forecast file not found: {config.forecast_filename}"
                )
            bind_dirs.add(str(config.forecast_filename.parent.resolve()))
        if bind_paths:
            bind_dirs.update(bind_paths)

        bind_str = ",".join(bind_dirs)

        cmd = [
            self.container_cmd,
            "exec",
            "--bind",
            bind_str,
            str(self.container_path),
            "WindNinja_cli",
        ] + config.to_cli_args()

        logger.info(f"Running WindNinja: {' '.join(cmd)}")
        logger.info(f"Elevation file: {config.elevation_file}")
        logger.info(f"Output directory: {config.output_path}")
        logger.info(f"Binding directories: {bind_str}")

        # Don't set cwd since we're using absolute paths now
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        # Combine stdout and stderr for better error reporting
        combined_output = result.stdout + result.stderr

        if verbose or result.returncode != 0:
            print("\n" + "=" * 60)
            print("STDOUT:")
            print(result.stdout if result.stdout else "(empty)")
            print("\nSTDERR:")
            print(result.stderr if result.stderr else "(empty)")
            print("=" * 60 + "\n")

        if result.returncode != 0:
            error_msg = (
                f"WindNinja execution failed with return code {result.returncode}"
            )
            if combined_output:
                error_msg += f"\n{combined_output}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        logger.info("WindNinja completed successfully")
        logger.debug(result.stdout)

        output_files = list(config.output_path.glob("*"))

        return {
            "success": True,
            "output_path": config.output_path,
            "output_files": output_files,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run_with_hrrr(
        self,
        elevation_file: Path,
        hrrr_file: Path,
        output_path: Path,
        mesh_resolution: float = 100.0,
        num_threads: int = 4,
    ) -> Dict[str, Any]:
        """
        Convenience method to run WindNinja with HRRR initialization.

        Parameters
        ----------
        elevation_file : Path
            DEM file (GeoTIFF)
        hrrr_file : Path
            HRRR GRIB2 or NetCDF file
        output_path : Path
            Output directory
        mesh_resolution : float
            Output resolution in meters
        num_threads : int
            Number of threads for computation

        Returns
        -------
        dict
            Run results
        """
        config = WindNinjaConfig(
            elevation_file=Path(elevation_file),
            output_path=Path(output_path),
            initialization_method="wxModelInitialization",
            wx_model_type="HRRR",
            forecast_filename=Path(hrrr_file),
            mesh_resolution=mesh_resolution,
            num_threads=num_threads,
            write_ascii_output=True,
            write_shapefile_output=True,
        )

        return self.run(config)

    def test_container(self) -> bool:
        """
        Test that the container is working and WindNinja_cli is available.

        Returns
        -------
        bool
            True if container works, False otherwise
        """
        if not self.container_path or not self.container_path.exists():
            logger.error("Container file does not exist")
            return False

        cmd = [
            self.container_cmd,
            "exec",
            str(self.container_path),
            "WindNinja_cli",
            "--help",
        ]

        logger.info(f"Testing container with: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.error(
                f"Container test failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            return False

        logger.info("Container test passed!")
        return True


def generate_slurm_script(
    config: WindNinjaConfig,
    container_path: Path,
    job_name: str = "windninja",
    partition: str = "standard",
    nodes: int = 1,
    cpus_per_task: int = 8,
    time_limit: str = "01:00:00",
    memory: str = "16G",
) -> str:
    """
    Generate a SLURM batch script for running WindNinja on HPC.

    Parameters
    ----------
    config : WindNinjaConfig
        WindNinja configuration
    container_path : Path
        Path to .sif container
    job_name : str
        SLURM job name
    partition : str
        SLURM partition
    nodes : int
        Number of nodes
    cpus_per_task : int
        CPUs per task
    time_limit : str
        Time limit (HH:MM:SS)
    memory : str
        Memory request

    Returns
    -------
    str
        SLURM script content
    """
    bind_dirs = [
        str(config.elevation_file.parent.resolve()),
        str(config.output_path.resolve()),
    ]
    if config.forecast_filename:
        bind_dirs.append(str(config.forecast_filename.parent.resolve()))

    bind_str = ",".join(set(bind_dirs))
    cli_args = " ".join(config.to_cli_args())

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --output={job_name}_%j.out
#SBATCH --error={job_name}_%j.err

module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting WindNinja job on $(hostname) at $(date)"
echo "Using {cpus_per_task} threads"

apptainer exec \\
    --bind {bind_str} \\
    {container_path} \\
    WindNinja_cli {cli_args}

echo "WindNinja completed at $(date)"
"""
    return script


if __name__ == "__main__":
    config = WindNinjaConfig(
        elevation_file=Path("./dem.tif"),
        output_path=Path("./output"),
        input_speed=15.0,
        input_direction=225.0,
        mesh_resolution=50.0,
        num_threads=8,
    )

    script = generate_slurm_script(
        config,
        container_path=Path("./windninja.sif"),
        cpus_per_task=8,
        time_limit="02:00:00",
    )

    print(script)
