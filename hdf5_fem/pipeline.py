"""End-to-end processing orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .config import FemConfig
from .fenics_solver import solve_temperature_history
from .hdf5_io import (
    prepare_output_file,
    prepare_output_file_to_path,
    read_case_parameters,
    validate_input_h5,
    write_fem_results,
)
from .topology import orient_triangles_3d, read_or_recover_triangles, validate_triangle_indices
from .units import length_mm_to_m


class DatasetTopologyCache:
    def __init__(self):
        self.triangles = None
        self.triangle_source = None
        self.num_nodes = None

    def resolve_for_file(self, h5_file):
        _, num_nodes = validate_input_h5(h5_file)
        if self.triangles is None:
            triangles, triangle_source = read_or_recover_triangles(h5_file)
            triangles = orient_triangles_3d(
                triangles,
                length_mm_to_m(h5_file["dynamic/xyz"][0, :, :]),
                h5_file["dynamic/normal"][0, :, :],
            )
            validate_triangle_indices(triangles, num_nodes)
            self.triangles = np.asarray(triangles, dtype=np.int64)
            self.triangle_source = triangle_source
            self.num_nodes = int(num_nodes)
        elif int(num_nodes) != int(self.num_nodes):
            raise ValueError(
                "Dataset topology cache expects "
                f"{self.num_nodes} nodes, but {h5_file.filename} has {num_nodes} nodes."
            )
        return self.triangles, self.triangle_source


def process_file(
    input_path,
    *,
    config: FemConfig,
    output_dir=None,
    output_path=None,
    overwrite=False,
    in_place=False,
    pvd_path=None,
    topology_cache: Optional[DatasetTopologyCache] = None,
    triangles=None,
    triangle_source=None,
    progress_callback=None,
) -> Path:
    if output_path is None:
        if output_dir is None:
            raise ValueError("Either output_dir or output_path must be provided.")
        output_path = prepare_output_file(input_path, output_dir, overwrite=overwrite, in_place=in_place)
    else:
        output_path = prepare_output_file_to_path(input_path, output_path, overwrite=overwrite, in_place=in_place)
    with h5py.File(output_path, "r+") as h5_file:
        num_frames, num_nodes = validate_input_h5(h5_file)
        if topology_cache is not None:
            triangles, triangle_source = topology_cache.resolve_for_file(h5_file)
        elif triangles is None:
            triangles, triangle_source = read_or_recover_triangles(h5_file)
            triangles = orient_triangles_3d(
                triangles,
                length_mm_to_m(h5_file["dynamic/xyz"][0, :, :]),
                h5_file["dynamic/normal"][0, :, :],
            )
        elif triangle_source is None:
            triangle_source = "provided_dataset_topology"
        validate_triangle_indices(triangles, num_nodes)
        case_parameters = read_case_parameters(h5_file)

        if config.output.write_pvd and pvd_path is None:
            pvd_dir = Path(config.output.pvd_dir)
            pvd_path = pvd_dir / Path(output_path).stem / "temperature.pvd"
        if not config.output.write_pvd:
            pvd_path = None

        result = solve_temperature_history(
            h5_file,
            config=config,
            triangles=triangles,
            case_parameters=case_parameters,
            pvd_path=pvd_path,
            progress_callback=progress_callback,
        )

        metadata = _build_metadata(
            input_path=Path(input_path),
            output_path=output_path,
            config=config,
            num_frames=num_frames,
            num_nodes=num_nodes,
            num_triangles=len(triangles),
            triangle_source=triangle_source,
            case_parameters=case_parameters,
            pvd_path=pvd_path,
            mesh_build_count=result.get("mesh_build_count"),
            mesh_geometry=result.get("mesh_geometry"),
            topology_cache_used=topology_cache is not None,
        )
        write_fem_results(
            h5_file,
            triangles=triangles,
            fem_time=result["time"],
            temperature=result["temperature"],
            temperature_unit=config.temperature.unit,
            valid_mask=result["valid_mask"],
            metadata=metadata,
        )
    return output_path


def _build_metadata(
    *,
    input_path: Path,
    output_path: Path,
    config: FemConfig,
    num_frames: int,
    num_nodes: int,
    num_triangles: int,
    triangle_source: str,
    case_parameters: dict,
    pvd_path,
    mesh_build_count,
    mesh_geometry,
    topology_cache_used: bool,
) -> dict:
    return {
        "solver": "fenics-dolfin",
        "equation": "transient convection diffusion heat equation",
        "temperature_unit": config.temperature.unit,
        "coordinate_unit_input": "mm",
        "coordinate_unit_solver": "m",
        "heat_flux_unit_input": "W/mm^2",
        "heat_flux_unit_solver": "W/m^2",
        "dt_source": "path/heat_center_step_distance / velocity_speed",
        "dt_s": case_parameters["dt_s"],
        "num_frames": int(num_frames),
        "num_nodes": int(num_nodes),
        "num_triangles": int(num_triangles),
        "triangle_source": triangle_source,
        "mesh_geometry": mesh_geometry or "3d_surface",
        "mesh_build_count": mesh_build_count,
        "mesh_topology_scope": "dataset_directory" if topology_cache_used else "single_file",
        "topology_validation": "node_count_only" if topology_cache_used else "triangle_indices",
        "parameter_priority": {
            "case_parameters": "HDF5",
            "material_parameters": "external_json",
            "solver_parameters": "external_json",
            "visualization_parameters": "external_json",
        },
        "external_config": {
            "path": str(config.source_path),
            "material": {
                "rho": config.material.rho,
                "Cp": config.material.Cp,
                "K_parallel": config.material.K_parallel,
                "k_ratio": config.material.k_ratio,
                "heat_source_effective_thickness": config.material.heat_source_effective_thickness,
                "heat_source_absorptivity": config.material.heat_source_absorptivity,
            },
            "temperature": {
                "T_amb": config.temperature.T_amb,
                "unit": config.temperature.unit,
            },
            "solver": {
                "time_scheme": config.solver.time_scheme,
                "linear_solver": config.solver.linear_solver,
            },
            "output": {
                "write_pvd": config.output.write_pvd,
                "pvd_dir": config.output.pvd_dir,
                "pvd_stride": config.output.pvd_stride,
            },
        },
        "hdf5_case_parameters": {
            "velocity_speed_mm_s": case_parameters["velocity_speed_mm_s"],
            "velocity_direction_local": case_parameters["velocity_direction_local"].tolist(),
            "heat_center_step_distance_mm": case_parameters["heat_center_step_distance_mm"],
            "time_step_attr_s": case_parameters["time_step_attr_s"],
        },
        "boundary_condition": {
            "upwind": "Dirichlet T_amb",
            "side": "Dirichlet T_amb",
            "downwind": "zero Neumann",
        },
        "node_alignment": "fem/temperature[t,i,0] matches dynamic/xyz[t,i,:]",
        "requires_mesh_triangles": True,
        "visualization": {
            "write_pvd": config.output.write_pvd,
            "pvd_path": None if pvd_path is None else str(pvd_path),
            "pvd_stride": config.output.pvd_stride,
        },
        "source_h5": str(input_path),
        "output_h5": str(output_path),
    }
