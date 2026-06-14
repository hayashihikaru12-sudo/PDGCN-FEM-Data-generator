"""HDF5 reading, writing, and validation helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np


REQUIRED_DATASETS = (
    "dynamic/xyz",
    "dynamic/fiber",
    "dynamic/normal",
    "dynamic/Q",
    "edge_index",
    "boundary_nodes/upwind",
    "boundary_nodes/downwind",
    "boundary_nodes/side",
    "path/heat_center_step_distance",
)


def prepare_output_file(input_path, output_dir, *, overwrite=False, in_place=False) -> Path:
    return prepare_output_file_to_path(
        input_path,
        _default_output_path(input_path, output_dir),
        overwrite=overwrite,
        in_place=in_place,
    )


def prepare_output_file_to_path(input_path, output_path, *, overwrite=False, in_place=False) -> Path:
    input_path = Path(input_path)
    if in_place:
        return input_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}")
        output_path.unlink()
    shutil.copy2(input_path, output_path)
    return output_path


def _default_output_path(input_path, output_dir) -> Path:
    input_path = Path(input_path)
    return Path(output_dir) / f"{input_path.stem}_fem{input_path.suffix}"


def validate_input_h5(h5_file) -> tuple[int, int]:
    missing = [key for key in REQUIRED_DATASETS if key not in h5_file]
    if missing:
        raise KeyError(f"Missing required HDF5 datasets: {missing}")

    xyz_shape = h5_file["dynamic/xyz"].shape
    fiber_shape = h5_file["dynamic/fiber"].shape
    normal_shape = h5_file["dynamic/normal"].shape
    q_shape = h5_file["dynamic/Q"].shape

    if len(xyz_shape) != 3 or xyz_shape[2] != 3:
        raise ValueError(f"dynamic/xyz must have shape [T, N, 3], got {xyz_shape}.")
    if fiber_shape != xyz_shape:
        raise ValueError(f"dynamic/fiber shape {fiber_shape} must match dynamic/xyz {xyz_shape}.")
    if normal_shape != xyz_shape:
        raise ValueError(f"dynamic/normal shape {normal_shape} must match dynamic/xyz {xyz_shape}.")
    if len(q_shape) != 3 or q_shape[:2] != xyz_shape[:2] or q_shape[2] != 1:
        raise ValueError(f"dynamic/Q must have shape [T, N, 1], got {q_shape}.")

    num_frames, num_nodes = int(xyz_shape[0]), int(xyz_shape[1])
    _validate_index_dataset(h5_file["edge_index"][()], num_nodes, "edge_index")
    for name in ("upwind", "downwind", "side"):
        _validate_node_ids(h5_file[f"boundary_nodes/{name}"][()], num_nodes, f"boundary_nodes/{name}")
    return num_frames, num_nodes


def read_case_parameters(h5_file) -> dict:
    velocity_speed = _attr_float(h5_file, "velocity_speed")
    step_distance = float(h5_file["path/heat_center_step_distance"][()])
    velocity_direction = np.asarray(h5_file.attrs["velocity_direction_local"], dtype=np.float64)
    if velocity_direction.shape != (3,):
        raise ValueError("velocity_direction_local must have shape [3].")

    dt_from_motion = step_distance / velocity_speed
    time_step_attr = h5_file.attrs.get("time_step", None)
    if time_step_attr is not None and not np.isclose(float(time_step_attr), dt_from_motion, rtol=1.0e-4, atol=1.0e-9):
        raise ValueError(
            "HDF5 time_step is inconsistent with path/heat_center_step_distance / velocity_speed: "
            f"{float(time_step_attr)} vs {dt_from_motion}."
        )

    return {
        "velocity_speed_mm_s": float(velocity_speed),
        "velocity_direction_local": velocity_direction,
        "heat_center_step_distance_mm": float(step_distance),
        "dt_s": float(dt_from_motion),
        "time_step_attr_s": None if time_step_attr is None else float(time_step_attr),
    }


def write_fem_results(
    h5_file,
    *,
    triangles,
    fem_time,
    temperature,
    temperature_unit,
    valid_mask,
    metadata,
) -> None:
    _replace_dataset(h5_file, "mesh/triangles", np.asarray(triangles, dtype=np.int64))
    _replace_dataset(h5_file, "fem/time", np.asarray(fem_time, dtype=np.float64))
    _replace_dataset(h5_file, "fem/temperature", np.asarray(temperature, dtype=np.float32))
    _replace_dataset(h5_file, "fem/valid_mask", np.asarray(valid_mask, dtype=np.uint8))
    _replace_scalar_string(h5_file, "fem/temperature_unit", str(temperature_unit))
    _replace_scalar_string(
        h5_file,
        "fem/metadata_json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
    )


def validate_processed_h5(path) -> list[str]:
    errors = []
    with h5py.File(path, "r") as h5_file:
        try:
            num_frames, num_nodes = validate_input_h5(h5_file)
        except Exception as exc:
            errors.append(str(exc))
            return errors

        for key in ("mesh/triangles", "fem/time", "fem/temperature", "fem/temperature_unit", "fem/metadata_json"):
            if key not in h5_file:
                errors.append(f"Missing required processed dataset: {key}")

        if errors:
            return errors

        triangles = h5_file["mesh/triangles"][()]
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            errors.append(f"mesh/triangles must have shape [M, 3], got {triangles.shape}.")
        elif triangles.size and (triangles.min() < 0 or triangles.max() >= num_nodes):
            errors.append("mesh/triangles contains out-of-range node indices.")

        temperature = h5_file["fem/temperature"][()]
        if temperature.shape != (num_frames, num_nodes, 1):
            errors.append(
                f"fem/temperature must have shape {(num_frames, num_nodes, 1)}, got {temperature.shape}."
            )
        elif not np.isfinite(temperature).all():
            errors.append("fem/temperature contains NaN or Inf.")

        fem_time = h5_file["fem/time"][()]
        if fem_time.shape != (num_frames,):
            errors.append(f"fem/time must have shape {(num_frames,)}, got {fem_time.shape}.")
        elif not np.isfinite(fem_time).all():
            errors.append("fem/time contains NaN or Inf.")
    return errors


def _replace_dataset(h5_file, path: str, data) -> None:
    if path in h5_file:
        del h5_file[path]
    parent, name = path.rsplit("/", 1)
    group = h5_file.require_group(parent)
    group.create_dataset(name, data=data)


def _replace_scalar_string(h5_file, path: str, value: str) -> None:
    if path in h5_file:
        del h5_file[path]
    parent, name = path.rsplit("/", 1)
    group = h5_file.require_group(parent)
    string_type = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=value, dtype=string_type)


def _attr_float(h5_file, name: str) -> float:
    if name not in h5_file.attrs:
        raise KeyError(f"Missing HDF5 root attribute: {name}")
    return float(h5_file.attrs[name])


def _validate_index_dataset(edge_index, num_nodes: int, name: str) -> None:
    edge_index = np.asarray(edge_index)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"{name} must have shape [2, E], got {edge_index.shape}.")
    if edge_index.size:
        _validate_node_ids(edge_index, num_nodes, name)


def _validate_node_ids(indices, num_nodes: int, name: str) -> None:
    indices = np.asarray(indices)
    if indices.size == 0:
        return
    min_idx = int(indices.min())
    max_idx = int(indices.max())
    if min_idx < 0 or max_idx >= int(num_nodes):
        raise ValueError(f"{name} values must be within [0, {num_nodes - 1}], got [{min_idx}, {max_idx}].")
