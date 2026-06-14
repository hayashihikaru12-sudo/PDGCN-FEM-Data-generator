"""Configuration loading for the FEM HDF5 preprocessing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MaterialConfig:
    rho: float
    Cp: float
    K_parallel: float
    k_ratio: float
    heat_source_effective_thickness: float
    heat_source_absorptivity: float


@dataclass(frozen=True)
class TemperatureConfig:
    T_amb: float
    unit: str = "degC"


@dataclass(frozen=True)
class SolverConfig:
    time_scheme: str = "backward_euler"
    linear_solver: str = "default"


@dataclass(frozen=True)
class OutputConfig:
    write_pvd: bool = False
    pvd_dir: str = "outputs/pvd"
    pvd_stride: int = 1


@dataclass(frozen=True)
class FemConfig:
    material: MaterialConfig
    temperature: TemperatureConfig
    solver: SolverConfig
    output: OutputConfig
    source_path: Path
    raw: Dict[str, Any]


def load_config(path: str | Path) -> FemConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    material = _require_mapping(raw, "material")
    temperature = _require_mapping(raw, "temperature")
    solver = raw.get("solver", {})
    output = raw.get("output", {})
    if not isinstance(solver, dict):
        raise ValueError("solver must be a JSON object.")
    if not isinstance(output, dict):
        raise ValueError("output must be a JSON object.")

    cfg = FemConfig(
        material=MaterialConfig(
            rho=_positive_float(material, "rho"),
            Cp=_positive_float(material, "Cp"),
            K_parallel=_positive_float(material, "K_parallel"),
            k_ratio=_positive_float(material, "k_ratio"),
            heat_source_effective_thickness=_positive_float(
                material,
                "heat_source_effective_thickness",
            ),
            heat_source_absorptivity=_nonnegative_float(
                material,
                "heat_source_absorptivity",
            ),
        ),
        temperature=TemperatureConfig(
            T_amb=_float(temperature, "T_amb"),
            unit=str(temperature.get("unit", "degC")),
        ),
        solver=SolverConfig(
            time_scheme=str(solver.get("time_scheme", "backward_euler")),
            linear_solver=str(solver.get("linear_solver", "default")),
        ),
        output=OutputConfig(
            write_pvd=bool(output.get("write_pvd", False)),
            pvd_dir=str(output.get("pvd_dir", "outputs/pvd")),
            pvd_stride=_positive_int(output, "pvd_stride", default=1),
        ),
        source_path=config_path,
        raw=raw,
    )
    _validate_config(cfg)
    return cfg


def _require_mapping(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object.")
    return value


def _float(raw: Dict[str, Any], key: str) -> float:
    if key not in raw:
        raise ValueError(f"Missing required config key: {key}.")
    try:
        return float(raw[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number.") from exc


def _positive_float(raw: Dict[str, Any], key: str) -> float:
    value = _float(raw, key)
    if value <= 0.0:
        raise ValueError(f"{key} must be positive, got {value}.")
    return value


def _nonnegative_float(raw: Dict[str, Any], key: str) -> float:
    value = _float(raw, key)
    if value < 0.0:
        raise ValueError(f"{key} must be non-negative, got {value}.")
    return value


def _positive_int(raw: Dict[str, Any], key: str, *, default: Optional[int] = None) -> int:
    if key not in raw:
        if default is None:
            raise ValueError(f"Missing required config key: {key}.")
        return int(default)
    value = int(raw[key])
    if value <= 0:
        raise ValueError(f"{key} must be positive, got {value}.")
    return value


def _validate_config(config: FemConfig) -> None:
    if config.temperature.unit not in {"degC", "K"}:
        raise ValueError("temperature.unit must be either 'degC' or 'K'.")
    if config.solver.time_scheme != "backward_euler":
        raise ValueError("Only solver.time_scheme='backward_euler' is supported.")
