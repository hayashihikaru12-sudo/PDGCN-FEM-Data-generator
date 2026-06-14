"""Runtime JSON configuration for VS Code driven preprocessing runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import FemConfig, OutputConfig, load_config
from .monitoring import MonitorConfig, parse_monitor_config


@dataclass(frozen=True)
class RunCase:
    input_hdf5_path: Path
    output_hdf5_path: Path
    write_pvd: bool
    pvd_path: Optional[Path]
    pvd_stride: int
    fem_config: FemConfig
    monitor: MonitorConfig


@dataclass(frozen=True)
class RunJob:
    input_hdf5_dir: Path
    output_hdf5_dir: Path
    write_pvd: bool
    pvd_dir: Optional[Path]
    pvd_stride: int
    fem_config: FemConfig
    monitor: MonitorConfig
    file_pattern: str
    recursive: bool
    cases: List[RunCase]


@dataclass(frozen=True)
class RunConfig:
    fem_config: FemConfig
    monitor: MonitorConfig
    overwrite: bool
    in_place: bool
    jobs: List[RunJob]
    cases: List[RunCase]
    source_path: Path


def load_run_config(path: str | Path) -> RunConfig:
    run_path = Path(path)
    with run_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Runtime config must be a JSON object.")

    fem_config_path = Path(_required_str(raw, "fem_config_path"))
    fem_config = load_config(fem_config_path)
    monitor = parse_monitor_config(raw.get("monitor"), default_enabled=True)
    overwrite = bool(raw.get("overwrite", False))
    in_place = bool(raw.get("in_place", False))
    if in_place:
        raise ValueError("VS runtime config does not support in_place=true; use explicit output_hdf5_path.")

    jobs_raw = raw.get("jobs")
    if not isinstance(jobs_raw, list) or not jobs_raw:
        raise ValueError("jobs must be a non-empty list.")

    jobs = [_parse_job(job_raw, index, fem_config, monitor) for index, job_raw in enumerate(jobs_raw)]
    cases = [case for job in jobs for case in job.cases]
    if not cases:
        raise ValueError("No HDF5 files were found in configured input directories.")
    return RunConfig(
        fem_config=fem_config,
        monitor=monitor,
        overwrite=overwrite,
        in_place=in_place,
        jobs=jobs,
        cases=cases,
        source_path=run_path,
    )


def fem_config_for_case(case: RunCase) -> FemConfig:
    output = OutputConfig(
        write_pvd=case.write_pvd,
        pvd_dir=str(case.pvd_path.parent) if case.pvd_path is not None else case.fem_config.output.pvd_dir,
        pvd_stride=case.pvd_stride,
    )
    return replace(case.fem_config, output=output)


def _parse_job(
    raw: Dict[str, Any],
    index: int,
    default_fem_config: FemConfig,
    default_monitor: MonitorConfig,
) -> RunJob:
    if not isinstance(raw, dict):
        raise ValueError(f"jobs[{index}] must be a JSON object.")

    fem_config = default_fem_config
    if "fem_config_path" in raw:
        fem_config = load_config(Path(_required_str(raw, "fem_config_path")))
    monitor = parse_monitor_config(raw.get("monitor"), default_enabled=default_monitor.enabled)
    if "monitor" not in raw:
        monitor = default_monitor

    input_dir = Path(_required_str(raw, "input_hdf5_dir"))
    output_dir = Path(_required_str(raw, "output_hdf5_dir"))
    file_pattern = str(raw.get("file_pattern", "*.h5"))
    recursive = bool(raw.get("recursive", False))
    write_pvd = bool(raw.get("write_pvd", False))
    pvd_dir_raw = raw.get("pvd_dir", None)
    pvd_dir = Path(pvd_dir_raw) if pvd_dir_raw else None
    pvd_stride = int(raw.get("pvd_stride", 1))
    if pvd_stride <= 0:
        raise ValueError(f"jobs[{index}].pvd_stride must be positive.")
    if write_pvd and pvd_dir is None:
        pvd_dir = output_dir / "pvd"

    input_files = _find_hdf5_files(input_dir, file_pattern=file_pattern, recursive=recursive)
    cases = [
        _case_from_input(
            input_path,
            input_dir=input_dir,
            output_dir=output_dir,
            write_pvd=write_pvd,
            pvd_dir=pvd_dir,
            pvd_stride=pvd_stride,
            fem_config=fem_config,
            monitor=monitor,
        )
        for input_path in input_files
    ]
    return RunJob(
        input_hdf5_dir=input_dir,
        output_hdf5_dir=output_dir,
        write_pvd=write_pvd,
        pvd_dir=pvd_dir,
        pvd_stride=pvd_stride,
        fem_config=fem_config,
        monitor=monitor,
        file_pattern=file_pattern,
        recursive=recursive,
        cases=cases,
    )


def _find_hdf5_files(input_dir: Path, *, file_pattern: str, recursive: bool) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"input_hdf5_dir must be an existing directory: {input_dir}")
    iterator = input_dir.rglob(file_pattern) if recursive else input_dir.glob(file_pattern)
    files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"})
    if not files:
        raise ValueError(f"No HDF5 files matched {file_pattern!r} in {input_dir}.")
    return files


def _case_from_input(
    input_path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    write_pvd: bool,
    pvd_dir: Optional[Path],
    pvd_stride: int,
    fem_config: FemConfig,
    monitor: MonitorConfig,
) -> RunCase:
    relative = input_path.relative_to(input_dir)
    output_relative = relative.with_name(f"{relative.stem}_fem{relative.suffix}")
    output_path = output_dir / output_relative
    pvd_path = None
    if write_pvd:
        base_pvd_dir = pvd_dir if pvd_dir is not None else output_dir / "pvd"
        pvd_case_dir = base_pvd_dir / relative.with_suffix("")
        pvd_path = pvd_case_dir / "temperature.pvd"
    return RunCase(
        input_hdf5_path=input_path,
        output_hdf5_path=output_path,
        write_pvd=write_pvd,
        pvd_path=pvd_path,
        pvd_stride=pvd_stride,
        fem_config=fem_config,
        monitor=monitor,
    )


def _required_str(raw: Dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value
