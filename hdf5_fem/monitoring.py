"""Runtime progress monitoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class MonitorConfig:
    enabled: bool = True
    step_stride: int = 1


def parse_monitor_config(raw: Optional[Dict[str, Any]], *, default_enabled: bool = True) -> MonitorConfig:
    if raw is None:
        return MonitorConfig(enabled=default_enabled, step_stride=1)
    if not isinstance(raw, dict):
        raise ValueError("monitor must be a JSON object.")
    enabled = bool(raw.get("enabled", default_enabled))
    step_stride = int(raw.get("step_stride", 1))
    if step_stride <= 0:
        raise ValueError("monitor.step_stride must be positive.")
    return MonitorConfig(enabled=enabled, step_stride=step_stride)


def make_progress_printer(config: MonitorConfig):
    if not config.enabled:
        return None

    def _print_progress(payload: dict) -> None:
        if payload.get("event") == "mesh_built":
            print(
                "[mesh] "
                f"geometry={payload['mesh_geometry']} "
                f"nodes={payload['num_nodes']} triangles={payload['num_triangles']} "
                f"steps={payload['num_steps']} build={payload['mesh_seconds']:.3f}s",
                flush=True,
            )
        elif payload.get("event") == "time_step":
            step = int(payload["step"])
            num_steps = int(payload["num_steps"])
            if step != 1 and step != num_steps and step % config.step_stride != 0:
                return
            print(
                "[step] "
                f"{step}/{num_steps} "
                f"t={payload['time_s']:.6g}s dt={payload['dt_s']:.6g}s "
                f"coeff={payload['coefficient_seconds']:.3f}s "
                f"assemble={payload['assembly_seconds']:.3f}s "
                f"solve={payload['solve_seconds']:.3f}s "
                f"total={payload['step_seconds']:.3f}s "
                f"linear_residual_l2={payload['linear_residual_l2']:.3e} "
                f"T=[{payload['temperature_min']:.6g}, {payload['temperature_max']:.6g}] "
                f"T_mean={payload['temperature_mean']:.6g}",
                flush=True,
            )

    return _print_progress
