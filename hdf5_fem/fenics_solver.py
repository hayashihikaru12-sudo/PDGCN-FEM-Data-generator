"""FEniCS transient heat solver on 3D embedded surface meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional

import numpy as np

from .config import FemConfig
from .units import heat_flux_w_per_mm2_to_w_per_m2, length_mm_to_m, velocity_mm_per_s_to_m_per_s

df = None


def _import_dolfin():
    global df
    if df is None:
        import dolfin as _df

        df = _df
    return df


class CellScalarExpression(_import_dolfin().UserExpression):
    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self.values = np.asarray(values, dtype=np.float64)

    def eval_cell(self, value, x, cell):
        value[0] = self.values[_cell_index(cell)]

    def value_shape(self):
        return ()


class CellVectorExpression(_import_dolfin().UserExpression):
    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self.values = np.asarray(values, dtype=np.float64)

    def eval_cell(self, value, x, cell):
        vector = self.values[_cell_index(cell)]
        value[0] = vector[0]
        value[1] = vector[1]
        value[2] = vector[2]

    def value_shape(self):
        return (3,)


class CellTensorExpression(_import_dolfin().UserExpression):
    def __init__(self, values, **kwargs):
        super().__init__(**kwargs)
        self.values = np.asarray(values, dtype=np.float64)

    def eval_cell(self, value, x, cell):
        tensor = self.values[_cell_index(cell)]
        flat = tensor.reshape(-1)
        for idx, item in enumerate(flat):
            value[idx] = item

    def value_shape(self):
        return (3, 3)


@dataclass
class SurfaceSolveContext:
    mesh: object
    function_space: object
    trial: object
    test: object
    vertex_to_dof: np.ndarray
    dirichlet_dofs: np.ndarray


def solve_temperature_history(
    h5_file,
    *,
    config: FemConfig,
    triangles,
    case_parameters,
    pvd_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    dolfin = _import_dolfin()
    dolfin.set_log_level(30)

    xyz_all = h5_file["dynamic/xyz"]
    fiber_all = h5_file["dynamic/fiber"]
    normal_all = h5_file["dynamic/normal"]
    q_all = h5_file["dynamic/Q"]
    num_frames, num_nodes = xyz_all.shape[:2]

    dt = float(case_parameters["dt_s"])
    fem_time = np.arange(num_frames, dtype=np.float64) * dt
    temperature = np.empty((num_frames, num_nodes, 1), dtype=np.float32)
    temperature[0, :, 0] = np.float32(config.temperature.T_amb)

    dirichlet_nodes = np.unique(
        np.concatenate(
            [
                np.asarray(h5_file["boundary_nodes/upwind"][()], dtype=np.int64),
                np.asarray(h5_file["boundary_nodes/side"][()], dtype=np.int64),
            ]
        )
    )
    mesh_start = perf_counter()
    context = _build_surface_context(
        length_mm_to_m(xyz_all[0, :, :]),
        triangles,
        dirichlet_nodes,
    )
    mesh_seconds = perf_counter() - mesh_start
    _emit_progress(
        progress_callback,
        {
            "event": "mesh_built",
            "num_frames": int(num_frames),
            "num_steps": int(max(num_frames - 1, 0)),
            "num_nodes": int(num_nodes),
            "num_triangles": int(len(triangles)),
            "mesh_seconds": mesh_seconds,
            "mesh_geometry": "3d_surface",
        },
    )

    pvd_file = None
    if pvd_path is not None:
        pvd_path.parent.mkdir(parents=True, exist_ok=True)
        pvd_file = dolfin.File(str(pvd_path))

    previous_values = np.full(num_nodes, float(config.temperature.T_amb), dtype=np.float64)
    _write_pvd_if_needed(
        pvd_file,
        context,
        previous_values,
        fem_time[0],
        0,
        config.output.pvd_stride,
    )

    for frame_idx in range(num_frames - 1):
        step_start = perf_counter()
        _update_mesh_coordinates(context.mesh, length_mm_to_m(xyz_all[frame_idx, :, :]))

        coefficient_start = perf_counter()
        u_prev = dolfin.Function(context.function_space)
        _set_function_from_vertex_values(u_prev, previous_values, context.vertex_to_dof)

        xyz_frame = length_mm_to_m(np.asarray(xyz_all[frame_idx, :, :], dtype=np.float64))
        normal_frame = np.asarray(normal_all[frame_idx, :, :], dtype=np.float64)
        fiber_frame = np.asarray(fiber_all[frame_idx, :, :], dtype=np.float64)

        velocity_cells = _build_velocity_cells(
            normal_frame,
            triangles,
            case_parameters,
            xyz_frame,
        )
        alpha_cells = _build_alpha_cells_3d(
            fiber_frame,
            normal_frame,
            xyz_frame,
            triangles,
            config,
        )
        source_cells = _build_source_cells(
            heat_flux_w_per_mm2_to_w_per_m2(np.asarray(q_all[frame_idx, :, 0], dtype=np.float64)),
            triangles,
            config,
        )
        coefficient_seconds = perf_counter() - coefficient_start

        assembly_start = perf_counter()
        velocity_expr = CellVectorExpression(velocity_cells, degree=0)
        alpha_expr = CellTensorExpression(alpha_cells, degree=0)
        source_expr = CellScalarExpression(source_cells, degree=0)
        dt_const = dolfin.Constant(dt)

        u = context.trial
        w = context.test
        lhs = (
            (u / dt_const) * w * dolfin.dx
            + dolfin.dot(velocity_expr, dolfin.grad(u)) * w * dolfin.dx
            + dolfin.inner(dolfin.dot(alpha_expr, dolfin.grad(u)), dolfin.grad(w)) * dolfin.dx
        )
        rhs = (u_prev / dt_const) * w * dolfin.dx + source_expr * w * dolfin.dx

        matrix = dolfin.assemble(lhs)
        vector = dolfin.assemble(rhs)
        _apply_dirichlet_dofs(
            matrix,
            vector,
            context.dirichlet_dofs,
            float(config.temperature.T_amb),
        )
        assembly_seconds = perf_counter() - assembly_start

        solve_start = perf_counter()
        solution = dolfin.Function(context.function_space, name="temperature")
        _solve_linear_system(matrix, solution, vector, config)
        solve_seconds = perf_counter() - solve_start

        previous_values = _function_to_vertex_values(solution, num_nodes, context.vertex_to_dof)
        residual_l2 = _linear_residual_l2(matrix, solution, vector)
        temperature[frame_idx + 1, :, 0] = previous_values.astype(np.float32)
        _write_pvd_if_needed(
            pvd_file,
            context,
            previous_values,
            fem_time[frame_idx + 1],
            frame_idx + 1,
            config.output.pvd_stride,
        )
        step_seconds = perf_counter() - step_start
        _emit_progress(
            progress_callback,
            {
                "event": "time_step",
                "step": int(frame_idx + 1),
                "num_steps": int(num_frames - 1),
                "frame_from": int(frame_idx),
                "frame_to": int(frame_idx + 1),
                "time_s": float(fem_time[frame_idx + 1]),
                "dt_s": float(dt),
                "coefficient_seconds": coefficient_seconds,
                "assembly_seconds": assembly_seconds,
                "solve_seconds": solve_seconds,
                "step_seconds": step_seconds,
                "linear_residual_l2": residual_l2,
                "temperature_min": float(np.min(previous_values)),
                "temperature_max": float(np.max(previous_values)),
                "temperature_mean": float(np.mean(previous_values)),
            },
        )

    return {
        "time": fem_time,
        "temperature": temperature,
        "valid_mask": np.ones_like(temperature, dtype=np.uint8),
        "mesh_build_count": 1,
        "mesh_geometry": "3d_surface",
    }


def _solver_parameters(config: FemConfig) -> dict:
    if config.solver.linear_solver == "default":
        return {}
    return {"linear_solver": config.solver.linear_solver}


def _cell_index(cell) -> int:
    index = cell.index
    if callable(index):
        return int(index())
    return int(index)


def _build_surface_context(xyz_m, triangles, dirichlet_nodes) -> SurfaceSolveContext:
    dolfin = _import_dolfin()
    mesh = _build_surface_mesh_xyz(xyz_m, triangles)
    function_space = dolfin.FunctionSpace(mesh, "Lagrange", 1)
    vertex_to_dof = dolfin.vertex_to_dof_map(function_space)
    dirichlet_dofs = np.asarray(vertex_to_dof[np.asarray(dirichlet_nodes, dtype=np.int64)], dtype=np.intc)
    return SurfaceSolveContext(
        mesh=mesh,
        function_space=function_space,
        trial=dolfin.TrialFunction(function_space),
        test=dolfin.TestFunction(function_space),
        vertex_to_dof=vertex_to_dof,
        dirichlet_dofs=np.unique(dirichlet_dofs),
    )


def _build_surface_mesh_xyz(xyz_m, triangles):
    dolfin = _import_dolfin()
    xyz_m = np.asarray(xyz_m, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    mesh = dolfin.Mesh()
    editor = dolfin.MeshEditor()
    editor.open(mesh, "triangle", 2, 3)
    editor.init_vertices(int(xyz_m.shape[0]))
    for idx, point in enumerate(xyz_m):
        editor.add_vertex(int(idx), [float(point[0]), float(point[1]), float(point[2])])
    editor.init_cells(int(triangles.shape[0]))
    for idx, tri in enumerate(triangles):
        editor.add_cell(int(idx), [int(tri[0]), int(tri[1]), int(tri[2])])
    editor.close()
    return mesh


def _update_mesh_coordinates(mesh, xyz_m) -> None:
    xyz_m = np.asarray(xyz_m, dtype=np.float64)
    if mesh.coordinates().shape != xyz_m.shape:
        raise ValueError(f"Updated mesh coordinates shape {xyz_m.shape} does not match {mesh.coordinates().shape}.")
    mesh.coordinates()[:] = xyz_m


def _set_function_from_vertex_values(function, vertex_values, vertex_to_dof):
    values = np.asarray(vertex_values, dtype=np.float64)
    dof_values = np.empty(function.function_space().dim(), dtype=np.float64)
    dof_values[vertex_to_dof] = values
    function.vector().set_local(dof_values)
    function.vector().apply("insert")


def _function_to_vertex_values(function, num_nodes: int, vertex_to_dof) -> np.ndarray:
    dof_values = function.vector().get_local()
    values = np.empty(int(num_nodes), dtype=np.float64)
    values[:] = dof_values[vertex_to_dof]
    return values


def _apply_dirichlet_dofs(matrix, vector, dofs, value: float) -> None:
    dofs = np.asarray(dofs, dtype=np.intc)
    if dofs.size == 0:
        return
    matrix.ident(dofs.tolist())
    for dof in dofs:
        vector[int(dof)] = float(value)
    matrix.apply("insert")
    vector.apply("insert")


def _solve_linear_system(matrix, solution, vector, config: FemConfig) -> None:
    dolfin = _import_dolfin()
    if config.solver.linear_solver == "default":
        dolfin.solve(matrix, solution.vector(), vector)
    else:
        dolfin.solve(matrix, solution.vector(), vector, config.solver.linear_solver)


def _linear_residual_l2(matrix, solution, rhs) -> float:
    residual = matrix * solution.vector()
    residual.axpy(-1.0, rhs)
    return float(residual.norm("l2"))


def _emit_progress(progress_callback, payload: dict) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def _build_velocity_cells(normal, triangles, case_parameters, xyz_m) -> np.ndarray:
    speed_m_s = velocity_mm_per_s_to_m_per_s(case_parameters["velocity_speed_mm_s"])
    direction = _normalize(np.asarray(case_parameters["velocity_direction_local"], dtype=np.float64))
    velocities = np.empty((len(triangles), 3), dtype=np.float64)
    for cell_idx, tri in enumerate(triangles):
        n = _cell_normal(normal, xyz_m, tri)
        tangent_velocity = direction - float(np.dot(direction, n)) * n
        norm = float(np.linalg.norm(tangent_velocity))
        if norm <= 1.0e-15:
            raise ValueError("velocity_direction_local is normal to a surface cell; cannot build tangent velocity.")
        velocities[cell_idx] = speed_m_s * tangent_velocity / norm
    return velocities


def _build_alpha_cells_3d(fiber, normal, xyz_m, triangles, config: FemConfig) -> np.ndarray:
    material = config.material
    k_parallel = material.K_parallel
    k_perp = material.k_ratio * material.K_parallel
    rho_cp = material.rho * material.Cp
    eye = np.eye(3, dtype=np.float64)
    alpha = np.empty((len(triangles), 3, 3), dtype=np.float64)
    for cell_idx, tri in enumerate(triangles):
        n = _cell_normal(normal, xyz_m, tri)
        projection = eye - np.outer(n, n)

        f = np.mean(fiber[tri], axis=0)
        f = projection @ f
        if float(np.linalg.norm(f)) <= 1.0e-15:
            f = _fallback_tangent(xyz_m, tri)
        else:
            f = _normalize(f)
        fiber_projection = np.outer(f, f)
        k_tensor = k_parallel * fiber_projection + k_perp * (projection - fiber_projection)
        alpha[cell_idx] = k_tensor / rho_cp
    return alpha


def _build_source_cells(q_w_m2, triangles, config: FemConfig) -> np.ndarray:
    material = config.material
    denominator = material.rho * material.Cp * material.heat_source_effective_thickness
    source_node = material.heat_source_absorptivity * np.asarray(q_w_m2, dtype=np.float64) / denominator
    return np.asarray([float(np.mean(source_node[tri])) for tri in triangles], dtype=np.float64)


def _cell_normal(normal, xyz_m, tri) -> np.ndarray:
    n = np.mean(normal[tri], axis=0)
    if float(np.linalg.norm(n)) <= 1.0e-15:
        pts = xyz_m[tri]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    return _normalize(n)


def _fallback_tangent(xyz_m, tri) -> np.ndarray:
    pts = xyz_m[tri]
    tangent = pts[1] - pts[0]
    if float(np.linalg.norm(tangent)) <= 1.0e-15:
        tangent = pts[2] - pts[0]
    return _normalize(tangent)


def _normalize(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-15:
        raise ValueError("Cannot normalize near-zero vector.")
    return vector / norm


def _write_pvd_if_needed(pvd_file, context, vertex_values, timestamp, frame_idx: int, stride: int) -> None:
    if pvd_file is None or frame_idx % int(stride) != 0:
        return
    dolfin = _import_dolfin()
    function = dolfin.Function(context.function_space, name="temperature")
    _set_function_from_vertex_values(function, vertex_values, context.vertex_to_dof)
    pvd_file << (function, float(timestamp))
