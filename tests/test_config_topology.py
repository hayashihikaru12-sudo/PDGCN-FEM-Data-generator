from pathlib import Path
import sys

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hdf5_fem.config import load_config
from hdf5_fem.monitoring import make_progress_printer, parse_monitor_config
from hdf5_fem.run_config import load_run_config
from hdf5_fem.topology import recover_triangles_from_edge_index


def test_example_config_loads():
    config = load_config("configs/fem_batch_config.example.json")

    assert config.material.rho == 1575.0
    assert config.temperature.unit == "degC"
    assert config.output.write_pvd is False
    assert config.output.pvd_stride == 1


def test_recover_triangles_from_sample_edge_index():
    with h5py.File("HDF5_Files/case1_Q0_V20_dt0p05_F89.h5", "r") as h5_file:
        triangles = recover_triangles_from_edge_index(h5_file["edge_index"][()])

    assert triangles.shape == (5622, 3)
    assert triangles.min() == 0
    assert triangles.max() == 2909


def test_vscode_run_config_expands_directory_job():
    config = load_run_config("configs/vscode_run_config.example.json")

    assert len(config.jobs) == 1
    assert len(config.cases) >= 1
    case = config.cases[0]
    assert case.input_hdf5_path.name == "case1_Q0_V20_dt0p05_F89.h5"
    assert case.output_hdf5_path.as_posix() == "outputs/case1_Q0_V20_dt0p05_F89_fem.h5"
    assert case.pvd_path.as_posix() == "outputs/pvd/case1_Q0_V20_dt0p05_F89/temperature.pvd"
    assert case.monitor.enabled is True
    assert case.monitor.step_stride == 1


def test_monitor_config_can_disable_progress_callback():
    monitor = parse_monitor_config({"enabled": False, "step_stride": 5})

    assert monitor.enabled is False
    assert monitor.step_stride == 5
    assert make_progress_printer(monitor) is None


if __name__ == "__main__":
    test_example_config_loads()
    test_recover_triangles_from_sample_edge_index()
    test_vscode_run_config_expands_directory_job()
    test_monitor_config_can_disable_progress_callback()
    print("smoke tests passed")
