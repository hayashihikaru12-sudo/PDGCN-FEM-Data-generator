"""VS Code entrypoint for HDF5 FEM preprocessing.

Run this file directly in VS Code. Paths are configured in
configs/vscode_run_config.example.json.
"""

from __future__ import annotations

from pathlib import Path

from hdf5_fem.hdf5_io import validate_processed_h5
from hdf5_fem.monitoring import make_progress_printer
from hdf5_fem.pipeline import DatasetTopologyCache, process_file
from hdf5_fem.run_config import fem_config_for_case, load_run_config


RUN_CONFIG_PATH = Path("configs/vscode_run_config.example.json")


def main() -> int:
    run_config = load_run_config(RUN_CONFIG_PATH)
    print(f"Runtime config: {run_config.source_path}")
    print(f"Total HDF5 files: {len(run_config.cases)}")

    processed_count = 0
    for job_index, job in enumerate(run_config.jobs, start=1):
        topology_cache = DatasetTopologyCache()
        print(f"Dataset job {job_index}: {job.input_hdf5_dir} -> {job.output_hdf5_dir}")
        for case in job.cases:
            processed_count += 1
            print(f"[{processed_count}/{len(run_config.cases)}] Processing: {case.input_hdf5_path}")
            job_config = fem_config_for_case(case)
            progress_callback = make_progress_printer(case.monitor)
            output_path = process_file(
                case.input_hdf5_path,
                config=job_config,
                output_path=case.output_hdf5_path,
                overwrite=run_config.overwrite,
                in_place=False,
                pvd_path=case.pvd_path,
                topology_cache=topology_cache,
                progress_callback=progress_callback,
            )
            errors = validate_processed_h5(output_path)
            if errors:
                print(f"Validation failed: {output_path}")
                for error in errors:
                    print(f"  - {error}")
                return 1
            print(f"HDF5 output: {output_path}")
            if case.write_pvd:
                print(f"PVD output: {case.pvd_path}")
            print(f"Mesh builds for this file: 1")
            print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
