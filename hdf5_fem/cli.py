"""Command line interface for the HDF5 FEM preprocessing pipeline."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .config import load_config
from .hdf5_io import validate_processed_h5
from .monitoring import make_progress_printer, parse_monitor_config
from .pipeline import process_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate FEM supervision fields for PDGCN HDF5 files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Process one or more HDF5 files.")
    process_parser.add_argument("--input", nargs="+", required=True, help="Input HDF5 path(s) or glob pattern(s).")
    process_parser.add_argument("--config", required=True, help="FEM JSON config path.")
    process_parser.add_argument("--output-dir", default="outputs", help="Directory for processed HDF5 copies.")
    process_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output copies.")
    process_parser.add_argument("--in-place", action="store_true", help="Modify input HDF5 files directly.")

    validate_parser = subparsers.add_parser("validate", help="Validate processed HDF5 files.")
    validate_parser.add_argument("--input", nargs="+", required=True, help="Processed HDF5 path(s) or glob pattern(s).")

    args = parser.parse_args(argv)
    if args.command == "process":
        return _process(args)
    if args.command == "validate":
        return _validate(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


def _process(args) -> int:
    config = load_config(args.config)
    monitor = parse_monitor_config(config.raw.get("monitor"), default_enabled=True)
    progress_callback = make_progress_printer(monitor)
    input_paths = _expand_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No input files matched: {args.input}")

    for input_path in input_paths:
        output_path = process_file(
            input_path,
            config=config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            in_place=args.in_place,
            progress_callback=progress_callback,
        )
        print(f"processed: {input_path} -> {output_path}")
    return 0


def _validate(args) -> int:
    input_paths = _expand_inputs(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No input files matched: {args.input}")

    ok = True
    for path in input_paths:
        errors = validate_processed_h5(path)
        if errors:
            ok = False
            print(f"FAILED: {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"OK: {path}")
    return 0 if ok else 1


def _expand_inputs(patterns) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted({path for path in paths if path.exists()})


if __name__ == "__main__":
    raise SystemExit(main())
