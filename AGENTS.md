# Project Agent Instructions

## Runtime Environment

Use the `fenics-env` conda environment for all Python commands in this project.

- Environment name: `fenics-env`
- Python executable: `/root/miniconda3/envs/fenics-env/bin/python`
- Python version: `3.8.20`
- FEniCS/DOLFIN version: `2019.1.0`

Prefer running scripts with the explicit interpreter path:

```bash
/root/miniconda3/envs/fenics-env/bin/python test.py
```

Alternatively, activate the environment first:

```bash
conda activate fenics-env
python test.py
```

## Notes

- This project currently targets classic FEniCS/DOLFIN, not `dolfinx`.
- Avoid using the base Python environment for tests, because it does not include FEniCS.
