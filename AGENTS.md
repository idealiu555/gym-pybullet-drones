# Repository Guidelines

## Project Structure & Module Organization

The installable package lives in `gym_pybullet_drones/`. Simulation environments are in `envs/`, flight controllers in `control/`, shared logging, enums, and helpers in `utils/`, and runnable demonstrations in `examples/`. Vehicle models, URDF files, images, trajectories, and Betaflight setup scripts belong in `assets/`. Tests are kept at the repository root in `tests/`; add focused test modules there rather than beside production code. Generated runs and temporary output should go to the ignored `results/` or `tmp/` directories.

## Build, Test, and Development Commands

Use Python 3.12, as required by `pyproject.toml`.

- `python -m pip install -e .` installs the package and dependencies in editable mode.
- `python -m pytest tests/` runs the complete test suite from the repository root.
- `python gym_pybullet_drones/examples/pid.py` launches the position-control example with the PyBullet GUI.
- `python gym_pybullet_drones/examples/learn.py` runs the PPO learning example; expect output under `results/`.
- `python -m build` creates distribution artifacts when the optional `build` package is installed.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, grouped imports, and NumPy-style docstrings for public classes and methods. Use `snake_case` for functions, methods, arguments, and test names; use `PascalCase` for classes. Existing controller and environment modules also use class-matching names such as `DSLPIDControl.py` and `BaseAviary.py`; preserve that convention when adding peers. Keep simulation constants explicit and include units in comments or docstrings. No automatic formatter or linter is currently configured, so keep changes consistent with nearby code and avoid unrelated reformatting.

## Testing Guidelines

Tests use `pytest` and follow the `test_*.py` / `test_*` naming pattern. Add import smoke tests for new modules and headless behavioral tests for examples. Pass `gui=False`, disable plotting, and write artifacts to `tmp/` so CI does not require a display or pollute the tree. There is no declared coverage threshold; prioritize environment reset/step behavior, controller outputs, and regressions fixed by the change.

## Commit & Pull Request Guidelines

Recent history favors short, imperative subjects, sometimes with a conventional prefix such as `fix:` or `docs:` and a merged PR number. Keep each commit focused, for example `fix: clamp invalid motor RPM`. Pull requests should explain the problem and approach, list verification commands, link relevant issues, and call out platform-specific behavior. Include screenshots or recordings for GUI or visualization changes, and do not commit generated results, caches, or local environments.
