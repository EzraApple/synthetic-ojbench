# Synthetic OJBench

Generate verifiable programming problems and measure their difficulty against a small open model. Start with a local **OJBench baseline using Qwen3.5-9B through MLX**, then build synthetic data generation.

## Status

Scaffold only. No dependencies, models, or datasets have been downloaded for this project. No inference, grading, or training has run. Upstream support has been checked; local MLX inference and judge execution remain untested. Setup and execution are deferred.

## Structure

```text
pyproject.toml                 One Python package
src/synthetic_ojbench/        Implementation, added as needed
configs/ojbench-pilot.toml     One draft experiment configuration
docs/experiment-plan.md       Protocol and setup steps
```

Evaluation will load problems, ask the model for a solution, grade the code, and save results. Synthesis will generate and validate new problems, then reuse evaluation to measure their difficulty. Keep these as modules in the same package. Extract shared helpers when there is actual reuse.

MLX inference runs on macOS. OJBench grading needs a separate Linux environment; add its container definition with the grading implementation. Save local outputs under `artifacts/`, created by the runner and ignored by Git.

Add tests alongside behavior, and a lockfile when dependencies are selected. There is no CLI yet. Training is outside the current scope.

See the [experiment plan](docs/experiment-plan.md) for the baseline and synthetic-data milestones.
