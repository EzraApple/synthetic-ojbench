# Baseline and synthetic data plan

Current scope: establish a local OJBench baseline, then generate validated synthetic programming problems and measure their difficulty. Use Qwen3.5-9B with MLX as the preferred runtime. No paid compute is assumed. Training is deferred; there is no training scaffold or backend commitment.

The longer-term question is whether difficulty-selected synthetic tasks help learning more than randomly selected valid tasks. This phase produces the measurements and data needed to investigate that question. It does not claim learnability or training gains.

## Minimal design

Use one Python package. Implement evaluation first; add synthesis when the baseline works. Synthesis calls the same model-attempt and grading functions used by evaluation. Extract concrete shared functions only when both paths need them. There is no separate core library, provider framework, or general workflow engine.

Keep the first experiment in one config file. Add another config when there is another experiment. The only required runtime separation is MLX on macOS and the OJBench judge in Linux. Add the container definition and tests with the grading implementation, not as empty directories.

Use local files. Each run saves its resolved config, selected task IDs/revisions, raw model responses, extracted code, judge output, and a summary under `artifacts/<run-id>/`. Use JSONL for per-attempt records. Include model/data/code identity, stop reasons, available token usage, and timing. Preserve completed attempts when resuming; do not overwrite or silently rerun them. Choose concrete record fields with the runner instead of designing a general artifact system in advance.

## Setup, when execution is authorized

Downloads, dependency installation, container builds, and model runs remain deferred. The scaffold has no runnable experiment commands.

1. Verify an MLX LM release and Qwen3.5-9B weight conversion support text inference, the chat template, thinking output, and stop handling. Record upstream provenance and exact revisions.
2. Select precision and generation settings. BF16 is a candidate, not a measured memory or throughput guarantee. Freeze sampling, thinking mode, context/output budgets, and seed policy.
3. Resolve dependencies and create a lockfile. Keep Linux judge dependencies separate from the host Python package. Acquire weights and the pinned OJBench dataset only in the authorized setup phase.
4. Prepare the judge using OJBench's documented DMOJ revision and compatible language runtimes. Check ARM64 compatibility, compiler behavior, and resource limits in the local Linux VM. Record the image and architecture; emulation or CPU differences can change timeout verdicts.
5. Verify grading using independent known-correct, wrong, compile-error, and timeout submissions. Keep hidden tests and references out of model inputs. Expose only required files to submitted programs, without host credentials or the Docker socket.
6. Implement the model-to-judge path and check response preservation, code extraction, error classification, and resume with small fixtures.

Inference support does not establish training support. Review upstream redistribution terms before incorporating external code or datasets into published artifacts.

## OJBench pilot

Propose 12 C++17 problems, four from each difficulty level, with one attempt per problem. Freeze IDs before observing outcomes. Use the official prompts, fresh conversations, and one submitted solution without feedback or repair turns.

Resolve every draft field in [the pilot config](../configs/ojbench-pilot.toml) before dispatch. Record exact dataset/evaluator/model revisions, template, precision, token budgets, compiler, CPU/memory allocation, and program time limits. Generation token limits and submitted-program time limits are separate settings.

Use full-test acceptance as the result. Partial-test outcomes are diagnostics. Record truncation and incorrect submissions as attempt outcomes under the declared settings; distinguish judge/inference infrastructure failures. Missing or unresolved attempts make a run incomplete. Retain original outcomes and declare any infrastructure retry policy; never replace a genuine wrong answer with a new sample.

Report this as a diagnostic subset, not a full score. Equal difficulty counts may differ from the benchmark distribution. Measure wall time and token usage to estimate the complete run. If the pilot leads to config changes, retain its results and identify the next run as a new configuration.

## Complete baseline

OJBench contains 232 problems and supports C++ and Python. Confirm the pinned dataset's actual count and evaluate the chosen language as one condition. Choose repeat count based on pilot throughput before starting: one attempt per problem means 232 generations; five means 1,160.

Report mean single-attempt success separately from the fraction solved at least once across repeated attempts. Keep per-problem outcomes and uncertainty that accounts for repeated attempts on the same problem. Record missing trials and infrastructure errors without silently changing the denominator.

Qwen3.5-9B is already post-trained. Its model-card score is context, not our baseline. Do not claim reproduction without matching the publisher's protocol. Keep runtime and inference settings fixed for future comparisons; remeasure the baseline if they change.

## Synthetic data pilot

Start with 20 candidates across algorithm families such as arrays/strings, graphs, dynamic programming, and search. This is a development target, not a sufficient training-set size. Generator choice and prompts remain open; use local compute.

Each task includes a statement, constraints, input/output format, public examples, hidden tests, reference solution, and provenance. Where practical, add a brute-force oracle for small cases.

Validate before measuring difficulty:

1. Check that the statement, constraints, examples, and tests agree.
2. Compile and run the reference solution against every test.
3. Compare the efficient reference with the brute-force oracle on small generated inputs where feasible.
4. Check boundaries, plausible incorrect solutions, and alternative correct solutions.
5. Freeze and hash the validated task revision.

A passing reference or model review alone does not prove test quality. If the same model authors and reviews tasks, record that limitation. A separate teacher is not required for this phase.

Reuse evaluation for five independent attempts from the frozen student. Select tasks with one to three passes out of five valid attempts as an initial difficulty screen. Run five fresh confirmation attempts for selected tasks and retain both cohorts, including disagreements. Do not rerun until a desired count appears. Defective tasks and infrastructure failures cannot count as student failures.

Keep revision rounds bounded. Retain rejected candidates and reasons. Each revised task gets a new hash and measurements. Export validated tasks with provenance, validity evidence, and observed difficulty, including tasks outside the selected band. This preserves the option of a later random-selection control.

## Evaluation isolation

Keep OJBench prompts, solutions, tests, model responses, and task-level failure feedback out of generation inputs. Use independent synthetic development tasks to tune generation. Split synthetic families or source lineages before making variants, and check duplicates/benchmark overlap separately without exposing benchmark content to the generator.

These checks reduce new contamination; they cannot prove what was absent from the model's upstream training. One to three successes out of five is a noisy difficulty estimate, not evidence that training will improve performance.

## Sources

- [OJBench](https://github.com/He-Ren/OJBench) and its [dataset](https://huggingface.co/datasets/He-Ren/OJBench_testdata).
- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [MLX LM](https://github.com/ml-explore/mlx-lm).
- [DMOJ](https://github.com/DMOJ/judge-server) for the Linux judge.
- [Task-generation feedback loop](https://x.com/michaelrbock/status/2115589801491800474): generate, validate, measure student outcomes, and revise. Begin with explicit stages; automate feedback after each stage works.
