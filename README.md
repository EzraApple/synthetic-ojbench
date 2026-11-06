# Synthetic OJBench

This started after an onsite interview I didn't pass. The exercise involved generating synthetic tasks that were difficult enough for a model to fail on some attempts, but still within reach. I wanted to understand that process better, along with how the resulting data could be used for supervised fine-tuning (SFT) or reinforcement learning (RL).

I'm using this project to work through the whole process on a small scale: run a benchmark, generate useful training data, train a model, and see whether its score improves. Starting with a small model should make it easier to experiment locally and understand each step.

The starting point is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) and [OJBench](https://github.com/He-Ren/OJBench), a competitive programming benchmark. [MLX](https://github.com/ml-explore/mlx-lm) supports local experiments on Apple Silicon; the full baseline used hosted inference with local grading. Programming problems give me a concrete way to check whether a solution is correct and meets execution limits.

The rough plan is to:

1. Establish an audited baseline and understand how benchmark evaluation works.
2. Generate new programming problems with tests and reference solutions, then measure how often the model can solve them.
3. Use that data for an SFT or RL training run. I haven't decided which approach to start with yet.
4. Evaluate again on OJBench to see whether the synthetic data helped on problems kept out of training.

The goal is to complete an experiment and understand the choices along the way. The plan will change as I learn what works.

**Current status:** an audited hosted baseline and failure analysis are complete. The audit separated unfinished answers, compilation failures, and incorrect executable solutions. It also identified an ARM64 validator compatibility issue, which the local judge now handles explicitly. The next step is to plan a small verifier-first synthetic-data pilot. No model weights have been updated. Run outputs, benchmark data, and local research notes stay outside Git.

## Local judge

Requires Docker with its Linux engine running and Python 3.11+. The current image targets native ARM64 and grades C++17 submissions using pinned OJBench and DMOJ revisions. No model or benchmark dataset is needed for these checks.

Build the image:

```sh
docker build --platform linux/arm64 -t synthetic-ojbench-judge:local judge
```

Run the smoke tests:

```sh
OJ_JUDGE_SMOKE=1 PYTHONPATH=src python3.11 -m unittest discover -s tests -v
```

The tests check accepted, wrong-answer, compile-error, timeout, and nonzero-exit verdicts. They also check missing-problem errors, container cleanup after a batch timeout, and blocked access to test answers and result files. Logs and results stay in `artifacts/judge-smoke-*/`. Omit `OJ_JUDGE_SMOKE=1` to run only the host-side tests without Docker.

To grade saved submissions, use one JSON object per line with `id`, `language` (`cpp`), and `content` (the model response or code). Problem directories contain DMOJ problem packages, with an `init.yml` and test files per problem. Independent examples are in `tests/fixtures/problems/`.

```sh
PYTHONPATH=src python3.11 -m synthetic_ojbench.grading \
  --submissions submissions.jsonl \
  --problems path/to/problems \
  --output artifacts/my-run
```

Each batch uses one container and grades submissions sequentially. The output directory must be new. It records inputs, per-submission results, judge logs, and the image identity and run settings. `status: error` means grading failed; it is not a wrong answer. The wrapper removes the container after completion or timeout and leaves the files for inspection.

The pinned evaluator uses a 10-second program time limit and a 1 GiB program memory limit. The wrapper's `--timeout` is a separate whole-batch deadline, defaulting to 300 seconds. The ARM64 compiler target is explicit because this DMOJ version mishandles an unset target. Its nonfatal syscall-403 seccomp warning is retained in logs; the sandbox and verdict checks above pass with it.

Some packaged validators store end-of-file in plain `char`, which is unsigned under this ARM64 compiler. For auxiliary checkers and interactors with that specific pattern, the runner adds `-fsigned-char` through DMOJ's supported configuration. It creates a separate configuration overlay, keeps the source package read-only, and leaves contestant compiler settings unchanged. `compatibility.json` and `run.json` record each applied flag and the original and overlay configuration hashes. The smoke tests include valid and invalid checker controls and verify that submissions cannot read answer files through the overlay.

## One model-generated solution

Install the locked MLX dependencies on Apple Silicon:

```sh
uv sync --locked --extra inference --python python3.11
```

With the judge image built, generate and grade one solution to the independent addition fixture:

```sh
uv run --locked --extra inference python examples/grade_one_solution.py \
  --output artifacts/model-smoke
```

This downloads about 19 GB of BF16 weights on the first run and reuses the Hugging Face cache afterward. The example pins the model revision, enables thinking, and caps generation at 1,024 tokens. It saves the prompt, response, timing, memory use, and judge results in a new output directory. This checks the model-to-judge path; it does not download or evaluate the OJBench problem set.

## Small OJBench pilot

Prepare a fixed sample of 12 C++ problems, with four from each difficulty group:

```sh
uv run --locked --extra inference python -m synthetic_ojbench.pilot prepare \
  --output artifacts/pilot-data \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --presence-penalty 1.5 --max-tokens 65536
```

This pins the dataset revision, samples with seed 0 before inference, excludes the three problems skipped by the upstream evaluator, and downloads only the selected problem packages. The manifest records the prompts, problem file hashes, model revision, sampling settings, and per-attempt seeds.

Thinking is enabled by default; use `--no-thinking` for a separately labeled experiment. The command records explicit sampling values and a fixed output budget. The presence penalty applies once to each generated token type and excludes prompt tokens. Settings remain fixed throughout a run. The pinned MLX sampler applies top-p, then top-k, then temperature; matching parameter values alone does not establish equivalence with hosted backends.

With the model already cached and Docker running, start the pilot:

```sh
HF_HUB_OFFLINE=1 caffeinate -i uv run --locked --extra inference \
  python -m synthetic_ojbench.pilot run \
  --manifest artifacts/pilot-data/manifest.json \
  --output artifacts/pilot-run
```

The model loads once. Each problem gets a fresh conversation and cache, one generated solution, and one grading run. There are no repair attempts. MLX caches attention keys and values and the model's recurrent state during generation. The pilot uses BF16 weights without cache quantization. By default, the output limit is the remaining context window with one position reserved. Pass `--max-tokens` to `prepare` to record a lower limit in the manifest.

`progress.json` reports the current phase, token count, speed, and MLX peak memory. Each problem directory saves the prompt, streamed response, generated token IDs, generation metrics, and judge output. `summary.json` updates after each attempt. Output directories must be new; existing attempts are never overwritten or automatically retried.

Only the final response is submitted to the official evaluator. For thinking-mode runs, code inside reasoning is not scored. If generation ends before a final response, the attempt is recorded as `no_final` and counts as a failure in `accepted_fraction_of_scored`. Infrastructure errors and operator interruptions are separate. `accepted_fraction_of_graded` is the conditional rate among attempts actually sent to the judge.

To interrupt a runaway generation while continuing to the next problem, write `stop.json` in the run directory with the active `problem_id` and a `reason`. The runner checks it about every five seconds during generation and preserves the partial response and intervention. Interrupted attempts and judge errors are reported separately from wrong answers. A stalled inference call or grading process cannot consume this file; those require stopping the process directly.

This is a small, balanced pilot, not a full OJBench score. Results use the pinned evaluator's 10-second per-case time limit on local ARM64 hardware. The separate judge deadline scales with the problem package size so a problem with many cases is not cut off by the wrapper's default 300-second batch deadline.
