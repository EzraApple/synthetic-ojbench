# Synthetic OJBench

This started after an onsite interview I didn't pass. The exercise involved generating synthetic tasks that were difficult enough for a model to fail on some attempts, but still within reach. I wanted to understand that process better, along with how the resulting data could be used for supervised fine-tuning (SFT) or reinforcement learning (RL).

I'm using this project to work through the whole process on a small scale: run a benchmark, generate useful training data, train a model, and see whether its score improves. Starting with a small model should make it easier to experiment locally and understand each step.

The starting point is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), running locally with [MLX](https://github.com/ml-explore/mlx-lm), and [OJBench](https://github.com/He-Ren/OJBench), a competitive programming benchmark. Programming problems give me a concrete way to check whether a solution is correct and meets execution limits.

The rough plan is to:

1. Get a local baseline and understand how benchmark evaluation works.
2. Generate new programming problems with tests and reference solutions, then measure how often the model can solve them.
3. Use that data for an SFT or RL training run. I haven't decided which approach to start with yet.
4. Evaluate again on OJBench to see whether the synthetic data helped on problems kept out of training.

The goal is to complete an experiment and understand the choices along the way. The plan will change as I learn what works.

**Current status:** the local C++ judge passes smoke tests on Apple Silicon, and the first MLX-generated addition solution was accepted. This verifies the model-to-judge path. No OJBench baseline or training results yet.

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
