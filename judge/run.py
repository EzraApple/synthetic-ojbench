"""Grade a batch sequentially using OJBench's single-submission evaluator."""

import json
import sys
import traceback
from pathlib import Path

import ojbench
from compatibility import prepare_problem_dirs


def grade_entry(entry):
    result = dict(entry)
    try:
        submission = dict(entry)
        if type(submission["id"]) is int:
            submission["id"] = f"loj-{submission['id']}"
        verdict, cases = ojbench.judge_entry(submission, use_tqdm=False)
        infrastructure_error = verdict == "IE" or any(
            "IE" in case["readable_codes"] for case in cases
        )
        status = "error" if infrastructure_error else "skipped" if verdict == "Skip" else "graded"
        result.update(
            status=status,
            verdict=verdict,
            is_passed=(verdict == "AC") if status == "graded" else None,
            detailed_results={"final_verdict": verdict, "results": cases},
        )
    except Exception as error:
        traceback.print_exc()
        result.update(status="error", verdict=None, is_passed=None,
                      error=f"{type(error).__name__}: {error}")
    return result


def main():
    entries = [json.loads(line) for line in Path("/input/submissions.jsonl").read_text().splitlines() if line.strip()]
    problem_ids = {f"loj-{entry['id']}" if type(entry["id"]) is int else entry["id"] for entry in entries}
    problem_dirs = prepare_problem_dirs([Path(path) for path in sys.argv[1:]], problem_ids, Path("/output"))
    ojbench.init(
        problem_dirs=problem_dirs,
        config_path="/opt/judge/config.yaml",
        runtime_path="/opt/judge/runtime.yaml",
        compile_lock_path="/tmp/compile.lock",
    )
    with Path("/output/results.jsonl").open("w") as outputs:
        for entry in entries:
            result = grade_entry(entry)
            outputs.write(json.dumps(result) + "\n")
            outputs.flush()


if __name__ == "__main__":
    main()
