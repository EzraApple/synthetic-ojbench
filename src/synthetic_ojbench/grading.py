"""Run a local OJBench judge container and retain its inputs, results, and logs."""

import argparse
import json
import math
import subprocess
import time
import uuid
from pathlib import Path

DEFAULT_IMAGE = "synthetic-ojbench-judge:local"


class JudgeError(RuntimeError):
    """The grading run could not finish reliably; inspect its output directory."""


def grade_submissions(
    submissions: Path,
    problem_dirs: list[Path],
    output_dir: Path,
    *,
    image: str = DEFAULT_IMAGE,
    timeout: float = 300,
) -> list[dict]:
    """Grade C++ JSONL submissions into a new output directory; never overwrite a run."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a positive, finite number of seconds")
    sources = [path.resolve(strict=True) for path in problem_dirs]
    if not sources or any(not path.is_dir() for path in sources):
        raise ValueError("Provide at least one problem directory")
    raw = submissions.read_text()
    entries = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not entries:
        raise ValueError("No submissions supplied")
    for entry in entries:
        if not isinstance(entry, dict) or type(entry.get("id")) not in (str, int):
            raise ValueError("Each submission needs a string or integer id")
        identifier = entry["id"]
        if isinstance(identifier, str) and (not identifier or identifier in {".", ".."}
                                           or any(char in identifier for char in ("/", "\\", "\0"))):
            raise ValueError("Problem IDs must be simple directory names")
        if entry.get("language") != "cpp" or not isinstance(entry.get("content"), str):
            raise ValueError("Each submission needs language='cpp' and string content")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    request = output_dir / "submissions.jsonl"
    request.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    name = f"ojbench-{uuid.uuid4().hex}"
    metadata = {"container": name, "image": image, "timeout_seconds": timeout,
                "problem_dirs": [str(path) for path in sources], "status": "running"}
    started = time.monotonic()
    command = []
    cleanup_failed = False
    try:
        inspected = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True,
            check=True, timeout=15,
        )
        image_info = json.loads(inspected.stdout)[0]
        if image_info["Architecture"] != "arm64":
            raise JudgeError("The local judge requires an ARM64 image; rebuild with --platform linux/arm64")
        metadata["image_id"] = image_info["Id"]
        metadata["architecture"] = image_info["Architecture"]
        metadata["labels"] = image_info["Config"].get("Labels")
        command = [
            "docker", "run", "--name", name, "--pull=never", "--network=none",
            "--read-only", "--cap-drop=ALL", "--cap-add=SYS_PTRACE",
            "--security-opt=no-new-privileges", "--pids-limit=128",
            "--cpus=2", "--memory=2g", "--memory-swap=2g",
            "--tmpfs=/tmp:rw,exec,nosuid,size=512m,mode=1777",
            "--mount", f"type=bind,src={request},dst=/input/submissions.jsonl,readonly",
            "--mount", f"type=bind,src={output_dir},dst=/output",
        ]
        for index, path in enumerate(sources):
            command += ["--mount", f"type=bind,src={path},dst=/problems/{index},readonly"]
        command += [image_info["Id"], *[f"/problems/{index}" for index in range(len(sources))]]
        metadata["command"] = command
        with (output_dir / "judge.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        compatibility = output_dir / "compatibility.json"
        if compatibility.exists():
            metadata["compatibility"] = json.loads(compatibility.read_text())
        results = [json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines()]
        if len(results) != len(entries):
            raise JudgeError("Judge returned an incomplete batch")
        for entry, result in zip(entries, results):
            if any(result.get(key) != value for key, value in entry.items()):
                raise JudgeError("Judge output does not match submitted inputs")
            if result.get("status") not in {"graded", "error", "skipped"}:
                raise JudgeError("Judge output has an invalid status")
        metadata["status"] = "error" if any(r["status"] == "error" for r in results) else "complete"
    except (OSError, subprocess.SubprocessError, ValueError, JudgeError) as error:
        metadata.update(status="error", error=f"{type(error).__name__}: {error}")
        raise JudgeError(f"Grading failed; inspect {output_dir}: {error}") from error
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        raise
    finally:
        if command:
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "--force", name], capture_output=True, text=True, timeout=15,
                )
                metadata["cleanup"] = {"returncode": cleanup.returncode, "stderr": cleanup.stderr}
                cleanup_failed = cleanup.returncode != 0 and "No such container" not in cleanup.stderr
            except (OSError, subprocess.SubprocessError) as error:
                metadata["cleanup"] = {"error": str(error)}
                cleanup_failed = True
        if cleanup_failed:
            metadata["status"] = "error"
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 3)
        (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if cleanup_failed:
        raise JudgeError(f"Could not remove container {name}; inspect {output_dir / 'run.json'}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submissions", type=Path, required=True)
    parser.add_argument("--problems", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        results = grade_submissions(args.submissions, args.problems, args.output,
                                    image=args.image, timeout=args.timeout)
    except (ValueError, OSError, JudgeError) as error:
        parser.exit(1, f"{error}\n")
    for row in results:
        print(f"{row.get('attempt_id', row['id'])}: {row['status']} {row['verdict']}")
    parser.exit(1 if any(row["status"] == "error" for row in results) else 0)


if __name__ == "__main__":
    main()
