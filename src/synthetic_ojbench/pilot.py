"""Prepare a fixed OJBench subset, then generate and grade one attempt per task."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path

from synthetic_ojbench.grading import DEFAULT_IMAGE, JudgeError, grade_submissions

MODEL = "mlx-community/Qwen3.5-9B-bf16"
MODEL_REVISION = "27cdb77d9381d508abfe5d4e06bcb95471e312ac"
DATASET = "He-Ren/OJBench_testdata"
DATASET_REVISION = "61cf9986f22c25d08e1657b03742124099c74353"
UPSTREAM_SKIPS = {"nwerc2023_H", "swerc2023_L", "loj-3897"}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def problem_id(row):
    return f"loj-{row['id']}" if type(row["id"]) is int else row["id"]


def select_tasks(rows, per_difficulty=4, seed=0):
    eligible = [row for row in rows if row["language"] == "cpp"
                and problem_id(row) not in UPSTREAM_SKIPS]
    rng = random.Random(seed)
    selected = []
    for difficulty in ("easy", "medium", "hard"):
        pool = sorted((row for row in eligible if row["difficulty"] == difficulty),
                      key=problem_id)
        selected.extend(rng.sample(pool, per_difficulty))
    return selected


def prepare(output, *, thinking=True, temperature=0.6, top_p=0.95, top_k=20,
            presence_penalty=0.0, max_tokens=None):
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    output.mkdir(parents=True, exist_ok=False)
    prompts = Path(hf_hub_download(DATASET, "prompts/full.jsonl", repo_type="dataset",
                                  revision=DATASET_REVISION))
    rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    selected = select_tasks(rows)
    files = HfApi().list_repo_files(DATASET, repo_type="dataset", revision=DATASET_REVISION)
    selected_files = [name for name in files
                      if any(f"/{problem_id(row)}/" in name for row in selected)]
    snapshot = Path(snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION,
                                      allow_patterns=selected_files))
    hashes = {}
    tasks = []
    for index, row in enumerate(selected):
        identifier = problem_id(row)
        roots = list(snapshot.glob(f"*/{identifier}"))
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one package for {identifier}")
        target = output / "problems" / identifier
        # Materialize cache symlinks so Docker's read-only mount is self-contained.
        shutil.copytree(roots[0], target)
        archive_entries = 0
        for file in sorted(target.rglob("*")):
            if file.is_file():
                hashes[str(file.relative_to(output))] = hashlib.sha256(file.read_bytes()).hexdigest()
                if file.suffix == ".zip":
                    with zipfile.ZipFile(file) as archive:
                        archive_entries += len(archive.namelist())
        tasks.append({**row, "problem_id": identifier, "seed": index,
                      # Conservative batch deadline; the official per-case limit stays 10s.
                      "judge_timeout_seconds": max(300, 120 + 60 * archive_entries)})
    write_json(output / "manifest.json", {
        "dataset": DATASET, "dataset_revision": DATASET_REVISION,
        "selection_seed": 0, "per_difficulty": 4, "upstream_skips": sorted(UPSTREAM_SKIPS),
        "prompts_sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
        "model": MODEL, "model_revision": MODEL_REVISION, "precision": "bf16",
        "enable_thinking": thinking, "temperature": temperature, "top_p": top_p, "top_k": top_k,
        "presence_penalty": presence_penalty, "presence_scope": "all generated tokens; excludes prompt",
        "max_tokens": max_tokens,
        "tasks": tasks, "problem_file_sha256": hashes,
    })
    print(output / "manifest.json", flush=True)


def summarize(results, total):
    graded = [row for row in results if row["status"] == "graded"]
    scored = [row for row in results if row["status"] in {"graded", "no_final"}]
    accepted = sum(row["verdict"] == "AC" for row in graded)
    return {
        "planned": total, "finished": len(results), "graded": len(graded), "accepted": accepted,
        "scored": len(scored), "accepted_fraction_of_scored": accepted / len(scored) if scored else None,
        "accepted_fraction_of_graded": accepted / len(graded) if graded else None,
        "statuses": dict(Counter(row["status"] for row in results)),
        "by_difficulty": {
            difficulty: {"graded": sum(row["difficulty"] == difficulty for row in graded),
                         "scored": sum(row["difficulty"] == difficulty for row in scored),
                         "accepted": sum(row["difficulty"] == difficulty and row["verdict"] == "AC"
                                         for row in graded)}
            for difficulty in ("easy", "medium", "hard")},
        "results": results,
    }


def extract_final_response(response, *, thinking):
    """Keep reasoning drafts out of the scored submission."""
    if thinking:
        _, separator, final = response.partition("</think>")
        return final if separator else ""
    return response


def make_presence_processor(penalty):
    """Penalize tokens already generated, excluding the initial prompt chunk."""
    prompt_length = None

    def process(tokens, logits):
        nonlocal prompt_length
        if prompt_length is None:
            prompt_length = len(tokens)
        generated = tokens[prompt_length:]
        if len(generated):
            logits[:, generated] -= penalty
        return logits

    return process


def consume_stop_request(output, task_dir, identifier, finish_reason):
    control = output / "stop.json"
    if not control.exists():
        return None
    request = json.loads(control.read_text())
    if request.get("problem_id") != identifier:
        raise ValueError("Stop request must name the active problem")
    # A completed answer still gets judged if the stop request arrived too late.
    filename = "late-stop.json" if finish_reason else "intervention.json"
    write_json(task_dir / filename, request)
    control.unlink()
    return None if finish_reason else request


def run(manifest_path, output):
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    manifest = json.loads(manifest_path.read_text())
    output_budget = manifest.get("max_tokens")
    if output_budget is not None and (type(output_budget) is not int or output_budget <= 0):
        raise ValueError("max_tokens must be a positive integer or null")
    for name, expected in manifest["problem_file_sha256"].items():
        if hashlib.sha256((manifest_path.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Problem file changed: {name}")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", DEFAULT_IMAGE]))[0]["Id"]
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, output / "manifest.json")
    write_json(output / "environment.json", {
        "pid": os.getpid(), "image": image, "started_at": time.time(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("mlx", "mlx-lm", "transformers", "huggingface-hub")},
    })

    def progress(phase, **values):
        write_json(output / "progress.json", {"phase": phase, "updated_at": time.time(), **values})

    progress("loading")
    started = time.monotonic()
    model, tokenizer, config = load(manifest["model"], revision=manifest["model_revision"],
                                    return_config=True)
    context_size = config.get("text_config", config)["max_position_embeddings"]
    print(f"Model loaded in {time.monotonic() - started:.1f}s; context={context_size}", flush=True)
    results = []
    for index, task in enumerate(manifest["tasks"]):
        task_dir = output / task["problem_id"]
        task_dir.mkdir()
        messages = [{"role": "user", "content": task["prompt"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=manifest["enable_thinking"])
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=manifest["enable_thinking"])
        # Reserve one position for MLX's asynchronous look-ahead token.
        max_tokens = context_size - len(tokens) - 1
        if output_budget is not None:
            max_tokens = min(max_tokens, output_budget)
        if max_tokens <= 0:
            raise ValueError("Prompt exceeds the model context")
        cache = make_prompt_cache(model)
        request = {"id": task["id"], "seed": task["seed"], "messages": messages,
                   "prompt_tokens": len(tokens), "max_tokens": max_tokens,
                   "context_size": context_size, "prefill_step_size": 2048,
                   "cache_types": dict(Counter(type(layer).__name__ for layer in cache)),
                   "kv_bits": None}
        write_json(task_dir / "request.json", request)
        (task_dir / "rendered-prompt.txt").write_text(prompt)
        mx.random.seed(task["seed"])
        mx.reset_peak_memory()
        started = time.monotonic()
        identity = {"problem_id": task["problem_id"], "index": index + 1,
                    "total": len(manifest["tasks"])}
        progress("prefill", **identity)
        print(f"Starting {index + 1}/{len(manifest['tasks'])}: {task['problem_id']}", flush=True)
        final = None
        intervention = None
        last_progress = 0
        stream = stream_generate(
            model, tokenizer, tokens, max_tokens=max_tokens, prompt_cache=cache,
            prefill_step_size=2048,
            logits_processors=([make_presence_processor(manifest["presence_penalty"])]
                               if manifest.get("presence_penalty", 0) else []),
            sampler=make_sampler(temp=manifest["temperature"], top_p=manifest["top_p"],
                                 top_k=manifest["top_k"]),
            prompt_progress_callback=lambda done, total: progress(
                "prefill", **identity, prompt_processed=done, prompt_tokens=total),
        )
        try:
            with (task_dir / "response.txt").open("w") as response_file, \
                    (task_dir / "generated-token-ids.jsonl").open("w") as token_file:
                for response in stream:
                    response_file.write(response.text)
                    token_file.write(f"{response.token}\n")
                    final = response
                    now = time.monotonic()
                    if now - last_progress >= 5 or response.finish_reason:
                        response_file.flush()
                        token_file.flush()
                        progress("generating", **identity, generation_tokens=response.generation_tokens,
                                 tokens_per_second=response.generation_tps,
                                 elapsed_seconds=now - started, peak_memory_gb=response.peak_memory)
                        last_progress = now
                        intervention = consume_stop_request(
                            output, task_dir, task["problem_id"], response.finish_reason)
                        if intervention:
                            break
        finally:
            stream.close()
            del stream, cache
            mx.clear_cache()
        if final is None:
            raise RuntimeError("Model returned no tokens")
        generation = {"generation_tokens": final.generation_tokens,
                      "generation_seconds": time.monotonic() - started,
                      "tokens_per_second": final.generation_tps, "peak_memory_gb": final.peak_memory,
                      "finish_reason": "intervention" if intervention else final.finish_reason}
        write_json(task_dir / "generation.json", generation)
        record = {"id": task["id"], "problem_id": task["problem_id"],
                  "difficulty": task["difficulty"], **generation}
        if intervention:
            record.update(status="interrupted", verdict=None)
        else:
            answer = extract_final_response((task_dir / "response.txt").read_text(),
                                            thinking=manifest["enable_thinking"])
            (task_dir / "final.txt").write_text(answer)
            if not answer.strip():
                record.update(status="no_final", verdict=None)
                results.append(record)
                write_json(output / "summary.json", summarize(results, len(manifest["tasks"])))
                print(json.dumps(record), flush=True)
                continue
            submission = task_dir / "submissions.jsonl"
            submission.write_text(json.dumps({"id": task["id"], "language": "cpp",
                                             "channel": "final", "content": answer}) + "\n")
            progress("grading", **identity, **generation)
            try:
                judged = grade_submissions(submission, [manifest_path.parent / "problems"],
                                           task_dir / "judge", image=image,
                                           timeout=task["judge_timeout_seconds"])[0]
                record.update(status=judged["status"], verdict=judged["verdict"])
            except JudgeError as error:
                record.update(status="error", verdict=None, error=str(error))
        results.append(record)
        write_json(output / "summary.json", summarize(results, len(manifest["tasks"])))
        print(json.dumps(record), flush=True)
    progress("complete", **summarize(results, len(manifest["tasks"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--no-thinking", action="store_true")
    prep.add_argument("--temperature", type=float, default=0.6)
    prep.add_argument("--top-p", type=float, default=0.95)
    prep.add_argument("--top-k", type=int, default=20)
    prep.add_argument("--presence-penalty", type=float, default=0.0)
    prep.add_argument("--max-tokens", type=int, default=None)
    execute = commands.add_parser("run")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, thinking=not args.no_thinking, temperature=args.temperature,
                top_p=args.top_p, top_k=args.top_k,
                presence_penalty=args.presence_penalty, max_tokens=args.max_tokens)
    else:
        run(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
