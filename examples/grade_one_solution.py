"""Generate and grade one addition solution; this is not an OJBench evaluation."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from synthetic_ojbench.grading import grade_submissions
from synthetic_ojbench.pilot import extract_final_response

MODEL = "mlx-community/Qwen3.5-9B-bf16"
REVISION = "27cdb77d9381d508abfe5d4e06bcb95471e312ac"
PROMPT = """Write a complete C++17 program for this problem.

Read pairs of integers a and b from standard input until EOF. Each pair is on
one line. For each pair, print a + b on its own line. There are at most 100000
pairs, and each integer is between -1000000000 and 1000000000 inclusive.

Return your complete solution in one fenced code block labeled cpp.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {
        "model": MODEL, "revision": REVISION, "precision": "bf16",
        "enable_thinking": True, "seed": 0, "temperature": 0.6,
        "top_p": 0.95, "top_k": 20, "max_tokens": 1024,
        "messages": [{"role": "user", "content": PROMPT}],
        "versions": {name: importlib.metadata.version(name)
                     for name in ["mlx", "mlx-lm", "transformers", "huggingface-hub"]},
    }
    (args.output / "request.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.perf_counter()
    print(f"Loading {MODEL} at {REVISION}", flush=True)
    model, tokenizer = load(MODEL, revision=REVISION)
    load_seconds = time.perf_counter() - started
    prompt = tokenizer.apply_chat_template(settings["messages"], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
    (args.output / "rendered-prompt.txt").write_text(prompt)
    mx.random.seed(settings["seed"])
    sampler = make_sampler(temp=settings["temperature"], top_p=settings["top_p"],
                           top_k=settings["top_k"])
    started = time.perf_counter()
    final = None
    with (args.output / "response.txt").open("w") as output:
        for response in stream_generate(model, tokenizer, prompt, sampler=sampler,
                                        max_tokens=settings["max_tokens"]):
            output.write(response.text)
            output.flush()
            print(response.text, end="", flush=True)
            final = response
    if final is None:
        raise RuntimeError("Model returned no generation result")
    metrics = {
        "load_seconds": load_seconds, "generation_seconds": time.perf_counter() - started,
        "prompt_tokens": final.prompt_tokens, "generation_tokens": final.generation_tokens,
        "prompt_tokens_per_second": final.prompt_tps,
        "generation_tokens_per_second": final.generation_tps,
        "peak_memory_gb": final.peak_memory, "finish_reason": final.finish_reason,
    }
    (args.output / "generation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    answer = extract_final_response((args.output / "response.txt").read_text(), thinking=True)
    (args.output / "final.txt").write_text(answer)
    if not answer.strip():
        raise RuntimeError("No final answer within the token budget; reasoning drafts were not graded")
    submission = {"id": "addition", "language": "cpp", "channel": "final", "content": answer}
    path = args.output / "submissions.jsonl"
    path.write_text(json.dumps(submission) + "\n")
    problems = Path(__file__).resolve().parents[1] / "tests/fixtures/problems"
    results = grade_submissions(path, [problems], args.output / "judge")
    print("\n" + json.dumps({"generation": metrics, "status": results[0]["status"],
                             "verdict": results[0]["verdict"]}, indent=2))
    if results[0]["status"] != "graded":
        raise RuntimeError("Judge failed; inspect the saved artifacts")


if __name__ == "__main__":
    main()
