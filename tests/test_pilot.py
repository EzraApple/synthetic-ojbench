import json
import tempfile
import unittest
from pathlib import Path

from synthetic_ojbench.pilot import consume_stop_request, extract_final_response, make_presence_processor, select_tasks, summarize


class PilotTests(unittest.TestCase):
    def test_thinking_drafts_cannot_be_scored_without_a_final_answer(self):
        draft = "reasoning\n```cpp\nint main(){}\n```"
        self.assertEqual(extract_final_response(draft, thinking=True), "")
        final = "\n```cpp\nint main(){return 0;}\n```"
        self.assertEqual(extract_final_response(draft + "</think>" + final, thinking=True), final)
        literal = '```cpp\nconst char* tag = "</think>";\n```'
        self.assertEqual(extract_final_response(literal, thinking=False), literal)

    def test_stop_request_preserves_natural_completion_and_cannot_leak_to_next_task(self):
        for finish_reason in (None, "stop", "length"):
            with self.subTest(finish_reason=finish_reason), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                task_dir = output / "task-a"
                task_dir.mkdir()
                request = {"problem_id": "task-a", "reason": "repeated reasoning"}
                (output / "stop.json").write_text(json.dumps(request))
                result = consume_stop_request(output, task_dir, "task-a", finish_reason)
                self.assertEqual(result, None if finish_reason else request)
                saved = "late-stop.json" if finish_reason else "intervention.json"
                self.assertEqual(json.loads((task_dir / saved).read_text()), request)
                self.assertFalse((output / "stop.json").exists())
                self.assertIsNone(consume_stop_request(output, task_dir, "task-b", None))

    def test_presence_penalty_excludes_prompt_and_counts_each_token_once(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("install inference dependencies for the MLX sampling check")
        process = make_presence_processor(1.5)
        self.assertEqual(process(mx.array([0, 1]), mx.zeros((1, 4))).tolist(),
                         [[0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(process(mx.array([0, 1, 2, 2]), mx.zeros((1, 4))).tolist(),
                         [[0.0, 0.0, -1.5, 0.0]])
        self.assertEqual(process(mx.array([0, 1, 2, 2, 1]), mx.zeros((1, 4))).tolist(),
                         [[0.0, -1.5, -1.5, 0.0]])

    def test_selection_is_stratified_stable_and_excludes_upstream_skips(self):
        rows = [{"id": f"{difficulty}-{index}", "difficulty": difficulty, "language": "cpp"}
                for difficulty in ("easy", "medium", "hard") for index in range(8)]
        rows += [{"id": 3897, "difficulty": "easy", "language": "cpp"},
                 {"id": "python-only", "difficulty": "hard", "language": "python"}]
        selected = select_tasks(rows)
        self.assertEqual(selected, select_tasks(list(reversed(rows))))
        self.assertEqual(len({row["id"] for row in selected}), 12)
        for difficulty in ("easy", "medium", "hard"):
            self.assertEqual(sum(row["difficulty"] == difficulty for row in selected), 4)
        self.assertNotIn(3897, [row["id"] for row in selected])
        self.assertTrue(all(row["language"] == "cpp" for row in selected))

    def test_infrastructure_errors_and_interventions_are_not_wrong_answers(self):
        results = [
            {"status": "graded", "verdict": "AC", "difficulty": "easy"},
            {"status": "graded", "verdict": "WA", "difficulty": "medium"},
            {"status": "error", "verdict": None, "difficulty": "hard"},
            {"status": "interrupted", "verdict": None, "difficulty": "hard"},
            {"status": "no_final", "verdict": None, "difficulty": "medium"},
        ]
        summary = summarize(results, 12)
        self.assertEqual(summary["graded"], 2)
        self.assertEqual(summary["accepted_fraction_of_graded"], 0.5)
        self.assertEqual(summary["scored"], 3)
        self.assertEqual(summary["accepted_fraction_of_scored"], 1 / 3)
        self.assertEqual(summary["finished"], 5)
        self.assertEqual(summary["by_difficulty"]["hard"]["graded"], 0)
        self.assertIsNone(summarize([], 12)["accepted_fraction_of_graded"])


if __name__ == "__main__":
    unittest.main()
