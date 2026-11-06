import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthetic_ojbench.grading import JudgeError, grade_submissions

FIXTURES = Path(__file__).parent / "fixtures"
CPP = "#include <iostream>\nint main(){long long a,b;while(std::cin>>a>>b)std::cout<<a+b<<'\\n';}"


def write_submissions(path, programs):
    entries = [{"id": "addition", "attempt_id": name, "language": "cpp",
                "content": f"```cpp\n{source}\n```"} for name, source in programs]
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))


class GradingFailureTests(unittest.TestCase):
    def test_problem_ids_cannot_escape_the_problem_directory(self):
        for identifier in ("", ".", "..", "../addition", "/addition", "path\\addition", "bad\0id"):
            with self.subTest(identifier=identifier), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inputs, output = root / "input.jsonl", root / "output"
                inputs.write_text(json.dumps({"id": identifier, "language": "cpp", "content": CPP}) + "\n")
                with patch("synthetic_ojbench.grading.subprocess.run") as docker:
                    with self.assertRaisesRegex(ValueError, "simple directory names"):
                        grade_submissions(inputs, [FIXTURES / "problems"], output)
                    docker.assert_not_called()
                self.assertFalse(output.exists())

    def test_cleanup_failure_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "input.jsonl", root / "output"
            write_submissions(inputs, [("correct", CPP)])

            def docker(command, **kwargs):
                if command[1:3] == ["image", "inspect"]:
                    return subprocess.CompletedProcess(command, 0, json.dumps([
                        {"Id": "sha256:test", "Architecture": "arm64", "Config": {}}
                    ]))
                if command[1] == "run":
                    entry = json.loads(inputs.read_text())
                    entry.update(status="graded", verdict="AC", is_passed=True)
                    (output / "results.jsonl").write_text(json.dumps(entry) + "\n")
                    return subprocess.CompletedProcess(command, 0)
                return subprocess.CompletedProcess(command, 1, "", "engine unavailable")

            with patch("synthetic_ojbench.grading.subprocess.run", side_effect=docker):
                with self.assertRaisesRegex(JudgeError, "Could not remove container"):
                    grade_submissions(inputs, [FIXTURES / "problems"], output)
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["status"], "error")
            self.assertEqual(metadata["cleanup"]["returncode"], 1)
            self.assertTrue((output / "results.jsonl").exists())

    def test_timeout_removes_container_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, output = root / "input.jsonl", root / "output"
            write_submissions(inputs, [("hang", CPP)])
            calls = []

            def docker(command, **kwargs):
                calls.append(command)
                if command[1:3] == ["image", "inspect"]:
                    return subprocess.CompletedProcess(command, 0, json.dumps([
                        {"Id": "sha256:test", "Architecture": "arm64", "Config": {}}
                    ]))
                if command[1] == "run":
                    kwargs["stdout"].write("judge started\n")
                    raise subprocess.TimeoutExpired(command, kwargs["timeout"])
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("synthetic_ojbench.grading.subprocess.run", side_effect=docker):
                with self.assertRaises(JudgeError):
                    grade_submissions(inputs, [FIXTURES / "problems"], output, timeout=1)
            metadata = json.loads((output / "run.json").read_text())
            self.assertEqual(metadata["status"], "error")
            self.assertIn("TimeoutExpired", metadata["error"])
            self.assertEqual((output / "judge.log").read_text(), "judge started\n")
            self.assertTrue((output / "submissions.jsonl").exists())
            self.assertEqual(calls[-1], ["docker", "rm", "--force", metadata["container"]])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "input.jsonl"
            write_submissions(inputs, [("correct", CPP)])
            with patch("synthetic_ojbench.grading.subprocess.run") as docker:
                with self.assertRaises(FileExistsError):
                    grade_submissions(inputs, [FIXTURES / "problems"], root)
                docker.assert_not_called()


@unittest.skipUnless(os.environ.get("OJ_JUDGE_SMOKE") == "1", "set OJ_JUDGE_SMOKE=1 to run Docker checks")
class JudgeSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="judge-smoke-", dir=cls.artifact_root()))
        print(f"\nSmoke-test artifacts: {cls.root}", flush=True)

    @staticmethod
    def artifact_root():
        root = Path("artifacts")
        root.mkdir(exist_ok=True)
        return root

    def assert_container_removed(self, output):
        metadata = json.loads((output / "run.json").read_text())
        self.assertEqual(metadata["cleanup"]["returncode"], 0)
        inspected = subprocess.run(["docker", "container", "inspect", metadata["container"]],
                                   capture_output=True, text=True)
        self.assertNotEqual(inspected.returncode, 0)

    def test_known_verdicts(self):
        inputs, output = self.root / "known.jsonl", self.root / "known"
        write_submissions(inputs, [
            ("correct", CPP),
            ("wrong", CPP.replace("a+b", "a-b")),
            ("compile_error", "int main( {"),
            ("timeout", "int main(){for(;;){asm volatile(\"\");}}"),
            ("runtime_error", "int main(){return 1;}"),
        ])
        results = grade_submissions(inputs, [FIXTURES / "problems"], output, timeout=120)
        self.assertEqual([r["status"] for r in results], ["graded"] * 5)
        self.assertEqual([r["verdict"] for r in results], ["AC", "WA", "CE", "TLE", "IR"])
        self.assertEqual([r["is_passed"] for r in results], [True, False, False, False, False])
        self.assert_container_removed(output)

    def test_eof_compatibility_changes_only_validator_flags_and_keeps_tests_private(self):
        problem = self.root / "eof-problems" / "eof-checker"
        problem.mkdir(parents=True)
        config = '''checker:
  name: bridged
  args:
    files: [validator.cpp]
    lang: CPP17
    type: default
test_cases:
- in: cases.in
  out: cases.out
  points: 1
'''
        (problem / "init.yml").write_text(config)
        (problem / "cases.in").write_text("2 3\n")
        (problem / "cases.out").write_text("5\n")
        (problem / "validator.cpp").write_text('''#include <fstream>
#include <string>
bool eof(std::istream& in) {
    in >> std::ws;
    char c = in.get();
    if(c == std::char_traits<char>::eof()) return true;
    return false;
}
int main(int argc, char** argv) {
    std::ifstream output(argv[2]), answer(argv[3]);
    std::string got, expected;
    if(!(output >> got) || !(answer >> expected) || got != expected) return 1;
    return eof(output) ? 0 : 1;
}
''')
        # Contestant plain-char behavior must remain the ARM64 default.
        correct = "#include <limits>\n" + CPP.replace(
            "int main(){", "int main(){if(std::numeric_limits<char>::is_signed)return 1;")
        leak = '''#include <fstream>
#include <iostream>
int main(){std::ifstream f("/output/compatibility/problems/eof-checker/cases.out");
if(!f)return 42;std::cout<<f.rdbuf();}
'''
        inputs, output = self.root / "eof.jsonl", self.root / "eof-results"
        entries = [{"id": "eof-checker", "attempt_id": name, "language": "cpp", "content": code}
                   for name, code in (("correct", correct), ("wrong", CPP.replace("a+b", "a-b")), ("leak", leak))]
        inputs.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        results = grade_submissions(inputs, [problem.parent], output)
        self.assertEqual([row["verdict"] for row in results], ["AC", "WA", "IR"])
        metadata = json.loads((output / "run.json").read_text())
        overlay = metadata["compatibility"]["overlays"][0]
        self.assertEqual(overlay["problem_id"], "eof-checker")
        self.assertEqual(overlay["changes"][0]["flags"], ["-fsigned-char"])
        self.assertEqual((problem / "init.yml").read_text(), config)
        self.assert_container_removed(output)

    def test_submission_cannot_read_tests_or_write_results(self):
        inputs, output = self.root / "isolation.jsonl", self.root / "isolation"
        source = '''#include <fstream>
int main() {
    std::ifstream tests("/problems/0/addition/cases.out");
    std::ofstream results("/output/forged.txt");
    return tests.is_open() || results.is_open() ? 0 : 42;
}'''
        write_submissions(inputs, [("isolation", source)])
        results = grade_submissions(inputs, [FIXTURES / "problems"], output)
        self.assertEqual(results[0]["status"], "graded")
        self.assertEqual(results[0]["verdict"], "IR")
        self.assertFalse((output / "forged.txt").exists())
        self.assert_container_removed(output)

    def test_missing_problem_is_an_error_not_a_wrong_answer(self):
        inputs, output = self.root / "missing.jsonl", self.root / "missing"
        inputs.write_text(json.dumps({"id": "missing", "language": "cpp", "content": CPP}) + "\n")
        results = grade_submissions(inputs, [FIXTURES / "problems"], output)
        self.assertEqual(results[0]["status"], "error")
        self.assertIsNone(results[0]["is_passed"])
        self.assertIsNone(results[0]["verdict"])
        self.assert_container_removed(output)

    def test_invalid_problem_config_does_not_drop_other_results(self):
        problems = self.root / "mixed-problems"
        shutil.copytree(FIXTURES / "problems" / "addition", problems / "addition")
        invalid = problems / "invalid"
        invalid.mkdir()
        (invalid / "init.yml").write_text("checker: [\n")
        inputs, output = self.root / "mixed.jsonl", self.root / "mixed-results"
        inputs.write_text("".join(json.dumps({"id": name, "language": "cpp", "content": CPP}) + "\n"
                                  for name in ("invalid", "addition")))
        results = grade_submissions(inputs, [problems], output)
        self.assertEqual([(row["status"], row["verdict"]) for row in results],
                         [("error", None), ("graded", "AC")])
        self.assert_container_removed(output)

    def test_batch_deadline_stops_the_container(self):
        inputs, output = self.root / "deadline.jsonl", self.root / "deadline"
        write_submissions(inputs, [("hang", "int main(){for(;;){asm volatile(\"\");}}")])
        with self.assertRaises(JudgeError):
            grade_submissions(inputs, [FIXTURES / "problems"], output, timeout=2)
        metadata = json.loads((output / "run.json").read_text())
        self.assertIn("TimeoutExpired", metadata["error"])
        self.assert_container_removed(output)


if __name__ == "__main__":
    unittest.main()
