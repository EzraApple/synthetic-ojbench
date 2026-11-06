"""Preserve signed-character EOF semantics in affected ARM64 validators."""

import hashlib
import json
import platform
import re
from pathlib import Path

EOF_CHAR = re.compile(
    r"\bchar\s+(\w+)\s*=\s*\w+\.get\(\)\s*;\s*"
    r"if\s*\(\s*\1\s*==\s*std::char_traits<char>::eof\(\)"
)


def prepare_package(package, overlay_root):
    import yaml

    original = (package / "init.yml").read_bytes()
    try:
        config = yaml.safe_load(original)
    except yaml.YAMLError:
        # The official per-entry evaluator reports malformed packages as errors.
        return None
    if not isinstance(config, dict):
        return None
    helpers = []
    checker = config.get("checker")
    if isinstance(checker, dict) and checker.get("name") == "bridged":
        helpers.append(("checker.args.flags", checker.get("args", {})))
    if isinstance(config.get("interactive"), dict):
        helpers.append(("interactive.flags", config["interactive"]))
    changes = []
    for field, helper in helpers:
        files = helper.get("files", [])
        if isinstance(files, str):
            files = [files]
        matches = {}
        for name in files:
            source = package / name
            if source.is_file() and EOF_CHAR.search(source.read_text(errors="replace")):
                matches[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if not matches:
            continue
        flags = helper.get("flags", [])
        if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
            raise ValueError(f"{package.name}: validator flags must be a list of strings")
        if "-funsigned-char" in flags:
            raise ValueError(f"{package.name}: explicit unsigned-char validator conflicts with EOF compatibility")
        if "-fsigned-char" not in flags:
            helper["flags"] = [*flags, "-fsigned-char"]
            changes.append({"field": field, "previous_flags": flags,
                            "flags": helper["flags"], "source_sha256": matches})
    if not changes:
        return None
    target = overlay_root / package.name
    target.mkdir(parents=True, exist_ok=False)
    for child in package.iterdir():
        if child.name != "init.yml":
            (target / child.name).symlink_to(child.resolve(), target_is_directory=child.is_dir())
    updated = yaml.safe_dump(config, sort_keys=False).encode()
    (target / "init.yml").write_bytes(updated)
    return {"problem_id": package.name, "source_package": str(package),
            "original_config_sha256": hashlib.sha256(original).hexdigest(),
            "overlay_config_sha256": hashlib.sha256(updated).hexdigest(), "changes": changes}


def prepare_problem_dirs(problem_dirs, problem_ids, output):
    """Overlay only affected helper flags; keep sources, tests and submissions unchanged."""
    machine = platform.machine().lower()
    for identifier in problem_ids:
        if (not isinstance(identifier, str) or not identifier or identifier in {".", ".."}
                or any(char in identifier for char in ("/", "\\", "\0"))):
            raise ValueError("Problem IDs must be simple directory names")
    metadata = {"policy": "signed-char-eof-v1", "machine": machine, "overlays": []}
    overlay_root = output / "compatibility" / "problems"
    if machine in {"aarch64", "arm64"}:
        for problem_id in sorted(problem_ids):
            package = next((root / problem_id for root in problem_dirs
                            if (root / problem_id / "init.yml").is_file()), None)
            if package is not None:
                record = prepare_package(package, overlay_root)
                if record:
                    metadata["overlays"].append(record)
    (output / "compatibility.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return [overlay_root, *problem_dirs] if metadata["overlays"] else problem_dirs
