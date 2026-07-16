"""Navigation release provenance, JSON/CSV output, and repository audit."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_REQUIRED_FILES = (
    "pyproject.toml",
    "uv.lock",
    "conf/navigation/reproducibility.json",
    "docs/codex/implementation-roadmap.md",
    "docs/codex/current-state.md",
    "docs/codex/navigation-reproducibility.md",
    "scripts/navigation_repro.py",
)
_REQUIRED_IGNORE_RULES = (
    "logs/",
    "runs/",
    "wandb/",
    "checkpoints/",
    "events.out.tfevents.*",
    "*.ckpt",
)
_ARTIFACT_PATTERN = re.compile(
    r"(^|/)(logs|runs|wandb|checkpoints)(/|$)|"
    r"events\.out\.tfevents|\.(pt|pth|ckpt)$|"
    r"(^|/)(run_config|run_summary)\.json$"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _package_version(name: str, module_name: str | None = None) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        if module_name is None:
            return "unavailable"
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))


def build_navigation_provenance(
    root: str | Path,
    *,
    workflow: str,
    command: Sequence[str],
    config: Mapping[str, Any],
    seeds: Sequence[int],
    input_files: Sequence[str | Path] = (),
    output_files: Sequence[str | Path] = (),
    command_results: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a strict provenance object for one navigation workflow."""
    repository = Path(root).resolve()
    if not workflow or not command:
        raise ValueError("provenance workflow and command must be non-empty")
    normalized_seeds = [int(seed) for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds) or any(
        seed < 0 for seed in normalized_seeds
    ):
        raise ValueError("provenance seeds must be unique and non-negative")

    def hashes(paths: Sequence[str | Path]) -> dict[str, str]:
        result = {}
        for value in paths:
            path = Path(value).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"provenance file does not exist: {path}")
            result[str(path)] = file_sha256(path)
        return result

    status = _git(repository, "status", "--short")
    return {
        "schema_version": 1,
        "workflow": workflow,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command),
        "config": dict(config),
        "seeds": normalized_seeds,
        "git": {
            "commit": _git(repository, "rev-parse", "HEAD"),
            "branch": _git(repository, "branch", "--show-current"),
            "dirty": bool(status),
            "status": status.splitlines() if status else [],
        },
        "dependency_lock": {
            "path": str(repository / "uv.lock"),
            "sha256": file_sha256(repository / "uv.lock"),
        },
        "software_hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch", "torch"),
            "mujoco": _package_version("mujoco", "mujoco"),
        },
        "inputs": hashes(input_files),
        "outputs": hashes(output_files),
        "command_results": [dict(result) for result in command_results],
    }


def validate_navigation_provenance(payload: Mapping[str, Any]) -> None:
    """Reject incomplete release provenance before publication."""
    required = {
        "schema_version",
        "workflow",
        "created_at_utc",
        "command",
        "config",
        "seeds",
        "git",
        "dependency_lock",
        "software_hardware",
        "inputs",
        "outputs",
        "command_results",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"provenance is missing fields: {missing}")
    if int(payload["schema_version"]) != 1:
        raise ValueError("provenance schema_version must be 1")
    if not payload["workflow"] or not payload["command"]:
        raise ValueError("provenance workflow and command must be non-empty")
    lock = payload["dependency_lock"]
    if not lock.get("path") or not re.fullmatch(r"[0-9a-f]{64}", lock.get("sha256", "")):
        raise ValueError("provenance dependency lock is incomplete")
    for group in ("inputs", "outputs"):
        if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in payload[group].values()):
            raise ValueError(f"provenance {group} contain invalid hashes")


def write_navigation_provenance(
    payload: Mapping[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Write the versioned JSON object and one-row portable CSV summary."""
    validate_navigation_provenance(payload)
    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    row = {
        "schema_version": payload["schema_version"],
        "workflow": payload["workflow"],
        "created_at_utc": payload["created_at_utc"],
        "git_commit": payload["git"]["commit"],
        "git_branch": payload["git"]["branch"],
        "git_dirty": payload["git"]["dirty"],
        "lock_sha256": payload["dependency_lock"]["sha256"],
        "seeds_json": json.dumps(payload["seeds"], separators=(",", ":")),
        "config_json": json.dumps(payload["config"], sort_keys=True, separators=(",", ":")),
        "inputs_json": json.dumps(payload["inputs"], sort_keys=True, separators=(",", ":")),
        "outputs_json": json.dumps(payload["outputs"], sort_keys=True, separators=(",", ":")),
    }
    with csv_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return json_output, csv_output


def find_tracked_generated_artifacts(paths: Sequence[str]) -> list[str]:
    """Return tracked training/runtime artifacts forbidden by the release contract."""
    return sorted(path for path in paths if _ARTIFACT_PATTERN.search(path))


def audit_navigation_release(root: str | Path) -> dict[str, Any]:
    """Audit lock, release files, ignore rules, and tracked artifact hygiene."""
    repository = Path(root).resolve()
    tracked = _git(repository, "ls-files").splitlines()
    missing_files = [path for path in _REQUIRED_FILES if not (repository / path).is_file()]
    gitignore = (repository / ".gitignore").read_text()
    missing_ignore_rules = [rule for rule in _REQUIRED_IGNORE_RULES if rule not in gitignore]
    tracked_artifacts = find_tracked_generated_artifacts(tracked)
    errors = [
        *(f"missing required file: {path}" for path in missing_files),
        *(f"missing .gitignore rule: {rule}" for rule in missing_ignore_rules),
        *(f"tracked generated artifact: {path}" for path in tracked_artifacts),
    ]
    return {
        "schema_version": 1,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "dependency_lock": {
            "path": "uv.lock",
            "sha256": file_sha256(repository / "uv.lock")
            if (repository / "uv.lock").is_file()
            else None,
        },
        "required_files": list(_REQUIRED_FILES),
        "required_ignore_rules": list(_REQUIRED_IGNORE_RULES),
        "tracked_file_count": len(tracked),
        "tracked_generated_artifacts": tracked_artifacts,
    }
