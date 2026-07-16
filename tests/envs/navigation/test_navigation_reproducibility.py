"""Tests for navigation seed, provenance, and artifact release contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from scripts import navigation_repro

from unilab.reproducibility import (
    audit_navigation_release,
    build_navigation_provenance,
    find_tracked_generated_artifacts,
    validate_navigation_provenance,
    write_navigation_provenance,
)

ROOT_DIR = Path(__file__).parents[3]


def test_seed_manifest_versions_quick_and_formal_protocols() -> None:
    payload = json.loads(
        (ROOT_DIR / "conf" / "navigation" / "reproducibility.json").read_text()
    )
    assert payload["schema_version"] == 1
    assert payload["task"] == "DiffDrivePointGoal"
    assert payload["formal"]["training_seeds"] == [1, 2, 3]
    assert payload["formal"]["training_environment_steps_per_seed"] == 98_304
    assert payload["formal"]["benchmark_seeds"] == [101, 202, 303]
    assert payload["quick"]["evaluation_episodes"] < payload["formal"][
        "evaluation_episodes"
    ]


def test_provenance_writes_strict_json_and_one_row_csv(tmp_path) -> None:
    output = tmp_path / "result.json"
    output.write_text('{"status": "passed"}\n')
    payload = build_navigation_provenance(
        ROOT_DIR,
        workflow="test",
        command=["python", "test"],
        config={"episodes": 2},
        seeds=[1, 2],
        input_files=[ROOT_DIR / "conf" / "navigation" / "reproducibility.json"],
        output_files=[output],
        command_results=[{"status": "passed", "wall_time_seconds": 0.1}],
    )
    json_path, csv_path = write_navigation_provenance(
        payload,
        tmp_path / "provenance.json",
        tmp_path / "provenance.csv",
    )
    restored = json.loads(json_path.read_text())
    validate_navigation_provenance(restored)
    assert restored["dependency_lock"]["sha256"]
    assert list(restored["outputs"].values())[0]
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["workflow"] == "test"
    assert json.loads(rows[0]["seeds_json"]) == [1, 2]


def test_provenance_rejects_missing_fields_and_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="missing fields"):
        validate_navigation_provenance({"schema_version": 1})
    with pytest.raises(ValueError, match="unique"):
        build_navigation_provenance(
            ROOT_DIR,
            workflow="test",
            command=["test"],
            config={},
            seeds=[1, 1],
        )


def test_artifact_audit_rejects_generated_training_files() -> None:
    violations = find_tracked_generated_artifacts(
        [
            "src/module.py",
            "logs/run/model_1.pt",
            "wandb/latest-run/file.json",
            "events.out.tfevents.123",
            "run_summary.json",
        ]
    )
    assert violations == [
        "events.out.tfevents.123",
        "logs/run/model_1.pt",
        "run_summary.json",
        "wandb/latest-run/file.json",
    ]


def test_repository_navigation_release_audit_passes() -> None:
    report = audit_navigation_release(ROOT_DIR)
    assert report["status"] == "passed", report["errors"]
    assert report["tracked_generated_artifacts"] == []
    assert report["dependency_lock"]["sha256"]


def test_quick_and_formal_benchmark_commands_use_seed_manifest(tmp_path) -> None:
    manifest = navigation_repro._seed_manifest()
    quick = navigation_repro._benchmark_command(
        "quick",
        tmp_path / "quick.json",
        manifest,
    )
    formal = navigation_repro._benchmark_command(
        "formal",
        tmp_path / "formal.json",
        manifest,
    )
    assert "32" in quick
    assert "256" in formal
    assert formal[formal.index("--repetitions") + 1] == "5"
    seed_index = formal.index("--seeds") + 1
    assert formal[seed_index : seed_index + 3] == ["101", "202", "303"]
