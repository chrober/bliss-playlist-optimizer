#!/usr/bin/env python3
"""Verify stable, privacy-safe expectations after running the Python oracle."""

from __future__ import annotations

import json
import math
import pathlib


def main() -> None:
    fixture = pathlib.Path(__file__).resolve().parent
    expected = json.loads((fixture / "expected-python-oracle-v1.json").read_text(encoding="utf-8"))
    result_path = fixture / "oracle-output" / "run.json"
    if not result_path.exists():
        raise SystemExit("Missing oracle-output/run.json; run the command in manifest.json first")
    actual = json.loads(result_path.read_text(encoding="utf-8"))
    source_metrics = actual["candidates"]["original"]["adaptive"]
    for key, value in expected["source_order_scoring"].items():
        assert math.isclose(source_metrics[key], value, rel_tol=1e-12, abs_tol=1e-12), (
            f"source_order_scoring.{key}", source_metrics[key], value
        )
    selected_name = actual["selected"]
    selected = actual["candidates"][selected_name]
    metrics = selected["adaptive"]
    assert selected_name == expected["selected"]
    assert selected["order"] == expected["selected_order"]
    assert metrics["repeat_violations"] == expected["repeat_violations"]
    for key in ("objective", "transition_sum", "worst_transition"):
        assert math.isclose(metrics[key], expected[key], rel_tol=1e-12, abs_tol=1e-12), (
            key, metrics[key], expected[key]
        )
    print("Python oracle matches expected synthetic parity result.")


if __name__ == "__main__":
    main()
