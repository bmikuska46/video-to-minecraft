from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from app.worker_diagnostics import startup_self_test, write_health


def test_worker_self_test_reports_tool_and_cuda_versions() -> None:
    def version(command: list[str]) -> str:
        if command == ["nvidia-smi"]:
            return "NVIDIA-SMI 570.0 Driver Version: 570.0 CUDA Version: 12.9"
        if command[0] == "nvidia-smi":
            return "NVIDIA RTX fixture, 570.0"
        if command[0] == "colmap":
            return "COLMAP 4.0.4"
        return "ffmpeg version 7.1"

    with patch("app.worker_diagnostics.shutil.which", return_value="/usr/bin/tool"):
        result = startup_self_test(version_runner=version)

    assert result == {
        "colmapVersion": "COLMAP 4.0.4",
        "cudaVersion": "12.9",
        "gpu": "NVIDIA RTX fixture, 570.0",
        "ffmpegVersion": "ffmpeg version 7.1",
        "selfTest": "passed",
    }


def test_export_worker_self_test_needs_java_but_no_gpu() -> None:
    with patch("app.worker_diagnostics.shutil.which",
               side_effect=lambda name: "/usr/bin/java" if name == "java" else None):
        result = startup_self_test(version_runner=lambda command: "openjdk 25", gpu=False)
    assert result == {"javaVersion": "openjdk 25", "selfTest": "passed"}


def test_worker_health_file_is_atomically_parseable() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "worker-health.json"
        write_health(path, ready=True, diagnostics={"selfTest": "passed"})
        payload = json.loads(path.read_text())
        assert payload["ready"] is True
        assert payload["diagnostics"] == {"selfTest": "passed"}
        assert not path.with_suffix(".json.tmp").exists()
