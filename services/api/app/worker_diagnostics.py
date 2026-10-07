"""GPU worker startup diagnostics and a deliberately tiny local self-test."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable


class WorkerSelfTestError(RuntimeError):
    pass


def _run_version(command: list[str]) -> str:
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
        check=False,
    )
    if result.returncode != 0:
        raise WorkerSelfTestError(f"{command[0]} version check failed")
    return " ".join((result.stdout or "").split())[:500]


def startup_self_test(
    *,
    version_runner: Callable[[list[str]], str] = _run_version,
    gpu: bool = True,
) -> dict[str, Any]:
    """``gpu=False`` is the export worker, which needs Java but no GPU or COLMAP."""
    if not gpu:
        java = shutil.which("java")
        if java is None:
            raise WorkerSelfTestError("required worker executable is missing: java")
        return {"javaVersion": version_runner([java, "-version"]), "selfTest": "passed"}
    for executable in ("colmap", "ffmpeg", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise WorkerSelfTestError(f"required worker executable is missing: {executable}")

    # Exercise the same numerical and scratch-file basics used before COLMAP starts,
    # while keeping startup independent of a fixture video.
    import numpy as np

    transformed = np.eye(4) @ np.array([1.0, 2.0, 3.0, 1.0])
    if not np.allclose(transformed, [1.0, 2.0, 3.0, 1.0]):
        raise WorkerSelfTestError("NumPy transform self-test failed")
    with tempfile.TemporaryDirectory(prefix="vtm-worker-self-test-") as temporary:
        marker = Path(temporary) / "scratch-ok"
        marker.write_bytes(b"ok")
        if marker.read_bytes() != b"ok":
            raise WorkerSelfTestError("worker scratch self-test failed")

    nvidia_smi = version_runner(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi, re.IGNORECASE)
    if cuda_match is None:
        raise WorkerSelfTestError("nvidia-smi did not report a CUDA version")
    return {
        "colmapVersion": version_runner(["colmap", "-h"]),
        "cudaVersion": cuda_match.group(1),
        "gpu": version_runner(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
        ),
        "ffmpegVersion": version_runner(["ffmpeg", "-version"]).splitlines()[0],
        "selfTest": "passed",
    }


def write_health(path: Path, *, ready: bool, diagnostics: dict[str, Any] | None = None,
                 failure: str | None = None) -> None:
    payload = {
        "ready": ready,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "diagnostics": diagnostics or {},
        "failure": failure,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, path)
