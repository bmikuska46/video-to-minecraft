#!/usr/bin/env python3
"""Convert an observed colored PLY point cloud to a compact GLB point primitive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from point_cloud import read_ply


def _pad(payload: bytes, byte: bytes) -> bytes:
    return payload + byte * ((-len(payload)) % 4)


def convert(source: Path, destination: Path) -> None:
    points = read_ply(source)
    names = set(points.dtype.names or ())
    if not {"x", "y", "z"} <= names or not {"red", "green", "blue"} <= names:
        raise ValueError("preview PLY must contain positions and RGB colors")
    positions = np.column_stack([points[name] for name in ("x", "y", "z")]).astype("<f4")
    colors = np.column_stack([points[name] for name in ("red", "green", "blue")]).astype("u1")
    if not len(positions) or not np.isfinite(positions).all():
        raise ValueError("preview PLY contains no finite points")
    position_bytes = positions.tobytes()
    color_offset = len(position_bytes)
    binary = _pad(position_bytes + colors.tobytes(), b"\0")
    document = {
        "asset": {"version": "2.0", "generator": "video-to-minecraft-preview"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}]}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": color_offset, "byteLength": len(colors.tobytes()), "target": 34962},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5121, "count": len(colors), "type": "VEC3", "normalized": True},
        ],
    }
    json_chunk = _pad(json.dumps(document, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(struct.pack("<4sII", b"glTF", 2, total))
        output.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        output.write(json_chunk)
        output.write(struct.pack("<I4s", len(binary), b"BIN\0"))
        output.write(binary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
