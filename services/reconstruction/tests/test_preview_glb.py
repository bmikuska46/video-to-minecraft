from __future__ import annotations

import json
from pathlib import Path
import struct

from preview_glb import convert


ROOT = Path(__file__).resolve().parents[3]


def test_observed_ply_is_encoded_as_colored_glb_points(tmp_path: Path):
    destination = tmp_path / "preview.glb"
    convert(ROOT / "fixtures/point-clouds/wall-with-window/cloud.ply", destination)
    payload = destination.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload)
    assert magic == b"glTF"
    assert version == 2
    assert total == len(payload)
    json_length, chunk_type = struct.unpack_from("<I4s", payload, 12)
    assert chunk_type == b"JSON"
    document = json.loads(payload[20:20 + json_length])
    primitive = document["meshes"][0]["primitives"][0]
    assert primitive["mode"] == 0
    assert primitive["attributes"] == {"POSITION": 0, "COLOR_0": 1}
    assert document["accessors"][0]["count"] == 2212
    assert document["accessors"][1]["normalized"] is True
