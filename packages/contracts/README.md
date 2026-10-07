# Capture package contract

Capture objects use server-assigned, create-only keys below. Original device
filenames are never included in storage paths or subprocess arguments.

```text
scans/{scanId}/source/video.mp4
scans/{scanId}/source/capture.json
scans/{scanId}/source/imu.jsonl.zst
scans/{scanId}/source/ar-frames.pb.zst
scans/{scanId}/source/depth/             # reserved for a later milestone
```

`video.mp4` and `capture.json` are required. The IMU and AR frame streams are
optional. Every upload is signed with `If-None-Match: *`, its exact byte length,
content type, and SHA-256 metadata, so an existing capture object cannot be
replaced through the API.

[`capture.schema.json`](capture.schema.json) is the version 1 JSON contract.
Timestamps in IMU JSONL and optional AR protobuf records are relative to
`monotonicRecordingStartNs`; `wallClockTimestamp` exists for audit/display and
must not be used to synchronize samples.

## Voxel handoff

[`voxel_world.proto`](voxel_world.proto) is the version 1 Python-to-Java world
generation contract. The reconstruction worker serializes deterministic
Protobuf, with palette entries sorted by ID and voxels sorted by chunk X,
chunk Z, vertical section, Y, Z, then X. The bytes are compressed as a Zstandard
frame with a content checksum and stored as `voxels.pb.zst`.

Coordinates use Protobuf `sint32` so negative Minecraft positions remain compact.
Bounds are inclusive occupied-cell bounds. Consumers must reject unsupported
schema versions, missing palette references, and non-canonical voxel order.
