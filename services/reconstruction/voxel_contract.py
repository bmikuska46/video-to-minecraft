"""Deterministic Protobuf + Zstandard handoff to the Java world generator."""

from __future__ import annotations

from pathlib import Path

import zstandard

from generated import voxel_world_pb2
from voxelizer import Palette, VoxelizationResult


SCHEMA_VERSION = 1
ZSTD_LEVEL = 10


class VoxelContractError(ValueError):
    """A voxel result cannot be represented by the stable handoff contract."""


def voxel_sort_key(voxel: object) -> tuple[int, ...]:
    """Order by chunk, vertical section, Y, then stable in-section coordinates."""
    x, y, z = int(voxel.x), int(voxel.y), int(voxel.z)
    return (x // 16, z // 16, y // 16, y, z, x)


def to_message(result: VoxelizationResult, palette: Palette):
    if not result.voxels:
        raise VoxelContractError("voxel world must contain at least one voxel")
    entries = sorted(palette.entries, key=lambda entry: entry.id)
    palette_ids = {entry.id for entry in entries}
    if len(palette_ids) != len(entries) or 0 in palette_ids:
        raise VoxelContractError("palette IDs must be non-zero and unique")
    unknown = {voxel.palette_id for voxel in result.voxels} - palette_ids
    if unknown:
        raise VoxelContractError(f"voxels reference unknown palette IDs: {sorted(unknown)}")

    ordered = sorted(result.voxels, key=voxel_sort_key)
    message = voxel_world_pb2.VoxelWorld(
        schema_version=SCHEMA_VERSION,
        minecraft_version=palette.minecraft_version,
    )
    message.palette.extend(
        voxel_world_pb2.PaletteEntry(id=entry.id, block_state=entry.block_state)
        for entry in entries
    )
    message.voxels.extend(
        voxel_world_pb2.Voxel(x=voxel.x, y=voxel.y, z=voxel.z,
                              palette_id=voxel.palette_id)
        for voxel in ordered
    )
    xs, ys, zs = ([int(getattr(voxel, axis)) for voxel in ordered]
                  for axis in ("x", "y", "z"))
    message.bounds.CopyFrom(voxel_world_pb2.Bounds(
        min_x=min(xs), min_y=min(ys), min_z=min(zs),
        max_x=max(xs), max_y=max(ys), max_z=max(zs),
    ))
    return message


def encode(result: VoxelizationResult, palette: Palette) -> bytes:
    protobuf = to_message(result, palette).SerializeToString(deterministic=True)
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_checksum=True).compress(protobuf)


def decode(payload: bytes):
    try:
        protobuf = zstandard.ZstdDecompressor().decompress(payload)
        message = voxel_world_pb2.VoxelWorld.FromString(protobuf)
    except (zstandard.ZstdError, ValueError) as error:
        raise VoxelContractError("invalid voxels.pb.zst payload") from error
    if message.schema_version != SCHEMA_VERSION:
        raise VoxelContractError(f"unsupported voxel schema version: {message.schema_version}")
    if not message.minecraft_version or not message.voxels or not message.HasField("bounds"):
        raise VoxelContractError("voxel contract is missing required semantic fields")
    palette_ids = {entry.id for entry in message.palette}
    if len(palette_ids) != len(message.palette) or 0 in palette_ids:
        raise VoxelContractError("voxel contract contains an invalid palette")
    if any(voxel.palette_id not in palette_ids for voxel in message.voxels):
        raise VoxelContractError("voxel contract references an unknown palette ID")
    if list(message.voxels) != sorted(message.voxels, key=voxel_sort_key):
        raise VoxelContractError("voxels are not in canonical chunk/section order")
    return message


def write(path: Path, result: VoxelizationResult, palette: Palette) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(result, palette))
