#!/usr/bin/env python3
"""Write a vanilla Minecraft Java 26.2 singleplayer save directly from voxels.pb.zst.

This replaces the two Paper server starts (generate, then validate) with plain
file writing. The save it produces is what the Paper path produces after
``worldgen_runner.convert_to_singleplayer``:

- ``dimensions/minecraft/overworld/region/r.X.Z.mca``: Anvil region files that
  hold only the chunks with platform or model blocks. Every other chunk is
  missing, so the game generates it from ``world_gen_settings.dat`` (a flat
  world with no layers, i.e. empty air) when a player gets near it.
- ``level.dat`` and the ``data/`` files from ``world-template/``, captured from a
  converted Paper 26.2 save (``capture-template``), with the spawn, world name
  and last-played time filled in.

Chunks use the same layout Paper 26.2 writes (``DataVersion`` 4903, status
``minecraft:full``, 24 sections from Y -4 to 19 with block-state and biome
palettes, four heightmaps) and are stored with ``isLightOn`` 0 and no light
arrays. Paper writes ``isLightOn`` 0 as well (it keeps its own Starlight data,
which vanilla ignores), so in both cases the game computes sky and block light
itself when it first loads a chunk.

``validate_world`` reads the region files back with a decoder that shares no
code with the writer and checks every expected platform and model block, that
nothing else is placed, the spawn and the world defaults, as the Paper
validation run did.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import time
import zlib

import numpy as np

import level_nbt
from level_nbt import BYTE, COMPOUND, END, INT, LIST, LONG, LONG_ARRAY, STRING, Tag


# From the Paper 26.2 build 112 save this replaces (chunk and level.dat DataVersion).
DATA_VERSION = 4903
MINECRAFT_VERSION = "26.2"
# Overworld build limits: 24 sections of 16 blocks from Y -64 to 319.
MIN_SECTION_Y = -4
MAX_SECTION_Y = 19
MIN_Y = MIN_SECTION_Y * 16
MAX_Y = MAX_SECTION_Y * 16 + 15
AIR = "minecraft:air"
BIOME = "minecraft:plains"
HEIGHTMAP_TYPES = ("MOTION_BLOCKING", "MOTION_BLOCKING_NO_LEAVES", "OCEAN_FLOOR", "WORLD_SURFACE")
# Heightmap entries hold 0..384, so 9 bits each, 7 per long, 37 longs per chunk.
HEIGHTMAP_BITS = 9
TEMPLATE = Path(__file__).resolve().with_name("world-template")
REGION_DIRECTORY = Path("dimensions") / "minecraft" / "overworld" / "region"
# The heightmaps count every placed block as solid and motion-blocking, which
# holds for the palette's full cubes but not for leaves, fluids or air variants.
NON_SOLID_MARKERS = ("leaves", "water", "lava", "bubble_column", "_air")


class WorldWriteError(RuntimeError):
    """The voxels cannot be written, or a written save failed validation."""


def layout(bounds, payload: dict) -> dict[str, int]:
    """Same coordinates as WorldLayout.create in the Paper plugin."""
    width = bounds.max_x - bounds.min_x + 1
    depth = bounds.max_z - bounds.min_z + 1
    margin = int(payload["platformMargin"])
    platform_y = int(payload["platformY"])
    return {
        "shiftX": -bounds.min_x,
        "shiftY": platform_y + 1 - bounds.min_y,
        "shiftZ": -bounds.min_z,
        "platformMinX": -margin,
        "platformMaxX": width - 1 + margin,
        "platformMinZ": -margin,
        "platformMaxZ": depth - 1 + margin,
        "platformY": platform_y,
        "spawnX": (width - 1) // 2,
        "spawnY": platform_y + 1,
        "spawnZ": -margin + max(1, margin // 2),
    }


def expected_blocks(world, payload: dict) -> tuple[dict[str, int], np.ndarray, list[str]]:
    """World layout, an (N, 4) array of x, y, z and block-state index, and the state names.

    Index 0 is air, 1 the platform block and 2.. the voxel palette in ID order.
    The platform comes first, so the array holds every placed block exactly once.
    """
    place = layout(world.bounds, payload)
    states = [AIR, payload["platformBlockState"]]
    index_of_id = {}
    for entry in sorted(world.palette, key=lambda entry: entry.id):
        if entry.block_state not in states:
            states.append(entry.block_state)
        index_of_id[entry.id] = states.index(entry.block_state)
    count = len(world.voxels)
    voxels = np.empty((count, 4), np.int64)
    for row, voxel in enumerate(world.voxels):
        voxels[row] = (voxel.x, voxel.y, voxel.z, voxel.palette_id)
    voxels[:, 0] += place["shiftX"]
    voxels[:, 1] += place["shiftY"]
    voxels[:, 2] += place["shiftZ"]
    lookup = np.zeros(max(index_of_id) + 1, np.int64)
    for identifier, index in index_of_id.items():
        lookup[identifier] = index
    voxels[:, 3] = lookup[voxels[:, 3]]
    xs = np.arange(place["platformMinX"], place["platformMaxX"] + 1)
    zs = np.arange(place["platformMinZ"], place["platformMaxZ"] + 1)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    platform = np.stack([grid_x.ravel(), np.full(grid_x.size, place["platformY"]), grid_z.ravel(),
                         np.ones(grid_x.size, np.int64)], 1)
    return place, np.concatenate([platform, voxels]), states


def palette_tag(name: str) -> Tag:
    """Block-state NBT; ``minecraft:x[k=v,...]`` becomes Name plus Properties."""
    entry = {}
    if "[" in name:
        base, properties = name[:-1].split("[", 1)
        entry["Name"] = Tag(STRING, base)
        entry["Properties"] = Tag(COMPOUND, {
            key: Tag(STRING, value)
            for key, value in sorted(part.split("=", 1) for part in properties.split(","))
        })
    else:
        entry["Name"] = Tag(STRING, name)
    return Tag(COMPOUND, entry)


def pack(values: np.ndarray, bits: int) -> list[int]:
    """Pack unsigned values into signed longs, ``64 // bits`` per long, none spanning two."""
    per_long = 64 // bits
    longs = -(-len(values) // per_long)
    padded = np.zeros(longs * per_long, np.uint64)
    padded[:len(values)] = values
    shifts = (np.arange(per_long, dtype=np.uint64) * np.uint64(bits))
    packed = np.bitwise_or.reduce(padded.reshape(longs, per_long) << shifts, axis=1)
    return packed.view(np.int64).tolist()


def section_tag(section_y: int, indices: np.ndarray | None, states: list[str]) -> Tag:
    """One 16x16x16 section; ``indices`` holds 4096 state indices in YZX order, or None for air."""
    if indices is None:
        local = [0]
        data = None
    else:
        local, inverse = np.unique(indices, return_inverse=True)
        local = local.tolist()
        data = None
        if len(local) > 1:
            data = pack(inverse.astype(np.uint64), max(4, math.ceil(math.log2(len(local)))))
    block_states = {"palette": Tag(LIST, [palette_tag(states[index]) for index in local], COMPOUND)}
    if data is not None:
        block_states["data"] = Tag(LONG_ARRAY, data)
    return Tag(COMPOUND, {
        "Y": Tag(BYTE, section_y),
        "block_states": Tag(COMPOUND, block_states),
        "biomes": Tag(COMPOUND, {"palette": Tag(LIST, [Tag(STRING, BIOME)], STRING)}),
    })


def chunk_tag(chunk_x: int, chunk_z: int, sections: dict[int, np.ndarray], states: list[str],
              last_update: int) -> Tag:
    # Heightmaps: per column (index z * 16 + x) the top block's Y + 1 - MIN_Y, 0 if empty.
    tops = np.full(256, MIN_Y - 1, np.int64)
    for section_y, indices in sections.items():
        filled = indices.reshape(16, 256) != 0
        highest = section_y * 16 + 15 - np.argmax(filled[::-1], axis=0)
        tops = np.maximum(tops, np.where(filled.any(axis=0), highest, MIN_Y - 1))
    heights = pack(np.where(tops >= MIN_Y, tops + 1 - MIN_Y, 0).astype(np.uint64), HEIGHTMAP_BITS)
    return Tag(COMPOUND, {
        "DataVersion": Tag(INT, DATA_VERSION),
        "xPos": Tag(INT, chunk_x),
        "yPos": Tag(INT, MIN_SECTION_Y),
        "zPos": Tag(INT, chunk_z),
        "Status": Tag(STRING, "minecraft:full"),
        "LastUpdate": Tag(LONG, last_update),
        "InhabitedTime": Tag(LONG, 0),
        # The game lights the chunk itself when it first loads it, as for Paper saves.
        "isLightOn": Tag(BYTE, 0),
        "sections": Tag(LIST, [section_tag(y, sections.get(y), states)
                               for y in range(MIN_SECTION_Y, MAX_SECTION_Y + 1)], COMPOUND),
        "Heightmaps": Tag(COMPOUND, {name: Tag(LONG_ARRAY, list(heights)) for name in HEIGHTMAP_TYPES}),
        "block_entities": Tag(LIST, [], END),
        "block_ticks": Tag(LIST, [], END),
        "fluid_ticks": Tag(LIST, [], END),
        "PostProcessing": Tag(LIST, [Tag(LIST, [], END) for _ in range(MAX_SECTION_Y - MIN_SECTION_Y + 1)], LIST),
        "structures": Tag(COMPOUND, {"References": Tag(COMPOUND, {}), "starts": Tag(COMPOUND, {})}),
    })


def write_region(path: Path, chunks: dict[tuple[int, int], bytes], timestamp: int) -> None:
    """Anvil region: 4 KiB location table, 4 KiB timestamps, then zlib chunks in 4 KiB sectors."""
    locations = bytearray(4096)
    timestamps = bytearray(4096)
    body = bytearray()
    sector = 2
    for (chunk_x, chunk_z), payload in sorted(chunks.items(), key=lambda item: (item[0][1], item[0][0])):
        compressed = zlib.compress(payload, 6)
        record = struct.pack(">IB", len(compressed) + 1, 2) + compressed
        sectors = -(-len(record) // 4096)
        if sectors > 255:
            raise WorldWriteError(f"chunk {chunk_x},{chunk_z} is too large for a region sector run")
        index = (chunk_x & 31) + (chunk_z & 31) * 32
        locations[index * 4:index * 4 + 4] = struct.pack(">I", (sector << 8) | sectors)
        timestamps[index * 4:index * 4 + 4] = struct.pack(">I", timestamp)
        body += record + bytes(sectors * 4096 - len(record))
        sector += sectors
    path.write_bytes(bytes(locations) + bytes(timestamps) + bytes(body))


def write_regions(world_dir: Path, blocks: np.ndarray, states: list[str], timestamp: int) -> dict[str, int]:
    """Group blocks into chunks and sections and write one region file per 32x32 chunks."""
    if blocks[:, 1].min() < MIN_Y or blocks[:, 1].max() > MAX_Y:
        raise WorldWriteError("translated model is outside the target world's build height")
    x, y, z, state = blocks.T
    chunk_x, chunk_z, section_y = x >> 4, z >> 4, y >> 4
    local = ((y & 15) << 8) | ((z & 15) << 4) | (x & 15)
    order = np.lexsort((section_y, chunk_z, chunk_x))
    keys = np.stack([chunk_x[order], chunk_z[order], section_y[order]], 1)
    starts = np.flatnonzero(np.r_[True, (keys[1:] != keys[:-1]).any(1)])
    ends = np.r_[starts[1:], len(order)]
    chunks: dict[tuple[int, int], dict[int, np.ndarray]] = {}
    for start, end in zip(starts, ends):
        cx, cz, sy = (int(value) for value in keys[start])
        indices = np.zeros(4096, np.int64)
        members = order[start:end]
        indices[local[members]] = state[members]
        chunks.setdefault((cx, cz), {})[sy] = indices
    regions: dict[tuple[int, int], dict[tuple[int, int], bytes]] = {}
    for (cx, cz), sections in chunks.items():
        tag = chunk_tag(cx, cz, sections, states, last_update=0)
        regions.setdefault((cx >> 5, cz >> 5), {})[(cx, cz)] = level_nbt.dumps("", tag, compress=False)
    region_dir = world_dir / REGION_DIRECTORY
    region_dir.mkdir(parents=True, exist_ok=True)
    for (rx, rz), region_chunks in regions.items():
        write_region(region_dir / f"r.{rx}.{rz}.mca", region_chunks, timestamp)
    return {"chunks": len(chunks), "sections": len(starts), "regions": len(regions)}


def write_level_data(world_dir: Path, place: dict[str, int], display_name: str, last_played_ms: int) -> None:
    """Copy the captured level-wide files and fill in spawn, name and time in level.dat."""
    if not (TEMPLATE / "level.dat").is_file():
        raise WorldWriteError(f"world template is missing: {TEMPLATE}")
    for item in sorted(TEMPLATE.rglob("*")):
        if item.is_file() and item.name != "README.md":
            target = world_dir / item.relative_to(TEMPLATE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
    name, root = level_nbt.load(world_dir / "level.dat")
    data = root.value["Data"].value
    if data["DataVersion"].value != DATA_VERSION:
        raise WorldWriteError("world template does not match the writer's DataVersion")
    # Assign values only, so every tag keeps the type the game wrote.
    data["spawn"].value["pos"].value = [place["spawnX"], place["spawnY"], place["spawnZ"]]
    data["LevelName"].value = display_name
    data["LastPlayed"].value = last_played_ms
    level_nbt.save(world_dir / "level.dat", name, root)
    shutil.copy2(world_dir / "level.dat", world_dir / "level.dat_old")


def decode_voxels(voxel_path: Path):
    """Decode voxels.pb.zst with the reconstruction service's contract module.

    The worker images put services/reconstruction on PYTHONPATH; in a checkout it
    sits next to this service.
    """
    try:
        from voxel_contract import decode
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reconstruction"))
        from voxel_contract import decode
    return decode(voxel_path.read_bytes())


def write_world(world_dir: Path, voxel_path: Path, payload: dict, display_name: str,
                *, now: float | None = None) -> dict[str, object]:
    """Write the save and return counts plus the expected blocks for ``validate_world``."""
    world = decode_voxels(voxel_path)
    if world.minecraft_version != MINECRAFT_VERSION or payload["minecraftVersion"] != MINECRAFT_VERSION:
        raise WorldWriteError("voxels or manifest do not target Minecraft 26.2")
    if len(world.voxels) > int(payload["maxBlockCount"]):
        raise WorldWriteError("voxel artifact exceeds maxBlockCount")
    place, blocks, states = expected_blocks(world, payload)
    platform_blocks = (place["platformMaxX"] - place["platformMinX"] + 1) * (
        place["platformMaxZ"] - place["platformMinZ"] + 1)
    if platform_blocks > int(payload["maxPlatformBlocks"]):
        raise WorldWriteError("platform exceeds maxPlatformBlocks")
    unsupported = [state for state in states[1:] if any(marker in state for marker in NON_SOLID_MARKERS)]
    if unsupported:
        raise WorldWriteError(f"direct writer only places full solid blocks, not {', '.join(unsupported)}")
    unique_positions = np.unique(blocks[:, :3], axis=0)
    if len(unique_positions) != len(blocks):
        raise WorldWriteError("voxel artifact contains duplicate or platform-overlapping coordinates")
    now = time.time() if now is None else now
    world_dir.mkdir(parents=True, exist_ok=True)
    counts = write_regions(world_dir, blocks, states, int(now))
    write_level_data(world_dir, place, display_name, int(now * 1000))
    return {"layout": place, "blocks": blocks, "states": states, "platformBlocks": platform_blocks,
            "modelBlocks": len(world.voxels), **counts}


# --- Independent read-back --------------------------------------------------
# Nothing below reuses the writer: chunks are parsed as generic NBT and block
# states are unpacked from the long arrays bit by bit, as the game reads them.


def read_region_chunks(path: Path):
    """Yield (chunk x, chunk z, root NBT tag) for every chunk stored in a region file."""
    data = path.read_bytes()
    _, region_x, region_z, _ = path.name.split(".")
    region_x, region_z = int(region_x), int(region_z)
    for index in range(1024):
        entry = int.from_bytes(data[index * 4:index * 4 + 4], "big")
        offset, sectors = entry >> 8, entry & 0xFF
        if not offset:
            continue
        start = offset * 4096
        length, compression = struct.unpack(">IB", data[start:start + 5])
        if length + 4 > sectors * 4096 or compression != 2:
            raise WorldWriteError(f"{path.name}: chunk {index} has an invalid sector record")
        _, root = level_nbt.loads(zlib.decompress(data[start + 5:start + 4 + length]))
        chunk_x, chunk_z = region_x * 32 + index % 32, region_z * 32 + index // 32
        yield chunk_x, chunk_z, root


def unpack_block_states(block_states: Tag) -> list[str]:
    """The 4096 block-state names of one section in YZX order."""
    palette = []
    for entry in block_states.value["palette"].value:
        name = entry.value["Name"].value
        properties = entry.value.get("Properties")
        if properties is not None:
            name += "[" + ",".join(f"{key}={value.value}" for key, value in sorted(properties.value.items())) + "]"
        palette.append(name)
    if "data" not in block_states.value:
        if len(palette) != 1:
            raise WorldWriteError("a section without data must have exactly one palette entry")
        return [palette[0]] * 4096
    longs = block_states.value["data"].value
    bits = max(4, (len(palette) - 1).bit_length())
    per_long = 64 // bits
    if len(longs) != -(-4096 // per_long):
        raise WorldWriteError("section data length does not match its palette size")
    mask = (1 << bits) - 1
    names = []
    for index in range(4096):
        value = (longs[index // per_long] & 0xFFFFFFFFFFFFFFFF) >> ((index % per_long) * bits) & mask
        if value >= len(palette):
            raise WorldWriteError("section data refers past its palette")
        names.append(palette[value])
    return names


def read_world_blocks(world_dir: Path) -> tuple[dict[tuple[int, int, int], str], dict[tuple[int, int], dict]]:
    """Every non-air block of a save by position, plus per-chunk heightmaps and flags."""
    blocks: dict[tuple[int, int, int], str] = {}
    chunks: dict[tuple[int, int], dict] = {}
    for region in sorted((world_dir / REGION_DIRECTORY).glob("r.*.*.mca")):
        for chunk_x, chunk_z, root in read_region_chunks(region):
            values = root.value
            if values["xPos"].value != chunk_x or values["zPos"].value != chunk_z:
                raise WorldWriteError(f"chunk {chunk_x},{chunk_z} is stored at the wrong position")
            chunks[(chunk_x, chunk_z)] = {
                "DataVersion": values["DataVersion"].value, "Status": values["Status"].value,
                # Absent in chunks a server saved before they reached full status.
                "isLightOn": values["isLightOn"].value if "isLightOn" in values else None,
                "Heightmaps": {name: tag.value for name, tag in values["Heightmaps"].value.items()}
                if "Heightmaps" in values else {},
            }
            for section in values["sections"].value:
                if "block_states" not in section.value:
                    continue
                base_y = section.value["Y"].value * 16
                names = unpack_block_states(section.value["block_states"])
                for index, name in enumerate(names):
                    if name != AIR:
                        x = chunk_x * 16 + index % 16
                        z = chunk_z * 16 + (index // 16) % 16
                        blocks[(x, base_y + index // 256, z)] = name
    return blocks, chunks


def validate_world(world_dir: Path, written: dict[str, object]) -> dict[str, int]:
    """Check a written save block for block against the voxels it was written from."""
    blocks, chunks = read_world_blocks(world_dir)
    states: list[str] = written["states"]  # type: ignore[assignment]
    expected_array: np.ndarray = written["blocks"]  # type: ignore[assignment]
    expected = {(int(x), int(y), int(z)): states[int(index)] for x, y, z, index in expected_array}
    for position, state in expected.items():
        actual = blocks.get(position)
        if actual != state:
            raise WorldWriteError(f"block validation failed at {position}: expected {state}, found {actual or AIR}")
    unexpected = len(blocks) - len(expected)
    if unexpected:
        raise WorldWriteError(f"the save holds {unexpected} blocks that are not in the voxels or platform")
    for position, info in chunks.items():
        if info["DataVersion"] != DATA_VERSION or info["Status"] != "minecraft:full":
            raise WorldWriteError(f"chunk {position} has an unexpected DataVersion or status")
    place: dict[str, int] = written["layout"]  # type: ignore[assignment]
    _, root = level_nbt.load(world_dir / "level.dat")
    data = root.value["Data"].value
    spawn = data["spawn"].value["pos"].value
    if list(spawn) != [place["spawnX"], place["spawnY"], place["spawnZ"]]:
        raise WorldWriteError("level.dat spawn does not match the export contract")
    if (data["GameType"].value != 1 or data["allowCommands"].value != 1
            or data["difficulty_settings"].value["difficulty"].value != "peaceful"
            or data["DataVersion"].value != DATA_VERSION):
        raise WorldWriteError("level.dat defaults do not match the export contract")
    _, rules = level_nbt.load(world_dir / "data" / "minecraft" / "game_rules.dat")
    rule_values = rules.value["data"].value
    for rule in ("minecraft:advance_time", "minecraft:advance_weather", "minecraft:spawn_mobs"):
        if rule_values[rule].value != 0:
            raise WorldWriteError(f"game rule {rule} does not match the export contract")
    below = (place["platformMinX"], place["platformY"] - 1, place["platformMinZ"])
    beside = (place["platformMinX"] - 1, place["platformY"], place["platformMinZ"])
    if below in blocks or beside in blocks:
        raise WorldWriteError("the save contains blocks outside or below the explicit platform")
    return {"checkedBlocks": len(expected), "chunks": len(chunks)}


def capture_template(world_zip: Path, destination: Path) -> None:
    """Store the non-region files of a converted Paper save (world.zip) as the writer's template."""
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(world_zip) as archive:
        for name in archive.namelist():
            if name.endswith(".dat") and "/region/" not in name and name != "level.dat_old":
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    _, root = level_nbt.load(destination / "level.dat")
    data = root.value["Data"].value
    data["LevelName"].value = "minecraft-video-world"
    data["LastPlayed"].value = 0
    level_nbt.save(destination / "level.dat", "", root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture-template", help="refresh world-template/ from a Paper world.zip")
    capture.add_argument("world_zip", type=Path)
    capture.add_argument("--destination", type=Path, default=TEMPLATE)
    dump = commands.add_parser("blocks", help="print a save's non-air blocks and chunk metadata as JSON")
    dump.add_argument("world_dir", type=Path)
    args = parser.parse_args()
    if args.command == "capture-template":
        capture_template(args.world_zip, args.destination)
    else:
        blocks, chunks = read_world_blocks(args.world_dir)
        print(json.dumps({
            "blocks": {f"{x},{y},{z}": state for (x, y, z), state in sorted(blocks.items())},
            "distribution": dict(Counter(blocks.values()).most_common()),
            "chunks": {f"{x},{z}": info for (x, z), info in sorted(chunks.items())},
        }, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
