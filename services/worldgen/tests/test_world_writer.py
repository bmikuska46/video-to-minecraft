from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "services" / "worldgen"))
sys.path.insert(0, str(ROOT / "services" / "reconstruction"))
import level_nbt  # noqa: E402
import world_writer  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "paper-26.2-small-world.json.gz"


def payload(**overrides) -> dict:
    values = {
        "schemaVersion": 1, "jobId": "550e8400-e29b-41d4-a716-446655440000", "minecraftVersion": "26.2",
        "voxelSha256": "a" * 64, "worldName": "minecraft-video-world", "platformY": 64,
        "platformMargin": 8, "platformBlockState": "minecraft:smooth_stone", "batchSize": 20_000,
        "maxBlockCount": 1_000_000, "maxPlatformBlocks": 4_000_000,
    }
    values.update(overrides)
    return values


def fixture_voxels(seed: int = 26) -> list[tuple[int, int, int, int]]:
    """A small model crossing chunk and section borders, with negative coordinates and 20 block types.

    The Paper golden (GOLDEN) was generated from exactly these voxels; see
    PaperEquivalenceTests.
    """
    rng = np.random.default_rng(seed)
    cells = {}
    # A hollow box, a solid column through several sections and scattered blocks.
    for x in range(-20, 21):
        for z in range(-6, 30):
            for y in (-3, 9):
                cells[(x, y, z)] = int(rng.integers(1, 21))
    for y in range(-3, 40):
        cells[(15, y, 15)] = 1 + y % 20
    for _ in range(300):
        cells[tuple(int(v) for v in rng.integers([-20, -3, -6], [21, 40, 30]))] = int(rng.integers(1, 21))
    return sorted((x, y, z, palette_id) for (x, y, z), palette_id in cells.items())


def write_voxels(path: Path, voxels: list[tuple[int, int, int, int]], palette_size: int = 20) -> None:
    import zstandard
    from generated import voxel_world_pb2
    from voxel_contract import voxel_sort_key

    states = [entry["blockState"] for entry in
              json.loads((ROOT / "packages" / "block-palette" / "palette-v1.json").read_text())["entries"]]
    message = voxel_world_pb2.VoxelWorld(schema_version=1, minecraft_version="26.2")
    message.palette.extend(voxel_world_pb2.PaletteEntry(id=index + 1, block_state=states[index])
                           for index in range(palette_size))
    ordered = sorted((voxel_world_pb2.Voxel(x=x, y=y, z=z, palette_id=p) for x, y, z, p in voxels),
                     key=voxel_sort_key)
    message.voxels.extend(ordered)
    xs, ys, zs = zip(*((x, y, z) for x, y, z, _ in voxels))
    message.bounds.CopyFrom(voxel_world_pb2.Bounds(min_x=min(xs), min_y=min(ys), min_z=min(zs),
                                                   max_x=max(xs), max_y=max(ys), max_z=max(zs)))
    path.write_bytes(zstandard.ZstdCompressor(level=10, write_checksum=True).compress(
        message.SerializeToString(deterministic=True)))


class WorldWriterTests(unittest.TestCase):
    def write(self, root: Path, voxels, **payload_overrides):
        voxel_path = root / "voxels.pb.zst"
        write_voxels(voxel_path, voxels)
        world = root / "world"
        written = world_writer.write_world(world, voxel_path, payload(**payload_overrides), "Test world", now=1.0)
        return world, written

    def test_every_block_reads_back_at_its_layout_position(self):
        voxels = fixture_voxels()
        with tempfile.TemporaryDirectory() as temporary:
            world, written = self.write(Path(temporary), voxels)
            checked = world_writer.validate_world(world, written)
            blocks, chunks = world_writer.read_world_blocks(world)
            place = written["layout"]
            states = written["states"]
            # Same arithmetic as WorldLayout.java, written out independently.
            min_x, min_y, min_z = (min(v[i] for v in voxels) for i in range(3))
            for x, y, z, palette_id in voxels:
                position = (x - min_x, y - min_y + 65, z - min_z)
                self.assertEqual(blocks[position], states[1 + palette_id])
            width = max(v[0] for v in voxels) - min_x + 1
            depth = max(v[2] for v in voxels) - min_z + 1
            platform = [(x, 64, z) for x in range(-8, width + 8) for z in range(-8, depth + 8)]
            self.assertTrue(all(blocks[position] == "minecraft:smooth_stone" for position in platform))
            self.assertEqual(len(blocks), len(voxels) + len(platform))
            self.assertEqual(checked["checkedBlocks"], len(blocks))
            self.assertEqual((place["spawnX"], place["spawnY"], place["spawnZ"]), ((width - 1) // 2, 65, -4))
            self.assertTrue(all(info["isLightOn"] == 0 and info["Status"] == "minecraft:full"
                                and info["DataVersion"] == world_writer.DATA_VERSION for info in chunks.values()))

    def test_heightmaps_hold_the_top_block_of_every_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            world, _ = self.write(Path(temporary), [(0, 0, 0, 1), (0, 5, 0, 2), (3, 1, 2, 3)])
            blocks, chunks = world_writer.read_world_blocks(world)
            heights = chunks[(0, 0)]["Heightmaps"]["WORLD_SURFACE"]
            values = [(heights[i // 7] & 0xFFFFFFFFFFFFFFFF) >> (9 * (i % 7)) & 511 for i in range(256)]
            # Platform at 64 everywhere in this chunk, model columns at x=0,z=0 (top y 70) and x=3,z=2 (66).
            self.assertEqual(values[0 * 16 + 0], 70 + 1 + 64)
            self.assertEqual(values[2 * 16 + 3], 66 + 1 + 64)
            self.assertEqual(values[5 * 16 + 9], 64 + 1 + 64)
            self.assertEqual(set(chunks[(0, 0)]["Heightmaps"]), set(world_writer.HEIGHTMAP_TYPES))

    def test_section_palettes_pack_with_vanilla_bit_widths(self):
        for count in (1, 2, 16, 17, 33):
            values = np.arange(4096) % count
            bits = max(4, (count - 1).bit_length()) if count > 1 else 4
            longs = world_writer.pack(values.astype(np.uint64), bits)
            self.assertEqual(len(longs), -(-4096 // (64 // bits)))
            per_long = 64 // bits
            unpacked = [(longs[i // per_long] & 0xFFFFFFFFFFFFFFFF) >> (bits * (i % per_long)) & ((1 << bits) - 1)
                        for i in range(4096)]
            self.assertEqual(unpacked, values.tolist())

    def test_level_dat_carries_spawn_name_and_vanilla_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            world, written = self.write(Path(temporary), [(0, 0, 0, 1), (40, 3, 7, 2)])
            _, root = level_nbt.load(world / "level.dat")
            data = root.value["Data"].value
            self.assertEqual(data["LevelName"].value, "Test world")
            self.assertEqual(data["spawn"].value["pos"].value, [20, 65, -4])
            self.assertEqual(data["spawn"].value["pos"].type, level_nbt.INT_ARRAY)
            self.assertEqual(data["ServerBrands"].value[0].value, "vanilla")
            self.assertEqual(data["allowCommands"].value, 1)
            self.assertEqual(data["GameType"].value, 1)
            self.assertTrue((world / "data" / "minecraft" / "world_gen_settings.dat").is_file())
            self.assertTrue((world / "level.dat_old").is_file())
            self.assertFalse((world / "README.md").exists())

    def test_validation_catches_a_changed_block(self):
        with tempfile.TemporaryDirectory() as temporary:
            world, written = self.write(Path(temporary), [(0, 0, 0, 1), (1, 0, 0, 2)])
            region = next((world / world_writer.REGION_DIRECTORY).glob("*.mca"))
            chunk_x, chunk_z, root = next(world_writer.read_region_chunks(region))
            # Swap the platform block for stone in the stored chunk and write it back.
            for section in root.value["sections"].value:
                for entry in section.value["block_states"].value["palette"].value:
                    if entry.value["Name"].value == "minecraft:smooth_stone":
                        entry.value["Name"].value = "minecraft:stone"
            world_writer.write_region(region, {(chunk_x, chunk_z): level_nbt.dumps("", root, compress=False)}, 1)
            with self.assertRaisesRegex(world_writer.WorldWriteError, "block validation failed"):
                world_writer.validate_world(world, written)

    def test_refuses_models_outside_the_build_height_and_oversized_platforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(world_writer.WorldWriteError, "build height"):
                self.write(Path(temporary), [(0, 0, 0, 1), (0, 300, 0, 1)])
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(world_writer.WorldWriteError, "maxPlatformBlocks"):
                self.write(Path(temporary), [(0, 0, 0, 1), (100, 0, 0, 1)], maxPlatformBlocks=100)

    def test_block_states_with_properties_become_name_and_properties(self):
        tag = world_writer.palette_tag("minecraft:oak_log[axis=x]")
        self.assertEqual(tag.value["Name"].value, "minecraft:oak_log")
        self.assertEqual(tag.value["Properties"].value["axis"].value, "x")


class PaperEquivalenceTests(unittest.TestCase):
    """The direct writer must place exactly the blocks the Paper plugin placed for the same voxels."""

    @unittest.skipUnless(GOLDEN.is_file(), "Paper golden world fixture is missing")
    def test_matches_the_paper_generated_golden_world(self):
        golden = json.loads(gzip.decompress(GOLDEN.read_bytes()))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voxel_path = root / "voxels.pb.zst"
            write_voxels(voxel_path, fixture_voxels())
            written = world_writer.write_world(root / "world", voxel_path, payload(), "Test world", now=1.0)
            world_writer.validate_world(root / "world", written)
            blocks, chunks = world_writer.read_world_blocks(root / "world")
        self.assertEqual({f"{x},{y},{z}": state for (x, y, z), state in blocks.items()}, golden["blocks"])
        for key, info in chunks.items():
            self.assertEqual(info["Heightmaps"], golden["chunks"][f"{key[0]},{key[1]}"]["Heightmaps"])

    @unittest.skipUnless(os.environ.get("RUN_PAPER_EQUIVALENCE") == "1",
                         "set RUN_PAPER_EQUIVALENCE=1 (with VTM_PAPER_JAR_PATH and the plugin JAR) to run Paper")
    def test_live_paper_run_matches_and_refreshes_golden(self):
        import worldgen_runner

        paper_jar = Path(os.environ["VTM_PAPER_JAR_PATH"])
        plugin_jar = Path(os.environ.get("VTM_WORLDGEN_PLUGIN_JAR",
                                         ROOT / "services" / "worldgen" / "target" / "worldgen-plugin-0.1.0.jar"))
        key = "paper-equivalence-test-key-0123456789abcdef"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            voxel_path = root / "voxels.pb.zst"
            write_voxels(voxel_path, fixture_voxels())
            job = payload(voxelSha256=worldgen_runner.sha256_file(voxel_path))
            manifest = root / "job-manifest.json"
            manifest.write_text(json.dumps({"payload": job, "signature": worldgen_runner.sign_payload(job, key.encode())}))
            os.environ["VTM_WORLDGEN_MANIFEST_KEY"] = key
            worlds = {}
            for writer in ("paper", "direct"):
                arguments = ["generate", "--writer", writer, "--manifest", str(manifest), "--voxels", str(voxel_path),
                             "--work-root", str(root / f"work-{writer}"), "--progress", str(root / f"{writer}.json"),
                             "--output-zip", str(root / f"{writer}.zip")]
                if writer == "paper":
                    arguments += ["--paper-jar", str(paper_jar), "--plugin-jar", str(plugin_jar),
                                  "--template", str(ROOT / "services" / "worldgen" / "server-template"),
                                  "--xmx", "1G"]
                    if os.environ.get("VTM_PAPER_CACHE_DIR"):
                        arguments += ["--paper-cache-dir", os.environ["VTM_PAPER_CACHE_DIR"]]
                args = worldgen_runner.parser().parse_args(arguments)
                args.function(args)
                with zipfile.ZipFile(root / f"{writer}.zip") as archive:
                    archive.extractall(root / f"world-{writer}")
                worlds[writer] = world_writer.read_world_blocks(root / f"world-{writer}")
        paper_blocks, paper_chunks = worlds["paper"]
        direct_blocks, direct_chunks = worlds["direct"]
        self.assertEqual(paper_blocks, direct_blocks)
        for key_xz, info in direct_chunks.items():
            self.assertEqual(info["Heightmaps"], paper_chunks[key_xz]["Heightmaps"])
        if os.environ.get("REFRESH_PAPER_GOLDEN") == "1":
            GOLDEN.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN.write_bytes(gzip.compress(json.dumps({
                "blocks": {f"{x},{y},{z}": state for (x, y, z), state in sorted(paper_blocks.items())},
                "chunks": {f"{x},{z}": {"Heightmaps": paper_chunks[(x, z)]["Heightmaps"]}
                           for (x, z) in sorted(direct_chunks)},
            }, sort_keys=True).encode(), mtime=0))


if __name__ == "__main__":
    unittest.main()
