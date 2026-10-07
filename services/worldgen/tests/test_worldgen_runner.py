from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

MODULE_PATH = Path(__file__).resolve().parents[1] / "worldgen_runner.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("worldgen_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
import level_nbt  # noqa: E402
from level_nbt import BYTE, COMPOUND, LIST, STRING, Tag  # noqa: E402


def payload() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "jobId": "550e8400-e29b-41d4-a716-446655440000",
        "minecraftVersion": "26.2",
        "voxelSha256": "a" * 64,
        "worldName": "minecraft-video-world",
        "platformY": 64,
        "platformMargin": 8,
        "platformBlockState": "minecraft:smooth_stone",
        "batchSize": 20_000,
        "maxBlockCount": 1_000_000,
        "maxPlatformBlocks": 4_000_000,
    }


class ManifestTest(unittest.TestCase):
    def test_signed_manifest_round_trips(self) -> None:
        key = b"this-is-a-test-key-that-is-at-least-32-bytes"
        envelope = {
            "payload": payload(),
            "signature": runner.sign_payload(payload(), key),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job-manifest.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            self.assertEqual(payload(), runner.verify_manifest(path, key)["payload"])

    def test_tampered_manifest_is_rejected(self) -> None:
        key = b"this-is-a-test-key-that-is-at-least-32-bytes"
        original = payload()
        envelope = {
            "payload": original,
            "signature": runner.sign_payload(original, key),
        }
        envelope["payload"]["worldName"] = "tampered"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "job-manifest.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(runner.WorldgenError, "signature"):
                runner.verify_manifest(path, key)

    def test_short_secret_is_rejected(self) -> None:
        with self.assertRaisesRegex(runner.WorldgenError, "at least 32"):
            runner.sign_payload(payload(), b"short")


class PackagingTest(unittest.TestCase):
    def test_level_dat_is_at_zip_root_and_transient_files_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world = root / "world"
            (world / "region").mkdir(parents=True)
            (world / "level.dat").write_bytes(b"level")
            (world / "session.lock").write_bytes(b"lock")
            (world / "uid.dat").write_bytes(b"uid")
            (world / "region" / "r.0.0.mca").write_bytes(b"region")
            output = root / "world.zip"

            runner.zip_world(world, output)

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    ["level.dat", "region/r.0.0.mca"], sorted(archive.namelist())
                )

    def test_template_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "real").write_text("data", encoding="utf-8")
            (source / "link").symlink_to(source / "real")
            with self.assertRaisesRegex(runner.WorldgenError, "symlink"):
                runner.copy_tree_without_links(source, destination)


def paper_level() -> Tag:
    strings = lambda *values: Tag(LIST, [Tag(STRING, value) for value in values], STRING)  # noqa: E731
    return Tag(COMPOUND, {"Data": Tag(COMPOUND, {
        "Bukkit.Version": Tag(STRING, "Paper/26.2-112"),
        "paperSpawnDimension": Tag(STRING, "minecraft:overworld"),
        "DataPacks": Tag(COMPOUND, {"Enabled": strings("vanilla", "paper"), "Disabled": strings()}),
        "ServerBrands": strings("Paper"),
        "WasModded": Tag(BYTE, 1),
        "allowCommands": Tag(BYTE, 0),
        "GameType": Tag(level_nbt.INT, 1),
        "LevelName": Tag(STRING, "minecraft-video-world"),
    })})


def write_paper_save(world: Path) -> None:
    level_nbt.save(world / "level.dat", "", paper_level())
    (world / "data" / "minecraft").mkdir(parents=True)
    (world / "data" / "minecraft" / "world_clocks.dat").write_bytes(b"stale root clock")
    for dimension in ("overworld", "the_nether", "the_end"):
        base = world / "dimensions" / "minecraft" / dimension
        (base / "data" / "minecraft").mkdir(parents=True)
        (base / "data" / "paper").mkdir(parents=True)
        (base / "data" / "paper" / "metadata.dat").write_bytes(b"paper")
        (base / "paper-world.yml").write_text("paper: true\n", encoding="utf-8")
        (base / "region").mkdir()
        (base / "region" / "r.0.0.mca").write_bytes(dimension.encode())
        (base / "data" / "minecraft" / "world_border.dat").write_bytes(b"border")
        for name in runner.LEVEL_WIDE_DATA_FILES:
            (base / "data" / "minecraft" / name).write_bytes(f"{dimension}/{name}".encode())


class SingleplayerConversionTest(unittest.TestCase):
    def test_paper_save_becomes_vanilla_singleplayer_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            world = Path(temporary)
            write_paper_save(world)

            runner.convert_to_singleplayer(world, "Barn scan")

            files = sorted(path.relative_to(world).as_posix() for path in world.rglob("*") if path.is_file())
            self.assertEqual(files, [
                "data/minecraft/game_rules.dat",
                "data/minecraft/scheduled_events.dat",
                "data/minecraft/weather.dat",
                "data/minecraft/world_clocks.dat",
                "data/minecraft/world_gen_settings.dat",
                "dimensions/minecraft/overworld/data/minecraft/world_border.dat",
                "dimensions/minecraft/overworld/region/r.0.0.mca",
                "level.dat",
                "level.dat_old",
            ])
            # Level-wide state comes from the overworld, including the export's game rules and clock.
            for name in runner.LEVEL_WIDE_DATA_FILES:
                self.assertEqual((world / "data" / "minecraft" / name).read_bytes(), f"overworld/{name}".encode())

            name, root = level_nbt.load(world / "level.dat")
            data = root.value["Data"].value
            self.assertEqual(name, "")
            self.assertNotIn("Bukkit.Version", data)
            self.assertNotIn("paperSpawnDimension", data)
            self.assertEqual([tag.value for tag in data["DataPacks"].value["Enabled"].value], ["vanilla"])
            self.assertEqual([tag.value for tag in data["ServerBrands"].value], ["vanilla"])
            self.assertEqual(data["WasModded"].value, 0)
            self.assertEqual(data["allowCommands"].value, 1)
            self.assertEqual(data["GameType"].value, 1)
            self.assertEqual(data["LevelName"].value, "Barn scan")
            self.assertEqual((world / "level.dat_old").read_bytes(), (world / "level.dat").read_bytes())

    def test_incomplete_paper_save_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            world = Path(temporary)
            write_paper_save(world)
            (world / "dimensions" / "minecraft" / "overworld" / "data" / "minecraft" / "game_rules.dat").unlink()
            with self.assertRaisesRegex(runner.WorldgenError, "game_rules.dat"):
                runner.convert_to_singleplayer(world, "Barn scan")


if __name__ == "__main__":
    unittest.main()
