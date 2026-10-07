#!/usr/bin/env python3
"""Turn a video into a Minecraft Java Edition world (world.zip) in one command.

This runs the same stages as the app, without the API, database or storage:
reconstruction (COLMAP + depth) -> optional orientation and crop -> voxelization
into palette blocks -> Paper world generation and validation -> world.zip.

It needs COLMAP, CUDA, FFmpeg, Java and the reconstruction Python environment, so
it is normally started through scripts/video_to_world.sh, which runs it inside
the pipeline-worker image. Everything lands in OUTPUT_DIR:

    reconstruction/   manifest.json, canonical.ply, preview.ply, logs
    voxels.pb.zst     the block grid handed to world generation
    world.zip         the singleplayer save; extract it into .minecraft/saves/
    export.json       the settings used and the resulting block count
    logs/             voxelizer and world-generation logs

A completed reconstruction of the same video (same SHA-256, mode and scan type)
in OUTPUT_DIR is reused, so re-exporting at another size, rotation or crop only
takes the seconds that voxelization and world generation need.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION = ROOT / "services" / "reconstruction"
sys.path.insert(0, str(ROOT / "services" / "api"))
sys.path.insert(0, str(RECONSTRUCTION))

from app.export_contract import (  # noqa: E402
    AXIS_INDEX, ContractError, estimate_blocks, transformed_bounds,
    validate_crop_within_bounds,
)


MINECRAFT_VERSION = "26.2"
PAPER_JAR_NAME = "paper-26.2-build.112-stable.jar"
# World generation refuses manifests above this many placed blocks.
WORLDGEN_BLOCK_LIMIT = 5_000_000


def first_existing(*candidates: Path) -> Path | None:
    return next((path for path in candidates if path.exists()), None)


def default_paper_jar() -> Path | None:
    configured = os.environ.get("VTM_PAPER_JAR_PATH")
    if configured:
        return Path(configured)
    return first_existing(Path("/opt/paper") / PAPER_JAR_NAME, ROOT / "infra" / "paper" / PAPER_JAR_NAME)


def default_plugin_jar() -> Path | None:
    # A local Maven build or the image's copy, whichever was built last.
    builds = [path for path in (ROOT / "services" / "worldgen" / "target" / "worldgen-plugin-0.1.0.jar",
                                Path("/opt/worldgen/worldgen-plugin.jar")) if path.is_file()]
    return max(builds, key=lambda path: path.stat().st_mtime, default=None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(command: list[str], log_path: Path, *, environment: dict[str, str] | None = None,
        echo: bool = True) -> None:
    """Run a stage, copying its output to a log and, when echo is set, the terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if echo:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
        returncode = process.wait()
    if returncode != 0:
        if not echo:
            sys.stdout.write(log_path.read_text(errors="replace")[-3000:])
        raise SystemExit(f"error: {Path(command[1]).name} failed with code {returncode}; see {log_path}")


def quarter_turn(axis: str, turns: int) -> list[list[int]]:
    """Same rotation matrices as the app (apps/mobile/src/features/preview/geometry.ts)."""
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][turns % 4]
    if axis == "x":
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == "y":
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def rigid_transform(turns: tuple[int, int, int]) -> list[float]:
    """Row-major 4x4 matrix Rz * Ry * Rx, the order the app applies its 90 degree turns in."""
    def multiply(left, right):
        return [[sum(left[r][k] * right[k][c] for k in range(3)) for c in range(3)] for r in range(3)]
    rotation = multiply(multiply(quarter_turn("z", turns[2]), quarter_turn("y", turns[1])), quarter_turn("x", turns[0]))
    return [float(value) for row in rotation for value in (*row, 0)] + [0.0, 0.0, 0.0, 1.0]


def reconstruction_matches(manifest: dict, video_sha256: str, mode: str, scan_type: str) -> bool:
    source = manifest.get("source") or {}
    hashes = {source.get("sha256"), (source.get("original") or {}).get("sha256")}
    settings = manifest.get("settings") or {}
    return (video_sha256 in hashes and settings.get("processingMode") == mode
            and settings.get("scanType") == scan_type)


def reconstruct(args: argparse.Namespace, output: Path) -> dict:
    target = output / "reconstruction"
    manifest_path = target / "manifest.json"
    if target.exists() and any(target.iterdir()):
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        reusable = manifest.get("status") == "COMPLETED" and (target / "canonical.ply").is_file()
        if reusable and not args.force_reconstruct:
            if reconstruction_matches(manifest, sha256_file(args.video), args.mode, args.scan_type):
                step(f"Reusing the reconstruction in {target}")
                return manifest
            raise SystemExit(
                f"error: {target} holds a reconstruction of another video, mode or scan type; "
                "choose a new output directory or pass --force-reconstruct"
            )
        if not manifest_path.is_file():
            raise SystemExit(f"error: {target} exists but is not a reconstruction; refusing to replace it")
        print(f"Replacing the {'previous' if reusable else 'unfinished'} reconstruction in {target}")
        shutil.rmtree(target)

    step(f"Reconstructing {args.video.name} ({args.scan_type} scan, {args.mode} mode)")
    command = [sys.executable, str(RECONSTRUCTION / "reconstruct_fixture.py"), str(args.video), str(target),
               "--mode", args.mode, "--scan-type", args.scan_type, "--frame-rate", str(args.frame_rate)]
    if args.cpu:
        command.append("--cpu")
    started = time.monotonic()
    run(command, output / "logs" / "reconstruction.log")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETED":
        raise SystemExit(f"error: reconstruction did not complete; see {manifest_path}")
    metrics = manifest.get("metrics", {})
    print(f"Reconstruction finished in {time.monotonic() - started:.0f} s: "
          f"{metrics.get('registeredFrames', '?')} registered frames, {metrics.get('filteredPoints', '?')} points")
    return manifest


def export_selection(args: argparse.Namespace, manifest: dict) -> dict:
    """Resolve the transform, crop and scale exactly as the API's export preflight does."""
    bounds = manifest["metrics"]["bounds"]
    transform = rigid_transform(tuple(args.rotate))
    available_min, available_max = transformed_bounds(bounds["min"], bounds["max"], transform)
    crop_min = list(args.crop_min or available_min)
    crop_max = list(args.crop_max or available_max)
    if args.top is not None:
        crop_max[1] = args.top
    try:
        if any(low >= high for low, high in zip(crop_min, crop_max, strict=True)):
            raise ContractError("crop minimum must be below its maximum on every axis")
        validate_crop_within_bounds(crop_min, crop_max, available_min, available_max)
    except ContractError as error:
        raise SystemExit(
            f"error: {error}\nAvailable bounds after rotation: min {fmt(available_min)} max {fmt(available_max)}"
        ) from error
    axis = args.axis
    if axis == "longest":
        axis = max(AXIS_INDEX, key=lambda name: crop_max[AXIS_INDEX[name]] - crop_min[AXIS_INDEX[name]])
    estimate = estimate_blocks(crop_min, crop_max, axis=axis, blocks=args.blocks)
    print(f"Reconstruction bounds after rotation: min {fmt(available_min)} max {fmt(available_max)}")
    print(f"Crop: min {fmt(crop_min)} max {fmt(crop_max)}")
    print(f"Scale: {args.blocks} blocks along {axis} -> grid "
          f"{' x '.join(map(str, estimate.voxel_dimensions))} (at most {estimate.block_count_upper_bound:,} blocks)")
    if estimate.block_count_upper_bound > args.max_blocks:
        raise SystemExit(
            f"error: the crop could hold {estimate.block_count_upper_bound:,} blocks, above --max-blocks "
            f"{args.max_blocks:,}; lower --blocks, crop tighter, or raise --max-blocks"
        )
    return {"transform": transform, "cropMin": crop_min, "cropMax": crop_max, "axis": axis,
            "blocks": args.blocks, "voxelSize": estimate.voxel_size,
            "gridDimensions": list(estimate.voxel_dimensions)}


def fmt(values) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "]"


def voxelize(args: argparse.Namespace, output: Path, selection: dict) -> tuple[Path, int, Counter]:
    step("Voxelizing into Minecraft blocks")
    voxels = output / "voxels.pb.zst"
    voxels.unlink(missing_ok=True)
    run([sys.executable, str(RECONSTRUCTION / "voxelizer.py"),
         str(output / "reconstruction" / "canonical.ply"), str(voxels),
         "--crop-min", *map(str, selection["cropMin"]), "--crop-max", *map(str, selection["cropMax"]),
         "--axis", selection["axis"], "--blocks", str(selection["blocks"]),
         "--transform", *map(str, selection["transform"]), "--palette", str(args.palette)],
        output / "logs" / "voxelizer.log")
    from voxel_contract import decode
    world = decode(voxels.read_bytes())
    states = {entry.id: entry.block_state for entry in world.palette}
    distribution = Counter(states.get(voxel.palette_id, "unknown") for voxel in world.voxels)
    print(f"{len(world.voxels):,} blocks, {len(distribution)} block types")
    return voxels, len(world.voxels), distribution


def generate_world(args: argparse.Namespace, output: Path, voxels: Path, occupied: int) -> tuple[Path, dict]:
    step("Generating the Minecraft world with Paper (generate, then validate every block)")
    if occupied > args.max_blocks:
        raise SystemExit(f"error: {occupied:,} blocks exceed --max-blocks {args.max_blocks:,}")
    # The runner only accepts HMAC-signed job manifests. This process is both the
    # signer and the caller, so a throwaway key per run is enough.
    key = secrets.token_hex(32)
    payload = {
        "schemaVersion": 1, "jobId": str(uuid.uuid4()), "minecraftVersion": MINECRAFT_VERSION,
        "voxelSha256": sha256_file(voxels), "worldName": "minecraft-video-world", "platformY": 64,
        "platformMargin": 8, "platformBlockState": "minecraft:smooth_stone", "batchSize": 20000,
        "maxBlockCount": min(args.max_blocks, WORLDGEN_BLOCK_LIMIT), "maxPlatformBlocks": 4_000_000,
    }
    sys.path.insert(0, str(args.worldgen_runner.parent))
    from worldgen_runner import sign_payload
    world_zip = output / "world.zip"
    environment = os.environ.copy()
    environment["VTM_WORLDGEN_MANIFEST_KEY"] = key
    with tempfile.TemporaryDirectory(prefix=".worldgen-", dir=output) as temporary:
        work = Path(temporary)
        manifest = work / "job-manifest.json"
        manifest.write_text(json.dumps(
            {"payload": payload, "signature": sign_payload(payload, key.encode())}, indent=2) + "\n")
        progress = work / "progress.json"
        command = [sys.executable, str(args.worldgen_runner), "generate",
                   "--paper-jar", str(args.paper_jar), "--plugin-jar", str(args.plugin_jar),
                   "--template", str(args.worldgen_template), "--manifest", str(manifest),
                   "--voxels", str(voxels), "--work-root", str(work / "jobs"), "--progress", str(progress),
                   "--output-zip", str(world_zip), "--display-name", args.world_name,
                   "--xmx", args.xmx]
        if args.paper_cache_dir:
            command += ["--paper-cache-dir", str(args.paper_cache_dir)]
        run(command, output / "logs" / "worldgen.log", environment=environment, echo=args.verbose)
        result = json.loads(progress.read_text()) if progress.is_file() else {}
    return world_zip, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="input video (MP4/MOV, H.264 or HEVC)")
    parser.add_argument("output", type=Path, help="output directory; created if missing")

    capture = parser.add_argument_group("reconstruction")
    capture.add_argument("--scan-type", choices=("object", "scene"), default="object",
                         help="object: walk around one thing (default); scene: a room, closed into a shell")
    capture.add_argument("--mode", choices=("fast", "detailed"), default="fast",
                         help="fast: monocular depth, minutes (default); detailed: PatchMatch stereo, tens of minutes")
    capture.add_argument("--frame-rate", type=float, default=4.0, help="selected frames per second, 1-6 (default 4)")
    capture.add_argument("--cpu", action="store_true", help="run COLMAP features and matching on the CPU")
    capture.add_argument("--force-reconstruct", action="store_true",
                         help="redo the reconstruction even if OUTPUT_DIR already has a matching one")

    export = parser.add_argument_group("export")
    export.add_argument("--blocks", type=int, default=100,
                        help="size in blocks along --axis (default 100)")
    export.add_argument("--axis", choices=("longest", *AXIS_INDEX), default="longest",
                        help="axis that --blocks measures (default: the crop's longest axis)")
    export.add_argument("--rotate", nargs=3, type=int, default=(0, 0, 0), metavar=("X", "Y", "Z"),
                        help="90 degree turns around x, y and z, as in the app's orientation buttons")
    export.add_argument("--crop-min", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="crop box minimum after rotation (default: the full reconstruction)")
    export.add_argument("--crop-max", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="crop box maximum after rotation (default: the full reconstruction)")
    export.add_argument("--top", type=float, metavar="Y",
                        help="cut everything above this height, e.g. to remove a room's ceiling")
    export.add_argument("--max-blocks", type=int, default=1_000_000,
                        help="refuse exports whose crop could hold more blocks (default 1,000,000)")
    export.add_argument("--world-name", help="name in Minecraft's world list (default: the video's file name)")
    export.add_argument("--palette", type=Path, default=ROOT / "packages" / "block-palette" / "palette-v1.json")

    worldgen = parser.add_argument_group("world generation")
    worldgen.add_argument("--paper-jar", type=Path, default=default_paper_jar(),
                          help=f"Paper server JAR (default: $VTM_PAPER_JAR_PATH or infra/paper/{PAPER_JAR_NAME})")
    worldgen.add_argument("--plugin-jar", type=Path, default=default_plugin_jar(),
                          help="worldgen plugin JAR (default: the newer of services/worldgen/target/ and the image's copy)")
    worldgen.add_argument("--paper-cache-dir", type=Path,
                          default=Path.home() / ".cache" / "video-to-minecraft" / "paper-cache",
                          help="persistent cache of the patched Paper server, shared between runs")
    worldgen.add_argument("--xmx", default="2G", help="Paper heap size (default 2G)")
    worldgen.add_argument("--verbose", action="store_true", help="show Paper's server output")
    args = parser.parse_args()

    args.video = args.video.resolve()
    args.output = args.output.resolve()
    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not 1.0 <= args.frame_rate <= 6.0:
        parser.error("--frame-rate must be between 1 and 6")
    if not 1 <= args.blocks <= 2048:
        parser.error("--blocks must be between 1 and 2048")
    if (args.crop_min is None) != (args.crop_max is None):
        parser.error("--crop-min and --crop-max must be given together")
    if args.paper_jar is None or not args.paper_jar.is_file():
        parser.error(f"Paper JAR not found; download {PAPER_JAR_NAME} from papermc.io and pass --paper-jar")
    if args.plugin_jar is None or not args.plugin_jar.is_file():
        parser.error("worldgen plugin JAR not found; build it with: mvn -pl services/worldgen -am package")
    # The repository copy wins over the one baked into the image, which can be older.
    args.worldgen_runner = ROOT / "services" / "worldgen" / "worldgen_runner.py"
    args.worldgen_template = ROOT / "services" / "worldgen" / "server-template"
    args.world_name = args.world_name or args.video.stem
    missing = [name for name in ("ffmpeg", "ffprobe", "colmap", "java") if not shutil.which(name)]
    if missing:
        parser.error(f"required executables not found: {', '.join(missing)} "
                     "(run scripts/video_to_world.sh to use the pipeline-worker image)")
    return args


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = reconstruct(args, args.output)
    step("Choosing the export region")
    selection = export_selection(args, manifest)
    voxels, occupied, distribution = voxelize(args, args.output, selection)
    world_zip, worldgen = generate_world(args, args.output, voxels, occupied)
    summary = {
        "video": os.environ.get("VTM_SOURCE_VIDEO", str(args.video)), "scanType": args.scan_type, "mode": args.mode,
        "reconstructionSettingsHash": manifest.get("settingsHash"), **selection,
        "occupiedBlocks": occupied, "blockDistribution": dict(distribution.most_common()),
        "worldName": args.world_name, "worldZip": str(world_zip), "worldZipBytes": world_zip.stat().st_size,
        "paperGenerationDurationMs": worldgen.get("paperGenerationDurationMs"),
        "paperValidationDurationMs": worldgen.get("paperValidationDurationMs"),
    }
    (args.output / "export.json").write_text(json.dumps(summary, indent=2) + "\n")
    step(f"Done in {time.monotonic() - started:.0f} s")
    print(f"World: {world_zip} ({occupied:,} blocks)")
    print("Extract it into a new folder in your Minecraft saves/ directory; level.dat is at the ZIP's root.")


if __name__ == "__main__":
    main()
