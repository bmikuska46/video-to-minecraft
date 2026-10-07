#!/usr/bin/env python3
"""Isolated, fail-closed Paper lifecycle for a single Minecraft world export."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any
import zipfile

import level_nbt


PAPER_VERSION = "26.2"
MANIFEST_SCHEMA_VERSION = 1
MAX_COMPRESSED_VOXEL_BYTES = 256 * 1024 * 1024
TRANSIENT_WORLD_FILES = ("session.lock", "uid.dat")
# Paper 26.2 stores level-wide state in each dimension; vanilla, and therefore
# singleplayer, reads these files only from the save's root data/minecraft/.
LEVEL_WIDE_DATA_FILES = (
    "game_rules.dat",
    "scheduled_events.dat",
    "weather.dat",
    "world_clocks.dat",
    "world_gen_settings.dat",
)
PAPER_LEVEL_KEYS = ("Bukkit.Version", "paperSpawnDimension")
UNUSED_DIMENSIONS = ("the_nether", "the_end")


class WorldgenError(RuntimeError):
    """A stable failure reported by the isolated world-generation runner."""


def canonical_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sign_payload(payload: dict[str, Any], key: bytes) -> str:
    if len(key) < 32:
        raise WorldgenError("manifest HMAC key must contain at least 32 bytes")
    digest = hmac.new(key, canonical_payload(payload), hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def verify_manifest(path: Path, key: bytes) -> dict[str, Any]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorldgenError("job manifest is not valid JSON") from error
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
        raise WorldgenError("job manifest must contain only payload and signature")
    payload = envelope["payload"]
    signature = envelope["signature"]
    if not isinstance(payload, dict) or not isinstance(signature, str):
        raise WorldgenError("job manifest payload/signature types are invalid")
    expected = sign_payload(payload, key)
    if not hmac.compare_digest(signature, expected):
        raise WorldgenError("job manifest signature is invalid")
    required = {
        "schemaVersion",
        "jobId",
        "minecraftVersion",
        "voxelSha256",
        "worldName",
        "platformY",
        "platformMargin",
        "platformBlockState",
        "batchSize",
        "maxBlockCount",
        "maxPlatformBlocks",
    }
    if set(payload) != required:
        raise WorldgenError("job manifest payload fields do not match schema version 1")
    if payload["schemaVersion"] != MANIFEST_SCHEMA_VERSION:
        raise WorldgenError("unsupported job manifest schema")
    if payload["minecraftVersion"] != PAPER_VERSION:
        raise WorldgenError("manifest does not target the pinned Paper version")
    return envelope


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree_without_links(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise WorldgenError("server template must be a real directory")
    for item in source.rglob("*"):
        if item.is_symlink():
            raise WorldgenError(f"server template contains a symlink: {item.name}")
        relative = item.relative_to(source)
        target = destination / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def update_properties(path: Path, values: dict[str, str]) -> None:
    existing: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                existing[key.strip()] = value.strip()
    existing.update(values)
    content = "".join(f"{key}={existing[key]}\n" for key in sorted(existing))
    path.write_text(content, encoding="utf-8")


def atomic_status(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(values, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_paper(
    job_dir: Path,
    java_bin: str,
    xmx: str,
    timeout_seconds: int,
    mode: str,
    log_path: Path,
    child_environment: dict[str, str] | None = None,
    paper_cache_dir: Path | None = None,
) -> int:
    result_path = job_dir / "worldgen-result.json"
    result_path.unlink(missing_ok=True)
    # Paperclip downloads the vanilla server from Mojang and patches it into
    # versions/ and libraries/ under its repository directory, which defaults to
    # the fresh job directory. A shared directory keeps the patched server between
    # jobs; Paperclip re-checks every file's hash before reusing it.
    cache = [f"-DbundlerRepoDir={paper_cache_dir}"] if paper_cache_dir else []
    command = [
        java_bin,
        f"-Xms{xmx}",
        f"-Xmx{xmx}",
        "-XX:+ExitOnOutOfMemoryError",
        *cache,
        f"-Dminecraftvideo.mode={mode}",
        f"-Dminecraftvideo.manifest={job_dir / 'job-manifest.json'}",
        f"-Dminecraftvideo.voxels={job_dir / 'voxels.pb.zst'}",
        f"-Dminecraftvideo.progress={job_dir / 'plugin-progress.json'}",
        f"-Dminecraftvideo.result={result_path}",
        "-jar",
        str(job_dir / "paper.jar"),
        "--nogui",
    ]
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=job_dir,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_environment,
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise WorldgenError(
                f"Paper {mode} process exceeded {timeout_seconds} seconds"
            ) from error
    elapsed = time.monotonic() - started
    if return_code != 0:
        raise WorldgenError(f"Paper {mode} process exited with code {return_code}")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorldgenError(f"Paper {mode} process did not write a valid result") from error
    if result.get("status") != "SUCCEEDED":
        reason = result.get("error", "unknown plugin failure")
        raise WorldgenError(f"Paper {mode} plugin failed: {reason}")
    if elapsed <= 0:
        raise WorldgenError("invalid Paper process duration")
    return round(elapsed * 1000)


def clean_world(world: Path) -> None:
    if not (world / "level.dat").is_file():
        raise WorldgenError("generated save is missing level.dat")
    for relative in TRANSIENT_WORLD_FILES:
        (world / relative).unlink(missing_ok=True)
    for item in world.rglob("*"):
        if item.is_symlink():
            raise WorldgenError(f"generated save contains a symlink: {item.name}")


def convert_to_singleplayer(world: Path, display_name: str) -> None:
    """Rewrite a Paper save into the vanilla singleplayer layout with cheats enabled."""
    level_path = world / "level.dat"
    dimensions = world / "dimensions" / "minecraft"
    overworld_data = dimensions / "overworld" / "data" / "minecraft"
    root_data = world / "data" / "minecraft"
    missing = [name for name in LEVEL_WIDE_DATA_FILES if not (overworld_data / name).is_file()]
    if not level_path.is_file() or missing:
        raise WorldgenError(f"Paper save is missing level data: {', '.join(missing) or 'level.dat'}")
    root_data.mkdir(parents=True, exist_ok=True)
    for name in LEVEL_WIDE_DATA_FILES:
        os.replace(overworld_data / name, root_data / name)
    # The export only places blocks in the overworld; singleplayer creates the
    # Nether and End from world_gen_settings.dat if a player ever visits them.
    for name in UNUSED_DIMENSIONS:
        shutil.rmtree(dimensions / name, ignore_errors=True)
    for dimension in dimensions.iterdir():
        shutil.rmtree(dimension / "data" / "paper", ignore_errors=True)
        (dimension / "paper-world.yml").unlink(missing_ok=True)

    try:
        name, root = level_nbt.load(level_path)
        data = root.value["Data"].value
        for key in PAPER_LEVEL_KEYS:
            data.pop(key, None)
        enabled = data["DataPacks"].value["Enabled"]
        enabled.value = [pack for pack in enabled.value if pack.value != "paper"]
    except (KeyError, AttributeError, level_nbt.NbtError) as error:
        raise WorldgenError("generated level.dat does not have the expected structure") from error
    # With every Paper-specific file and key removed the save is a plain vanilla world.
    data["ServerBrands"] = level_nbt.Tag(level_nbt.LIST, [level_nbt.Tag(level_nbt.STRING, "vanilla")],
                                         level_nbt.STRING)
    data["WasModded"] = level_nbt.Tag(level_nbt.BYTE, 0)
    data["allowCommands"] = level_nbt.Tag(level_nbt.BYTE, 1)
    data["LevelName"] = level_nbt.Tag(level_nbt.STRING, display_name)
    level_nbt.save(level_path, name, root)
    shutil.copy2(level_path, world / "level.dat_old")


def zip_world(world: Path, destination: Path) -> None:
    clean_world(world)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for item in sorted(world.rglob("*")):
                if item.is_file():
                    archive.write(item, item.relative_to(world).as_posix())
        with zipfile.ZipFile(temporary) as archive:
            names = archive.namelist()
            if "level.dat" not in names or any(
                name.startswith("/") or ".." in Path(name).parts for name in names
            ):
                raise WorldgenError("world ZIP failed its root/path validation")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def generate(args: argparse.Namespace) -> None:
    key_value = os.environ.get(args.hmac_key_env)
    if key_value is None:
        raise WorldgenError(f"required secret environment variable is missing: {args.hmac_key_env}")
    key = key_value.encode("utf-8")
    manifest_path = args.manifest.resolve(strict=True)
    voxel_path = args.voxels.resolve(strict=True)
    paper_jar = args.paper_jar.resolve(strict=True)
    plugin_jar = args.plugin_jar.resolve(strict=True)
    template = args.template.resolve(strict=True)
    paper_cache = None
    if args.paper_cache_dir:
        paper_cache = args.paper_cache_dir.resolve()
        paper_cache.mkdir(parents=True, exist_ok=True)
    envelope = verify_manifest(manifest_path, key)
    payload = envelope["payload"]
    if voxel_path.stat().st_size > MAX_COMPRESSED_VOXEL_BYTES:
        raise WorldgenError("compressed voxel artifact exceeds the runner limit")
    if sha256_file(voxel_path) != payload["voxelSha256"]:
        raise WorldgenError("voxel artifact checksum does not match the signed manifest")

    args.work_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress.resolve()
    atomic_status(progress_path, status="RUNNING", phase="PREPARING")
    job_dir = Path(tempfile.mkdtemp(prefix=f"worldgen-{payload['jobId']}-", dir=args.work_root))
    try:
        copy_tree_without_links(template, job_dir)
        shutil.copy2(paper_jar, job_dir / "paper.jar")
        plugins = job_dir / "plugins"
        plugins.mkdir(exist_ok=True)
        shutil.copy2(plugin_jar, plugins / "worldgen-plugin.jar")
        shutil.copy2(manifest_path, job_dir / "job-manifest.json")
        shutil.copy2(voxel_path, job_dir / "voxels.pb.zst")
        (job_dir / "eula.txt").write_text("eula=true\n", encoding="utf-8")
        common_properties = {
            "allow-nether": "false",
            "difficulty": "peaceful",
            "enable-command-block": "true",
            "force-gamemode": "true",
            "gamemode": "creative",
            "generate-structures": "false",
            "generator-settings": json.dumps(
                {
                    "biome": "minecraft:plains",
                    "features": False,
                    "lakes": False,
                    "layers": [],
                },
                separators=(",", ":"),
            ),
            "level-type": "flat",
            "max-players": "1",
            "online-mode": "false",
            "server-ip": "127.0.0.1",
            "spawn-monsters": "false",
        }
        child_environment = os.environ.copy()
        child_environment.pop(args.hmac_key_env, None)
        update_properties(
            job_dir / "server.properties",
            common_properties | {"level-name": payload["worldName"]},
        )
        atomic_status(progress_path, status="RUNNING", phase="GENERATING_WORLD")
        generation_duration_ms = run_paper(
            job_dir, args.java_bin, args.xmx, args.timeout_seconds,
            "generate", job_dir / "paper-generation.log", child_environment, paper_cache,
        )
        world = job_dir / payload["worldName"]
        if not world.is_dir() or world.is_symlink():
            raise WorldgenError("Paper did not produce the expected world directory")
        update_properties(
            job_dir / "server.properties",
            common_properties | {"level-name": payload["worldName"]},
        )
        atomic_status(progress_path, status="RUNNING", phase="VALIDATING_WORLD")
        validation_duration_ms = run_paper(
            job_dir, args.java_bin, args.xmx, args.timeout_seconds,
            "validate", job_dir / "paper-validation.log", child_environment, paper_cache,
        )
        atomic_status(progress_path, status="RUNNING", phase="PACKAGING_WORLD")
        convert_to_singleplayer(world, args.display_name or payload["worldName"])
        zip_world(world, args.output_zip.resolve())
        atomic_status(
            progress_path,
            status="SUCCEEDED",
            phase="READY",
            output=str(args.output_zip.resolve()),
            byteSize=args.output_zip.resolve().stat().st_size,
            paperGenerationDurationMs=generation_duration_ms,
            paperValidationDurationMs=validation_duration_ms,
        )
    except Exception as error:
        atomic_status(progress_path, status="FAILED", phase="WORLD_GENERATION", error=str(error))
        raise
    finally:
        if args.keep_work_dir:
            print(job_dir)
        else:
            shutil.rmtree(job_dir)


def sign_manifest(args: argparse.Namespace) -> None:
    key_value = os.environ.get(args.hmac_key_env)
    if key_value is None:
        raise WorldgenError(f"required secret environment variable is missing: {args.hmac_key_env}")
    try:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorldgenError("payload is not valid JSON") from error
    if not isinstance(payload, dict):
        raise WorldgenError("payload must be a JSON object")
    envelope = {"payload": payload, "signature": sign_payload(payload, key_value.encode())}
    args.output.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_manifest(args.output, key_value.encode())


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("generate")
    run.add_argument("--paper-jar", type=Path, required=True)
    run.add_argument("--paper-cache-dir", type=Path,
                     help="persistent Paperclip repository (patched server and libraries) shared by jobs")
    run.add_argument("--plugin-jar", type=Path, required=True)
    run.add_argument("--template", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--voxels", type=Path, required=True)
    run.add_argument("--output-zip", type=Path, required=True)
    run.add_argument("--work-root", type=Path, required=True)
    run.add_argument("--progress", type=Path, required=True)
    run.add_argument("--java-bin", default="java")
    run.add_argument("--xmx", default="4G")
    run.add_argument("--timeout-seconds", type=int, default=1800)
    run.add_argument("--hmac-key-env", default="VTM_WORLDGEN_MANIFEST_KEY")
    run.add_argument("--keep-work-dir", action="store_true")
    run.add_argument("--display-name", help="world name shown in the singleplayer list")
    run.set_defaults(function=generate)
    sign = subparsers.add_parser("sign-manifest")
    sign.add_argument("--payload", type=Path, required=True)
    sign.add_argument("--output", type=Path, required=True)
    sign.add_argument("--hmac-key-env", default="VTM_WORLDGEN_MANIFEST_KEY")
    sign.set_defaults(function=sign_manifest)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except (WorldgenError, OSError, subprocess.SubprocessError) as error:
        print(f"worldgen failed: {error}", file=os.sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
