# Minecraft world generator

This service turns the versioned `voxels.pb.zst` contract into a Minecraft Java
26.2 singleplayer save (`world.zip`, with `level.dat` at its root). The runner
verifies the HMAC-signed job manifest and the voxel checksum, and then uses one
of two writers (`worldgen_runner.py generate --writer`):

- **`direct`** (the default of the CLI and the API workers): `world_writer.py`
  writes the Anvil region files itself. Every chunk that holds platform or model
  blocks is written in the layout Paper 26.2 saves (`DataVersion` 4903, status
  `minecraft:full`, 24 sections with block-state and biome palettes, four
  heightmaps); all other chunks are left out, and the game generates them as
  empty air from `world_gen_settings.dat` (a flat world with no layers) when a
  player gets near. `level.dat` and the other level-wide files come from
  `world-template/`, captured from a converted Paper save, with the spawn, world
  name and time filled in. The save is then checked by reading the region files
  back with a separate decoder: every expected model and platform block, nothing
  else, the spawn and the world defaults, as the Paper validation run did.
- **`paper`**: a one-shot Paper plugin creates an empty flat world and a smooth
  stone base, places model blocks in chunk-sorted server-thread batches, applies
  the export defaults, saves, and exits. The runner then starts a second Paper
  process against that save, verifies every expected model and platform block
  plus spawn/world defaults, removes transient server files, and converts the
  save to the vanilla layout (below).

On the benchmark exports the direct writer wrote
the reference room at 100 blocks (53-54k blocks, 56 chunks) in 0.22 s and read
it back in 0.30-0.36 s; the whole world step, runner start and zip included,
took 0.6-0.7 s, where the two Paper runs took 13.6-14.3 s. It produced the same
blocks at every position of the world and the same heightmaps as Paper for the
same `voxels.pb.zst` on every benchmark export, and a 77 KB instead of a 381 KB
`world.zip`, since only chunks with blocks are stored
(`tests/test_world_writer.py` keeps a Paper-made golden of a small model;
`RUN_PAPER_EQUIVALENCE=1` regenerates it with Paper).

Chunks are written with `isLightOn` 0 and no light arrays, so the game lights
them itself when it first loads them. Paper's saves carry `isLightOn` 0 too, but
with its own (Starlight) light data, which the game then starts from. Opened in
the vanilla 26.2 server, both writers' worlds load without errors; the light
the server computed for the direct writer's world matched an independent sky
light calculation in every one of 20,000 sampled air blocks, while the Paper
world kept up to level 7 sky light in places no sky reaches, such as inside a
closed room (9.6% of the room scan's air blocks). Block light is the same in
both.

The Paper save conversion: Paper 26.2 keeps game rules, world clocks, weather
and world-generation settings in each dimension, whereas vanilla reads them from
the root `data/minecraft/`. The runner moves the overworld's copies there,
deletes the unused Nether/End and Paper-only files, removes the Paper keys and
`paper` datapack from `level.dat`, enables cheats, and sets the world-list name
(`--display-name`, defaulting to the manifest `worldName`). The save was checked
by loading it in the vanilla 26.2 server, which is the same engine singleplayer
runs internally.

The Maven build pins Paper API `26.2.build.112-stable` and Java 25. The runnable
Paper JAR, needed only for `--writer paper`, is intentionally not downloaded at
job runtime; mount the reviewed, pinned server JAR into the worker image. The
direct writer needs the reconstruction service's Python packages (NumPy,
zstandard, protobuf) and `services/reconstruction` on the Python path, as in
the reconstruction image; the Paper-only `worldgen.Dockerfile` image has
neither, so it runs `--writer paper`.

`world-template/` holds the level-wide files of a converted Paper 26.2 save.
After a Minecraft or Paper upgrade, export one world with `--writer paper` and
refresh it with `python3 services/worldgen/world_writer.py capture-template
world.zip`, then update `DATA_VERSION` in `world_writer.py`.

## Build and test

```bash
mvn -B -pl services/worldgen -am test
python3 -m pytest services/worldgen/tests
docker build -f infra/containers/worldgen.Dockerfile -t vtm-worldgen .
```

`tests/test_world_writer.py` compares the direct writer with a Paper-generated
golden world. To run Paper itself and refresh that golden:

```bash
mvn -q -pl services/worldgen -am package -DskipTests
RUN_PAPER_EQUIVALENCE=1 REFRESH_PAPER_GOLDEN=1 \
VTM_PAPER_JAR_PATH=infra/paper/paper-26.2-build.112-stable.jar \
  python3 -m pytest services/worldgen/tests/test_world_writer.py -k live
```

## Manifest and run

The queue worker creates the payload shown in `example-manifest-payload.json`.
The voxel SHA-256 and all generation limits are covered by an HMAC-SHA256
signature. Keep the signing key outside the Paper process.

```bash
export VTM_WORLDGEN_MANIFEST_KEY='replace-with-at-least-32-random-bytes'
python3 services/worldgen/worldgen_runner.py sign-manifest \
  --payload services/worldgen/example-manifest-payload.json \
  --output /work/job-manifest.json

python3 services/worldgen/worldgen_runner.py generate --writer direct \
  --manifest /work/job-manifest.json \
  --voxels /work/voxels.pb.zst \
  --work-root /scratch/worldgen \
  --progress /work/worldgen-progress.json \
  --output-zip /work/world.zip

# Or with Paper (the runner's default writer):
python3 services/worldgen/worldgen_runner.py generate --writer paper \
  --paper-jar /opt/paper/paper-26.2-build.112-stable.jar \
  --plugin-jar services/worldgen/target/worldgen-plugin-0.1.0.jar \
  --template services/worldgen/server-template \
  --manifest /work/job-manifest.json \
  --voxels /work/voxels.pb.zst \
  --work-root /scratch/worldgen \
  --progress /work/worldgen-progress.json \
  --output-zip /work/world.zip
```

Generation uses an isolated temporary directory; Paper runs also use a
subprocess argument array, an explicit heap limit and a wall-time limit. The template enables creative
mode and commands (the single-player equivalent of cheats) and disables online
mode because this worker must run without network exposure. Keep failed work
directories only during controlled debugging; they can contain scan-derived
geometry.
