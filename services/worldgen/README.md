# Minecraft world generator

This service generates the Minecraft world. A one-shot Paper plugin reads
the versioned `voxels.pb.zst` contract, creates an empty flat world and a smooth
stone base, places model blocks in chunk-sorted server-thread batches, applies
the export defaults, saves, and exits. The runner then starts a second Paper
process against that save, verifies every expected model and platform block plus
spawn/world defaults, removes transient server files, and creates a ZIP whose
root contains `level.dat`.

Before zipping, the runner converts the Paper save into a vanilla singleplayer
world. Paper 26.2 keeps game rules, world clocks, weather and world-generation
settings in each dimension, whereas vanilla reads them from the root
`data/minecraft/`. The runner moves the overworld's copies there, deletes the
unused Nether/End and Paper-only files, removes the Paper keys and `paper`
datapack from `level.dat`, enables cheats, and sets the world-list name
(`--display-name`, defaulting to the manifest `worldName`). The save was checked by loading it
in the vanilla 26.2 server, which is the same engine singleplayer runs internally.

The Maven build pins Paper API `26.2.build.112-stable` and Java 25. The runnable
Paper JAR is intentionally not downloaded at job runtime; mount the reviewed,
pinned server JAR into the worker image.

## Build and test

```bash
mvn -B -pl services/worldgen -am test
python3 -m unittest discover -s services/worldgen/tests -v
docker build -f infra/containers/worldgen.Dockerfile -t vtm-worldgen .
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

python3 services/worldgen/worldgen_runner.py generate \
  --paper-jar /opt/paper/paper-26.2-build.112-stable.jar \
  --plugin-jar services/worldgen/target/worldgen-plugin-0.1.0.jar \
  --template services/worldgen/server-template \
  --manifest /work/job-manifest.json \
  --voxels /work/voxels.pb.zst \
  --work-root /scratch/worldgen \
  --progress /work/worldgen-progress.json \
  --output-zip /work/world.zip
```

Generation uses an isolated temporary directory, a subprocess argument array,
an explicit heap limit and a wall-time limit. The template enables creative
mode and commands (the single-player equivalent of cheats) and disables online
mode because this worker must run without network exposure. Keep failed work
directories only during controlled debugging; they can contain scan-derived
geometry.
