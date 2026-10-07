# Standalone reconstruction

This service has no API dependency. It validates a capture with `ffprobe`,
decodes candidates at 8 FPS, scores exposure/sharpness/similarity, and selects
30-240 temporally distributed keyframes (up to 600 with `--scan-type scene`, see
below). It then runs shared-camera sparse
reconstruction, enforces the sparse quality gate, runs CUDA PatchMatch stereo,
completes depth holes with aligned monocular depth, fuses the result, and
conservatively filters/aligns the points into `canonical.ply`. A deterministic
`preview.ply` is capped at 100,000 points.

PatchMatch runs at 960 px with an 11-sample window at step 2 (spanning about
38 px of the full frame) and `filter_min_ncc` 0.05. On weakly textured objects
this finds far more depth than COLMAP's defaults: on a smooth white air purifier
valid depth on its plain faces rose from 1% to 39%, in about half the time.
It runs 3 iterations over the 10 best-overlapping source views per frame
(`patch-match.cfg` is rewritten after undistortion); COLMAP samples a fixed number
of sources per pixel, so this halved PatchMatch time (754 s to 401 s for 60
frames on an RTX 4060) with essentially the same coverage. Stereo fusion uses the
same `max_image_size`.

Surfaces with no texture at all still defeat stereo matching. `depth_completion.py`
predicts relative depth for every frame with Depth Anything V2 Large, fits it to
that frame's observed MVS depth (robust affine in inverse depth plus a smooth
local correction) and writes it only into pixels MVS left empty; observed depth
is never modified. A frame is left untouched when fewer than 5% of its pixels
have observed depth or when the fit misses held-out observed pixels by more than
5% (median); filled depths outside the frame's observed depth range are dropped.
Stereo fusion then keeps only points that agree across several views. The
observed maps are fused first into `fused-observed.ply`, and the filtering step
uses it to add an `observed` vertex property to `canonical.ply` (1 = backed by
multi-view stereo, 0 = monocular only); the manifest reports both counts. Pass
`--no-depth-completion` to fuse observed depth only.

Validation inspects the source once with `ffprobe` (byte size, a video stream,
dimensions, duration). Nothing is re-encoded: a single FFmpeg pass decodes the
source (any codec FFmpeg reads, including HEVC and 10-bit HDR phone recordings;
on the GPU with NVDEC where it supports the codec), applies the display rotation,
trims to the scan type's limit, and writes both the 8 FPS candidate JPEGs and
their grey scoring thumbnails. On the reference 62 s, 120 fps HEVC room clip this
replaced a 17 s H.264 transcode plus a 12 s extraction with 3.5 s. The source's
checksum and stream are kept in `input.json` and the manifest.

`--scan-type` selects the capture shape. `object` (default) is an orbit around
one thing: at most 60.25 s and 240 keyframes, later cropped with a box in the
preview. `scene` is a room or several objects walked along the walls and
exported uncut: at most 300.25 s and 600 keyframes, so neighbouring views still
overlap over the longer path. Both caps keep the default 4 FPS (object) and
2 FPS (scene) keyframe density over the whole allowed clip; shorter clips select
proportionally fewer frames. Inputs may be up to 2 GiB. Up to 100 keyframes are matched exhaustively;
longer captures match every frame with its 20 neighbours and with frames 2, 4,
8, ... away (COLMAP's sequential matcher with quadratic overlap), since
exhaustive matching grows with the square of the frame count: 13-17 s instead of
162 s on the 250-frame reference room, with the same frames registered. The
mapper runs global bundle adjustment after every 40% (not 10%) growth of the
model, which took 55-61 s instead of 89 s there.

Plain painted walls and ceilings give COLMAP's default SIFT almost nothing to
match, so frames facing them fail to register. On the reference bedroom (62 s,
fast mode) the old 180-frame limit and default thresholds registered 159 frames
and reconstructed roughly half the room. Scene scans therefore extract SIFT with
`peak_threshold` 0.002 and register with relaxed absolute-pose thresholds (15
inliers, 15% inlier ratio, 5 trials), and when frames remain unregistered a
`SPARSE_EXTENSION` stage continues mapping from the main model (about 17 s; it
added no frames on the bedroom, where the thresholds alone did the work). The
same capture now registers 239 of 250 frames; the missing 11 face a blank white
ceiling.

`--mode fast` skips PatchMatch: every frame's depth is Depth Anything V2 Large
(at a 392 px network input, half the time of its native 518 px for a 1.88 to
1.91% change in held-out error) aligned to that frame's SfM points, written at
960 px from frames undistorted straight to 960 px. Fast scene scans also skip
COLMAP stereo fusion: scene completion rebuilds every surface from the depth
maps, so `fused.ply` is instead every 8th non-edge pixel of each depth map
back-projected with its normal and color (about as many points), which supplies
the bounds, floor and wall directions filtering and completion need.

Scene scans end with `SCENE_COMPLETION` (`scene_completion.py`), which turns the
fused cloud into a closed room without gaps:

1. Every frame's depth map is fused into a TSDF on a 320-voxel grid in the
   floor-levelled frame, rotated about +Y so the dominant walls lie on X and Z.
   Before the final fusion, two rounds render the fused surface into every frame
   and refit that frame's depth to it (affine in inverse depth plus a smooth
   correction): monocular depth puts a wall a few percent nearer or farther in
   every frame, which otherwise smears it over many voxels. Each frame only
   visits the voxels in the box around its view pyramid (cut a truncation band
   behind its deepest pixel in the refinement rounds), and the refinement rounds
   skip colors and free-space counts, which only the final fusion needs.
2. Free space needs at least two views that saw past a voxel and at least half
   of all views that saw it; voxels behind a wall get "occluded" votes from every
   view of it, so frames that overshoot a wall or look out of a skylight cannot
   carve space outside the room, and a ghost surface reported by a few frames is
   carved away by the rest. The room is the free space connected to the cameras,
   grown up to the fused surfaces.
3. Its floor projection is the footprint; the top of its free space per column is
   the ceiling (sloped attic ceilings included). Unseen ceiling is interpolated
   harmonically from seen ceiling, and narrow downward dips (free space ending
   under a shelf) are closed.
4. The one-voxel shell outside that volume (floor, walls, ceiling) is closed by
   construction and added whole, colored by inverse-distance weighting from the
   nearest fused surfaces of the same kind. Surfaces more than three voxels
   outside it (balcony through a door, a room through a window) are dropped, and
   nothing is added inside: the interior keeps only fused object surfaces.

`measured.ply` keeps the filtered cloud; `canonical.ply` is the completed one,
with `synthesized` = 1 on shell points. `scene-completion.json` records the grid,
rotation, camera centers, room bounds and gap statistics.

External programs are always invoked with argument arrays and no shell
interpolation.

`frame-selection.json` retains per-candidate scores and the exact
image-to-video timestamp map for provenance and later AR alignment. Every run
creates `manifest.json` before invoking external tools and updates it
atomically after each stage. The manifest records the pipeline/settings hash,
stage status and duration, sparse reconstruction statistics, the initial
quality decision, fused point count, artifacts, and a structured failure when a
run stops. It is therefore safe for automation to inspect while reconstruction
is still running. `logs/model-analyzer.log` retains COLMAP's original metrics
output alongside the normalized values in the manifest. The metrics stage also
exports a text model to `metrics-model/` so the manifest can report the median
per-point reprojection error rather than relying only on COLMAP's aggregate
mean.

The sparse quality gate stops before dense processing when fewer than 15 frames
register, less than 60% register, median reprojection error exceeds 2 pixels,
camera baseline is inadequate, or the camera trajectory is disconnected. These
calibration thresholds are embedded in each manifest. Camera poses also provide
an estimated up vector. Phones are held without roll, so every camera's right
axis is horizontal and up is the direction perpendicular to all of them
(`upVectorMethod: cameraRightVectors`). Averaging the cameras' image-up axes
instead is biased by pitch: on a room scan looking ~43 degrees down at the floor
it tilted the whole room by ~22 degrees. When the camera never turned (a straight
strafe) the right axes are parallel, so the mean image-up is projected
perpendicular to them instead (`cameraUpOrthogonalToRight`).

The filtering stage rigidly aligns that up vector to +Y, then snaps the floor
exactly level: every tilt within 10 degrees is tried, and the fullest
floor-thick slab with at least 70% of the scene above it is least-squares fitted
and rotated to horizontal (`floorLeveling` in the filter metrics). On the
reference room this removed the remaining 2 degrees, which would otherwise step
a 100-block floor by about 3 blocks; larger corrections and unsupported planes
are refused. Alignment is one rigid rotation that preserves every PLY vertex
property, including color and normals. Filtering only removes non-finite and
unequivocally isolated points; it never fills or creates geometry.

When recorded gravity has already been transformed into reconstruction
coordinates, pass it with `--up-vector X Y Z`; it takes precedence over the up
estimate from registered camera poses.

Build and lock the CUDA image:

```bash
scripts/build_reconstruction_image.sh
```

The build pins CUDA base-image digests and the COLMAP 4.0.4 commit. The script
loads the image, verifies the binary, and writes the immutable local image ID to
`infra/containers/reconstruction-image.lock`. A registry deployment should push
the image and replace/add its registry `RepoDigest`; the local image ID already
prevents tag drift on this host.

The default build targets CUDA compute capability 8.9 (the local RTX 4060).
Set `RECONSTRUCTION_CUDA_ARCHITECTURES` when building for another deployment,
for example `86` for an Ampere host; the selected value is recorded in the
lockfile.

Generate and reconstruct a fixture on an NVIDIA Container Toolkit host:

```bash
python3 scripts/generate_capture_fixtures.py
scripts/reconstruct_fixture.sh brick-facade-strafe
```

Output is written beneath `artifacts/reconstructions/`. The wrapper refuses to
overwrite an existing output. To run the Python entry point directly with a
host-installed COLMAP, use:

```bash
python3 services/reconstruction/reconstruct_fixture.py \
  fixtures/captures/brick-facade-strafe/video.mp4 \
  artifacts/reconstructions/brick-facade-strafe
```

`--cpu` disables GPU SIFT extraction and matching for diagnostics. Dense
PatchMatch still requires CUDA.

## Real-capture acceptance

`tests/test_real_capture.py` checks the real `barn-gable-arc-real` fixture (see
`fixtures/captures/README.md`) whenever it is present. With
`RUN_REAL_CAPTURE_ACCEPTANCE=1` it also reconstructs the clip through
`scripts/reconstruct_fixture.sh` and asserts that every stage completes, the
sparse model is well constrained, the canonical cloud is upright (lawn plane
within 8 degrees of +Y), and voxelization emits only observed surface cells that
round-trip through `voxels.pb.zst`. On the local RTX 4060 the reconstruction
takes about 32 minutes, 30 of them in PatchMatch stereo. Set
`REAL_CAPTURE_RECONSTRUCTION=<output dir>` to re-check an existing run.

## Complete flow verification

`services/api/tests/test_pipeline_flow.py` exercises the worker boundary from a
queued scan through preview generation, support-preserving voxelization and a
downloadable world ZIP. It uses the real GLB and voxelizer implementations with
controlled reconstruction/Paper fixture executables, so it is safe to run in CI
without a GPU or a licensed server binary.

## Surface voxelization

`voxelizer.py` transforms and crops the canonical PLY, derives the requested
voxel size, and emits only cells that pass explicit observation-support,
distance, confidence, and known-free checks. It aggregates robust linear-RGB
color and normals, maps color to the versioned geometry-safe block palette,
then rests and centers the model for world generation. Optional splatting is
bounded to half a voxel and disabled by default; no connectivity or completion
operation exists.

```bash
python3 services/reconstruction/voxelizer.py canonical.ply voxels.json \
  --crop-min -3 0 -0.1 --crop-max 3 4 0.1 --axis x --blocks 60
```

The JSON artifact is an auditable intermediate for development. Production
handoff uses the versioned `packages/contracts/voxel_world.proto` schema and
`voxel_contract.py` to write deterministic, chunk/section-sorted
`voxels.pb.zst` files. Both Python and Java compatibility tests decode the same
golden artifact.
