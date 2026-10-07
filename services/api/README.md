# API service

FastAPI control plane for scan uploads, reconstruction state and Minecraft exports.
Large capture files travel directly between the mobile client and S3/MinIO; the API
only signs canonical immutable object keys and verifies the stored size and SHA-256
metadata before queueing work.

The capture package requires `video.mp4` and a schema-versioned `capture.json`;
compressed IMU and AR-frame sidecars are optional. See
`../../packages/contracts/capture.schema.json`. Signed PUTs include
`If-None-Match: *`, making every canonical object key create-only.

## Run locally

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e '.[test]'
VTM_DATABASE_URL=sqlite:///./api.db .venv/bin/uvicorn app.main:app --reload
```

Production defaults expect PostgreSQL at `postgres`, Redis at `redis`, and MinIO at
`minio`. All configuration uses the `VTM_` prefix; see `app/config.py`.

When clients run outside Docker, set `VTM_S3_PUBLIC_ENDPOINT_URL` to an S3/MinIO
address they can reach (for example `http://192.168.1.20:9000` for a phone on the
same LAN). `VTM_S3_ENDPOINT_URL` remains the private address used by API and worker
containers. Signed upload and download URLs use the public endpoint while storage
operations continue over the private endpoint.

The reconstruction and export queue messages use deterministic Celery task IDs.
Workers must additionally make each stage idempotent by checking their output
manifest/settings hash before doing work.

`app.pipeline` now provides both Celery tasks. The GPU worker downloads the
immutable capture, runs reconstruction, converts the observed preview points to
GLB, uploads the canonical artifacts, and advances the scan to `PREVIEW_READY`.
The export task downloads that canonical cloud, voxelizes the saved crop/scale,
runs the isolated Paper generation plus load validation, uploads `world.zip`, and
advances the export to `READY`. Terminal records are treated idempotently.

The complete local topology is in `infra/compose.yaml`. Set
`VTM_PAPER_JAR_PATH` to the reviewed Paper 26.2 server JAR and use a non-default
`VTM_WORLDGEN_MANIFEST_KEY` outside local development, then run:

```bash
docker compose -f infra/compose.yaml up --build
```

For day-to-day development, `scripts/start_app.sh` builds and starts this stack,
waits for the API and GPU worker to be ready, and runs the Expo dev server with
`EXPO_PUBLIC_API_URL` and `VTM_S3_PUBLIC_ENDPOINT_URL` pointing at this machine's
LAN address. Ctrl+C stops everything in a safe order: Expo, the API, the worker
(after its current job; a second Ctrl+C abandons the job), then MinIO, Redis and
Postgres. Use `--backend-only` to skip Expo and `--no-build` to skip the rebuild.

## Observability

`GET /health/live` proves that the API process can serve requests. `GET
/health/ready` returns HTTP 200 only when PostgreSQL, Redis, and object storage are
reachable; otherwise it returns HTTP 503 with a per-dependency status map.

Uploads and worker stages are stored append-only in `pipeline_stage_metrics`.
Records include scan/job correlation IDs, status, duration, stable failure reason,
and stage-specific JSON metrics. Worker logs mirror these records as single-line
JSON events and omit raw filenames, object keys, signed URLs, and frame data.

The reconstruction container becomes healthy only after its startup self-test
reports COLMAP, CUDA/GPU, and FFmpeg versions and verifies NumPy plus scratch-disk
operation. Detailed reconstruction manifest stages are imported into the metric
table. Export records add occupied voxel count, palette distribution, Paper
generation/validation duration, ZIP size, and total end-to-end duration.

## Preview and scale preflight

`PATCH /v1/scans/{scanId}/reconstruction` accepts a row-major proper rigid 4x4
transform and a crop expressed in transformed coordinates. The API rejects scale,
shear, reflection, non-finite matrices, and crops outside the transformed source
bounds.

Before submission, call `POST /v1/scans/{scanId}/exports/estimate` with the same
`scale` and `palette` fields used by the export request. It returns the voxel size,
voxel dimensions, and a conservative block-count upper bound. Counts above
`VTM_BLOCK_COUNT_WARNING_THRESHOLD` require `confirmLargeExport: true` on the final
request; counts above `VTM_BLOCK_COUNT_HARD_LIMIT` are rejected. The true occupied
count is recorded later by support-filtered voxelization.

## Test

```bash
.venv/bin/pytest
```
