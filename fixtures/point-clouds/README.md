# Synthetic open-surface point clouds

These deterministic fixtures are golden inputs for crop, scale and
support-preserving voxelization work. Generate them with:

```bash
python3 scripts/generate_point_cloud_fixtures.py
```

Each directory contains an ASCII `cloud.ply`, its checksum, and `fixture.json`.
The PLY stores position, normal, sRGB color, confidence, support count and a
source-view ID for every sample. `fixture.json` records bounds and named regions
that must remain empty. The fixtures deliberately contain no inferred back
faces, interiors, hole filling or watertight completion.

The five scenes cover a wall with a window opening, a two-wall corner, three
observed faces of a cube, strongly supported disconnected railing details, and
a facade with observed ground/adjacent geometry requiring a crop.
