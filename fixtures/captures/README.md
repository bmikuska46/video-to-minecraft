# Capture fixtures

The three fixtures in this directory are deterministic, consent-free synthetic
captures used to smoke-test the standalone reconstruction pipeline. Generate
them with:

```bash
python3 scripts/generate_capture_fixtures.py
```

Each fixture contains `video.mp4`, a contract-valid `capture.json`, fixture-only
details in `fixture.json`, and `sha256.txt`. The camera path is expressed in
right-handed scene coordinates in `fixture.json`; +Y is up,
and the camera looks at the stated target. The clips are 1280x720 H.264, 30 FPS,
silent, and exactly 15 seconds long.

| Fixture | Scene | Physical capture path represented by the trajectory |
|---|---|---|
| `brick-facade-strafe` | Textured facade with two open windows | Walk 3.2 m left-to-right, staying about 5.5 m from the wall and keeping the facade centered. |
| `building-corner-arc` | Two textured walls meeting at a corner | Walk a roughly 70-degree, 4.4 m arc around the corner, keeping the corner centered. |
| `painted-facade-shallow` | Low-texture painted facade | Walk 2.0 m sideways about 6.5 m from the wall; this is intentionally challenging. |

These are engineering smoke fixtures, not acceptance captures. In practice the
repetitive synthetic brick texture yields an erratic camera trajectory, and
`brick-facade-strafe` is (correctly) rejected by the sparse quality gate. Before tuning
quality thresholds, replace or supplement them with three consented physical
phone captures following the same paths. Raw physical videos should remain
outside source control; add their immutable checksums and capture metadata here.

## Real capture

`barn-gable-arc-real` is 15 seconds (110-125 s) of the Tanks and Temples
"Barn" video: a real handheld walk around the corner of a park building, ending
front-on to its gable end. Fetch it with:

```bash
python3 scripts/fetch_real_capture.py
```

Only the MP4 index and the segment's byte range (about 185 MB of the 4.4 GB
source) are downloaded; those bytes are checked against a pinned SHA-256 before
the clip is downscaled to phone-like 1920x1080 H.264. Pass `--source Barn.mp4`
to cut from a local copy instead.

The source is published under
[CC BY 4.0](https://www.tanksandtemples.org/license/): Knapitsch, Park, Zhou
and Koltun, "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction",
ACM Transactions on Graphics 36(4), 2017. The clip is trimmed, downscaled and
re-encoded; `fixture.json` records the exact changes.
