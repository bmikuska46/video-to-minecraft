import { describe, expect, it } from "vitest";

import { boundedDurationMs, captureMetadata, MAX_CAPTURE_SECONDS } from "./model";

describe("capture metadata", () => {
  it("creates the canonical version-one camera sidecar", () => {
    const metadata = captureMetadata({
      uri: "file:///capture.mp4",
      durationMs: 8_000,
      monotonicStartNs: 1_234_000_000,
      wallClockStart: "2026-08-24T18:00:00.000Z",
    });

    expect(metadata).toMatchObject({
      schemaVersion: 1,
      monotonicRecordingStartNs: 1_234_000_000,
      wallClockTimestamp: "2026-08-24T18:00:00.000Z",
      orientation: "portrait",
      width: 1920,
      height: 1080,
      fps: 30,
      codec: "h264",
    });
  });

  it("keeps measured durations inside the API capture limit", () => {
    expect(boundedDurationMs(0, "object")).toBe(1);
    expect(boundedDurationMs(7_654.4, "object")).toBe(7_654);
    expect(boundedDurationMs(99_000, "object")).toBe(60_000);
    expect(boundedDurationMs(240_000, "scene")).toBe(240_000);
    expect(boundedDurationMs(399_000, "scene")).toBe(MAX_CAPTURE_SECONDS.scene * 1000);
  });
});
