import type { ScanType } from "@/api/client";

/** A room walk needs far longer than an orbit around one object. Mirrors the API limits. */
export const MAX_CAPTURE_SECONDS: Record<ScanType, number> = { object: 60, scene: 300 };
export const MIN_CAPTURE_SECONDS = 3;

export type RecordedCapture = {
  uri: string;
  durationMs: number;
  monotonicStartNs: number;
  wallClockStart: string;
};

export type CaptureMetadata = {
  schemaVersion: 1;
  monotonicRecordingStartNs: number;
  wallClockTimestamp: string;
  orientation: "portrait";
  cameraFormat: "1920x1080-30-wide";
  width: 1920;
  height: 1080;
  fps: 30;
  codec: "h264";
  physicalLens: "wide-angle-camera";
  appVersion: "0.1.0";
  nativeScannerModuleVersion: string;
};

export function captureMetadata(capture: RecordedCapture): CaptureMetadata {
  return {
    schemaVersion: 1,
    monotonicRecordingStartNs: capture.monotonicStartNs,
    wallClockTimestamp: capture.wallClockStart,
    orientation: "portrait",
    cameraFormat: "1920x1080-30-wide",
    width: 1920,
    height: 1080,
    fps: 30,
    codec: "h264",
    physicalLens: "wide-angle-camera",
    appVersion: "0.1.0",
    nativeScannerModuleVersion: "expo-camera-56",
  };
}

export function boundedDurationMs(elapsedMs: number, scanType: ScanType): number {
  return Math.max(1, Math.min(MAX_CAPTURE_SECONDS[scanType] * 1000, Math.round(elapsedMs)));
}
