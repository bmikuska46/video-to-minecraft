import { sha256 } from "@noble/hashes/sha2.js";
import { bytesToHex } from "@noble/hashes/utils.js";
import * as Crypto from "expo-crypto";
import * as Device from "expo-device";
import { File, FileMode, Paths } from "expo-file-system";
import { Platform } from "react-native";

import { ApiError, completeUpload, createScan, createUploadUrl, type ProcessingMode, type Scan, type ScanType, type UploadUrl } from "@/api/client";
import { captureMetadata, type RecordedCapture } from "./model";

const HASH_CHUNK_BYTES = 1024 * 1024;

export type CaptureUploadProgress = {
  fraction: number;
  label: string;
};

const pauseForUi = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

export async function sha256File(file: File, onProgress?: (fraction: number) => void): Promise<string> {
  if (!file.exists || file.size <= 0) throw new Error("The recorded video file is empty or unavailable.");
  const hash = sha256.create();
  const handle = file.open(FileMode.ReadOnly);
  let read = 0;

  try {
    while (read < file.size) {
      const chunk = handle.readBytes(Math.min(HASH_CHUNK_BYTES, file.size - read));
      if (!chunk.length) break;
      hash.update(chunk);
      read += chunk.length;
      onProgress?.(read / file.size);
      await pauseForUi();
    }
  } finally {
    handle.close();
  }

  if (read !== file.size) throw new Error("The recorded video could not be read completely.");
  return bytesToHex(hash.digest());
}

async function putFile(signed: UploadUrl, file: File, onProgress?: (fraction: number) => void) {
  const result = await file.upload(signed.url, {
    httpMethod: "PUT",
    headers: signed.requiredHeaders,
    mimeType: signed.requiredHeaders["content-type"] ?? file.type,
    onProgress: ({ bytesSent, totalBytes }) => onProgress?.(totalBytes > 0 ? bytesSent / totalBytes : 0),
    sessionType: "background",
  });
  if (result.status < 200 || result.status >= 300) {
    throw new ApiError(result.status, "UPLOAD_FAILED", `Storage upload failed (${result.status}).`);
  }
}

export async function uploadCapture(
  capture: RecordedCapture,
  processingMode: ProcessingMode,
  scanType: ScanType,
  onProgress: (progress: CaptureUploadProgress) => void,
): Promise<Scan> {
  const video = new File(capture.uri);
  onProgress({ fraction: 0.01, label: "Checking the video" });
  const videoSha256 = await sha256File(video, (fraction) => {
    onProgress({ fraction: 0.02 + fraction * 0.18, label: "Checking the video" });
  });

  onProgress({ fraction: 0.22, label: "Creating scan" });
  const created = await createScan({
    devicePlatform: Platform.OS === "ios" ? "ios" : "android",
    deviceModel: Device.modelName ?? `${Platform.OS} device`,
    captureDurationMs: capture.durationMs,
    processingMode,
    scanType,
  });
  if (video.size > created.uploadConstraints.maxVideoBytes) {
    throw new Error(`The video is too large. Maximum size is ${Math.round(created.uploadConstraints.maxVideoBytes / 1024 / 1024)} MB.`);
  }

  const metadataJson = JSON.stringify(captureMetadata(capture));
  const metadata = new File(Paths.cache, `capture-${created.id}.json`);
  metadata.create({ overwrite: true });
  metadata.write(metadataJson);
  const metadataSha256 = await Crypto.digestStringAsync(Crypto.CryptoDigestAlgorithm.SHA256, metadataJson);

  try {
    onProgress({ fraction: 0.25, label: "Preparing secure upload" });
    const videoUpload = await createUploadUrl(created.id, {
      kind: "video",
      contentType: "video/mp4",
      byteSize: video.size,
      sha256: videoSha256,
    });
    const metadataUpload = await createUploadUrl(created.id, {
      kind: "metadata",
      contentType: "application/json",
      byteSize: metadata.size,
      sha256: metadataSha256,
    });

    await putFile(metadataUpload, metadata);
    onProgress({ fraction: 0.3, label: "Uploading video" });
    await putFile(videoUpload, video, (fraction) => {
      onProgress({ fraction: 0.3 + fraction * 0.65, label: "Uploading video" });
    });

    onProgress({ fraction: 0.97, label: "Starting reconstruction" });
    const scan = await completeUpload(created.id, {
      videoObjectKey: videoUpload.objectKey,
      metadataObjectKey: metadataUpload.objectKey,
      sha256: videoSha256,
      byteSize: video.size,
    });
    onProgress({ fraction: 1, label: "Upload complete" });
    return scan;
  } finally {
    if (metadata.exists) metadata.delete();
  }
}
