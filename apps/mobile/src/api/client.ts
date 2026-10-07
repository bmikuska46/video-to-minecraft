export type Vector3 = [number, number, number];
export type Bounds = { min: Vector3; max: Vector3 };
export type Axis = "x" | "y" | "z";

export type ScanStatus =
  | "CREATED"
  | "UPLOADING"
  | "UPLOADED"
  | "QUEUED"
  | "EXTRACTING_FRAMES"
  | "SPARSE_RECONSTRUCTION"
  | "DENSE_RECONSTRUCTION"
  | "FILTERING"
  | "PREVIEW_READY"
  | "FAILED";

/** detailed: multi-view stereo + hole filling (~8 min). fast: monocular depth only (~2 min, softer edges). */
export type ProcessingMode = "detailed" | "fast";

/** object: one thing, cropped with a box. scene: a room or several objects, kept uncut. */
export type ScanType = "object" | "scene";

export type Scan = {
  id: string;
  status: ScanStatus;
  failureCode: string | null;
  progress: number;
  captureDurationMs: number;
  processingMode: ProcessingMode;
  scanType: ScanType;
  createdAt: string;
  updatedAt: string;
};

export type CreateScanResponse = {
  id: string;
  status: ScanStatus;
  uploadConstraints: {
    maxVideoBytes: number;
    maxCaptureDurationMs: number;
    allowedVideoContentTypes: string[];
  };
  retentionDeadline: string;
};

export type UploadKind = "video" | "metadata";

export type UploadUrl = {
  method: "PUT";
  url: string;
  objectKey: string;
  expiresAt: string;
  requiredHeaders: Record<string, string>;
};

export type Preview = {
  previewUrl: string;
  bounds: Bounds;
  units: "reconstruction_units";
  scanType: ScanType;
  registeredFrames: number;
  selectedFrames: number;
  quality: string;
  warnings: string[];
  expiresAt: string;
};

export type ReconstructionSelection = { transform: number[]; crop: Bounds };
export type ExportEstimate = {
  voxelSize: number;
  voxelDimensions: Vector3;
  blockCountEstimate: number;
  estimateIsUpperBound: true;
  warningThreshold: number;
  hardLimit: number;
  requiresConfirmation: boolean;
};
export type ExportStatus = "QUEUED" | "VOXELIZING" | "GENERATING_WORLD" | "READY" | "FAILED";
export type MinecraftExport = {
  id: string;
  scanId: string;
  status: ExportStatus;
  failureCode: string | null;
  selectedAxis: Axis;
  targetBlockDimension: number;
  voxelSize: number | null;
  occupiedBlockCount: number | null;
};

type ApiProblem = { detail?: { code?: string; message?: string } };

export class ApiError extends Error {
  constructor(public readonly status: number, public readonly code: string, message: string) {
    super(message);
  }
}

const apiUrl = (process.env.EXPO_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiUrl}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let problem: ApiProblem = {};
    try { problem = await response.json() as ApiProblem; } catch { /* non-JSON upstream error */ }
    throw new ApiError(
      response.status,
      problem.detail?.code ?? "REQUEST_FAILED",
      problem.detail?.message ?? `Request failed (${response.status})`,
    );
  }
  return response.json() as Promise<T>;
}

export const getPreview = (scanId: string) => request<Preview>(`/v1/scans/${scanId}/preview`);

export const getScan = (scanId: string) => request<Scan>(`/v1/scans/${scanId}`);

export const createScan = (capture: {
  devicePlatform: "ios" | "android";
  deviceModel: string;
  captureDurationMs: number;
  processingMode: ProcessingMode;
  scanType: ScanType;
}) =>
  request<CreateScanResponse>("/v1/scans", { method: "POST", body: JSON.stringify(capture) });

export const createUploadUrl = (
  scanId: string,
  upload: { kind: UploadKind; contentType: string; byteSize: number; sha256: string },
) => request<UploadUrl>(`/v1/scans/${scanId}/upload-url`, {
  method: "POST",
  body: JSON.stringify(upload),
});

export const completeUpload = (
  scanId: string,
  upload: { videoObjectKey: string; metadataObjectKey: string; sha256: string; byteSize: number },
) => request<Scan>(`/v1/scans/${scanId}/upload-complete`, {
  method: "POST",
  body: JSON.stringify(upload),
});

export const saveSelection = (scanId: string, selection: ReconstructionSelection) =>
  request<ReconstructionSelection & { scanId: string }>(`/v1/scans/${scanId}/reconstruction`, {
    method: "PATCH", body: JSON.stringify(selection),
  });

export const estimateExport = (scanId: string, axis: Axis, blocks: number) =>
  request<ExportEstimate>(`/v1/scans/${scanId}/exports/estimate`, {
    method: "POST",
    body: JSON.stringify({ scale: { axis, blocks }, palette: "geometry-safe-v1" }),
  });

export const createExport = (scanId: string, axis: Axis, blocks: number, confirmLargeExport: boolean) =>
  request<MinecraftExport>(`/v1/scans/${scanId}/exports`, {
    method: "POST",
    body: JSON.stringify({
      scale: { axis, blocks }, palette: "geometry-safe-v1", confirmLargeExport,
    }),
  });

export const getExport = (exportId: string) => request<MinecraftExport>(`/v1/exports/${exportId}`);

export const getDownloadUrl = (exportId: string) =>
  request<{ url: string; expiresAt: string }>(`/v1/exports/${exportId}/download-url`);
