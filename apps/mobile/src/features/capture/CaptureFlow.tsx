import { useMutation, useQuery } from "@tanstack/react-query";
import { CameraView, useCameraPermissions } from "expo-camera";
import { File } from "expo-file-system";
import { useKeepAwake } from "expo-keep-awake";
import { useRouter } from "expo-router";
import { useVideoPlayer, VideoView } from "expo-video";
import { useEffect, useRef, useState } from "react";
import { ActivityIndicator, Alert, Platform, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";

import { ApiError, getScan, type ProcessingMode, type ScanStatus, type ScanType } from "@/api/client";
import { Button } from "@/ui/Button";
import { Segmented } from "@/ui/Segmented";
import { colors, radius } from "@/ui/theme";
import { useBackHandler } from "@/ui/useBackHandler";
import { boundedDurationMs, MAX_CAPTURE_SECONDS, MIN_CAPTURE_SECONDS, type RecordedCapture } from "./model";
import { uploadCapture, type CaptureUploadProgress } from "./upload";
import { saveRecentScanId } from "./recentScan";

type Step = "prepare" | "camera" | "review" | "upload" | "processing";

const statusLabels: Record<ScanStatus, string> = {
  CREATED: "Preparing upload",
  UPLOADING: "Uploading capture",
  UPLOADED: "Upload verified",
  QUEUED: "Waiting for reconstruction",
  EXTRACTING_FRAMES: "Selecting sharp video frames",
  SPARSE_RECONSTRUCTION: "Matching camera positions",
  DENSE_RECONSTRUCTION: "Building the point cloud",
  FILTERING: "Preparing the preview",
  PREVIEW_READY: "Preview ready",
  FAILED: "Reconstruction failed",
};

const message = (error: unknown) => error instanceof ApiError ? error.message : error instanceof Error ? error.message : "Something went wrong.";

export function CaptureFlow({ onClose }: { onClose: () => void }) {
  const router = useRouter();
  const [step, setStep] = useState<Step>("prepare");
  const [scanType, setScanType] = useState<ScanType>("object");
  const [capture, setCapture] = useState<RecordedCapture>();
  const [scanId, setScanId] = useState<string>();
  const [uploadProgress, setUploadProgress] = useState<CaptureUploadProgress>({ fraction: 0, label: "Preparing upload" });
  const upload = useMutation({
    mutationFn: ({ recorded, mode }: { recorded: RecordedCapture; mode: ProcessingMode }) =>
      uploadCapture(recorded, mode, scanType, setUploadProgress),
    onSuccess: (scan, { recorded }) => {
      void saveRecentScanId(scan.id).catch(() => undefined);
      setScanId(scan.id);
      setStep("processing");
      const video = new File(recorded.uri);
      if (video.exists) video.delete();
    },
  });
  const scan = useQuery({
    queryKey: ["scan", scanId],
    queryFn: () => getScan(scanId!),
    enabled: Boolean(scanId) && step === "processing",
    retry: false,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "PREVIEW_READY" || status === "FAILED" ? false : 2500;
    },
  });

  useEffect(() => {
    if (scan.data?.status === "PREVIEW_READY") {
      router.replace({ pathname: "/preview/[scanId]", params: { scanId: scan.data.id } });
    }
  }, [router, scan.data?.id, scan.data?.status]);

  const retake = () => {
    if (capture) {
      const video = new File(capture.uri);
      if (video.exists) video.delete();
    }
    upload.reset();
    setCapture(undefined);
    setStep("camera");
  };
  const confirmDiscard = () => Alert.alert("Discard this recording?", "You will go back to the camera to record again.", [
    { text: "Keep", style: "cancel" },
    { text: "Discard", style: "destructive", onPress: retake },
  ]);

  // Steps are local state rather than routes, so Android back has to walk them here.
  useBackHandler(() => {
    if (step === "prepare") onClose();
    else if (step === "review") confirmDiscard();
    else if (step === "upload") { if (upload.isError) setStep("review"); }
    else if (step === "processing") onClose();
    // The camera step handles back itself (it must not leave mid-recording).
  });

  if (step === "prepare") return <Preparation scanType={scanType} onScanType={setScanType} onClose={onClose} onContinue={() => setStep("camera")} />;
  if (step === "camera") return <Recorder scanType={scanType} onClose={() => setStep("prepare")} onRecorded={(recorded) => { setCapture(recorded); setStep("review"); }} />;
  if (step === "review" && capture) {
    return <Review scanType={scanType} capture={capture} onRetake={confirmDiscard} onUpload={(mode) => { setStep("upload"); upload.mutate({ recorded: capture, mode }); }} />;
  }
  if (step === "upload") {
    return (
      <ProgressScreen
        title="Uploading your scan"
        label={uploadProgress.label}
        progress={uploadProgress.fraction}
        error={upload.isError ? message(upload.error) : undefined}
        onRetry={upload.isError && upload.variables ? () => upload.mutate(upload.variables) : undefined}
        onBack={upload.isError ? () => setStep("review") : undefined}
        backLabel="Back to review"
      />
    );
  }

  const status = scan.data?.status;
  return (
    <ProgressScreen
      title={status === "FAILED" ? "Reconstruction failed" : "Building your reconstruction"}
      label={status ? statusLabels[status] : "Checking reconstruction status"}
      progress={scan.data?.progress ?? 0}
      stage={status === "FAILED" ? undefined : stageIndex(status)}
      detail={scanId ? `Scan ID: ${scanId}` : undefined}
      note={status !== "FAILED" ? "This keeps running on the server. You can leave and resume it from the home screen." : undefined}
      error={status === "FAILED" ? `Reason: ${scan.data?.failureCode ?? "UNKNOWN"}. Try recording again with more sideways movement and overlap.` : scan.isError ? message(scan.error) : undefined}
      onRetry={status === "FAILED" ? retake : scan.isError ? () => scan.refetch() : undefined}
      retryLabel={status === "FAILED" ? "Record again" : undefined}
      onBack={onClose}
    />
  );
}

const SCAN_TYPES: { type: ScanType; label: string; lead: string; tips: { title: string; body: string }[] }[] = [
  {
    type: "object",
    label: "Object",
    lead: `Record one continuous video of the visible exterior. You can crop it to the object afterwards. Recording stops by itself after ${formatSeconds(MAX_CAPTURE_SECONDS.object)}.`,
    tips: [
      { title: "Use even daylight", body: "Avoid darkness, glare, glass, moving people, and moving vegetation." },
      { title: "Keep the object in frame", body: "Fill most of the guide while leaving a little space around every edge." },
      { title: "Move sideways", body: "Walk in an arc. Keep roughly two-thirds of the previous view visible as you move." },
      { title: "Move slowly", body: "Avoid zooming, standing in one spot, sudden turns, and motion blur." },
    ],
  },
  {
    type: "scene",
    label: "Room / scene",
    lead: `For a room or several objects. Everything the camera sees is kept, and nothing is cut out. Recording stops by itself after ${formatSeconds(MAX_CAPTURE_SECONDS.scene)}.`,
    tips: [
      { title: "Turn the lights on", body: "Avoid darkness, mirrors, glare on windows, and people walking through." },
      { title: "Walk, never spin in place", body: "Move along the walls pointing the camera across the room. Turning on the spot gives no depth." },
      { title: "Cover everything you want", body: "Sweep each wall, the floor and every object. Anything never seen stays empty." },
      { title: "Move slowly", body: "Keep roughly two-thirds of the previous view visible. Avoid zooming and sudden turns." },
    ],
  },
];

function Preparation({ scanType, onScanType, onClose, onContinue }: { scanType: ScanType; onScanType: (type: ScanType) => void; onClose: () => void; onContinue: () => void }) {
  const selected = SCAN_TYPES.find((option) => option.type === scanType)!;
  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.preparation}>
        <Button variant="link" label="‹ Home" onPress={onClose} style={styles.back} />
        <Text style={styles.eyebrow}>BEFORE YOU RECORD</Text>
        <Text style={styles.title}>Scan with a slow sideways walk</Text>
        <Text style={styles.modeHeading}>What are you scanning?</Text>
        <Segmented
          accessibilityLabel="What are you scanning?"
          options={SCAN_TYPES.map((option) => ({ value: option.type, label: option.label, detail: `Up to ${formatSeconds(MAX_CAPTURE_SECONDS[option.type])}` }))}
          value={scanType}
          onChange={onScanType}
        />
        <Text style={[styles.lead, styles.scanTypeLead]}>{selected.lead}</Text>
        <View style={styles.tipList}>
          {selected.tips.map((tip, index) => <Tip key={tip.title} number={String(index + 1)} title={tip.title} body={tip.body} />)}
        </View>
        <Button label="Open camera" onPress={onContinue} />
      </ScrollView>
    </SafeAreaView>
  );
}

function Tip({ number, title, body }: { number: string; title: string; body: string }) {
  return <View style={styles.tip}><View style={styles.tipNumber}><Text style={styles.tipNumberText}>{number}</Text></View><View style={styles.tipCopy}><Text style={styles.tipTitle}>{title}</Text><Text style={styles.tipBody}>{body}</Text></View></View>;
}

function Recorder({ scanType, onClose, onRecorded }: { scanType: ScanType; onClose: () => void; onRecorded: (capture: RecordedCapture) => void }) {
  const maxSeconds = MAX_CAPTURE_SECONDS[scanType];
  useKeepAwake("object-scan");
  const insets = useSafeAreaInsets();
  const camera = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [recordingError, setRecordingError] = useState<string>();
  // Leaving mid-recording would orphan the camera session, so back is ignored then.
  useBackHandler(() => { if (!recording) onClose(); });

  useEffect(() => {
    if (!recording) return;
    const started = Date.now() - elapsedMs;
    const interval = setInterval(() => setElapsedMs(Math.min(maxSeconds * 1000, Date.now() - started)), 100);
    return () => clearInterval(interval);
  }, [recording]);

  const start = async () => {
    if (!camera.current || !cameraReady || recording) return;
    setRecordingError(undefined);
    const startedAtMs = Date.now();
    const monotonicStartNs = Math.max(0, Math.round(performance.now() * 1_000_000));
    setElapsedMs(0);
    setRecording(true);
    try {
      const result = await camera.current.recordAsync({
        maxDuration: maxSeconds,
        ...(Platform.OS === "ios" ? { codec: "avc1" as const } : {}),
      });
      if (!result?.uri) throw new Error("The camera did not return a video.");
      onRecorded({
        uri: result.uri,
        durationMs: boundedDurationMs(Date.now() - startedAtMs, scanType),
        monotonicStartNs,
        wallClockStart: new Date(startedAtMs).toISOString(),
      });
    } catch (error) {
      setRecordingError(message(error));
    } finally {
      setRecording(false);
    }
  };

  const stop = () => camera.current?.stopRecording();

  if (!permission) return <ProgressScreen title="Opening camera" label="Checking camera permission" progress={0} />;
  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.permission}>
        <Text style={styles.permissionIcon}>◉</Text>
        <Text style={styles.title}>Camera access is required</Text>
        <Text style={styles.lead}>The app records a muted video and uploads it only after you review it.</Text>
        <Button label="Allow camera" onPress={requestPermission} />
        <Button variant="link" label="Cancel" onPress={onClose} />
      </SafeAreaView>
    );
  }

  const elapsedSeconds = elapsedMs / 1000;
  const canStop = elapsedSeconds >= MIN_CAPTURE_SECONDS;
  const remaining = maxSeconds - elapsedSeconds;
  const scene = scanType === "scene";
  return (
    <View style={styles.cameraRoot}>
      <CameraView
        ref={camera}
        style={StyleSheet.absoluteFill}
        facing="back"
        mode="video"
        mute
        videoQuality="1080p"
        videoBitrate={8_000_000}
        videoStabilizationMode="auto"
        onCameraReady={() => setCameraReady(true)}
      />
      <View style={[styles.cameraTop, { paddingTop: insets.top + 8 }]}>
        <View style={styles.cameraTopRow}>
          <Pressable accessibilityRole="button" disabled={recording} onPress={onClose} hitSlop={12} style={styles.cameraActionSlot}>
            <Text style={[styles.cameraAction, recording && styles.disabledText]}>Cancel</Text>
          </Pressable>
          <View style={styles.timerPill}>
            <View style={[styles.recordDot, recording && styles.recordDotActive]} />
            <Text style={styles.timer}>{formatClock(elapsedSeconds)} / {formatClock(maxSeconds)}</Text>
          </View>
          <View style={styles.cameraActionSlot} />
        </View>
        <View style={styles.recordTrack}><View style={[styles.recordFill, { width: `${Math.min(100, (elapsedSeconds / maxSeconds) * 100)}%` }]} /></View>
      </View>
      <View pointerEvents="none" style={[styles.frameGuide, scene && styles.sceneGuide]}>
        {!scene ? <><View style={[styles.corner, styles.cornerTL]} /><View style={[styles.corner, styles.cornerTR]} /><View style={[styles.corner, styles.cornerBL]} /><View style={[styles.corner, styles.cornerBR]} /></> : null}
        <Text style={styles.guideText}>{recording ? (scene ? "Keep walking · never spin in place" : "Move sideways · keep overlap") : (scene ? "Point across the room" : "Keep the object inside the guide")}</Text>
      </View>
      <View style={[styles.cameraBottom, { paddingBottom: insets.bottom + 18 }]}>
        {recordingError ? <Text style={styles.cameraError}>{recordingError}</Text> : null}
        <Text style={styles.cameraHelp}>
          {recording
            ? (!canStop ? "Keep moving slowly" : remaining <= 10 ? `Stopping automatically in ${Math.ceil(remaining)} s` : scene ? "Tap stop when you have covered every wall" : "Tap stop when you have covered the visible sides")
            : !cameraReady ? "Starting the camera…" : scene ? "Start, then walk slowly along the walls" : "Start, then walk slowly in an arc"}
        </Text>
        <Pressable
          accessibilityLabel={recording ? "Stop recording" : "Start recording"}
          accessibilityRole="button"
          disabled={!cameraReady || (recording && !canStop)}
          onPress={recording ? stop : start}
          style={({ pressed }) => [styles.shutterOuter, pressed && styles.shutterPressed, (!cameraReady || (recording && !canStop)) && styles.buttonDisabled]}
        >
          <View style={recording ? styles.stopIcon : styles.shutterInner} />
        </Pressable>
      </View>
    </View>
  );
}

const PROCESSING_MODES: { mode: ProcessingMode; label: string; detail: string; help: string }[] = [
  { mode: "detailed", label: "Detailed", detail: "Sharpest", help: "Measures depth from many views for the sharpest edges. Takes the longest: about 10 minutes for a 15-second clip, and more for longer recordings." },
  { mode: "fast", label: "Fast", detail: "Much quicker", help: "Estimates depth from single frames: several times quicker, with softer edges and less accuracy on dark or shiny surfaces. Recommended for rooms." },
];

function Review({ scanType, capture, onRetake, onUpload }: { scanType: ScanType; capture: RecordedCapture; onRetake: () => void; onUpload: (mode: ProcessingMode) => void }) {
  const subject = scanType === "scene" ? "room" : "object";
  const player = useVideoPlayer(capture.uri, (instance) => { instance.loop = true; instance.muted = true; instance.play(); });
  const [mode, setMode] = useState<ProcessingMode>(scanType === "scene" ? "fast" : "detailed");
  const seconds = capture.durationMs / 1000;
  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.review}>
        <Text style={styles.eyebrow}>REVIEW CAPTURE</Text>
        <Text style={styles.title}>Is the {subject} clear and in frame?</Text>
        <VideoView player={player} style={styles.video} contentFit="cover" nativeControls />
        <Text style={styles.reviewDuration}>{formatClock(seconds)} · muted · 1080p</Text>
        <Text style={styles.lead}>Retake if the video is blurry, the {subject} leaves the frame, or you did not move sideways.</Text>
        <Text style={styles.modeHeading}>Processing</Text>
        <Segmented
          accessibilityLabel="Processing"
          options={PROCESSING_MODES.map((option) => ({ value: option.mode, label: option.label, detail: option.detail }))}
          value={mode}
          onChange={setMode}
        />
        <Text style={styles.modeHelp}>{PROCESSING_MODES.find((option) => option.mode === mode)?.help}</Text>
        {seconds > 30 ? <Text style={styles.modeHelp}>Longer recordings use more frames, so they take longer to process.</Text> : null}
        <View style={styles.reviewActions}>
          <Button variant="secondary" label="Retake" onPress={onRetake} />
          <Button label="Upload & reconstruct" onPress={() => onUpload(mode)} style={styles.flexButton} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const STAGES = ["Upload", "Select sharp frames", "Match camera positions", "Build the point cloud", "Prepare the preview"];

/** Index of the stage in progress (STAGES.length once everything is done). */
function stageIndex(status?: ScanStatus): number {
  switch (status) {
    case undefined: case "CREATED": case "UPLOADING": return 0;
    case "UPLOADED": case "QUEUED": case "EXTRACTING_FRAMES": return 1;
    case "SPARSE_RECONSTRUCTION": return 2;
    case "DENSE_RECONSTRUCTION": return 3;
    case "FILTERING": return 4;
    default: return STAGES.length;
  }
}

function ProgressScreen({
  title, label, progress, stage, detail, note, error, onRetry, retryLabel, onBack, backLabel,
}: {
  title: string; label: string; progress: number; stage?: number; detail?: string; note?: string; error?: string;
  onRetry?: () => void; retryLabel?: string; onBack?: () => void; backLabel?: string;
}) {
  const percent = Math.round(Math.max(0, Math.min(1, progress)) * 100);
  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.progressRoot}>
        {!error ? <ActivityIndicator size="large" color={colors.accent} /> : <Text style={styles.failureIcon}>!</Text>}
        <Text style={styles.progressTitle}>{title}</Text>
        <Text style={error ? styles.error : styles.progressLabel}>{error ?? label}</Text>
        {!error ? (
          <View style={styles.progressBlock}>
            <View style={styles.progressTrack}><View style={[styles.progressFill, { width: `${percent}%` }]} /></View>
            <Text style={styles.progressPercent}>{percent}%</Text>
          </View>
        ) : null}
        {stage !== undefined ? (
          <View style={styles.stages}>
            {STAGES.map((name, index) => {
              const state = index < stage ? "done" : index === stage ? "active" : "pending";
              return (
                <View key={name} style={styles.stageRow}>
                  <View style={[styles.stageDot, state === "done" && styles.stageDone, state === "active" && styles.stageActive]}>
                    {state === "done" ? <Text style={styles.stageCheck}>✓</Text> : null}
                  </View>
                  <Text style={[styles.stageName, state === "pending" && styles.stagePending, state === "active" && styles.stageNameActive]}>{name}</Text>
                </View>
              );
            })}
          </View>
        ) : null}
        {note ? <Text style={styles.note}>{note}</Text> : null}
        {detail ? <Text selectable style={styles.scanId}>{detail}</Text> : null}
        <View style={styles.progressActions}>
          {onRetry ? <Button label={retryLabel ?? "Try again"} onPress={onRetry} /> : null}
          {onBack ? <Button variant="link" label={backLabel ?? "Back to home"} onPress={onBack} /> : null}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function formatClock(seconds: number) {
  const whole = Math.max(0, Math.floor(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function formatSeconds(seconds: number) {
  if (seconds < 60) return `${seconds} s`;
  return seconds % 60 === 0 ? `${seconds / 60} min` : `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
}

const CORNER = 28;

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  preparation: { padding: 22, paddingTop: 12, paddingBottom: 36 },
  back: { alignSelf: "flex-start", marginLeft: -12, marginBottom: 16 },
  eyebrow: { color: colors.accent, fontSize: 12, fontWeight: "800", letterSpacing: 1.4, marginBottom: 9 },
  title: { color: colors.text, fontSize: 29, lineHeight: 35, fontWeight: "700", marginBottom: 12 },
  lead: { color: colors.textSecondary, fontSize: 15, lineHeight: 22 },
  tipList: { marginVertical: 26, gap: 20 },
  tip: { flexDirection: "row", gap: 13 },
  tipNumber: { width: 30, height: 30, borderRadius: 15, backgroundColor: "#233529", alignItems: "center", justifyContent: "center" },
  tipNumberText: { color: colors.accent, fontSize: 14, fontWeight: "800" },
  tipCopy: { flex: 1, gap: 3 },
  tipTitle: { color: colors.text, fontSize: 16, fontWeight: "700" },
  tipBody: { color: colors.textMuted, fontSize: 14, lineHeight: 20 },
  flexButton: { flex: 1 },
  permission: { flex: 1, backgroundColor: colors.background, justifyContent: "center", padding: 24, gap: 16 },
  permissionIcon: { color: colors.accent, fontSize: 44, marginBottom: 4 },
  cameraRoot: { flex: 1, backgroundColor: "#000" },
  cameraTop: { backgroundColor: "rgba(0,0,0,0.55)", paddingHorizontal: 16, paddingBottom: 12, gap: 12 },
  cameraTopRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  cameraActionSlot: { width: 72, minHeight: 36, justifyContent: "center" },
  cameraAction: { color: "#fff", fontSize: 15, fontWeight: "700" },
  disabledText: { opacity: 0.35 },
  timerPill: { flexDirection: "row", alignItems: "center", gap: 8, backgroundColor: "rgba(0,0,0,0.6)", paddingHorizontal: 14, paddingVertical: 7, borderRadius: 18 },
  timer: { color: "#fff", fontSize: 14, fontVariant: ["tabular-nums"], fontWeight: "700" },
  recordDot: { width: 9, height: 9, borderRadius: 5, backgroundColor: colors.textMuted },
  recordDotActive: { backgroundColor: colors.record },
  recordTrack: { height: 3, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.18)", overflow: "hidden" },
  recordFill: { height: "100%", backgroundColor: colors.record },
  frameGuide: { flex: 1, margin: 28, alignItems: "center", justifyContent: "flex-end" },
  sceneGuide: { marginHorizontal: 0 },
  corner: { position: "absolute", width: CORNER, height: CORNER, borderColor: colors.accent },
  cornerTL: { top: 0, left: 0, borderTopWidth: 3, borderLeftWidth: 3, borderTopLeftRadius: 12 },
  cornerTR: { top: 0, right: 0, borderTopWidth: 3, borderRightWidth: 3, borderTopRightRadius: 12 },
  cornerBL: { bottom: 0, left: 0, borderBottomWidth: 3, borderLeftWidth: 3, borderBottomLeftRadius: 12 },
  cornerBR: { bottom: 0, right: 0, borderBottomWidth: 3, borderRightWidth: 3, borderBottomRightRadius: 12 },
  guideText: { color: "#fff", backgroundColor: "rgba(0,0,0,0.65)", paddingHorizontal: 14, paddingVertical: 8, marginBottom: 14, borderRadius: 16, fontSize: 13, fontWeight: "700", overflow: "hidden" },
  cameraBottom: { backgroundColor: "rgba(0,0,0,0.55)", alignItems: "center", paddingTop: 16, gap: 14 },
  cameraHelp: { color: "#fff", fontSize: 14, textAlign: "center", paddingHorizontal: 24 },
  cameraError: { color: colors.danger, fontSize: 13, textAlign: "center", paddingHorizontal: 20 },
  shutterOuter: { width: 78, height: 78, borderRadius: 39, borderWidth: 4, borderColor: "#fff", alignItems: "center", justifyContent: "center" },
  shutterPressed: { transform: [{ scale: 0.94 }] },
  shutterInner: { width: 62, height: 62, borderRadius: 31, backgroundColor: colors.record },
  stopIcon: { width: 30, height: 30, borderRadius: 6, backgroundColor: colors.record },
  buttonDisabled: { opacity: 0.45 },
  review: { padding: 22, paddingBottom: 36 },
  video: { alignSelf: "center", width: "100%", aspectRatio: 9 / 16, maxHeight: 440, backgroundColor: "#000", borderRadius: radius.large, marginTop: 10, overflow: "hidden" },
  reviewDuration: { color: colors.accent, fontSize: 13, fontWeight: "700", marginVertical: 12, textAlign: "center", fontVariant: ["tabular-nums"] },
  reviewActions: { flexDirection: "row", gap: 10, marginTop: 24 },
  modeHeading: { color: "#d7dfd9", fontSize: 13, fontWeight: "800", marginTop: 20, marginBottom: 8, letterSpacing: 0.5 },
  scanTypeLead: { marginTop: 14 },
  modeHelp: { color: colors.textSecondary, fontSize: 13, lineHeight: 19, marginTop: 8 },
  progressRoot: { flexGrow: 1, justifyContent: "center", alignItems: "center", padding: 28, gap: 14 },
  progressTitle: { color: colors.text, fontSize: 26, lineHeight: 32, fontWeight: "700", textAlign: "center" },
  progressLabel: { color: colors.textSecondary, fontSize: 15, textAlign: "center" },
  progressBlock: { width: "100%", maxWidth: 360, alignItems: "center", gap: 10, marginTop: 6 },
  progressTrack: { width: "100%", height: 8, borderRadius: 4, backgroundColor: colors.border, overflow: "hidden" },
  progressFill: { height: "100%", backgroundColor: colors.accent, borderRadius: 4 },
  progressPercent: { color: colors.accent, fontSize: 13, fontWeight: "800", fontVariant: ["tabular-nums"] },
  stages: { width: "100%", maxWidth: 360, marginTop: 10, padding: 16, gap: 12, backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, borderRadius: radius.large },
  stageRow: { flexDirection: "row", alignItems: "center", gap: 12 },
  stageDot: { width: 22, height: 22, borderRadius: 11, borderWidth: 2, borderColor: colors.borderStrong, alignItems: "center", justifyContent: "center" },
  stageDone: { backgroundColor: colors.accent, borderColor: colors.accent },
  stageActive: { borderColor: colors.accent },
  stageCheck: { color: colors.onAccent, fontSize: 12, fontWeight: "900", lineHeight: 14 },
  stageName: { color: "#d7dfd9", fontSize: 14, fontWeight: "600" },
  stageNameActive: { color: colors.text, fontWeight: "800" },
  stagePending: { color: colors.textFaint },
  note: { color: colors.textMuted, fontSize: 13, lineHeight: 19, textAlign: "center", maxWidth: 340 },
  scanId: { color: colors.textFaint, fontSize: 12, textAlign: "center" },
  progressActions: { width: "100%", maxWidth: 360, gap: 4, marginTop: 6 },
  error: { color: colors.danger, fontSize: 14, lineHeight: 20, textAlign: "center", marginBottom: 6 },
  failureIcon: { width: 54, height: 54, borderRadius: 27, backgroundColor: colors.dangerSurface, color: colors.danger, fontSize: 34, fontWeight: "800", textAlign: "center", lineHeight: 54, overflow: "hidden" },
});
