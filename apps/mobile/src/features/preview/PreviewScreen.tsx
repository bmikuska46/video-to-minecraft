import { useMutation, useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { useEffect, useMemo, useState } from "react";
import { ActivityIndicator, Keyboard, KeyboardAvoidingView, Linking, Platform, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { ApiError, createExport, estimateExport, getDownloadUrl, getExport, getPreview, saveSelection, type Axis, type Bounds, type ExportStatus } from "@/api/client";
import { Button } from "@/ui/Button";
import { colors, radius } from "@/ui/theme";
import { useBackHandler } from "@/ui/useBackHandler";
import { CeilingControl, CropControls } from "./CropControls";
import { IDENTITY_TURNS, rigidTransform, transformedBounds, type QuarterTurns } from "./geometry";
import { PointCloudScene } from "./PointCloudScene";
import { ScaleControls } from "./ScaleControls";

const message = (error: unknown) => error instanceof ApiError ? error.message : "Something went wrong. Try again.";

const exportLabels: Record<ExportStatus, string> = {
  QUEUED: "Waiting for the world generator",
  VOXELIZING: "Turning points into blocks",
  GENERATING_WORLD: "Placing blocks in a new world",
  READY: "World ready",
  FAILED: "Export failed",
};

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timeout = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timeout);
  }, [delayMs, value]);
  return debounced;
}

export function PreviewScreen({ scanId }: { scanId: string }) {
  const router = useRouter();
  const preview = useQuery({ queryKey: ["preview", scanId], queryFn: () => getPreview(scanId) });
  const home = () => router.replace("/");
  // Preview is opened with replace(), so without this Android back would close the app.
  useBackHandler(home);

  if (preview.isPending) return <Centered><ActivityIndicator size="large" color={colors.accent} /><Text style={styles.help}>Loading your reconstruction…</Text></Centered>;
  if (preview.isError) {
    return (
      <Centered>
        <Text style={styles.failureIcon}>!</Text>
        <Text style={styles.errorTitle}>Could not open the preview</Text>
        <Text style={styles.error}>{message(preview.error)}</Text>
        <Button label="Try again" onPress={() => preview.refetch()} style={styles.centeredButton} />
        <Button variant="link" label="Back to home" onPress={home} />
      </Centered>
    );
  }
  return <Editor scanId={scanId} preview={preview.data} onHome={home} />;
}

function Editor({ scanId, preview, onHome }: { scanId: string; preview: Awaited<ReturnType<typeof getPreview>>; onHome: () => void }) {
  const scene = preview.scanType === "scene";
  const [turns, setTurns] = useState<QuarterTurns>(IDENTITY_TURNS);
  const transform = useMemo(() => rigidTransform(turns), [turns]);
  const available = useMemo(() => transformedBounds(preview.bounds, transform), [preview.bounds, transform]);
  const [crop, setCrop] = useState<Bounds>(available);
  const [savedFingerprint, setSavedFingerprint] = useState<string>();
  const [axis, setAxis] = useState<Axis>("y");
  const [blocks, setBlocks] = useState("");
  const [dragging, setDragging] = useState(false);
  const debouncedBlocks = useDebouncedValue(blocks, 350);
  const fingerprint = JSON.stringify({ transform, crop });
  const selectionSaved = savedFingerprint === fingerprint;
  const parsedBlocks = Number(debouncedBlocks);
  const validBlocks = Number.isInteger(parsedBlocks) && parsedBlocks > 0;

  const save = useMutation({
    mutationFn: ({ selection }: { selection: { transform: number[]; crop: Bounds }; fingerprint: string }) =>
      saveSelection(scanId, selection),
    onSuccess: (_, variables) => setSavedFingerprint(variables.fingerprint),
  });
  const estimate = useQuery({
    queryKey: ["export-estimate", scanId, fingerprint, axis, parsedBlocks],
    queryFn: () => estimateExport(scanId, axis, parsedBlocks),
    enabled: selectionSaved && validBlocks,
    retry: false,
  });
  const [exportId, setExportId] = useState<string>();
  const create = useMutation({
    mutationFn: () => createExport(scanId, axis, parsedBlocks, estimate.data?.requiresConfirmation ?? false),
    onSuccess: (created) => setExportId(created.id),
  });
  const minecraftExport = useQuery({
    queryKey: ["export", exportId],
    queryFn: () => getExport(exportId!),
    enabled: Boolean(exportId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "READY" || status === "FAILED" ? false : 2500;
    },
  });
  const download = useMutation({
    mutationFn: async () => {
      const signed = await getDownloadUrl(exportId!);
      await Linking.openURL(signed.url);
    },
  });

  const rotate = (axisName: Axis) => {
    const next = { ...turns, [axisName]: turns[axisName] + 1 };
    setTurns(next);
    setCrop(transformedBounds(preview.bounds, rigidTransform(next)));
    setSavedFingerprint(undefined);
  };
  const updateCrop = (next: Bounds) => {
    setCrop(next);
    setSavedFingerprint(undefined);
  };

  return (
    <SafeAreaView style={styles.safe} edges={["top", "left", "right"]}>
      <KeyboardAvoidingView style={styles.safe} behavior="padding">
      <ScrollView contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled" scrollEnabled={!dragging}>
        <View style={styles.header}>
          <Button variant="link" label="‹ Home" onPress={onHome} style={styles.back} />
          <View style={styles.titleRow}>
            <View style={styles.titleBlock}><Text style={styles.eyebrow}>RECONSTRUCTION READY</Text><Text style={styles.title}>Frame your build</Text></View>
            <View style={styles.chip}><Text style={styles.chipText}>{preview.registeredFrames}/{preview.selectedFrames} frames</Text></View>
          </View>
        </View>
        {preview.warnings.map((warning) => <Text key={warning} style={styles.notice}>{warning}</Text>)}
        <View style={styles.viewer}>
          <PointCloudScene url={preview.previewUrl} transform={transform} crop={crop} showCrop={!scene || crop.max[1] < available.max[1]} onInteracting={setDragging} />
        </View>
        <View style={styles.panel}>
          <Text style={styles.step}>STEP 1</Text>
          <Text style={styles.heading}>Rotate upright</Text>
          <Text style={styles.help}>Each tap turns the model 90° around that axis. Stop when the floor is at the bottom.</Text>
          <View style={styles.row}>
            {(["x", "y", "z"] as Axis[]).map((value) => (
              <Pressable
                key={value}
                accessibilityRole="button"
                accessibilityLabel={`Rotate around ${value}`}
                style={({ pressed }) => [styles.rotateButton, pressed && styles.rotatePressed]}
                onPress={() => rotate(value)}
              >
                <Text style={styles.rotateIcon}>↻</Text>
                <Text style={styles.rotateText}>{value.toUpperCase()}</Text>
              </Pressable>
            ))}
          </View>
          <View style={styles.divider} />
          {scene
            ? <CeilingControl available={available} crop={crop} onChange={updateCrop} onDragging={setDragging} />
            : <CropControls available={available} crop={crop} onChange={updateCrop} onDragging={setDragging} />}
          {save.isError ? <Text style={styles.error}>{message(save.error)}</Text> : null}
          <Button
            label={scene ? (selectionSaved ? "✓  Orientation saved" : "Save orientation") : (selectionSaved ? "✓  Crop saved" : "Save crop & orientation")}
            variant={selectionSaved ? "secondary" : "primary"}
            disabled={selectionSaved}
            loading={save.isPending}
            onPress={() => save.mutate({ selection: { transform, crop }, fingerprint })}
          />
        </View>
        <View style={styles.panel}>
          <Text style={styles.step}>STEP 2</Text>
          <ScaleControls
            axis={axis}
            blocks={blocks}
            disabled={!selectionSaved}
            estimate={validBlocks && selectionSaved && blocks === debouncedBlocks ? estimate.data : undefined}
            error={estimate.isError ? message(estimate.error) : undefined}
            onAxis={setAxis}
            onBlocks={setBlocks}
          />
          {estimate.isFetching ? <ActivityIndicator color={colors.accent} /> : null}
          {create.isError ? <Text style={styles.error}>{message(create.error)}</Text> : null}
          {!exportId ? (
            <Button
              label={estimate.data?.requiresConfirmation ? `Confirm & build ${estimate.data.blockCountEstimate.toLocaleString()} blocks` : "Create Minecraft world"}
              disabled={!estimate.data || estimate.isFetching}
              loading={create.isPending}
              onPress={() => { Keyboard.dismiss(); create.mutate(); }}
            />
          ) : minecraftExport.data?.status === "READY" ? (
            <View style={styles.ready}>
              <Text style={styles.success}>✓  World ready</Text>
              <Text style={styles.help}>{minecraftExport.data.occupiedBlockCount?.toLocaleString() ?? "—"} blocks placed. Extract the ZIP into a new folder inside Minecraft's saves folder.</Text>
              {download.isError ? <Text style={styles.error}>{message(download.error)}</Text> : null}
              <Button label="Download world ZIP" loading={download.isPending} onPress={() => download.mutate()} />
            </View>
          ) : minecraftExport.data?.status === "FAILED" ? (
            <View style={styles.ready}>
              <Text style={styles.error}>Export failed: {minecraftExport.data.failureCode ?? "UNKNOWN"}</Text>
              <Button variant="secondary" label="Try a different size" onPress={() => { setExportId(undefined); create.reset(); }} />
            </View>
          ) : (
            <View style={styles.exportProgress}>
              <ActivityIndicator color={colors.accent} />
              <Text style={styles.exportLabel}>{exportLabels[minecraftExport.data?.status ?? "QUEUED"]}</Text>
            </View>
          )}
        </View>
      </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <SafeAreaView style={styles.centered}>{children}</SafeAreaView>;
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  centered: { flex: 1, backgroundColor: colors.background, alignItems: "center", justifyContent: "center", gap: 14, padding: 28 },
  centeredButton: { alignSelf: "stretch" },
  content: { paddingBottom: 40 },
  header: { paddingHorizontal: 18, paddingTop: 4, paddingBottom: 14 },
  back: { alignSelf: "flex-start", marginLeft: -12 },
  titleRow: { flexDirection: "row", alignItems: "flex-end", justifyContent: "space-between", gap: 12 },
  titleBlock: { flexShrink: 1 },
  eyebrow: { color: colors.accent, fontSize: 11, fontWeight: "800", letterSpacing: 1.2, marginBottom: 4 },
  title: { color: colors.text, fontSize: 27, fontWeight: "700" },
  chip: { borderRadius: 12, backgroundColor: colors.surface, borderWidth: 1, borderColor: colors.border, paddingHorizontal: 10, paddingVertical: 5, marginBottom: 4 },
  chipText: { color: colors.textMuted, fontSize: 12, fontWeight: "700", fontVariant: ["tabular-nums"] },
  viewer: { borderTopWidth: 1, borderBottomWidth: 1, borderColor: "#2d3b31" },
  panel: { margin: 14, marginBottom: 0, padding: 16, backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, borderRadius: radius.large, gap: 12 },
  step: { color: colors.accent, fontSize: 11, fontWeight: "800", letterSpacing: 1.2, marginBottom: -6 },
  heading: { color: colors.text, fontWeight: "700", fontSize: 17 },
  help: { color: colors.textSecondary, fontSize: 13, lineHeight: 18 },
  row: { flexDirection: "row", gap: 8 },
  rotateButton: { flex: 1, minHeight: 48, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 6, borderWidth: 1, borderColor: colors.borderStrong, borderRadius: radius.small },
  rotatePressed: { backgroundColor: colors.accentMuted, borderColor: colors.accent },
  rotateIcon: { color: colors.accent, fontSize: 18, fontWeight: "700" },
  rotateText: { color: "#d7dfd9", fontWeight: "800", fontSize: 15 },
  divider: { height: 1, backgroundColor: colors.border, marginVertical: 4 },
  error: { color: colors.danger, fontSize: 14, lineHeight: 20, textAlign: "center" },
  errorTitle: { color: colors.text, fontSize: 22, fontWeight: "700", textAlign: "center" },
  failureIcon: { width: 54, height: 54, borderRadius: 27, backgroundColor: colors.dangerSurface, color: colors.danger, fontSize: 34, fontWeight: "800", textAlign: "center", lineHeight: 54, overflow: "hidden" },
  notice: { color: colors.warning, backgroundColor: colors.warningSurface, marginHorizontal: 14, marginBottom: 10, borderRadius: radius.small, padding: 10, lineHeight: 19 },
  success: { color: colors.accent, fontSize: 16, fontWeight: "800" },
  ready: { gap: 10 },
  exportProgress: { flexDirection: "row", alignItems: "center", gap: 12, minHeight: 52, paddingHorizontal: 4 },
  exportLabel: { color: "#d7dfd9", fontSize: 14, fontWeight: "600" },
});
