import { useQuery } from "@tanstack/react-query";
import { useRouter } from "expo-router";
import { useEffect, useState } from "react";
import { ActivityIndicator, Keyboard, KeyboardAvoidingView, Platform, ScrollView, StyleSheet, Text, TextInput, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { ApiError, getScan, type ScanStatus } from "@/api/client";
import { CaptureFlow } from "@/features/capture/CaptureFlow";
import { getRecentScanId } from "@/features/capture/recentScan";
import { Button } from "@/ui/Button";
import { GrassBlock } from "@/ui/GrassBlock";
import { colors, radius } from "@/ui/theme";
import { useBackHandler } from "@/ui/useBackHandler";

const SCAN_ID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

const statusLabels: Record<ScanStatus, string> = {
  CREATED: "Waiting for upload",
  UPLOADING: "Uploading capture",
  UPLOADED: "Upload complete",
  QUEUED: "Waiting for reconstruction",
  EXTRACTING_FRAMES: "Selecting video frames",
  SPARSE_RECONSTRUCTION: "Matching camera positions",
  DENSE_RECONSTRUCTION: "Building the point cloud",
  FILTERING: "Preparing your preview",
  PREVIEW_READY: "Preview ready",
  FAILED: "Reconstruction failed",
};

const errorMessage = (error: unknown) => {
  if (error instanceof ApiError && error.status === 404) return "No reconstruction was found for that scan ID.";
  if (error instanceof ApiError) return error.message;
  return "Could not reach the API. Check EXPO_PUBLIC_API_URL and your network connection.";
};

export default function Index() {
  const [screen, setScreen] = useState<"home" | "capture" | "lookup">("home");
  const [recentScanId, setRecentScanId] = useState<string>();
  const [lookupScanId, setLookupScanId] = useState<string>();
  useEffect(() => { void getRecentScanId().then((value) => setRecentScanId(value ?? undefined)).catch(() => undefined); }, []);
  if (screen === "capture") return <CaptureFlow onClose={() => setScreen("home")} />;
  if (screen === "lookup") return <ScanLookup initialScanId={lookupScanId} onClose={() => setScreen("home")} />;
  return <Home
    recentScanId={recentScanId}
    onCapture={() => setScreen("capture")}
    onLookup={() => { setLookupScanId(undefined); setScreen("lookup"); }}
    onResume={() => { setLookupScanId(recentScanId); setScreen("lookup"); }}
  />;
}

function Home({ recentScanId, onCapture, onLookup, onResume }: { recentScanId?: string; onCapture: () => void; onLookup: () => void; onResume: () => void }) {
  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.home}>
        <View>
          <Text style={styles.eyebrow}>VIDEO TO MINECRAFT</Text>
          <Text style={styles.heroTitle}>Turn a real object or room into a Minecraft build</Text>
          <Text style={styles.body}>Walk around an object or through a room with your phone. The server rebuilds it in 3D, then you orient it, pick a size, and download a world made of real blocks.</Text>
        </View>
        <View style={styles.heroGraphic}><GrassBlock size={88} /></View>
        <View style={styles.homeActions}>
          <Button label="Start a scan" onPress={onCapture} />
          {recentScanId ? <Button variant="secondary" label="Resume last scan" onPress={onResume} /> : null}
          <Button variant="secondary" label="Open existing scan" onPress={onLookup} />
          <Text style={styles.caption}>Needs the Video to Minecraft server running on your network.</Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function ScanLookup({ initialScanId, onClose }: { initialScanId?: string; onClose: () => void }) {
  const router = useRouter();
  const [scanId, setScanId] = useState(initialScanId ?? "");
  const [submittedId, setSubmittedId] = useState(initialScanId ?? "");
  const normalizedId = scanId.trim();
  const validId = SCAN_ID_PATTERN.test(normalizedId);
  const scan = useQuery({
    queryKey: ["scan", submittedId],
    queryFn: () => getScan(submittedId),
    enabled: Boolean(submittedId),
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

  const openScan = () => {
    if (!validId) return;
    Keyboard.dismiss();
    setSubmittedId(normalizedId);
  };
  const status = scan.data?.status;
  const progress = scan.data ? Math.round(Math.max(0, Math.min(1, scan.data.progress)) * 100) : 0;
  useBackHandler(onClose);

  return (
    <SafeAreaView style={styles.safe}>
      <KeyboardAvoidingView style={styles.safe} behavior="padding">
      <ScrollView keyboardShouldPersistTaps="handled" contentContainerStyle={styles.lookup}>
        <Button variant="link" label="‹ Home" onPress={onClose} style={styles.back} />
        <View style={styles.lookupBody}>
        <Text style={styles.eyebrow}>EXISTING SCAN</Text>
        <Text style={styles.title}>Open your reconstruction</Text>
        <Text style={styles.body}>Paste a scan ID to resume processing or open its preview.</Text>

        <View style={styles.panel}>
          <Text style={styles.label}>Scan ID</Text>
          <TextInput
            accessibilityLabel="Scan ID"
            autoCapitalize="none"
            autoCorrect={false}
            onChangeText={(value) => { setScanId(value); if (submittedId) setSubmittedId(""); }}
            onSubmitEditing={openScan}
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
            placeholderTextColor={colors.textFaint}
            returnKeyType="go"
            selectionColor={colors.accent}
            style={styles.input}
            value={scanId}
          />
          {normalizedId && !validId ? <Text style={styles.validation}>Enter a complete UUID scan ID.</Text> : null}
          <Button label="Open reconstruction" disabled={!validId} loading={scan.isFetching && !scan.data} onPress={openScan} />
          {scan.isError ? <Text style={styles.error}>{errorMessage(scan.error)}</Text> : null}
          {status && status !== "PREVIEW_READY" ? (
            <View style={styles.statusCard}>
              <View style={styles.statusRow}>
                {status !== "FAILED" ? <ActivityIndicator color="#72d68b" /> : null}
                <Text style={[styles.statusTitle, status === "FAILED" && styles.error]}>{statusLabels[status]}</Text>
                {status !== "FAILED" ? <Text style={styles.percent}>{progress}%</Text> : null}
              </View>
              <Text style={styles.statusDetail}>{status === "FAILED" ? `Reason: ${scan.data?.failureCode ?? "UNKNOWN"}` : "Status updates every few seconds."}</Text>
            </View>
          ) : null}
        </View>
        </View>
      </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: colors.background },
  home: { flexGrow: 1, justifyContent: "space-between", padding: 24, paddingTop: 40, paddingBottom: 28, gap: 24 },
  lookup: { flexGrow: 1, padding: 22, paddingTop: 12 },
  lookupBody: { flexGrow: 1, justifyContent: "center", paddingBottom: 48 },
  back: { alignSelf: "flex-start", marginLeft: -12, marginBottom: 16 },
  eyebrow: { color: colors.accent, fontSize: 12, fontWeight: "800", letterSpacing: 1.4, marginBottom: 9 },
  heroTitle: { color: colors.text, fontSize: 34, lineHeight: 40, fontWeight: "800", marginBottom: 14 },
  title: { color: colors.text, fontSize: 30, lineHeight: 36, fontWeight: "700", marginBottom: 12 },
  body: { color: colors.textSecondary, fontSize: 16, lineHeight: 23 },
  heroGraphic: { flexGrow: 1, minHeight: 200, alignItems: "center", justifyContent: "center" },
  homeActions: { gap: 11 },
  caption: { color: colors.textFaint, fontSize: 12, lineHeight: 17, textAlign: "center", marginTop: 3 },
  panel: { marginTop: 28, padding: 18, backgroundColor: colors.surface, borderColor: colors.border, borderWidth: 1, borderRadius: radius.large, gap: 12 },
  label: { color: "#d7dfd9", fontSize: 14, fontWeight: "700" },
  input: { minHeight: 52, borderWidth: 1, borderColor: colors.borderStrong, borderRadius: radius.medium, paddingHorizontal: 14, color: colors.text, fontSize: 15, backgroundColor: colors.surfaceRaised },
  validation: { color: colors.warning, fontSize: 13 },
  error: { color: colors.danger, fontSize: 14, lineHeight: 20 },
  statusCard: { marginTop: 4, padding: 13, backgroundColor: colors.background, borderRadius: radius.small, gap: 8 },
  statusRow: { flexDirection: "row", alignItems: "center", gap: 10 },
  statusTitle: { color: colors.text, fontSize: 14, fontWeight: "700", flex: 1 },
  statusDetail: { color: colors.textMuted, fontSize: 13, lineHeight: 18 },
  percent: { color: colors.accent, fontSize: 13, fontWeight: "800" },
});
