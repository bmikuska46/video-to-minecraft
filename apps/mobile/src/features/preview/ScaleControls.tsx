import { StyleSheet, Text, TextInput, View } from "react-native";

import type { Axis, ExportEstimate } from "@/api/client";
import { Segmented } from "@/ui/Segmented";
import { colors, radius } from "@/ui/theme";

const AXIS_OPTIONS: { value: Axis; label: string; detail: string }[] = [
  { value: "x", label: "Width", detail: "X" },
  { value: "y", label: "Height", detail: "Y" },
  { value: "z", label: "Depth", detail: "Z" },
];

export function ScaleControls({
  axis, blocks, estimate, error, disabled, onAxis, onBlocks,
}: {
  axis: Axis;
  blocks: string;
  estimate?: ExportEstimate;
  error?: string;
  disabled: boolean;
  onAxis: (axis: Axis) => void;
  onBlocks: (blocks: string) => void;
}) {
  return (
    <View style={[styles.group, disabled && styles.disabled]}>
      <Text style={styles.heading}>Minecraft scale</Text>
      <Text style={styles.help}>Pick a side of the model and how many blocks long it should be. The other sides scale to match.</Text>
      <Segmented accessibilityLabel="Measured side" options={AXIS_OPTIONS} value={axis} onChange={onAxis} disabled={disabled} />
      <TextInput
        accessibilityLabel="Target dimension in blocks"
        editable={!disabled}
        keyboardType="number-pad"
        maxLength={4}
        placeholder="Blocks, for example 80"
        placeholderTextColor={colors.textFaint}
        selectionColor={colors.accent}
        returnKeyType="done"
        value={blocks}
        onChangeText={(value) => onBlocks(value.replace(/[^0-9]/g, ""))}
        style={styles.input}
      />
      {disabled ? <Text style={styles.help}>Save step 1 first.</Text> : null}
      {error ? <Text style={styles.error}>{error}</Text> : null}
      {estimate ? (
        <View style={[styles.estimate, estimate.requiresConfirmation && styles.warning]}>
          <Text style={styles.estimateTitle}>{estimate.blockCountEstimate.toLocaleString()} blocks or fewer</Text>
          <Text style={styles.help}>{estimate.voxelDimensions.join(" × ")} blocks (width × height × depth)</Text>
          {estimate.requiresConfirmation ? <Text style={styles.warningText}>Large build: world generation will take longer, so you will be asked to confirm.</Text> : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  group: { gap: 10 }, disabled: { opacity: 0.55 },
  heading: { color: colors.text, fontWeight: "700", fontSize: 17 },
  help: { color: colors.textSecondary, fontSize: 13, lineHeight: 18 },
  input: { color: colors.text, backgroundColor: colors.surfaceRaised, borderColor: colors.borderStrong, borderWidth: 1, borderRadius: radius.medium, paddingHorizontal: 14, minHeight: 52, fontSize: 17, fontWeight: "700" },
  error: { color: colors.danger, fontSize: 13 },
  estimate: { backgroundColor: "#17291d", borderRadius: radius.small, padding: 12, gap: 3 },
  warning: { backgroundColor: colors.warningSurface },
  estimateTitle: { color: "#e8f8eb", fontWeight: "700", fontSize: 15 },
  warningText: { color: colors.warning, fontSize: 13, marginTop: 4, lineHeight: 18 },
});
