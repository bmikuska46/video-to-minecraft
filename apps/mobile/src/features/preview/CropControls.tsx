import { StyleSheet, Text, View } from "react-native";

import type { Bounds, Vector3 } from "@/api/client";
import { RangeSlider } from "@/ui/RangeSlider";
import { colors } from "@/ui/theme";
import { normalizedCrop } from "./geometry";

const AXES = [
  { name: "X", label: "Left – right" },
  { name: "Y", label: "Bottom – top" },
  { name: "Z", label: "Front – back" },
];

type Props = {
  available: Bounds;
  crop: Bounds;
  onChange: (crop: Bounds) => void;
  onDragging?: (dragging: boolean) => void;
};

const keptPercent = (available: Bounds, crop: Bounds, index: number) => {
  const range = available.max[index]! - available.min[index]!;
  return range > 0 ? Math.round(((crop.max[index]! - crop.min[index]!) / range) * 100) : 100;
};

export function CropControls({ available, crop, onChange, onDragging }: Props) {
  const change = (index: number, low: number, high: number) => {
    const next: Bounds = { min: [...crop.min] as Vector3, max: [...crop.max] as Vector3 };
    next.min[index] = low;
    next.max[index] = high;
    onChange(normalizedCrop(available, next));
  };

  return (
    <View style={styles.group}>
      <Text style={styles.heading}>Crop to your object</Text>
      <Text style={styles.help}>Drag both ends of each bar until the green box holds only what you want to build.</Text>
      {AXES.map((axis, index) => {
        const range = available.max[index]! - available.min[index]!;
        return (
          <View key={axis.name} style={styles.axis}>
            <View style={styles.axisHeader}>
              <Text style={styles.axisName}>{axis.name}</Text>
              <Text style={styles.axisLabel}>{axis.label}</Text>
              <Text style={styles.axisValue}>{keptPercent(available, crop, index)}% kept</Text>
            </View>
            <RangeSlider
              accessibilityLabel={`${axis.label} crop`}
              min={available.min[index]!}
              max={available.max[index]!}
              low={crop.min[index]!}
              high={crop.max[index]!}
              minGap={range / 50}
              onChange={(low, high) => change(index, low, high)}
              onDragging={onDragging}
            />
          </View>
        );
      })}
    </View>
  );
}

/** Room/scene scans keep everything; the only optional trim lowers the top to remove a ceiling. */
export function CeilingControl({ available, crop, onChange, onDragging }: Props) {
  const range = available.max[1]! - available.min[1]!;
  const lowered = crop.max[1]! < available.max[1]! - range / 400;
  return (
    <View style={styles.group}>
      <Text style={styles.heading}>Everything is kept</Text>
      <Text style={styles.help}>Nothing is cut out of a room scan. After rotating it upright, lower the top only if you want to remove the ceiling and see inside.</Text>
      <View style={styles.axis}>
        <View style={styles.axisHeader}>
          <Text style={styles.axisName}>Y</Text>
          <Text style={styles.axisLabel}>Height</Text>
          <Text style={styles.axisValue}>{lowered ? `Ceiling removed · ${keptPercent(available, crop, 1)}%` : "Full height"}</Text>
        </View>
        <RangeSlider
          accessibilityLabel="Top of the scan"
          highOnly
          min={available.min[1]!}
          max={available.max[1]!}
          low={available.min[1]!}
          high={crop.max[1]!}
          minGap={range / 20}
          onChange={(_, high) => onChange({ min: [...available.min] as Vector3, max: [available.max[0], high, available.max[2]] })}
          onDragging={onDragging}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  group: { gap: 8 },
  heading: { color: colors.text, fontWeight: "700", fontSize: 17 },
  help: { color: colors.textSecondary, fontSize: 13, lineHeight: 18 },
  axis: { marginTop: 4 },
  axisHeader: { flexDirection: "row", alignItems: "baseline", gap: 8 },
  axisName: { color: colors.accent, fontWeight: "800", fontSize: 13, width: 14 },
  axisLabel: { color: "#d7dfd9", fontSize: 13, fontWeight: "600", flex: 1 },
  axisValue: { color: colors.textMuted, fontSize: 12, fontVariant: ["tabular-nums"] },
});
