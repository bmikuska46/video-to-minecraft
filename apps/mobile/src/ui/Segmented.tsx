import { Pressable, StyleSheet, Text, View } from "react-native";

import { colors, radius } from "./theme";

export type SegmentOption<T extends string> = { value: T; label: string; detail?: string };

/** A full-width row of mutually exclusive options; each may carry a one-line detail. */
export function Segmented<T extends string>({
  options, value, onChange, disabled, accessibilityLabel,
}: {
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  accessibilityLabel?: string;
}) {
  return (
    <View accessibilityRole="radiogroup" accessibilityLabel={accessibilityLabel} style={styles.row}>
      {options.map((option) => {
        const selected = option.value === value;
        return (
          <Pressable
            key={option.value}
            accessibilityRole="radio"
            accessibilityState={{ selected, disabled: Boolean(disabled) }}
            disabled={disabled}
            onPress={() => onChange(option.value)}
            style={({ pressed }) => [styles.option, selected && styles.selected, pressed && !selected && styles.pressed]}
          >
            <Text style={[styles.label, selected && styles.selectedLabel]}>{option.label}</Text>
            {option.detail ? <Text style={[styles.detail, selected && styles.selectedDetail]}>{option.detail}</Text> : null}
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  row: { flexDirection: "row", gap: 8 },
  option: { flex: 1, minHeight: 48, alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: colors.borderStrong, borderRadius: radius.small, paddingVertical: 9, paddingHorizontal: 8, gap: 2 },
  selected: { borderColor: colors.accent, backgroundColor: colors.accentMuted },
  pressed: { backgroundColor: colors.surfaceRaised },
  label: { color: colors.textSecondary, fontSize: 15, fontWeight: "700" },
  selectedLabel: { color: "#d7f7df" },
  detail: { color: colors.textFaint, fontSize: 12, fontWeight: "600" },
  selectedDetail: { color: colors.accent },
});
