import type { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, type StyleProp, type ViewStyle } from "react-native";

import { colors, radius } from "./theme";

type Variant = "primary" | "secondary" | "link";

export function Button({
  label, onPress, variant = "primary", disabled, loading, icon, style, accessibilityLabel,
}: {
  label: string;
  onPress?: () => void;
  variant?: Variant;
  disabled?: boolean;
  loading?: boolean;
  icon?: ReactNode;
  style?: StyleProp<ViewStyle>;
  accessibilityLabel?: string;
}) {
  const inactive = disabled || loading;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={accessibilityLabel ?? label}
      accessibilityState={{ disabled: Boolean(inactive), busy: Boolean(loading) }}
      disabled={inactive}
      hitSlop={variant === "link" ? 8 : undefined}
      onPress={onPress}
      style={({ pressed }) => [styles.base, styles[variant], pressed && pressedStyles[variant], disabled && styles.disabled, style]}
    >
      {loading ? <ActivityIndicator color={variant === "primary" ? colors.onAccent : colors.accent} /> : (
        <>
          {icon}
          <Text numberOfLines={1} style={[styles.label, labelStyles[variant]]}>{label}</Text>
        </>
      )}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  base: { minHeight: 52, borderRadius: radius.medium, alignItems: "center", justifyContent: "center", flexDirection: "row", gap: 8, paddingHorizontal: 18 },
  primary: { backgroundColor: colors.accent },
  secondary: { borderWidth: 1, borderColor: colors.borderStrong, backgroundColor: colors.surface },
  link: { minHeight: 44, paddingHorizontal: 12 },
  disabled: { opacity: 0.45 },
  label: { fontSize: 15, fontWeight: "800" },
});

const pressedStyles = StyleSheet.create({
  primary: { backgroundColor: colors.accentPressed },
  secondary: { backgroundColor: colors.surfaceRaised, borderColor: colors.accent },
  link: { opacity: 0.6 },
});

const labelStyles = StyleSheet.create({
  primary: { color: colors.onAccent },
  secondary: { color: "#d7dfd9" },
  link: { color: colors.accent, fontSize: 14, fontWeight: "700" },
});
