import { useMemo, useRef, useState } from "react";
import { PanResponder, StyleSheet, View, type LayoutChangeEvent } from "react-native";

import { colors } from "./theme";

const THUMB = 28;

type Props = {
  min: number;
  max: number;
  low: number;
  high: number;
  /** Smallest allowed distance between the thumbs, in value units. */
  minGap?: number;
  /** Hide the low thumb; only the high end is adjustable. */
  highOnly?: boolean;
  onChange: (low: number, high: number) => void;
  /** Called with true while a thumb is dragged, so a parent can pause scrolling. */
  onDragging?: (dragging: boolean) => void;
  accessibilityLabel?: string;
};

/** A dual-thumb slider in plain React Native: one track, the kept range highlighted. */
export function RangeSlider({ min, max, low, high, minGap = 0, highOnly, onChange, onDragging, accessibilityLabel }: Props) {
  const [width, setWidth] = useState(0);
  const span = Math.max(max - min, Number.EPSILON);
  const latest = useRef({ low, high, width, min, max, span, minGap, onChange, onDragging });
  latest.current = { low, high, width, min, max, span, minGap, onChange, onDragging };
  const start = useRef(0);

  const responder = (edge: "low" | "high") => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,
    onPanResponderTerminationRequest: () => false,
    onPanResponderGrant: () => {
      start.current = latest.current[edge];
      latest.current.onDragging?.(true);
    },
    onPanResponderMove: (_, gesture) => {
      const s = latest.current;
      if (s.width <= 0) return;
      const value = start.current + (gesture.dx / s.width) * s.span;
      if (edge === "low") s.onChange(Math.min(Math.max(value, s.min), s.high - s.minGap), s.high);
      else s.onChange(s.low, Math.max(Math.min(value, s.max), s.low + s.minGap));
    },
    onPanResponderRelease: () => latest.current.onDragging?.(false),
    onPanResponderTerminate: () => latest.current.onDragging?.(false),
  });
  const lowResponder = useMemo(() => responder("low"), []);
  const highResponder = useMemo(() => responder("high"), []);

  const position = (value: number) => ((value - min) / span) * width;
  const lowX = highOnly ? 0 : position(low);
  const highX = position(high);
  return (
    <View
      accessibilityLabel={accessibilityLabel}
      style={styles.root}
      onLayout={(event: LayoutChangeEvent) => setWidth(event.nativeEvent.layout.width - THUMB)}
    >
      <View style={styles.track} />
      <View style={[styles.fill, { left: THUMB / 2 + lowX, width: Math.max(0, highX - lowX) }]} />
      {!highOnly ? <View {...lowResponder.panHandlers} hitSlop={12} style={[styles.thumb, { left: lowX }]} /> : null}
      <View {...highResponder.panHandlers} hitSlop={12} style={[styles.thumb, { left: highX }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { height: 40, justifyContent: "center" },
  track: { position: "absolute", left: THUMB / 2, right: THUMB / 2, height: 4, borderRadius: 2, backgroundColor: "#34443a" },
  fill: { position: "absolute", height: 4, borderRadius: 2, backgroundColor: colors.accent },
  thumb: { position: "absolute", width: THUMB, height: THUMB, borderRadius: THUMB / 2, backgroundColor: "#d7f7df", borderWidth: 3, borderColor: colors.accent, elevation: 3 },
});
