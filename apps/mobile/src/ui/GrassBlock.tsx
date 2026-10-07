import { useEffect, useRef, type ReactNode } from "react";
import { Animated, Easing, StyleSheet, View } from "react-native";

// 4x4 pixel textures, row-major. Letters index into the palettes below.
const TOP = "abca" + "cbab" + "bacb" + "acba";
const SIDE = "gggg" + "dgdg" + "eded" + "dfed";
const TOP_COLORS: Record<string, string> = { a: "#7fd36a", b: "#6cc25a", c: "#8fe07a", base: "#78cc64" };
const SIDE_COLORS: Record<string, string> = { g: "#6cc25a", d: "#8b5e3c", e: "#7a5032", f: "#9c6b45", base: "#835839" };
const COS30 = Math.cos(Math.PI / 6);

function Face({ texture, palette, shade }: { texture: string; palette: Record<string, string>; shade: number }) {
  // The base colour fills the hairline gaps anti-aliasing leaves between pixels.
  return (
    <View style={[styles.face, { backgroundColor: palette.base }]}>
      {[...texture].map((key, index) => <View key={index} style={[styles.pixel, { backgroundColor: palette[key] }]} />)}
      <View pointerEvents="none" style={[StyleSheet.absoluteFill, { backgroundColor: `rgba(0,0,0,${shade})` }]} />
    </View>
  );
}

/**
 * Draws a size x size square mapped by the 2x2 matrix [[a, b], [c, d]] (columns are
 * where the square's x and y edges end up) and centred on (x, y). Android views
 * cannot skew, so the map is split by SVD into rotate·scale·rotate across two
 * nested views, which Android composes exactly.
 */
function Projected({ matrix: [a, b, c, d], x, y, size, children }: { matrix: [number, number, number, number]; x: number; y: number; size: number; children: ReactNode }) {
  const e = (a + d) / 2, f = (a - d) / 2, g = (c + b) / 2, h = (c - b) / 2;
  const q = Math.hypot(e, h), r = Math.hypot(f, g);
  const a1 = Math.atan2(g, f), a2 = Math.atan2(h, e);
  return (
    <View style={{ position: "absolute", left: x - size / 2, top: y - size / 2, width: size, height: size, transform: [{ rotate: `${(a2 + a1) / 2}rad` }, { scaleX: q + r }, { scaleY: q - r }] }}>
      <View style={{ flex: 1, transform: [{ rotate: `${(a2 - a1) / 2}rad` }] }}>{children}</View>
    </View>
  );
}

/** An isometric Minecraft grass block drawn with plain views, gently floating. */
export function GrassBlock({ size = 96 }: { size?: number }) {
  const float = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    const loop = Animated.loop(Animated.sequence([
      Animated.timing(float, { toValue: 1, duration: 1800, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
      Animated.timing(float, { toValue: 0, duration: 1800, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
    ]));
    loop.start();
    return () => loop.stop();
  }, [float]);

  const a = size;
  const width = 2 * a * COS30;
  const translateY = float.interpolate({ inputRange: [0, 1], outputRange: [0, -a * 0.1] });
  const shadowScale = float.interpolate({ inputRange: [0, 1], outputRange: [1, 0.82] });
  return (
    <View accessibilityElementsHidden importantForAccessibility="no-hide-descendants" style={{ width, height: a * 2.3 }}>
      <Animated.View style={[styles.shadow, { left: width * 0.1, top: a * 2.05 - width * 0.4, width: width * 0.8, height: width * 0.8, borderRadius: width * 0.4, transform: [{ scaleY: 0.16 }, { scaleX: shadowScale }] }]} />
      <Animated.View style={{ width, height: a * 2, transform: [{ translateY }] }}>
        <Projected matrix={[COS30, COS30, -0.5, 0.5]} x={a * COS30} y={a * 0.5} size={a * 1.02}>
          <Face texture={TOP} palette={TOP_COLORS} shade={0} />
        </Projected>
        <Projected matrix={[COS30, 0, 0.5, 1]} x={a * COS30 / 2} y={a * 1.25} size={a * 1.02}>
          <Face texture={SIDE} palette={SIDE_COLORS} shade={0.14} />
        </Projected>
        <Projected matrix={[COS30, 0, -0.5, 1]} x={a * COS30 * 1.5} y={a * 1.25} size={a * 1.02}>
          <Face texture={SIDE} palette={SIDE_COLORS} shade={0.34} />
        </Projected>
      </Animated.View>
    </View>
  );
}

const styles = StyleSheet.create({
  face: { flex: 1, flexDirection: "row", flexWrap: "wrap" },
  pixel: { width: "25%", height: "25%" },
  shadow: { position: "absolute", backgroundColor: "rgba(0,0,0,0.55)" },
});
