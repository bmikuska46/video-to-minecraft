import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber/native";
import { Suspense, useMemo, useRef, useState } from "react";
import { ActivityIndicator, PanResponder, Pressable, StyleSheet, Text, View } from "react-native";
import { BoxGeometry, EdgesGeometry, LineBasicMaterial, LineSegments, Matrix4, Object3D } from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

import type { Bounds } from "@/api/client";
import { colors } from "@/ui/theme";

type CameraPose = { azimuth: number; elevation: number; distance: number };

function CameraRig({ bounds, pose }: { bounds: Bounds; pose: CameraPose }) {
  const { camera } = useThree();
  const center = useMemo(() => bounds.min.map((value, i) => (value + bounds.max[i]!) / 2), [bounds]);
  useFrame(() => {
    const cosElevation = Math.cos(pose.elevation);
    camera.position.set(
      center[0]! + pose.distance * Math.sin(pose.azimuth) * cosElevation,
      center[1]! + pose.distance * Math.sin(pose.elevation),
      center[2]! + pose.distance * Math.cos(pose.azimuth) * cosElevation,
    );
    camera.lookAt(center[0]!, center[1]!, center[2]!);
    camera.updateProjectionMatrix();
  });
  return null;
}

function Model({ url, transform }: { url: string; transform: number[] }) {
  const gltf = useLoader(GLTFLoader, url);
  const object = useMemo(() => {
    const clone = gltf.scene.clone(true);
    clone.matrixAutoUpdate = false;
    clone.matrix.copy(new Matrix4().set(
      transform[0]!, transform[1]!, transform[2]!, transform[3]!,
      transform[4]!, transform[5]!, transform[6]!, transform[7]!,
      transform[8]!, transform[9]!, transform[10]!, transform[11]!,
      transform[12]!, transform[13]!, transform[14]!, transform[15]!,
    ));
    return clone;
  }, [gltf.scene, transform]);
  return <primitive object={object as Object3D} />;
}

function CropBox({ crop }: { crop: Bounds }) {
  const lines = useMemo(() => {
    const size = crop.max.map((value, i) => value - crop.min[i]!) as [number, number, number];
    const geometry = new EdgesGeometry(new BoxGeometry(...size));
    const material = new LineBasicMaterial({ color: "#72d68b", depthTest: false });
    const object = new LineSegments(geometry, material);
    object.renderOrder = 10;
    object.position.set(...crop.min.map((value, i) => value + size[i]! / 2) as [number, number, number]);
    return object;
  }, [crop]);
  return <primitive object={lines} />;
}

export function PointCloudScene({ url, transform, crop, showCrop = true, onInteracting }: { url: string; transform: number[]; crop: Bounds; showCrop?: boolean; onInteracting?: (active: boolean) => void }) {
  const diagonal = Math.hypot(...crop.max.map((value, i) => value - crop.min[i]!));
  const minDistance = Math.max(diagonal * 0.35, 0.01);
  const maxDistance = Math.max(diagonal * 8, 1);
  const initialPose: CameraPose = { azimuth: 0.6, elevation: 0.35, distance: diagonal * 1.3 };
  const [pose, setPose] = useState<CameraPose>(initialPose);
  const gesture = useRef<{ x: number; y: number; distance: number; pose: CameraPose } | undefined>(undefined);
  const responder = useMemo(() => PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,
    // Keep the gesture: the surrounding ScrollView is paused through onInteracting.
    onPanResponderTerminationRequest: () => false,
    onPanResponderRelease: () => onInteracting?.(false),
    onPanResponderTerminate: () => onInteracting?.(false),
    onPanResponderGrant: (event) => {
      onInteracting?.(true);
      const [first, second] = event.nativeEvent.touches;
      const distance = first && second ? Math.hypot(first.pageX - second.pageX, first.pageY - second.pageY) : 0;
      gesture.current = { x: first?.pageX ?? 0, y: first?.pageY ?? 0, distance, pose };
    },
    onPanResponderMove: (event) => {
      const [first, second] = event.nativeEvent.touches;
      const start = gesture.current;
      if (!first || !start) return;
      if (second && start.distance > 0) {
        const distance = Math.hypot(first.pageX - second.pageX, first.pageY - second.pageY);
        setPose({ ...start.pose, distance: Math.min(maxDistance, Math.max(minDistance, start.pose.distance * start.distance / distance)) });
      } else {
        setPose({
          ...start.pose,
          azimuth: start.pose.azimuth - (first.pageX - start.x) * 0.01,
          elevation: Math.max(-1.45, Math.min(1.45, start.pose.elevation + (first.pageY - start.y) * 0.01)),
        });
      }
    },
  }), [maxDistance, minDistance, pose, onInteracting]);

  const zoom = (factor: number) => setPose((current) => ({
    ...current, distance: Math.min(maxDistance, Math.max(minDistance, current.distance * factor)),
  }));

  return (
    <View style={styles.root} {...responder.panHandlers}>
      <Canvas camera={{ fov: 45, near: Math.max(diagonal / 1000, 0.001), far: Math.max(diagonal * 30, 10) }}>
        <color attach="background" args={["#111814"]} />
        <ambientLight intensity={1.8} />
        <directionalLight position={[2, 4, 3]} intensity={2} />
        <CameraRig bounds={crop} pose={pose} />
        <Suspense fallback={null}>
          <Model url={url} transform={transform} />
        </Suspense>
        {showCrop ? <CropBox crop={crop} /> : null}
      </Canvas>
      <View pointerEvents="none" style={styles.loading}><ActivityIndicator color={colors.accent} /></View>
      <View style={styles.zoom}>
        <Pressable accessibilityRole="button" accessibilityLabel="Zoom in" style={({ pressed }) => [styles.zoomButton, pressed && styles.zoomPressed]} onPress={() => zoom(0.8)}><Text style={styles.zoomText}>+</Text></Pressable>
        <Pressable accessibilityRole="button" accessibilityLabel="Zoom out" style={({ pressed }) => [styles.zoomButton, pressed && styles.zoomPressed]} onPress={() => zoom(1.25)}><Text style={styles.zoomText}>−</Text></Pressable>
        <Pressable accessibilityRole="button" accessibilityLabel="Reset view" style={({ pressed }) => [styles.zoomButton, pressed && styles.zoomPressed]} onPress={() => setPose(initialPose)}><Text style={styles.resetText}>⟲</Text></Pressable>
      </View>
      <Text pointerEvents="none" style={styles.hint}>Drag to orbit · pinch to zoom</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { height: 360, overflow: "hidden", backgroundColor: "#111814" },
  loading: { ...StyleSheet.absoluteFill, alignItems: "center", justifyContent: "center", zIndex: -1 },
  zoom: { position: "absolute", right: 12, top: 12, gap: 8 },
  zoomButton: { width: 44, height: 44, borderRadius: 22, backgroundColor: "#243128e6", borderWidth: 1, borderColor: "#3a4a3f", alignItems: "center", justifyContent: "center" },
  zoomPressed: { backgroundColor: colors.accentMuted, borderColor: colors.accent },
  zoomText: { color: colors.text, fontSize: 24, lineHeight: 28, fontWeight: "500" },
  resetText: { color: colors.text, fontSize: 20, lineHeight: 24 },
  hint: { position: "absolute", bottom: 10, alignSelf: "center", color: "#dce4dd", backgroundColor: "#0c110dcc", paddingHorizontal: 10, paddingVertical: 5, borderRadius: 12, fontSize: 12 },
});
