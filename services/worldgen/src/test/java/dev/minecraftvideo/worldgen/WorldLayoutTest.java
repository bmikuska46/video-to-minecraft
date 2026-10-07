package dev.minecraftvideo.worldgen;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.minecraftvideo.contracts.v1.Bounds;
import dev.minecraftvideo.contracts.v1.Voxel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import org.junit.jupiter.api.Test;

final class WorldLayoutTest {
  private static JobManifest.Payload manifest(int margin) {
    return new JobManifest.Payload(
        1, "550e8400-e29b-41d4-a716-446655440000", "26.2", "a".repeat(64),
        "minecraft-video-world", 64, margin, "minecraft:smooth_stone", 20_000,
        1_000_000, 4_000_000);
  }

  @Test
  void translatesMinimumVoxelDirectlyAbovePlatform() {
    Bounds bounds = Bounds.newBuilder()
        .setMinX(-3).setMinY(-7).setMinZ(10)
        .setMaxX(6).setMaxY(12).setMaxZ(14).build();
    WorldLayout layout = WorldLayout.create(bounds, manifest(8));
    Voxel minimum = Voxel.newBuilder().setX(-3).setY(-7).setZ(10).build();

    assertEquals(0, layout.x(minimum));
    assertEquals(65, layout.y(minimum));
    assertEquals(0, layout.z(minimum));
    assertEquals(-8, layout.platformMinX());
    assertEquals(17, layout.platformMaxX());
    assertEquals(546, layout.platformBlockCount());
  }

  @Test
  void platformCursorVisitsEveryCoordinateOnceInChunkOrder() {
    Bounds bounds = Bounds.newBuilder()
        .setMinX(0).setMinY(0).setMinZ(0)
        .setMaxX(20).setMaxY(0).setMaxZ(18).build();
    WorldLayout layout = WorldLayout.create(bounds, manifest(2));
    PlatformCursor cursor = new PlatformCursor(layout);
    var coordinates = new ArrayList<PlatformCursor.Coordinate>();
    while (cursor.hasNext()) coordinates.add(cursor.next());

    assertEquals(layout.platformBlockCount(), coordinates.size());
    assertEquals(coordinates.size(), new HashSet<>(coordinates).size());
    Comparator<PlatformCursor.Coordinate> chunkOrder = Comparator
        .comparingInt((PlatformCursor.Coordinate value) -> Math.floorDiv(value.x(), 16))
        .thenComparingInt(value -> Math.floorDiv(value.z(), 16));
    for (int index = 1; index < coordinates.size(); index++) {
      assertTrue(chunkOrder.compare(coordinates.get(index - 1), coordinates.get(index)) <= 0);
    }
  }
}
