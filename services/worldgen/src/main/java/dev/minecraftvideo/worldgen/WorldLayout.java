package dev.minecraftvideo.worldgen;

import dev.minecraftvideo.contracts.v1.Bounds;
import dev.minecraftvideo.contracts.v1.Voxel;

/** Pure coordinate transform shared by generation and validation. */
public record WorldLayout(
    int shiftX,
    int shiftY,
    int shiftZ,
    int platformMinX,
    int platformMaxX,
    int platformMinZ,
    int platformMaxZ,
    int platformY,
    int spawnX,
    int spawnY,
    int spawnZ) {

  public static WorldLayout create(Bounds bounds, JobManifest.Payload manifest) {
    int width = Math.addExact(Math.subtractExact(bounds.getMaxX(), bounds.getMinX()), 1);
    int depth = Math.addExact(Math.subtractExact(bounds.getMaxZ(), bounds.getMinZ()), 1);
    int margin = manifest.platformMargin();
    int shiftX = Math.negateExact(bounds.getMinX());
    int shiftY = Math.subtractExact(Math.addExact(manifest.platformY(), 1), bounds.getMinY());
    int shiftZ = Math.negateExact(bounds.getMinZ());
    int minX = -margin;
    int maxX = Math.addExact(width - 1, margin);
    int minZ = -margin;
    int maxZ = Math.addExact(depth - 1, margin);
    int spawnX = Math.floorDiv(width - 1, 2);
    int spawnZ = minZ + Math.max(1, margin / 2);
    return new WorldLayout(
        shiftX, shiftY, shiftZ, minX, maxX, minZ, maxZ, manifest.platformY(),
        spawnX, manifest.platformY() + 1, spawnZ);
  }

  public long platformBlockCount() {
    long width = (long) platformMaxX - platformMinX + 1;
    long depth = (long) platformMaxZ - platformMinZ + 1;
    return Math.multiplyExact(width, depth);
  }

  public int x(Voxel voxel) { return Math.addExact(voxel.getX(), shiftX); }
  public int y(Voxel voxel) { return Math.addExact(voxel.getY(), shiftY); }
  public int z(Voxel voxel) { return Math.addExact(voxel.getZ(), shiftZ); }
}
