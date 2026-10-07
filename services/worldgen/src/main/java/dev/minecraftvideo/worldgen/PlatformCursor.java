package dev.minecraftvideo.worldgen;

/** Iterates a rectangle chunk-by-chunk without allocating one object per platform block. */
final class PlatformCursor {
  record Coordinate(int x, int z) {}

  private final WorldLayout layout;
  private final int minChunkX;
  private final int maxChunkX;
  private final int maxChunkZ;
  private int chunkX;
  private int chunkZ;
  private int x;
  private int z;
  private Coordinate next;

  PlatformCursor(WorldLayout layout) {
    this.layout = layout;
    minChunkX = Math.floorDiv(layout.platformMinX(), 16);
    maxChunkX = Math.floorDiv(layout.platformMaxX(), 16);
    chunkX = minChunkX;
    chunkZ = Math.floorDiv(layout.platformMinZ(), 16);
    maxChunkZ = Math.floorDiv(layout.platformMaxZ(), 16);
    resetCell();
    seek();
  }

  boolean hasNext() { return next != null; }

  Coordinate next() {
    if (next == null) throw new IllegalStateException("platform cursor is exhausted");
    Coordinate result = next;
    z++;
    seek();
    return result;
  }

  private void resetCell() {
    x = Math.max(layout.platformMinX(), chunkX * 16);
    z = Math.max(layout.platformMinZ(), chunkZ * 16);
  }

  private void seek() {
    next = null;
    while (chunkX <= maxChunkX) {
      int chunkMaxX = Math.min(layout.platformMaxX(), chunkX * 16 + 15);
      int chunkMaxZ = Math.min(layout.platformMaxZ(), chunkZ * 16 + 15);
      if (x <= chunkMaxX && z <= chunkMaxZ) {
        next = new Coordinate(x, z);
        return;
      }
      if (z > chunkMaxZ) {
        x++;
        z = Math.max(layout.platformMinZ(), chunkZ * 16);
        continue;
      }
      chunkZ++;
      if (chunkZ > maxChunkZ) {
        chunkX++;
        if (chunkX > maxChunkX) return;
        chunkZ = Math.floorDiv(layout.platformMinZ(), 16);
      }
      resetCell();
    }
  }
}
