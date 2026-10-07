package dev.minecraftvideo.contracts;

import com.github.luben.zstd.ZstdInputStream;
import dev.minecraftvideo.contracts.v1.Bounds;
import dev.minecraftvideo.contracts.v1.Voxel;
import dev.minecraftvideo.contracts.v1.VoxelWorld;
import java.io.IOException;
import java.io.InputStream;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Set;

/** Strict reader shared by the Paper generator and contract compatibility tests. */
public final class VoxelWorldReader {
  public static final int SCHEMA_VERSION = 1;

  private static final Comparator<Voxel> CANONICAL_ORDER = Comparator
      .comparingInt((Voxel voxel) -> Math.floorDiv(voxel.getX(), 16))
      .thenComparingInt(voxel -> Math.floorDiv(voxel.getZ(), 16))
      .thenComparingInt(voxel -> Math.floorDiv(voxel.getY(), 16))
      .thenComparingInt(Voxel::getY)
      .thenComparingInt(Voxel::getZ)
      .thenComparingInt(Voxel::getX);

  private VoxelWorldReader() {}

  public static VoxelWorld read(InputStream compressed) throws IOException {
    final VoxelWorld world;
    try (var zstandard = new ZstdInputStream(compressed)) {
      world = VoxelWorld.parseFrom(zstandard);
    }
    validate(world);
    return world;
  }

  static void validate(VoxelWorld world) throws IOException {
    if (world.getSchemaVersion() != SCHEMA_VERSION) {
      throw new IOException("unsupported voxel schema version: " + world.getSchemaVersion());
    }
    if (world.getMinecraftVersion().isBlank() || world.getVoxelsCount() == 0 || !world.hasBounds()) {
      throw new IOException("voxel contract is missing required semantic fields");
    }
    Set<Integer> paletteIds = new HashSet<>();
    for (var entry : world.getPaletteList()) {
      if (entry.getId() == 0 || entry.getBlockState().isBlank() || !paletteIds.add(entry.getId())) {
        throw new IOException("voxel contract contains an invalid palette");
      }
    }
    Voxel previous = null;
    for (Voxel voxel : world.getVoxelsList()) {
      if (!paletteIds.contains(voxel.getPaletteId())) {
        throw new IOException("voxel contract references an unknown palette ID");
      }
      if (previous != null && CANONICAL_ORDER.compare(previous, voxel) > 0) {
        throw new IOException("voxels are not in canonical chunk/section order");
      }
      previous = voxel;
    }
    Bounds bounds = world.getBounds();
    if (bounds.getMinX() > bounds.getMaxX()
        || bounds.getMinY() > bounds.getMaxY()
        || bounds.getMinZ() > bounds.getMaxZ()) {
      throw new IOException("voxel contract contains invalid bounds");
    }
  }
}
