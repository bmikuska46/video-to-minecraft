package dev.minecraftvideo.worldgen;

import dev.minecraftvideo.contracts.VoxelWorldReader;
import dev.minecraftvideo.contracts.v1.Voxel;
import dev.minecraftvideo.contracts.v1.VoxelWorld;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.bukkit.Bukkit;
import org.bukkit.Difficulty;
import org.bukkit.GameMode;
import org.bukkit.GameRule;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.block.data.BlockData;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;

/** One-shot Paper plugin used by both the generation and independent validation processes. */
public final class WorldgenPlugin extends JavaPlugin {
  private ProgressReporter progress;
  private JobManifest manifest;
  private VoxelWorld voxels;
  private WorldLayout layout;
  private Map<Integer, BlockData> palette;
  private Path resultPath;

  @Override
  public void onEnable() {
    Path progressPath = configuredPath("minecraftvideo.progress", "worldgen-progress.json");
    resultPath = configuredPath("minecraftvideo.result", "worldgen-result.json");
    progress = new ProgressReporter(progressPath);
    Bukkit.getScheduler().runTask(this, () -> {
      try {
        initialize();
        if ("generate".equals(JobManifest.mode())) generate();
        else validate();
      } catch (Throwable error) {
        fail(error);
      }
    });
  }

  private void initialize() throws Exception {
    Path manifestPath = configuredPath("minecraftvideo.manifest", "job-manifest.json");
    Path voxelPath = configuredPath("minecraftvideo.voxels", "voxels.pb.zst");
    manifest = JobManifest.read(manifestPath);
    if (!sha256(voxelPath).equals(manifest.payload().voxelSha256())) {
      throw new IOException("voxel artifact checksum does not match the signed manifest");
    }
    try (InputStream input = Files.newInputStream(voxelPath)) {
      voxels = VoxelWorldReader.read(input);
    }
    if (!manifest.payload().minecraftVersion().equals(voxels.getMinecraftVersion())) {
      throw new IOException("manifest and voxel Minecraft versions differ");
    }
    if (!manifest.payload().minecraftVersion().equals(Bukkit.getMinecraftVersion())) {
      throw new IOException("Paper version does not match the pinned Minecraft target");
    }
    if (voxels.getVoxelsCount() > manifest.payload().maxBlockCount()) {
      throw new IOException("voxel artifact exceeds maxBlockCount");
    }
    layout = WorldLayout.create(voxels.getBounds(), manifest.payload());
    if (layout.platformBlockCount() > manifest.payload().maxPlatformBlocks()) {
      throw new IOException("platform exceeds maxPlatformBlocks");
    }
    palette = new HashMap<>();
    for (var entry : voxels.getPaletteList()) {
      try {
        palette.put(entry.getId(), Bukkit.createBlockData(entry.getBlockState()));
      } catch (IllegalArgumentException error) {
        throw new IOException("invalid palette block state: " + entry.getBlockState(), error);
      }
    }
    requireUniquePositions();
  }

  private void requireUniquePositions() throws IOException {
    Set<Position> positions = new HashSet<>(Math.max(16, voxels.getVoxelsCount() * 4 / 3));
    for (Voxel voxel : voxels.getVoxelsList()) {
      Position position = new Position(layout.x(voxel), layout.y(voxel), layout.z(voxel));
      if (!positions.add(position)) throw new IOException("voxel artifact contains duplicate coordinates");
    }
  }

  private void generate() throws IOException {
    String worldName = manifest.payload().worldName();
    World world = Bukkit.getWorld(worldName);
    if (world == null) throw new IOException("Paper did not load the isolated output world");
    configureWorld(world);
    validateBuildHeight(world);

    BlockData platformData;
    try {
      platformData = Bukkit.createBlockData(manifest.payload().platformBlockState());
    } catch (IllegalArgumentException error) {
      throw new IOException("invalid platform block state", error);
    }
    List<Voxel> ordered = new ArrayList<>(voxels.getVoxelsList());
    ordered.sort(Comparator
        .comparingInt((Voxel value) -> Math.floorDiv(layout.x(value), 16))
        .thenComparingInt(value -> Math.floorDiv(layout.z(value), 16))
        .thenComparingInt(value -> Math.floorDiv(layout.y(value), 16))
        .thenComparingInt(layout::y)
        .thenComparingInt(layout::z)
        .thenComparingInt(layout::x));
    placeInBatches(world, platformData, ordered);
  }

  private void configureWorld(World world) {
    world.setDifficulty(Difficulty.PEACEFUL);
    world.setGameRule(GameRule.DO_DAYLIGHT_CYCLE, false);
    world.setGameRule(GameRule.DO_WEATHER_CYCLE, false);
    world.setGameRule(GameRule.DO_MOB_SPAWNING, false);
    world.setTime(6000L);
    world.setStorm(false);
    world.setThundering(false);
    world.setSpawnLocation(new Location(
        world, layout.spawnX() + 0.5, layout.spawnY(), layout.spawnZ() + 0.5, 0.0f, 0.0f));
    Bukkit.setDefaultGameMode(GameMode.CREATIVE);
  }

  private void validateBuildHeight(World world) throws IOException {
    int minY = layout.platformY();
    int maxY = Math.addExact(voxels.getBounds().getMaxY(), layout.shiftY());
    if (minY < world.getMinHeight() || maxY >= world.getMaxHeight()) {
      throw new IOException("translated model is outside the target world's build height");
    }
  }

  private void placeInBatches(World world, BlockData platformData, List<Voxel> ordered) {
    PlatformCursor platform = new PlatformCursor(layout);
    long total = layout.platformBlockCount() + ordered.size();
    progress.write("RUNNING", "PLACING_BLOCKS", 0, total, null);
    new BukkitRunnable() {
      private int voxelIndex;
      private long completed;

      @Override
      public void run() {
        try {
          int batch = 0;
          while (batch < manifest.payload().batchSize() && platform.hasNext()) {
            PlatformCursor.Coordinate coordinate = platform.next();
            world.getBlockAt(coordinate.x(), layout.platformY(), coordinate.z())
                .setBlockData(platformData, false);
            batch++;
            completed++;
          }
          while (batch < manifest.payload().batchSize() && voxelIndex < ordered.size()) {
            Voxel voxel = ordered.get(voxelIndex++);
            world.getBlockAt(layout.x(voxel), layout.y(voxel), layout.z(voxel))
                .setBlockData(palette.get(voxel.getPaletteId()), false);
            batch++;
            completed++;
          }
          progress.write("RUNNING", "PLACING_BLOCKS", completed, total, null);
          if (!platform.hasNext() && voxelIndex == ordered.size()) {
            cancel();
            finishGeneration(world, total);
          }
        } catch (Throwable error) {
          cancel();
          fail(error);
        }
      }
    }.runTaskTimer(this, 1L, 1L);
  }

  private void finishGeneration(World world, long total) throws IOException {
    progress.write("RUNNING", "SAVING_WORLD", total, total, null);
    world.save();
    progress.write("SUCCEEDED", "GENERATION_COMPLETE", total, total, null);
    writeResult("SUCCEEDED", "GENERATION_COMPLETE", total, null);
    Bukkit.shutdown();
  }

  private void validate() throws IOException {
    World world = Bukkit.getWorld(manifest.payload().worldName());
    if (world == null) throw new IOException("validation server could not load the generated world");
    validateBuildHeight(world);
    progress.write("RUNNING", "VALIDATING_WORLD", 0, voxels.getVoxelsCount(), null);
    assertDefaults(world);
    BlockData platform = Bukkit.createBlockData(manifest.payload().platformBlockState());
    PlatformCursor cursor = new PlatformCursor(layout);
    while (cursor.hasNext()) {
      PlatformCursor.Coordinate coordinate = cursor.next();
      if (!world.getBlockAt(coordinate.x(), layout.platformY(), coordinate.z()).getBlockData().equals(platform)) {
        throw new IOException("platform validation failed at " + coordinate.x() + "," + coordinate.z());
      }
    }
    long checked = 0;
    for (Voxel voxel : voxels.getVoxelsList()) {
      BlockData actual = world.getBlockAt(layout.x(voxel), layout.y(voxel), layout.z(voxel)).getBlockData();
      if (!actual.equals(palette.get(voxel.getPaletteId()))) {
        throw new IOException("model validation failed at source coordinate "
            + voxel.getX() + "," + voxel.getY() + "," + voxel.getZ());
      }
      checked++;
      if (checked % manifest.payload().batchSize() == 0) {
        progress.write("RUNNING", "VALIDATING_WORLD", checked, voxels.getVoxelsCount(), null);
      }
    }
    world.save();
    progress.write("SUCCEEDED", "VALIDATION_COMPLETE", checked, checked, null);
    writeResult("SUCCEEDED", "VALIDATION_COMPLETE", checked, null);
    Bukkit.shutdown();
  }

  private void assertDefaults(World world) throws IOException {
    if (world.getDifficulty() != Difficulty.PEACEFUL
        || !Boolean.FALSE.equals(world.getGameRuleValue(GameRule.DO_DAYLIGHT_CYCLE))
        || !Boolean.FALSE.equals(world.getGameRuleValue(GameRule.DO_WEATHER_CYCLE))
        || world.hasStorm() || world.isThundering()
        || Bukkit.getDefaultGameMode() != GameMode.CREATIVE) {
      throw new IOException("generated world defaults do not match the export contract");
    }
    Location spawn = world.getSpawnLocation();
    if (spawn.getBlockX() != layout.spawnX() || spawn.getBlockY() != layout.spawnY()
        || spawn.getBlockZ() != layout.spawnZ()) {
      throw new IOException("generated world spawn does not match the export contract");
    }
    if (!world.getBlockAt(layout.platformMinX() - 1, layout.platformY(), layout.platformMinZ()).isEmpty()
        || !world.getBlockAt(layout.platformMinX(), layout.platformY() - 1, layout.platformMinZ()).isEmpty()) {
      throw new IOException("generated world contains terrain outside/below the explicit platform");
    }
  }

  private void fail(Throwable error) {
    getLogger().severe(error.getClass().getSimpleName() + ": " + error.getMessage());
    if (progress != null) progress.write("FAILED", "WORLDGEN", 0, 0, error.getMessage());
    if (resultPath != null) writeResult("FAILED", "WORLDGEN", 0, error.getMessage());
    Bukkit.shutdown();
  }

  private void writeResult(String status, String phase, long checked, String error) {
    new ProgressReporter(resultPath).write(status, phase, checked, checked, error);
  }

  private static Path configuredPath(String property, String fallback) {
    return Path.of(System.getProperty(property, fallback)).toAbsolutePath().normalize();
  }

  private static String sha256(Path path) throws Exception {
    MessageDigest digest = MessageDigest.getInstance("SHA-256");
    try (InputStream input = Files.newInputStream(path)) {
      byte[] buffer = new byte[64 * 1024];
      int read;
      while ((read = input.read(buffer)) >= 0) digest.update(buffer, 0, read);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private record Position(int x, int y, int z) {}
}
