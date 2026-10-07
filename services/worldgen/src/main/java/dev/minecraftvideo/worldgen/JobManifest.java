package dev.minecraftvideo.worldgen;

import com.google.gson.Gson;
import com.google.gson.JsonParseException;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.UUID;
import java.util.regex.Pattern;

/** The signed envelope is verified by the runner; Paper consumes only its validated payload. */
public record JobManifest(Payload payload, String signature) {
  public static final int SCHEMA_VERSION = 1;
  private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");
  private static final Pattern WORLD_NAME = Pattern.compile("[a-zA-Z0-9_-]{1,48}");
  private static final Pattern BLOCK_STATE = Pattern.compile("minecraft:[a-z0-9_]+(?:\\[[^]\\r\\n]+])?");
  private static final Set<String> MODES = Set.of("generate", "validate");

  public record Payload(
      int schemaVersion,
      String jobId,
      String minecraftVersion,
      String voxelSha256,
      String worldName,
      int platformY,
      int platformMargin,
      String platformBlockState,
      int batchSize,
      int maxBlockCount,
      long maxPlatformBlocks) {}

  public static JobManifest read(Path path) throws IOException {
    try (Reader reader = Files.newBufferedReader(path)) {
      JobManifest manifest = new Gson().fromJson(reader, JobManifest.class);
      validate(manifest);
      return manifest;
    } catch (JsonParseException | NullPointerException error) {
      throw new IOException("invalid job manifest JSON", error);
    }
  }

  private static void validate(JobManifest manifest) throws IOException {
    if (manifest == null || manifest.payload == null || manifest.signature == null) {
      throw new IOException("job manifest is missing its payload or signature");
    }
    Payload payload = manifest.payload;
    if (payload.schemaVersion != SCHEMA_VERSION) {
      throw new IOException("unsupported job manifest schema: " + payload.schemaVersion);
    }
    try {
      UUID.fromString(payload.jobId);
    } catch (IllegalArgumentException error) {
      throw new IOException("jobId must be a UUID", error);
    }
    if (!"26.2".equals(payload.minecraftVersion)) {
      throw new IOException("unsupported Minecraft target: " + payload.minecraftVersion);
    }
    if (!SHA256.matcher(payload.voxelSha256).matches()) {
      throw new IOException("voxelSha256 must be lowercase hexadecimal");
    }
    if (!WORLD_NAME.matcher(payload.worldName).matches()) {
      throw new IOException("unsafe worldName");
    }
    if (payload.platformY < -63 || payload.platformY > 300
        || payload.platformMargin < 1 || payload.platformMargin > 64) {
      throw new IOException("invalid platform bounds");
    }
    if (!BLOCK_STATE.matcher(payload.platformBlockState).matches()) {
      throw new IOException("invalid platform block state");
    }
    if (payload.batchSize < 100 || payload.batchSize > 100_000
        || payload.maxBlockCount < 1 || payload.maxBlockCount > 5_000_000
        || payload.maxPlatformBlocks < 1 || payload.maxPlatformBlocks > 20_000_000) {
      throw new IOException("invalid world generation limits");
    }
    if (!manifest.signature.startsWith("hmac-sha256:") || manifest.signature.length() != 76) {
      throw new IOException("invalid job manifest signature encoding");
    }
  }

  public static String mode() throws IOException {
    String mode = System.getProperty("minecraftvideo.mode", "generate");
    if (!MODES.contains(mode)) {
      throw new IOException("invalid worldgen mode: " + mode);
    }
    return mode;
  }
}
