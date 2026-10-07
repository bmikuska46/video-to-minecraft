package dev.minecraftvideo.worldgen;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class JobManifestTest {
  @TempDir Path temporary;

  @Test
  void readsRunnerVerifiedEnvelope() throws Exception {
    Path path = temporary.resolve("manifest.json");
    Files.writeString(path, json("26.2", "a".repeat(64)));
    JobManifest manifest = JobManifest.read(path);
    assertEquals("minecraft-video-world", manifest.payload().worldName());
    assertEquals(20_000, manifest.payload().batchSize());
  }

  @Test
  void rejectsWrongTargetVersion() throws Exception {
    Path path = temporary.resolve("manifest.json");
    Files.writeString(path, json("26.3", "a".repeat(64)));
    assertThrows(IOException.class, () -> JobManifest.read(path));
  }

  private static String json(String version, String checksum) {
    return """
        {
          "payload": {
            "schemaVersion": 1,
            "jobId": "550e8400-e29b-41d4-a716-446655440000",
            "minecraftVersion": "%s",
            "voxelSha256": "%s",
            "worldName": "minecraft-video-world",
            "platformY": 64,
            "platformMargin": 8,
            "platformBlockState": "minecraft:smooth_stone",
            "batchSize": 20000,
            "maxBlockCount": 1000000,
            "maxPlatformBlocks": 4000000
          },
          "signature": "hmac-sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        }
        """.formatted(version, checksum);
  }
}
