package dev.minecraftvideo.worldgen;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;

final class ProgressReporter {
  private final Path path;
  private final Gson gson = new Gson();

  ProgressReporter(Path path) { this.path = path; }

  void write(String status, String phase, long completed, long total, String error) {
    Map<String, Object> value = new LinkedHashMap<>();
    value.put("status", status);
    value.put("phase", phase);
    value.put("completedBlocks", completed);
    value.put("totalBlocks", total);
    if (error != null) value.put("error", error);
    try {
      Path parent = path.toAbsolutePath().getParent();
      Files.createDirectories(parent);
      Path temporary = Files.createTempFile(parent, path.getFileName().toString(), ".tmp");
      Files.writeString(temporary, gson.toJson(value));
      try {
        Files.move(temporary, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
      } catch (IOException unsupportedAtomicMove) {
        Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING);
      }
    } catch (IOException errorDuringReporting) {
      // Reporting must not leave a successful world half-written solely due to a progress callback.
    }
  }
}
