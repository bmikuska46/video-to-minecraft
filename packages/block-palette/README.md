# Geometry-safe block palette

`palette-v1.json` contains representative sRGB colors measured/curated for
full-cube, non-directional vanilla blocks. The service converts these values to
CIE Lab at load time. `textureNoisePenalty` is added to Delta E so a similarly
colored quiet material wins over a visually noisy texture.

The file intentionally contains no Minecraft texture assets. Falling,
transparent, liquid, biome-tinted, emissive, directional, neighbor-dependent,
and block-entity materials are excluded.
