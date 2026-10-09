Level-wide files of a Minecraft Java 26.2 singleplayer save (`DataVersion`
4903), captured from a Paper 26.2 build 112 export after
`worldgen_runner.convert_to_singleplayer` with
`python3 services/worldgen/world_writer.py capture-template world.zip`.
`world_writer.py` copies them into every direct-written save and fills in
`level.dat`'s spawn, `LevelName` and `LastPlayed`. This file is not copied.
