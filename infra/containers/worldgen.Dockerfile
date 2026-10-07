FROM maven:3.9.11-eclipse-temurin-25 AS build
WORKDIR /src
COPY pom.xml ./
COPY packages/contracts/java packages/contracts/java
COPY services/worldgen services/worldgen
RUN mvn -B -pl services/worldgen -am package -DskipTests

FROM eclipse-temurin:25-jdk-noble
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 worldgen
WORKDIR /opt/worldgen
COPY --from=build /src/services/worldgen/target/worldgen-plugin-0.1.0.jar ./worldgen-plugin.jar
COPY services/worldgen/worldgen_runner.py ./worldgen_runner.py
COPY services/worldgen/level_nbt.py ./level_nbt.py
COPY services/worldgen/server-template ./server-template
USER worldgen
ENTRYPOINT ["python3", "/opt/worldgen/worldgen_runner.py"]
