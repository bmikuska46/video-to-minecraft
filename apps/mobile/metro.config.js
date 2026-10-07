const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// three.js ships separate CommonJS and ES module builds. @react-three/fiber/native
// require()s the CommonJS one and swaps in React Native-safe loaders on it, while
// three/examples (GLTFLoader) import the ES module one, which never gets those
// loaders: GLB previews then fail with "JSON Parse error". Resolve every "three"
// import to the single CommonJS build so there is exactly one copy.
const threePath = require.resolve("three");
const resolveRequest = config.resolver.resolveRequest;
config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (moduleName === "three") return { type: "sourceFile", filePath: threePath };
  return (resolveRequest ?? context.resolveRequest)(context, moduleName, platform);
};

module.exports = config;
