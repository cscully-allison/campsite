const esbuild = require("esbuild");
const path = require("path");

esbuild.build({
  entryPoints: [path.join("campsite", "src", "campsite.ts")],
  bundle: true,
  format: "esm",
  outdir: path.join("campsite", "static"),
  platform: "browser",
  sourcemap: true,
}).catch(() => process.exit(1));
