import * as fs from "node:fs/promises";
import * as path from "node:path";

async function createAlias() {
  const srcDist = path.resolve(__dirname, "../dist");
  const targetDist = path.resolve(__dirname, "../alias/dist");

  await fs.cp(srcDist, targetDist, { recursive: true });

  const pkgPath = path.resolve(targetDist, "package.json");

  const pkg = JSON.parse(await fs.readFile(pkgPath, { encoding: "utf-8" }));
  pkg["name"] = "@dqbd/tiktoken";

  await fs.writeFile(pkgPath, JSON.stringify(pkg, null, 2), {
    encoding: "utf-8",
  });
}

createAlias();
