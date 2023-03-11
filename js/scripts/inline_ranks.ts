import fs from "node:fs/promises";
import path from "node:path";
import { load } from "../src/load";

async function main() {
  try {
    await fs.mkdir(path.resolve(__dirname, "../ranks"), { recursive: true });
  } catch {}

  const registry = JSON.parse(
    await fs.readFile(path.resolve(__dirname, "../../tiktoken/registry.json"), {
      encoding: "utf-8",
    })
  );

  for (const name in registry) {
    console.log(name);
    const data = registry[name];

    const targetFile = path.resolve(__dirname, `../ranks/${name}.tiktoken`);

    try {
      await fs.stat(targetFile);
      continue;
    } catch {}

    const result = await load(data);
    await fs.writeFile(targetFile, result.bpe_ranks, { encoding: "utf-8" });
  }
}

main();
