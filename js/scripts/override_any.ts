import { Project, ts } from "ts-morph";
import * as fs from "node:fs";
import * as path from "node:path";

const project = new Project();
project.addSourceFilesAtPaths(["./dist/**/*.ts", "./dist/**/*.js"]);

// make sure the types are correct
for (const filename of [
  "./dist/bundler/_tiktoken.d.ts",
  "./dist/node/_tiktoken.d.ts",
]) {
  const sourceFile = project.getSourceFileOrThrow(filename);
  const cls = sourceFile.getFirstDescendantByKindOrThrow(
    ts.SyntaxKind.ClassDeclaration
  );

  cls
    .getConstructors()[0]
    .getParameterOrThrow("special_tokens")
    .set({ type: "Record<string, number>" });

  for (const method of ["encode", "encode_with_unstable"]) {
    cls
      .getMethodOrThrow(method)
      .getParameterOrThrow("allowed_special")
      .set({ type: `"all" | string[]`, hasQuestionToken: true });

    cls
      .getMethodOrThrow(method)
      .getParameterOrThrow("disallowed_special")
      .set({ type: `"all" | string[]`, hasQuestionToken: true });
  }

  cls
    .getMemberOrThrow("token_byte_values")
    .set({ returnType: "Array<Array<number>>" });

  sourceFile.saveSync();
}

// use only a single WASM binary
fs.copyFileSync(
  path.resolve(__dirname, "../dist/bundler/_tiktoken_bg.wasm"),
  path.resolve(__dirname, "../dist/tiktoken.wasm")
);

fs.copyFileSync(
  path.resolve(__dirname, "../dist/bundler/_tiktoken_bg.wasm.d.ts"),
  path.resolve(__dirname, "../dist/tiktoken.wasm.d.ts")
);

// remove unnecessary files
for (const folder of ["bundler", "node"]) {
  fs.rmSync(path.resolve(__dirname, `../dist/${folder}/package.json`));
  fs.rmSync(path.resolve(__dirname, `../dist/${folder}/README.md`));
}

function replaceContent(file: string, transform: (content: string) => string) {
  const options = { encoding: "utf-8" } as const;
  fs.writeFileSync(
    path.resolve(__dirname, file),
    transform(fs.readFileSync(path.resolve(__dirname, file), options)),
    options
  );
}

// bundler
{
  replaceContent("../dist/bundler/_tiktoken.js", (src) =>
    src.replaceAll(`"./_tiktoken_bg.wasm"`, `"../tiktoken.wasm"`)
  );

  fs.rmSync(path.resolve(__dirname, "../dist/bundler/_tiktoken_bg.wasm"));
  fs.rmSync(path.resolve(__dirname, "../dist/bundler/_tiktoken_bg.wasm.d.ts"));
}

// node
{
  replaceContent("../dist/node/_tiktoken.js", (src) =>
    src
      .replaceAll("__wbindgen_placeholder__", `./_tiktoken_bg.js`)
      .replace("'_tiktoken_bg.wasm'", `'../tiktoken.wasm'`)
  );

  fs.rmSync(path.resolve(__dirname, "../dist/node/_tiktoken_bg.wasm"));
  fs.rmSync(path.resolve(__dirname, "../dist/node/_tiktoken_bg.wasm.d.ts"));
}

{
  fs.writeFileSync(
    path.resolve(__dirname, "../dist/init.js"),
    `
import * as imports from "./bundler/_tiktoken_bg.js";

export async function init(cb) {
  const res = await cb({
    "./_tiktoken_bg.js": imports,
  });

  const instance =
    "instance" in res && res.instance instanceof WebAssembly.Instance
      ? res.instance
      : res instanceof WebAssembly.Instance
      ? res
      : null;

  if (instance == null) throw new Error("Missing instance");
  imports.__wbg_set_wasm(instance.exports);
  return imports;
}

export * from "./bundler/_tiktoken_bg.js";
  `.trim(),
    { encoding: "utf-8" }
  );

  fs.writeFileSync(
    path.resolve(__dirname, "../dist/init.d.ts"),
    `
/* tslint:disable */
/* eslint-disable */
export * from "./bundler/_tiktoken";
export function init(
  callback: (
    imports: WebAssembly.Imports
  ) => Promise<WebAssembly.WebAssemblyInstantiatedSource | WebAssembly.Instance>
): Promise<void>;
  `.trim(),
    { encoding: "utf-8" }
  );
}

{
  const pkg = JSON.parse(
    fs.readFileSync(path.resolve(__dirname, "../package.json"), {
      encoding: "utf-8",
    })
  );

  delete pkg.devDependencies;
  delete pkg.scripts;
  pkg.files = ["**/*"];

  fs.writeFileSync(
    path.resolve(__dirname, "../dist/package.json"),
    JSON.stringify(pkg, null, 2),
    { encoding: "utf-8" }
  );
}
