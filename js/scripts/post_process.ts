import { Project, ts } from "ts-morph";
import * as fs from "node:fs";
import * as path from "node:path";

const project = new Project();
project.addSourceFilesAtPaths(["./dist/**/*.ts", "./dist/**/*.js"]);

// make sure the types are correct
for (const filename of ["./dist/tiktoken.d.ts", "./dist/node/tiktoken.d.ts"]) {
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

// bundler
{
  fs.writeFileSync(
    path.resolve(__dirname, "../dist/bundler.js"),
    `export * from "./tiktoken";  `.trim(),
    { encoding: "utf-8" }
  );

  fs.writeFileSync(
    path.resolve(__dirname, "../dist/bundler.d.ts"),
    `export * from "./tiktoken";  `.trim(),
    { encoding: "utf-8" }
  );
}

// node
{
  const options = { encoding: "utf-8" } as const;
  fs.writeFileSync(
    path.resolve(__dirname, "../dist/tiktoken.node.js"),
    fs
      .readFileSync(
        path.resolve(__dirname, "../dist/node/tiktoken.js"),
        options
      )
      .replaceAll("__wbindgen_placeholder__", `./tiktoken_bg.js`),
    options
  );

  fs.rmSync(path.resolve(__dirname, "../dist/node"), { recursive: true });
}

// package.json
{
  fs.writeFileSync(
    path.resolve(__dirname, "../dist/init.js"),
    `
import * as imports from "./tiktoken_bg.js";

export async function init(cb) {
  const res = await cb({
    "./tiktoken_bg.js": imports,
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

export * from "./tiktoken_bg.js";
  `.trim(),
    { encoding: "utf-8" }
  );

  fs.writeFileSync(
    path.resolve(__dirname, "../dist/init.d.ts"),
    `
/* tslint:disable */
/* eslint-disable */
export * from "./tiktoken";
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

  pkg["main"] = "tiktoken.node.js";
  pkg["types"] = "tiktoken.d.ts";
  pkg["exports"] = {
    ".": {
      types: "./tiktoken.d.ts",
      node: "./tiktoken.node.js",
      default: "./tiktoken.js",
    },
    "./bundler": {
      types: "./bundler.d.ts",
      default: "./bundler.js",
    },
    "./init": {
      types: "./init.d.ts",
      default: "./init.js",
    },
    "./tiktoken_bg.wasm": {
      types: "./tiktoken_bg.wasm.d.ts",
      default: "./tiktoken_bg.wasm",
    },
  };

  fs.writeFileSync(
    path.resolve(__dirname, "../dist/package.json"),
    JSON.stringify(pkg, null, 2),
    { encoding: "utf-8" }
  );
}
