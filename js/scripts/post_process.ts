import { Project, StructureKind, ts } from "ts-morph";
import * as fs from "node:fs";
import * as path from "node:path";

const project = new Project();
project.addSourceFilesAtPaths(["./dist/**/*.ts", "./dist/**/*.js"]);

// make sure the types are correct
for (const filename of ["./dist/tiktoken.d.ts", "./dist/lite/tiktoken.d.ts"]) {
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

for (const filename of [
  "./dist/tiktoken_bg.js",
  "./dist/lite/tiktoken_bg.js",
]) {
  const targetFileName = filename.replace("_bg", ".node");
  const sourceFile = project.getSourceFileOrThrow(filename);

  sourceFile.insertStatements(0, [
    `let imports = {};`,
    `imports["./tiktoken_bg.js"] = module.exports;`,
  ]);

  for (const cls of sourceFile.getClasses().filter((x) => x.isExported())) {
    cls.set({
      ...cls.getStructure(),
      kind: StructureKind.Class,
      isExported: false,
    });

    sourceFile.insertStatements(cls.getChildIndex() + 1, [
      `module.exports.${cls.getName()} = ${cls.getName()};`,
    ]);
  }

  for (const fn of sourceFile.getFunctions().filter((f) => f.isExported())) {
    fn.set({
      ...fn.getStructure(),
      kind: StructureKind.Function,
      isExported: false,
    });

    sourceFile.insertStatements(fn.getChildIndex(), [
      `module.exports.${fn.getName()} = ${fn.getText()};`,
    ]);

    sourceFile
      .getDescendantsOfKind(ts.SyntaxKind.FunctionExpression)
      .filter((x) => x.getName() === fn.getName())
      .forEach((f) => f.removeName());

    fn.remove();
  }

  sourceFile.addStatements([
    `const path = require("path").join(__dirname, "tiktoken_bg.wasm");`,
    `const bytes = require("fs").readFileSync(path);`,

    `const wasmModule = new WebAssembly.Module(bytes);`,
    `const wasmInstance = new WebAssembly.Instance(wasmModule, imports);`,
    `wasm = wasmInstance.exports;`,
    `module.exports.__wasm = wasm;`,
  ]);

  sourceFile.copy(targetFileName, { overwrite: true }).saveSync();
}

for (const targetFile of [
  path.resolve(__dirname, "../dist"),
  path.resolve(__dirname, "../dist/lite"),
]) {
  // bundler
  {
    fs.writeFileSync(
      path.resolve(targetFile, "bundler.js"),
      `export * from "./tiktoken";`.trim(),
      { encoding: "utf-8" }
    );

    fs.writeFileSync(
      path.resolve(targetFile, "bundler.d.ts"),
      `export * from "./tiktoken";`.trim(),
      { encoding: "utf-8" }
    );
  }

  // init.js
  {
    fs.writeFileSync(
      path.resolve(targetFile, "init.js"),
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
      path.resolve(targetFile, "init.d.ts"),
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
}

// package.json, README.md
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
    "./lite": {
      types: "./lite/tiktoken.d.ts",
      node: "./lite/tiktoken.node.js",
      default: "./lite/tiktoken.js",
    },
    "./lite/bundler": {
      types: "./lite/bundler.d.ts",
      default: "./lite/bundler.js",
    },
    "./lite/init": {
      types: "./lite/init.d.ts",
      default: "./lite/init.js",
    },
    "./lite/tiktoken_bg.wasm": {
      types: "./lite/tiktoken_bg.wasm.d.ts",
      default: "./lite/tiktoken_bg.wasm",
    },
  };

  fs.writeFileSync(
    path.resolve(__dirname, "../dist/package.json"),
    JSON.stringify(pkg, null, 2),
    { encoding: "utf-8" }
  );

  fs.copyFileSync(
    path.resolve(__dirname, "../README.md"),
    path.resolve(__dirname, "../dist/README.md")
  );
}
