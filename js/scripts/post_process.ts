import { Project, ScriptTarget, StructureKind, ts } from "ts-morph";
import * as fs from "node:fs";
import * as path from "node:path";

for (const baseDir of [
  path.resolve(__dirname, "../dist"),
  path.resolve(__dirname, "../dist/lite"),
]) {
  // fix `any` types
  {
    const sourceFile = new Project().addSourceFileAtPath(
      path.resolve(baseDir, "tiktoken.d.ts")
    );
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

  // tiktoken.cjs
  {
    const sourceFile = new Project().addSourceFileAtPath(
      path.resolve(baseDir, "tiktoken_bg.js")
    );
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

    sourceFile
      .copy(path.resolve(baseDir, "tiktoken.cjs"), { overwrite: true })
      .saveSync();
  }

  // init.js
  {
    const sourceFile = new Project({
      compilerOptions: {
        target: ScriptTarget.ES2022,
        moduleResolution: ts.ModuleResolutionKind.NodeJs,
        strict: true,
        declaration: true,
      },
    }).addSourceFileAtPath(path.resolve(__dirname, "../src/init.ts"));

    const emitOutput = sourceFile.getEmitOutput();
    for (const file of emitOutput.getOutputFiles()) {
      fs.writeFileSync(
        path.resolve(baseDir, path.basename(file.getFilePath())),
        file.getText(),
        { encoding: "utf-8" }
      );
    }
  }

  // load.js and load.cjs
  {
    for (const module of [ts.ModuleKind.CommonJS, ts.ModuleKind.ES2022]) {
      const sourceFile = new Project({
        compilerOptions: {
          target: ScriptTarget.ES2022,
          module,
          moduleResolution: ts.ModuleResolutionKind.NodeJs,
          strict: true,
          declaration: true,
        },
      }).addSourceFileAtPath(path.resolve(__dirname, "../src/load.ts"));

      const emitOutput = sourceFile.getEmitOutput();
      for (const file of emitOutput.getOutputFiles()) {
        let targetFile = path.basename(file.getFilePath());

        if (module === ts.ModuleKind.CommonJS) {
          targetFile = targetFile.replace(".js", ".cjs");
        }

        fs.writeFileSync(path.resolve(baseDir, targetFile), file.getText(), {
          encoding: "utf-8",
        });
      }
    }
  }

  // bundler.js
  {
    fs.writeFileSync(
      path.resolve(baseDir, "bundler.js"),
      `export * from "./tiktoken";`.trim(),
      { encoding: "utf-8" }
    );

    fs.writeFileSync(
      path.resolve(baseDir, "bundler.d.ts"),
      `export * from "./tiktoken";`.trim(),
      { encoding: "utf-8" }
    );

    fs.writeFileSync(
      path.resolve(baseDir, "tiktoken_bg.d.ts"),
      `export * from "./tiktoken";`.trim(),
      { encoding: "utf-8" }
    );
  }

  if (!baseDir.includes("/lite")) {
    fs.writeFileSync(
      path.resolve(baseDir, "lite.d.ts"),
      `export * from "./lite/tiktoken";`.trim(),
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

  pkg["main"] = "tiktoken.cjs";
  pkg["types"] = "tiktoken.d.ts";
  pkg["exports"] = {
    ".": {
      types: "./tiktoken.d.ts",
      node: "./tiktoken.cjs",
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
    "./load": {
      types: "./load.d.ts",
      node: "./load.cjs",
      default: "./load.js",
    },
    "./tiktoken_bg.wasm": {
      types: "./tiktoken_bg.wasm.d.ts",
      default: "./tiktoken_bg.wasm",
    },
    "./lite": {
      types: "./lite/tiktoken.d.ts",
      node: "./lite/tiktoken.cjs",
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
    "./lite/load": {
      types: "./lite/load.d.ts",
      node: "./lite/load.cjs",
      default: "./lite/load.js",
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
