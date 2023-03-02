import { Project, ts } from "ts-morph";

const project = new Project();
project.addSourceFilesAtPaths("./dist/**/*.ts");

for (const filename of [
  "./dist/bundler/_tiktoken.d.ts",
  "./dist/node/_tiktoken.d.ts",
  "./dist/web/_tiktoken.d.ts",
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
