import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..");
const roots = ["src", "app"];
const files = [];

function walk(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) walk(fullPath);
    else if (/\.(ts|tsx)$/.test(entry.name)) files.push(fullPath);
  }
}

for (const directory of roots) walk(path.join(root, directory));
files.push(path.join(root, "tailwind.config.ts"));

const failures = [];
for (const file of files) {
  const source = fs.readFileSync(file, "utf8");
  const isTsx = file.endsWith(".tsx");
  const result = ts.transpileModule(source, {
    compilerOptions: {
      jsx: ts.JsxEmit.Preserve,
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
      isolatedModules: true,
    },
    fileName: file,
    reportDiagnostics: true,
  });

  for (const diagnostic of result.diagnostics ?? []) {
    if (diagnostic.category !== ts.DiagnosticCategory.Error) continue;
    const relative = path.relative(root, file);
    const position = diagnostic.start === undefined
      ? ""
      : (() => {
          const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.ES2022, true, isTsx ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
          const { line, character } = sourceFile.getLineAndCharacterOfPosition(diagnostic.start);
          return `:${line + 1}:${character + 1}`;
        })();
    failures.push(`${relative}${position} ${ts.flattenDiagnosticMessageText(diagnostic.messageText, " ")}`);
  }
}

if (failures.length) {
  console.error(`Syntax verification FAILED (${failures.length} diagnostics)`);
  for (const failure of failures) console.error(` - ${failure}`);
  process.exit(1);
}

console.log(`Syntax verification passed: ${files.length} TS/TSX files, 0 syntax diagnostics.`);
