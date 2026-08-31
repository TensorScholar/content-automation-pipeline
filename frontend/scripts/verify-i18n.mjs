import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import vm from "node:vm";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const ts = require("typescript");
const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..");
const source = fs.readFileSync(path.join(root, "src/i18n/messages.ts"), "utf8");
const output = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  fileName: "messages.ts",
  reportDiagnostics: true,
});
const diagnostics = (output.diagnostics ?? []).filter((item) => item.category === ts.DiagnosticCategory.Error);
if (diagnostics.length) {
  console.error("i18n transpilation failed");
  for (const diagnostic of diagnostics) console.error(ts.flattenDiagnosticMessageText(diagnostic.messageText, " "));
  process.exit(1);
}
const sandbox = { exports: {}, module: { exports: {} } };
sandbox.module.exports = sandbox.exports;
vm.runInNewContext(output.outputText, sandbox, { filename: "messages.js" });
const messages = sandbox.exports.messages;
if (!messages?.fa || !messages?.ar || !messages?.en) {
  console.error("i18n runtime maps are incomplete");
  process.exit(1);
}
const locales = ["fa", "ar", "en"];
const keySets = Object.fromEntries(locales.map((locale) => [locale, new Set(Object.keys(messages[locale]))]));
const union = new Set(locales.flatMap((locale) => [...keySets[locale]]));
const missing = [];
for (const locale of locales) {
  for (const key of union) if (!keySets[locale].has(key)) missing.push(`${locale}:${key}`);
}
if (missing.length) {
  console.error(`i18n runtime parity FAILED (${missing.length} missing keys)`);
  for (const item of missing.slice(0, 50)) console.error(` - ${item}`);
  process.exit(1);
}
const counts = Object.fromEntries(locales.map((locale) => [locale, keySets[locale].size]));
console.log(`i18n runtime parity passed: FA=${counts.fa}, AR=${counts.ar}, EN=${counts.en} keys.`);
