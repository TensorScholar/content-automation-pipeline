import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..");
const contract = JSON.parse(fs.readFileSync(path.join(root, "scripts/ui-contract.json"), "utf8"));
const read = (p) => fs.readFileSync(path.join(root, p), "utf8");
const sourceFiles = [];
function walk(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (/\.(ts|tsx)$/.test(entry.name)) sourceFiles.push(full);
  }
}
walk(path.join(root, "src"));
const allSource = sourceFiles.map((p) => fs.readFileSync(p, "utf8")).join("\n");
const studio = read("src/components/panels/content-studio-panel.tsx");
const tasks = read("src/components/panels/tasks-panel.tsx");
const projects = read("src/components/panels/projects-panel.tsx");
const messages = read("src/i18n/messages.ts");
const languageToggle = read("src/components/language-toggle.tsx");

const failures = [];
const assert = (ok, message) => { if (!ok) failures.push(message); };

for (const endpoint of contract.requiredEndpoints) {
  assert(allSource.includes(endpoint), `Missing API endpoint contract: ${endpoint}`);
}
for (const locale of contract.requiredLocales) {
  assert(languageToggle.includes(`"${locale}"`), `Language toggle lost locale: ${locale}`);
  assert(messages.includes(`${locale}:`) || messages.includes(`"${locale}"`), `Messages lost locale: ${locale}`);
}
for (const locale of contract.requiredLocales) {
  assert(studio.includes(`option value="${locale}"`), `Studio generation language missing: ${locale}`);
}
for (const marker of contract.studioOptions) assert(studio.includes(marker), `Studio option/state missing: ${marker}`);
for (const tab of contract.studioTabs) assert(studio.includes(`"${tab}"`), `Studio tab/output missing: ${tab}`);
for (const marker of contract.contentCapabilities) assert(tasks.includes(marker), `Content capability marker missing: ${marker}`);
for (const marker of contract.projectCapabilities) assert(projects.includes(marker), `Project capability marker missing: ${marker}`);

assert(!allSource.includes("--color-border-primary"), "Undefined legacy CSS variable --color-border-primary remains in source");
assert(studio.includes('type GenerationLanguage = "fa" | "ar" | "en"'), "GenerationLanguage is not explicitly trilingual");
assert(tasks.includes('onKeyDown={(event)'), "Content table rows lost keyboard activation");

if (failures.length) {
  console.error(`UI contract verification FAILED (${failures.length})`);
  for (const failure of failures) console.error(` - ${failure}`);
  process.exit(1);
}
console.log(`UI contract verification passed: ${contract.requiredEndpoints.length} API routes, 3 locales, Studio options/outputs, project/content workflow markers.`);
