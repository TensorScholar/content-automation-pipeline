import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..");
const roots = ["src", "app"];
const files = [];
const walk = (dir) => {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full);
    else if (/\.(ts|tsx|css)$/.test(entry.name)) files.push(full);
  }
};
for (const dir of roots) walk(path.join(root, dir));
files.push(path.join(root, "tailwind.config.ts"));

const textByFile = new Map(files.map((file) => [file, fs.readFileSync(file, "utf8")]));
const all = [...textByFile.values()].join("\n");
const failures = [];
const assert = (condition, message) => { if (!condition) failures.push(message); };

// Source hygiene: design must be semantic, not a pile of hard-coded framework palettes.
assert(!/(?:gray|slate|zinc|teal|emerald|amber|red|blue|rose|pink|purple|violet|indigo|cyan|sky|lime|orange|yellow)-\d{2,3}\b/.test(all), "Non-semantic Tailwind palette token remains");
assert(!/text-\[[^\]]+\]/.test(all), "Arbitrary typography remains outside the curated type scale");
assert(!/shadow-\[[^\]]+\]/.test(all), "Arbitrary shadow remains outside the curated elevation scale");
assert(!/\bdark:/.test(all), "Manual dark-mode utility branch remains; semantic tokens must own theme differences");
assert(!/\bas any\b|\bas unknown as\b|:\s*any\b|\bsafe_t\b/.test(all), "Unsafe any/cast or legacy translation helper remains");
assert(!/\b(?:glass-card|glassmorph|macos-|AI-Native|Flight Check|Tonal DNA|Polished Kebab)\b/i.test(all), "Prompt-era visual vocabulary remains");

// Interactive safety: explicit button type prevents accidental form submits.
for (const [file, source] of textByFile) {
  if (!file.endsWith(".tsx")) continue;
  for (const match of source.matchAll(/<button\b([^>]*)>/gs)) {
    assert(/\btype=/.test(match[1]), `${path.relative(root, file)} has <button> without explicit type`);
  }
}

// CSS custom properties: every var(--x) reference in authored CSS/TS must be declared in globals.css.
const globals = fs.readFileSync(path.join(root, "app/globals.css"), "utf8");
const defined = new Set([
  ...[...globals.matchAll(/(--[a-zA-Z0-9-]+)\s*:/g)].map((m) => m[1]),
  ...[...all.matchAll(/["\'](--[a-zA-Z0-9-]+)["\']\s*:/g)].map((m) => m[1]),
]);
const refs = new Set([...all.matchAll(/var\((--[a-zA-Z0-9-]+)/g)].map((m) => m[1]));
for (const ref of refs) assert(defined.has(ref), `Undefined CSS custom property referenced: ${ref}`);

// Translation consumption parity: every literal t("key") used by UI must exist in all three locale maps.
const messages = fs.readFileSync(path.join(root, "src/i18n/messages.ts"), "utf8");
const usedKeys = new Set([...all.matchAll(/\bt\(\s*["']([^"']+)["']/g)].map((m) => m[1]));
const baseFa = messages.slice(messages.indexOf("const fa:"), messages.indexOf("const ar:"));
const baseAr = messages.slice(messages.indexOf("const ar:"), messages.indexOf("const en:"));
const baseEn = messages.slice(messages.indexOf("const en:"), messages.indexOf("export const messages"));
const composite = messages.slice(messages.indexOf("export const messages"));
const overrideFa = composite.slice(composite.indexOf("  fa: {"), composite.indexOf("  ar: {"));
const overrideAr = composite.slice(composite.indexOf("  ar: {"), composite.indexOf("  en: {"));
const overrideEn = composite.slice(composite.indexOf("  en: {"));
for (const key of usedKeys) {
  const needleA = `"${key}"`;
  const needleB = `'${key}'`;
  const has = (source) => source.includes(needleA) || source.includes(needleB);
  assert(has(baseFa) || has(overrideFa), `Translation key ${key} is missing in FA`);
  assert(has(baseAr) || has(overrideAr), `Translation key ${key} is missing in AR`);
  assert(has(baseEn) || has(overrideEn), `Translation key ${key} is missing in EN`);
}


// Desktop security baseline: no unused transparent/private API path and no unsafe inline scripts.
const tauriConfig = JSON.parse(fs.readFileSync(path.join(root, "src-tauri/tauri.conf.json"), "utf8"));
const tauriCargo = fs.readFileSync(path.join(root, "src-tauri/Cargo.toml"), "utf8");
assert(tauriConfig.build?.beforeDevCommand === "npm run dev", "Tauri dev command is not cross-platform deterministic");
assert(tauriConfig.app?.macOSPrivateApi === false, "macOSPrivateApi must remain disabled unless a reviewed feature requires it");
assert(tauriConfig.app?.windows?.every((window) => window.transparent === false), "Desktop windows must not use unnecessary transparent rendering");
assert(!tauriCargo.includes("macos-private-api"), "Cargo still enables the macOS private API feature");
const nextConfigSource = fs.readFileSync(path.join(root, "next.config.mjs"), "utf8");
assert(/images:\s*\{\s*unoptimized:\s*true\s*\}/.test(nextConfigSource), "Static/Tauri export can still depend on the Next image optimizer");

const csp = tauriConfig.app?.security?.csp ?? "";
assert(!/script-src[^;]*'unsafe-inline'/.test(csp), "CSP allows unsafe-inline scripts");
assert(csp.includes("object-src 'none'"), "CSP object-src is not locked down");
assert(csp.includes("base-uri 'none'"), "CSP base-uri is not locked down");

// Trilingual generation must alter the actual request instruction, not just the UI selector.
const studioFiles = files.filter((f) => f.includes("src/components/panels/studio") || f.includes("src/components/panels/content-studio"));
const studio = studioFiles.map((f) => textByFile.get(f) ?? fs.readFileSync(f, "utf8")).join("\n");
assert(studio.includes('ar: "Output language must be Arabic."'), "Arabic generation instruction is missing");
assert(studio.includes('fa: "Output language must be Persian (Farsi)."'), "Persian generation instruction is missing");
assert(studio.includes('en: "Output language must be English."'), "English generation instruction is missing");
assert(!studio.includes('bg-white/70') && !studio.includes('dark:bg-white/10'), "Studio still contains manual light/dark surface forks");

if (failures.length) {
  console.error(`Source verification FAILED (${failures.length})`);
  for (const failure of failures) console.error(` - ${failure}`);
  process.exit(1);
}
console.log(`Source verification passed: ${files.length} authored files, ${usedKeys.size} literal i18n keys, semantic styling and interaction guards clean.`);
