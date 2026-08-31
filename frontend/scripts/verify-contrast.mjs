import fs from "node:fs";
import path from "node:path";
import url from "node:url";

const root = path.resolve(path.dirname(url.fileURLToPath(import.meta.url)), "..");
const css = fs.readFileSync(path.join(root, "app/globals.css"), "utf8");

function parseBlock(selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = css.match(new RegExp(`${escaped}\\s*\\{([\\s\\S]*?)\\n\\s*\\}`, "m"));
  if (!match) throw new Error(`Missing CSS block: ${selector}`);
  const map = new Map();
  for (const item of match[1].matchAll(/(--[a-zA-Z0-9-]+)\s*:\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*;/g)) {
    map.set(item[1], [Number(item[2]), Number(item[3]), Number(item[4])]);
  }
  return map;
}

function channel(value) {
  const c = value / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}
function luminance([r, g, b]) {
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}
function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const light = parseBlock(":root");
const dark = parseBlock(".dark");
const textTokens = [
  "--color-text-primary",
  "--color-text-secondary",
  "--color-text-muted",
  "--color-text-placeholder",
  "--color-primary",
  "--color-success",
  "--color-warning",
  "--color-error",
  "--color-info",
];
const backgrounds = ["--bg-primary", "--bg-elevated"];
const failures = [];

for (const [themeName, theme] of [["light", light], ["dark", new Map([...light, ...dark])]]) {
  for (const token of textTokens) {
    const foreground = theme.get(token);
    if (!foreground) { failures.push(`${themeName}:${token} missing`); continue; }
    for (const backgroundToken of backgrounds) {
      const background = theme.get(backgroundToken);
      if (!background) { failures.push(`${themeName}:${backgroundToken} missing`); continue; }
      const ratio = contrast(foreground, background);
      if (ratio < 4.5) failures.push(`${themeName}:${token} on ${backgroundToken} = ${ratio.toFixed(2)}:1`);
    }
  }
}

if (failures.length) {
  console.error(`Contrast verification FAILED (${failures.length})`);
  failures.forEach((failure) => console.error(` - ${failure}`));
  process.exit(1);
}
console.log("Contrast verification passed: semantic text/state tokens meet >= 4.5:1 on primary/elevated surfaces in light and dark themes.");
