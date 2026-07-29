const FRIENDLY_MODEL_NAMES: Record<string, string> = {
  "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
  "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
  "compatible/google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
};

export function formatModelDisplayName(model: string | null | undefined): string {
  const normalized = model?.trim().toLowerCase() ?? "";
  if (!normalized) return "Configured model";

  const identifier = normalized.replace(/^openai_compatible(?:[/:]|$)/, "");
  if (!identifier) return "Configured model";
  const knownName = FRIENDLY_MODEL_NAMES[identifier];
  if (knownName) return knownName;

  const modelSlug = identifier.split("/").at(-1) ?? identifier;
  return modelSlug
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => (/^gpt|^llama|^qwen$/i.test(part) ? part.toUpperCase() : `${part[0]?.toUpperCase() ?? ""}${part.slice(1)}`))
    .join(" ");
}
