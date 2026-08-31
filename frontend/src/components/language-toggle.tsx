"use client";

import clsx from "clsx";
import { useI18n } from "@/i18n/provider";
import { Locale } from "@/i18n/types";

const locales: Array<{ code: Locale; shortLabel: string }> = [
  { code: "fa", shortLabel: "FA" },
  { code: "ar", shortLabel: "AR" },
  { code: "en", shortLabel: "EN" },
];

export interface LanguageToggleProps { variant?: "light" | "dark"; }

export function LanguageToggle({ variant = "light" }: LanguageToggleProps) {
  const { locale, setLocale, t } = useI18n();
  const inverted = variant === "dark";

  return (
    // Container: 4px (rounded-sm); segment buttons: 4px (rounded-sm)
    // 2px (rounded-xs) is reserved for genuinely smallest geometry only (e.g. checkboxes)
    <div dir="ltr" className={clsx("inline-flex items-center gap-0.5 rounded-sm p-0.5", inverted ? "bg-white/[0.08]" : "bg-ink/[0.05]")} role="group" aria-label={t("lang.select")}>
      {locales.map((entry) => {
        const selected = locale === entry.code;
        return (
          <button
            key={entry.code}
            type="button"
            onClick={() => setLocale(entry.code)}
            aria-pressed={selected}
            aria-label={t(`lang.${entry.code}` as const)}
            className={clsx(
              "min-h-7 min-w-[34px] rounded-sm px-2 text-xs font-medium transition-colors duration-fast",
              selected
                ? inverted ? "bg-white/[0.14] text-white" : "bg-surface text-ink ring-1 ring-line"
                : inverted ? "text-white/60 hover:text-white" : "text-ink-tertiary hover:text-ink",
            )}
          >
            {entry.shortLabel}
          </button>
        );
      })}
    </div>
  );
}
