"use client";

import clsx from "clsx";
import { useI18n } from "@/i18n/provider";
import { Locale } from "@/i18n/types";

const locales: Array<{ code: Locale; nativeLabel: string }> = [
  { code: "fa", nativeLabel: "FA" },
  { code: "ar", nativeLabel: "AR" },
  { code: "en", nativeLabel: "EN" },
];

export interface LanguageToggleProps {
  variant?: "light" | "dark" | "macos";
}

export function LanguageToggle({ variant = "light" }: LanguageToggleProps) {
  const { locale, setLocale, t } = useI18n();
  const isDark = variant === "dark";

  return (
    <div
      dir="ltr"
      className={clsx(
        "inline-flex items-center gap-0.5 rounded-[10px] p-0.5",
        variant === "macos"
          ? "macos-segmented"
          : isDark
            ? "border border-white/10 bg-black/20"
            : "border border-black/[0.055] bg-black/[0.025] dark:border-white/[0.075] dark:bg-white/[0.045]",
      )}
      role="group"
      aria-label={t("lang.select")}
    >
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
              "min-h-[30px] min-w-[34px] rounded-[8px] px-2 text-[11px] font-semibold transition-[background-color,color,box-shadow] duration-150 focus-visible:outline-none sm:min-w-[38px]",
              selected
                ? isDark
                  ? "bg-white/[0.14] text-white"
                  : "bg-white text-ink shadow-[0_1px_2px_rgb(0_0_0/0.06)] dark:bg-white/[0.1] dark:text-white"
                : isDark
                  ? "text-white/60 hover:text-white"
                  : "text-ink-tertiary hover:text-ink",
            )}
          >
            {entry.nativeLabel}
          </button>
        );
      })}
    </div>
  );
}
