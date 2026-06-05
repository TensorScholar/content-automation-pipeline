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
  const isMacos = variant === "macos";

  return (
    <div
      className={clsx(
        "inline-flex w-full items-center gap-0.5 justify-between p-1 sm:w-auto",
        isMacos
          ? "macos-segmented rounded-[10px]"
          : isDark
            ? "border border-white/10 bg-black/20"
            : "rounded-[10px] border border-black/5 bg-white/70 dark:border-white/10 dark:bg-white/[0.04]"
      )}
      role="tablist"
      aria-label={t("lang.select")}
    >
      {locales.map((entry) => (
        <button
          key={entry.code}
          type="button"
          onClick={() => setLocale(entry.code)}
          role="tab"
          aria-selected={locale === entry.code}
          aria-label={t(`lang.${entry.code}` as const)}
          className={clsx(
            "flex-1 rounded-[8px] px-3 py-1.5 text-center text-[13px] font-medium transition-[background-color,color,box-shadow,transform] duration-150 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)] focus-visible:outline-none",
            isMacos
              ? locale === entry.code
                ? "bg-white text-ink shadow-[0_1px_2px_rgb(0_0_0/0.06)] dark:bg-white/10 dark:text-white"
                : "text-ink-secondary hover:text-ink dark:text-gray-300 dark:hover:text-white"
              : locale === entry.code
                ? isDark
                ? "bg-white/15 text-white shadow-[0_1px_2px_rgb(255_255_255/0.05)] border border-white/10"
                : "bg-brand text-white shadow-[0_1px_2px_rgb(0_0_0/0.06)]"
              : isDark
                ? "text-white/60 hover:bg-white/5 hover:text-white"
                : "text-ink-tertiary hover:bg-surface-alt hover:text-ink-secondary",
          )}
        >
          {entry.nativeLabel}
        </button>
      ))}
    </div>
  );
}
