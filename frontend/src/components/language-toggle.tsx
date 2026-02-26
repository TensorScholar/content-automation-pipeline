"use client";

import clsx from "clsx";
import { useI18n } from "@/i18n/provider";
import { Locale } from "@/i18n/types";

const locales: Array<{ code: Locale; nativeLabel: string }> = [
  { code: "fa", nativeLabel: "FA" },
  { code: "ar", nativeLabel: "AR" },
  { code: "en", nativeLabel: "EN" },
];

export function LanguageToggle() {
  const { locale, setLocale, t } = useI18n();

  return (
    <div
      className="inline-flex items-center gap-0.5 rounded-full border border-border bg-surface-alt p-1"
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
            "rounded-full px-3 py-1 text-body-sm font-semibold transition-all duration-fast",
            "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand",
            locale === entry.code
              ? "bg-brand text-white shadow-sm"
              : "text-ink-tertiary hover:bg-surface-alt hover:text-ink-secondary",
          )}
        >
          {entry.nativeLabel}
        </button>
      ))}
    </div>
  );
}
