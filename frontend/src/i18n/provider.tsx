"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { messages } from "./messages";
import { Locale, MessageKey } from "./types";

/* ═══════════════════════════════════════════════════════════════════
   Locale Reactivity Engine
   Runtime locale changes are synchronous and require zero page reload.
   ═══════════════════════════════════════════════════════════════════ */

interface I18nContextValue {
  locale: Locale;
  direction: "rtl" | "ltr";
  setLocale: (locale: Locale) => void;
  t: (key: MessageKey, vars?: Record<string, string | number>) => string;
}

const STORAGE_KEY = "smarlux_lang";
const LEGACY_STORAGE_KEY = "cap.locale";
const rtlLocales = new Set<Locale>(["fa", "ar"]);

const LOCALE_CONFIG: Record<Locale, { dir: "rtl" | "ltr"; lang: string }> = {
  fa: { dir: "rtl", lang: "fa" },
  ar: { dir: "rtl", lang: "ar" },
  en: { dir: "ltr", lang: "en" },
};

/**
 * applyLocaleChange — applies the locale contract synchronously.
 *
 * 1. document.documentElement.dir = rtl|ltr
 * 2. document.documentElement.lang = fa|ar|en
 * 3. Direction-sensitive icons: toggle .rtl-active on <html>
 * 4. Persist to localStorage['smarlux_lang']
 * 5. Trigger i18n re-render (handled by React state update)
 * Typography is intentionally owned by the canonical CSS token system.
 */
function applyLocaleChange(locale: Locale): void {
  if (typeof document === "undefined") return;

  const config = LOCALE_CONFIG[locale];

  // 1. dir attribute on <html>
  document.documentElement.setAttribute("dir", config.dir);

  // 2. lang attribute on <html>
  document.documentElement.setAttribute("lang", config.lang);

  // 3. RTL icon class toggle
  if (config.dir === "rtl") {
    document.documentElement.classList.add("rtl-active");
  } else {
    document.documentElement.classList.remove("rtl-active");
  }

  // 4. Persist to localStorage
  try {
    localStorage.setItem(STORAGE_KEY, locale);
  } catch {
    // localStorage may be unavailable in some contexts
  }

  // 5. React state triggers re-render (caller's responsibility via setLocale)
}

/** Read persisted locale from localStorage (with legacy fallback) */
function readPersistedLocale(): Locale {
  if (typeof window === "undefined") return "fa";

  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === "fa" || stored === "ar" || stored === "en") return stored;

    // Legacy compat: migrate from old key
    const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (legacy === "fa" || legacy === "ar" || legacy === "en") {
      localStorage.setItem(STORAGE_KEY, legacy);
      localStorage.removeItem(LEGACY_STORAGE_KEY);
      return legacy;
    }
  } catch {
    // ignore
  }

  return "fa";
}

const I18nContext = createContext<I18nContextValue | undefined>(undefined);

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("fa");

  // Boot: restore locale and apply side-effects before first paint
  useEffect(() => {
    const persisted = readPersistedLocale();
    applyLocaleChange(persisted);
    setLocaleState(persisted);
  }, []);

  const setLocale = useCallback((nextLocale: Locale) => {
    applyLocaleChange(nextLocale);
    setLocaleState(nextLocale);
  }, []);

  const t = useCallback(
    (key: MessageKey, vars?: Record<string, string | number>) => {
      let msg = messages[locale][key] ?? messages.en[key] ?? key;
      if (vars) {
        for (const [k, v] of Object.entries(vars)) {
          msg = msg.replace(`{${k}}`, String(v));
        }
      }
      return msg;
    },
    [locale]
  );

  const value = useMemo<I18nContextValue>(
    () => ({
      locale,
      direction: rtlLocales.has(locale) ? "rtl" : "ltr",
      setLocale,
      t,
    }),
    [locale, setLocale, t]
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used inside I18nProvider");
  }
  return ctx;
}

/* ── Dev-only verification (Task 2.7) ───────────────────────────── */

export function __devVerifyLocaleSwitch(): void {
  if (process.env.NODE_ENV !== "development") return;

  const checks: Array<{ name: string; pass: boolean }> = [];

  applyLocaleChange("ar");
  checks.push({ name: "AR: dir=rtl", pass: document.documentElement.dir === "rtl" });
  checks.push({ name: "AR: lang=ar", pass: document.documentElement.lang === "ar" });
  applyLocaleChange("en");
  checks.push({ name: "EN: dir=ltr", pass: document.documentElement.dir === "ltr" });
  checks.push({ name: "EN: lang=en", pass: document.documentElement.lang === "en" });
  applyLocaleChange("fa");
  checks.push({ name: "FA: dir=rtl", pass: document.documentElement.dir === "rtl" });
  checks.push({ name: "FA: lang=fa", pass: document.documentElement.lang === "fa" });

  const failedChecks = checks.filter((check) => !check.pass);
  if (failedChecks.length > 0) {
    throw new Error(`i18n locale verification failed: ${failedChecks.map((check) => check.name).join(", ")}`);
  }
}
