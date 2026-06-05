"use client";

import clsx from "clsx";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";
import { LanguageToggle } from "./language-toggle";
import { MessageKey } from "@/i18n/types";
import Image from "next/image";

/* ═══════════════════════════════════════════════════════════════
   Login Page v3 — Apple/Linear SaaS Form Split
   - STRICT RTL MACRO-LAYOUT: Uses flex-row to naturally flip sides.
   - ACCESSIBILITY: Explicit labels above inputs; dropped floating tricks.
   - RECOVERY ACTION: Added "Forgot Password?" below the password input.
   - BREATHING ROOM: Increased hero banner line-heights and padding.
   - BIDI PERFECTION: `inset-inline-end` vs `right`, no `order` classes.
   ═══════════════════════════════════════════════════════════════ */

// ── Error resolution severity system ──────────────────────────────

type ErrorSeverity = "server" | "credentials" | "network";
interface ApiLikeError { status: number; detail?: string; }

function isApiLikeError(error: unknown): error is ApiLikeError {
  return typeof error === "object" && error !== null && "status" in error && typeof (error as ApiLikeError).status === "number";
}

function getErrorKey(error: unknown): MessageKey {
  if (error instanceof ApiError || isApiLikeError(error)) {
    const status = error instanceof ApiError ? error.status : (error as ApiLikeError).status;
    if (status === 0) return "auth.networkError";
    if (status === 401) return "auth.wrongCredentials";
    if (status === 403) return "auth.accountDisabled";
    if (status === 429) return "auth.tooManyAttempts";
    if (status === 503) return "auth.serviceUnavailable";
    if (status >= 500) return "auth.serverError";
  }
  if (error instanceof TypeError) return "auth.networkError";
  return "auth.serverError";
}

function getErrorSeverity(error: unknown): ErrorSeverity {
  if (error instanceof ApiError || isApiLikeError(error)) {
    const s = error instanceof ApiError ? error.status : (error as ApiLikeError).status;
    if (s === 0) return "network";
    if (s === 401 || s === 403) return "credentials";
    if (s === 503) return "network";
    if (s >= 500) return "server";
  }
  if (error instanceof TypeError) return "network";
  return "server";
}

const SEVERITY_STYLES: Record<ErrorSeverity, string> = {
  server: "border-s-[4px] border-s-danger bg-danger/5 text-danger",
  credentials: "border-s-[4px] border-s-warning bg-warning/5 text-warning",
  network: "border-s-[4px] border-s-info bg-info/5 text-info",
};
const SEVERITY_ICON: Record<ErrorSeverity, string> = {
  server: "🛑", credentials: "🔑", network: "📡",
};

const FAILED_ATTEMPTS_LIMIT = 5;
const COOLDOWN_SECONDS = 60;

const LOCKOUT_KEY = "cap.login_lockout_until";

function getLockoutSecondsRemaining(): number {
  try {
    const until = window.sessionStorage.getItem(LOCKOUT_KEY);
    if (!until) return 0;
    const remaining = Math.ceil((parseInt(until, 10) - Date.now()) / 1000);
    if (remaining <= 0) {
      window.sessionStorage.removeItem(LOCKOUT_KEY);
      return 0;
    }
    return remaining;
  } catch {
    return 0;
  }
}

function setLockout(seconds: number) {
  try {
    window.sessionStorage.setItem(LOCKOUT_KEY, String(Date.now() + seconds * 1000));
  } catch {}
}

// ── SVG Icons ───────────────────────────────────────────────────

function EyeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden className="h-5 w-5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3.8-6 10-6 10 6 10 6-3.8 6-10 6-10-6-10-6z" /><circle cx="12" cy="12" r="3" />
    </svg>
  );
}
function EyeOffIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden className="h-5 w-5" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3l18 18" /><path d="M10.6 10.6A3 3 0 0 0 13.4 13.4" />
      <path d="M9.9 5.1A11 11 0 0 1 12 5c6.2 0 10 7 10 7a18.8 18.8 0 0 1-4.2 4.8" />
      <path d="M6.4 6.5A19 19 0 0 0 2 12s3.8 7 10 7c1.7 0 3.2-.4 4.6-1" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden className="h-3.5 w-3.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8.5 6.3 11.7 13 5" />
    </svg>
  );
}
function DismissIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden className="h-4 w-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round">
      <path d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

// ── SEO helpers ───────────────────────────────────────────────

function upsertMeta(name: string, content: string) {
  let meta = document.querySelector(`meta[name='${name}']`) as HTMLMetaElement | null;
  if (!meta) { meta = document.createElement("meta"); meta.name = name; document.head.appendChild(meta); }
  meta.content = content;
}
function upsertLink(rel: string, href: string, hrefLang?: string) {
  const selector = hrefLang ? `link[rel='${rel}'][hreflang='${hrefLang}']` : `link[rel='${rel}']:not([hreflang])`;
  let link = document.querySelector(selector) as HTMLLinkElement | null;
  if (!link) { link = document.createElement("link"); link.rel = rel; if (hrefLang) link.hreflang = hrefLang; document.head.appendChild(link); }
  link.href = href;
}

// ── Flowing mesh gradient animation ───────────────────────────

function HeroMeshGradient() {
  return (
    <div className="absolute inset-0 overflow-hidden" aria-hidden>
      <div className="absolute inset-0" style={{
        background: `
          radial-gradient(ellipse 60% 50% at 20% 30%, rgb(var(--color-text-primary) / 0.035) 0%, transparent 70%),
          radial-gradient(ellipse 50% 60% at 70% 60%, rgb(var(--color-text-secondary) / 0.025) 0%, transparent 70%),
          radial-gradient(ellipse 40% 40% at 50% 80%, rgb(var(--color-border-primary) / 0.45) 0%, transparent 70%)
        `,
        animation: "hero-mesh-drift 20s ease-in-out infinite alternate",
      }} />
      <svg viewBox="0 0 520 280" className="absolute inset-0 h-full w-full text-ink-tertiary opacity-[0.08]" fill="none" aria-hidden>
        <g stroke="currentColor" strokeWidth="0.5">
          <path d="M48 44h86" /><path d="M184 44h74" /><path d="M304 44h80" />
          <path d="M96 126h84" /><path d="M228 126h82" /><path d="M358 126h92" />
          <path d="M82 208h92" /><path d="M218 208h86" /><path d="M352 208h92" />
          <path d="M92 60v50" /><path d="M268 60v50" /><path d="M398 60v50" />
          <path d="M140 142v50" /><path d="M318 142v50" />
        </g>
        {[{ cx: 48, cy: 44 }, { cx: 142, cy: 44 }, { cx: 192, cy: 44 }, { cx: 268, cy: 44 }, { cx: 324, cy: 44 }, { cx: 398, cy: 44 }, { cx: 48, cy: 126 }, { cx: 96, cy: 126 }, { cx: 192, cy: 126 }, { cx: 324, cy: 126 }, { cx: 458, cy: 126 }, { cx: 82, cy: 208 }, { cx: 218, cy: 208 }, { cx: 318, cy: 208 }, { cx: 458, cy: 208 }].map(n => (
          <circle key={`${n.cx}-${n.cy}`} cx={n.cx} cy={n.cy} r="2.5" fill="currentColor" opacity="0.35" />
        ))}
      </svg>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
//  LoginCard component
// ═══════════════════════════════════════════════════════════════

export function LoginCard() {
  const { login } = useAuth();
  const { t, locale, direction } = useI18n();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [loginSuccess, setLoginSuccess] = useState(false);
  const [authError, setAuthError] = useState<unknown>(null);
  const [emailTouched, setEmailTouched] = useState(false);
  const [errorDismissed, setErrorDismissed] = useState(false);
  const [shakeButton, setShakeButton] = useState(false);
  const [, setFailedAttempts] = useState(0);
  const [cooldownRemaining, setCooldownRemaining] = useState(() => getLockoutSecondsRemaining());
  const errorTimerRef = useRef<number | null>(null);
  const shakeTimerRef = useRef<number | null>(null);

  // Email validation — ONLY emailInvalid triggers red on email field
  const emailInvalid = emailTouched && email.length > 0 && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const errorKey = authError && !errorDismissed ? getErrorKey(authError) : null;
  const localizedError = errorKey ? t(errorKey) : null;
  const errorSeverity = authError && !errorDismissed ? getErrorSeverity(authError) : null;

  const cooldownText = useMemo(() => t("auth.cooldown", { seconds: String(cooldownRemaining) }), [cooldownRemaining, t]);

  // Clean Persian tagline formatting (replace zero with bullet)
  const cleanTagline = useMemo(() => {
    let text = t("auth.heroTagline");
    return text ? text.replace(" 0 ", " • ") : text;
  }, [t]);

  // Safe fallback localization
  const safe_t = (key: string, fallback: string) => {
    const val = t(key as any);
    return val && val !== key ? val : fallback;
  };

  useEffect(() => {
    if (cooldownRemaining <= 0) return;
    const timer = window.setInterval(() => setCooldownRemaining(s => Math.max(0, s - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [cooldownRemaining]);

  // Auto-dismiss non-critical errors after 8s
  useEffect(() => {
    if (!authError || errorDismissed) return;
    if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current);
    const severity = getErrorSeverity(authError);
    if (severity !== "server") {
      errorTimerRef.current = window.setTimeout(() => setErrorDismissed(true), 8000);
    }
    return () => { if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current); };
  }, [authError, errorDismissed]);

  useEffect(() => {
    return () => {
      if (shakeTimerRef.current) window.clearTimeout(shakeTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.title = `${t("auth.title")} — ${t("app.name")}`;
    upsertMeta("description", t("auth.subtitle"));
    upsertMeta("robots", "noindex, nofollow");
    const origin = typeof window !== "undefined" ? window.location.origin : "";
    upsertLink("canonical", `${origin}/?lang=fa`);
    upsertLink("alternate", `${origin}/?lang=en`, "en");
    upsertLink("alternate", `${origin}/?lang=fa`, "fa");
    upsertLink("alternate", `${origin}/?lang=ar`, "ar");
  }, [locale, t]);

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (cooldownRemaining > 0) return;
    setAuthError(null);
    setErrorDismissed(false);
    setSubmitting(true);
    setLoginSuccess(false);
    try {
      await login(email, password, rememberMe);
      setFailedAttempts(0);
      try { window.sessionStorage.removeItem(LOCKOUT_KEY); } catch {}
      setLoginSuccess(true);
    } catch (error) {
      setAuthError(error);
      setShakeButton(true);
      if (shakeTimerRef.current) window.clearTimeout(shakeTimerRef.current);
      shakeTimerRef.current = window.setTimeout(() => setShakeButton(false), 400);
      // If backend returned 429, use its Retry-After duration; otherwise count locally
      const is429 = (error instanceof ApiError && error.status === 429) ||
                    (isApiLikeError(error) && (error as ApiLikeError).status === 429);
      if (is429) {
        const retryAfter = error instanceof ApiError
          ? (error.retryAfter ?? 60)
          : 60;
        setLockout(retryAfter);
        setCooldownRemaining(retryAfter);
        setFailedAttempts(0);
        return;
      }
      setFailedAttempts(prev => {
        const next = prev + 1;
        if (next >= FAILED_ATTEMPTS_LIMIT) {
          setLockout(COOLDOWN_SECONDS);
          setCooldownRemaining(COOLDOWN_SECONDS);
          return 0;
        }
        return next;
      });
    } finally {
      setSubmitting(false);
    }
  };

  // Dynamic system status — red when there was a recent server error
  const systemHealthy = !authError || getErrorSeverity(authError) !== "server";

  return (
    <main className="macos-app-bg mx-auto flex min-h-screen w-full max-w-6xl items-center px-4 py-6">
      <div
        dir={direction} /* 100% Native RTL Layout Mirroring via DOM Engine */
        className="macos-grouped-surface w-full flex flex-col lg:flex-row overflow-hidden rounded-[18px]"
        style={{ boxShadow: "0 8px 32px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04)" }}
      >

        {/* ═══ HERO PANEL (Takes exactly 50% width on Desktop) ═══ */}
        <section
          className="auth-hero-surface relative hidden min-h-[640px] flex-1 overflow-hidden text-ink lg:flex lg:flex-col"
        >
          {/* Ghost Watermark Logo - Pushed to the corner */}
          <div className="absolute -bottom-32 -end-32 w-[140%] opacity-[0.03] mix-blend-overlay pointer-events-none z-0" style={{ transform: direction === "rtl" ? "scaleX(-1)" : "none" }}>
            <Image src="/logo.png" alt="" width={500} height={500} className="w-full h-auto object-contain" style={{ imageRendering: "auto" }} />
          </div>

          <HeroMeshGradient />

          <div key={`hero-${locale}`} className="relative z-10 flex h-full flex-col p-12 lg:px-16 animate-fade-in">

            <div className="flex flex-1 items-center justify-center">
              {/* Increased breathing room gap and optical elevation */}
              <div className="flex flex-col items-center text-center gap-6 max-w-lg w-full mb-16">

                <div className="flex items-center justify-center gap-3 animate-fade-in" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                  <span className="block h-[3px] w-8 rounded-full bg-border" />
                  <span className="text-xs font-bold uppercase tracking-widest text-ink-tertiary">
                    AI-POWERED PLATFORM
                  </span>
                  <span className="block h-[3px] w-8 rounded-full bg-border" />
                </div>

                {/* Typography Confidence: text-3xl/4xl */}
                <h1
                  className="text-3xl md:text-4xl font-extrabold text-ink tracking-tight leading-snug animate-fade-in w-full text-balance text-center"
                  style={{ opacity: 0, animationDelay: "100ms", animationFillMode: 'forwards' }}
                >
                  {t("auth.heroHeadline").split("·").length === 2 ? (
                    <>
                      {t("auth.heroHeadline").split("·")[0].trim()}
                      <span className="mx-4 inline-block font-light text-ink-tertiary">&middot;</span>
                      {t("auth.heroHeadline").split("·")[1].trim()}
                    </>
                  ) : (
                    t("auth.heroHeadline")
                  )}
                </h1>

                {/* Decoupled from Glassmorphic Card */}
                <p
                  className="whitespace-pre-line text-lg text-ink-secondary font-medium max-w-md mt-2 leading-relaxed text-center animate-fade-in"
                  style={{ opacity: 0, animationDelay: "250ms", animationFillMode: 'forwards' }}
                >
                  {cleanTagline}
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ═══ MOBILE OVERRIDES ═══ */}
        <section className="auth-hero-surface relative hidden overflow-hidden border-b border-black/5 px-8 py-8 text-ink dark:border-white/10 md:block lg:hidden">
          <div className="absolute -bottom-16 -end-24 w-[130%] opacity-[0.04] mix-blend-overlay pointer-events-none" style={{ transform: direction === "rtl" ? "scaleX(-1)" : "none" }}>
            <Image src="/logo.png" alt="" width={500} height={500} className="w-full h-auto object-contain" style={{ imageRendering: "auto" }} />
          </div>
          <HeroMeshGradient />
          <div key={`tb-hero-${locale}`} className="relative z-10 space-y-3 animate-fade-in flex flex-col items-center text-center">
            <div className="flex items-center justify-center gap-3">
              <span className="block h-[2px] w-6 rounded-full bg-border" />
              <span className="text-[11px] font-bold uppercase tracking-widest text-ink-tertiary">AI-POWERED PLATFORM</span>
              <span className="block h-[2px] w-6 rounded-full bg-border" />
            </div>
            <p className="text-[22px] font-bold leading-snug text-ink">{cleanTagline}</p>
          </div>
        </section>

        <section className="auth-hero-surface relative overflow-hidden border-b border-black/5 px-6 py-5 text-ink dark:border-white/10 md:hidden">
          <div className="absolute -bottom-10 -end-10 w-[150%] opacity-[0.04] mix-blend-overlay pointer-events-none" style={{ transform: direction === "rtl" ? "scaleX(-1)" : "none" }}>
            <Image src="/logo.png" alt="" width={500} height={500} className="w-full h-auto object-contain" style={{ imageRendering: "auto" }} />
          </div>
          <div key={`mb-hero-${locale}`} className="relative z-10 flex flex-col items-center justify-center gap-2 animate-fade-in text-center">
            <div className="flex items-center justify-center gap-2">
              <span className="block h-[2px] w-4 rounded-full bg-border" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">PLATFORM</span>
              <span className="block h-[2px] w-4 rounded-full bg-border" />
            </div>
            <span className="inline-flex items-center gap-2 text-[12px] font-medium text-ink-secondary">
              <span className={clsx("h-2 w-2 rounded-full", systemHealthy ? "bg-ink-tertiary" : "bg-danger")} style={{ animation: "status-pulse 2s ease-in-out infinite" }} aria-hidden />{systemHealthy ? t("auth.systemOnline") : t("auth.systemDegraded")}
            </span>
          </div>
        </section>

        {/* ═══ FORM PANEL (Takes 50% width on Desktop, positioned purely by flex DOM logic) ═══ */}
        <section className="relative flex min-h-[640px] flex-col justify-center bg-surface px-8 py-10 text-ink sm:px-14 lg:w-1/2">

          {/* Strict logical end-alignment for Language Switcher */}
          <div className="absolute top-8 end-8 z-50 flex animate-fade-in items-center gap-2 text-ink-tertiary">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" /><path d="M2 12h20" /></svg>
            <LanguageToggle variant="macos" />
          </div>

          <div className="w-full max-w-sm mx-auto flex flex-col pt-8 lg:pt-0">
            <div key={`form-title-${locale}`} className="animate-fade-in mb-8">
              <h3 className="text-[28px] font-bold tracking-tight text-ink">{t("auth.title")}</h3>
              <p className="mt-2 text-[15px] font-medium text-ink-secondary">{t("auth.subtitle")}</p>
            </div>

            <form method="POST" autoComplete="on" onSubmit={onSubmit} className="flex flex-col gap-5">

              {/* ── Error alert removed from here ── */}

              {cooldownRemaining > 0 && (
                <div className="rounded-xl border border-warning/20 border-s-[4px] border-s-warning bg-warning/10 px-4 py-3">
                  <p className="text-[13px] font-medium text-warning">{cooldownText}</p>
                </div>
              )}

              {/* ── Email Field (Cognitive Structure: Explicit Labels Above) ── */}
              <div className="animate-fade-in flex flex-col items-start gap-1.5 w-full" style={{ animationDelay: "60ms", animationFillMode: 'forwards', opacity: 0 }}>
                <label htmlFor="login-email" className="text-[13px] font-semibold text-ink-secondary">{safe_t("auth.email", "Email Address")}</label>
                <div className="relative w-full flex items-center">
                  <input
                    id="login-email"
                    type="email"
                    dir="ltr"
                    autoComplete="username"
                    required
                    aria-invalid={emailInvalid}
                    aria-label={t("auth.email")}
                    className={clsx(
                      "auth-input h-[50px] w-full rounded-xl border px-4 text-[15px] outline-none transition-all duration-200 text-start",
                      "focus:border-brand focus:ring-2 focus:ring-brand/20",
                      emailInvalid ? "border-danger shadow-[0_0_0_3px_rgb(var(--color-error)/0.08)]" : "border-black/10 dark:border-white/10"
                    )}
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    onBlur={() => setEmailTouched(true)}
                  />
                </div>
                {emailInvalid && (
                  <p className="mt-1 flex items-center gap-1 text-[12px] font-semibold text-red-500"><span aria-hidden>⚠</span>{t("auth.invalidEmail")}</p>
                )}
              </div>

              {/* ── Password Field (Cognitive Structure) ── */}
              <div className="animate-fade-in flex flex-col items-start gap-1.5 w-full" style={{ animationDelay: "130ms", animationFillMode: 'forwards', opacity: 0 }}>
                <label htmlFor="login-password" className="text-[13px] font-semibold text-ink-secondary">{safe_t("auth.password", "Password")}</label>
                <div className="relative w-full flex items-center">
                  <input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    dir="ltr"
                    autoComplete="current-password"
                    required
                    aria-label={t("auth.password")}
                    className={clsx(
                      "auth-input h-[50px] w-full rounded-xl border border-black/10 px-4 pe-[48px] text-[15px] outline-none transition-all duration-200 text-start dark:border-white/10",
                      "focus:border-brand focus:ring-2 focus:ring-brand/20",
                      "[&::-ms-reveal]:hidden [&::-webkit-credentials-auto-fill-button]:hidden lg:[&::-webkit-contacts-auto-fill-button]:hidden"
                    )}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                  />
                  {/* BIDI Precise: end-3 guarantees RTL mirror */}
                  <button
                    type="button"
                    className="absolute end-3 z-10 rounded-lg p-1.5 text-ink-tertiary transition-colors duration-200 hover:text-ink focus:bg-surface focus:outline-none focus:ring-2 focus:ring-brand"
                    onClick={() => setShowPassword(o => !o)}
                    aria-label={showPassword ? t("auth.hidePassword") : t("auth.showPassword")}
                  >
                    {showPassword ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
              </div>

              {/* ── Recovery & Utility Row (Absolute Minimalism) ── */}
              <div className="animate-fade-in flex items-center justify-start mt-1" style={{ animationDelay: "190ms", animationFillMode: 'forwards', opacity: 0 }}>
                {/* Remember me logical property lock (strict flex container) */}
                <label className="group flex w-full cursor-pointer select-none items-center gap-2 text-[13px] font-medium text-ink-secondary">
                  <input type="checkbox" checked={rememberMe} onChange={e => setRememberMe(e.target.checked)} className="peer sr-only" />
                  <span className="grid h-[20px] w-[20px] shrink-0 place-items-center rounded-md border-[1.5px] border-black/10 bg-surface text-white transition-all duration-200 peer-checked:border-ink peer-checked:bg-ink peer-focus-visible:ring-2 peer-focus-visible:ring-brand/30 group-hover:border-ink dark:border-white/10 dark:peer-checked:border-white dark:peer-checked:bg-white dark:peer-checked:text-ink">
                    <CheckIcon />
                  </span>
                  <span>{t("auth.rememberMe")}</span>
                </label>
              </div>

              {/* ── Error alert (Moved down here and redesigned) ── */}
              {localizedError && errorSeverity && (
                <div
                  className="my-1 flex animate-fade-in items-center gap-2 rounded-lg border border-danger/20 bg-danger/10 p-3"
                  role="alert"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4 shrink-0 text-danger" aria-hidden><circle cx="12" cy="12" r="10" /><line x1="12" x2="12" y1="8" y2="12" /><line x1="12" x2="12.01" y1="16" y2="16" /></svg>
                  <p className="text-sm font-medium text-danger">{localizedError}</p>
                </div>
              )}

              {/* ── Submit CTA ── */}
              <button
                type="submit"
                disabled={submitting || cooldownRemaining > 0 || loginSuccess}
                className={clsx(
                  "flex w-full animate-fade-in items-center justify-center gap-2 rounded-xl bg-ink px-4 text-[15px] font-medium text-white shadow-md transition-all duration-200 hover:-translate-y-0.5 hover:bg-ink-secondary active:translate-y-0 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0 dark:bg-white dark:text-gray-950 dark:hover:bg-gray-200",
                  shakeButton && "animate-shake",
                  loginSuccess && "!bg-emerald-500 shadow-emerald-500/20",
                )}
                style={{ height: 50, animationDelay: "250ms", marginTop: localizedError ? 4 : 12, animationFillMode: 'forwards', opacity: 0 }}
              >
                {loginSuccess ? (
                  <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 10.5 8 14.5 16 6.5" /></svg>
                ) : submitting ? (
                  <><span className="h-[22px] w-[22px] rounded-full border-2 border-white/30 border-t-white animate-spin shrink-0" />{t("auth.submitting")}</>
                ) : (
                  safe_t("auth.submit", "Sign In")
                )}
              </button>

            </form>
          </div>
        </section>
      </div>
    </main>
  );
}
