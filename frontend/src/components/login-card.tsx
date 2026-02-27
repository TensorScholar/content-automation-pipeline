"use client";

import clsx from "clsx";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";
import { LanguageToggle } from "./language-toggle";
import { MessageKey } from "@/i18n/types";

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
    if (status === 401) return "auth.wrongCredentials";
    if (status === 403) return "auth.accountDisabled";
    if (status === 429) return "auth.tooManyAttempts";
    if (status >= 500) return "auth.serverError";
  }
  if (error instanceof TypeError) return "auth.networkError";
  return "auth.serverError";
}

function getErrorSeverity(error: unknown): ErrorSeverity {
  if (error instanceof ApiError || isApiLikeError(error)) {
    const s = error instanceof ApiError ? error.status : (error as ApiLikeError).status;
    if (s === 401 || s === 403) return "credentials";
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
          radial-gradient(ellipse 60% 50% at 20% 30%, rgba(14,110,110,0.35) 0%, transparent 70%),
          radial-gradient(ellipse 50% 60% at 70% 60%, rgba(6,78,78,0.3) 0%, transparent 70%),
          radial-gradient(ellipse 40% 40% at 50% 80%, rgba(20,140,140,0.25) 0%, transparent 70%)
        `,
        animation: "hero-mesh-drift 20s ease-in-out infinite alternate",
      }} />
      <svg viewBox="0 0 520 280" className="absolute inset-0 h-full w-full opacity-[0.08]" fill="none" aria-hidden>
        <g stroke="white" strokeWidth="0.5">
          <path d="M48 44h86" /><path d="M184 44h74" /><path d="M304 44h80" />
          <path d="M96 126h84" /><path d="M228 126h82" /><path d="M358 126h92" />
          <path d="M82 208h92" /><path d="M218 208h86" /><path d="M352 208h92" />
          <path d="M92 60v50" /><path d="M268 60v50" /><path d="M398 60v50" />
          <path d="M140 142v50" /><path d="M318 142v50" />
        </g>
        {[{ cx: 48, cy: 44 }, { cx: 142, cy: 44 }, { cx: 192, cy: 44 }, { cx: 268, cy: 44 }, { cx: 324, cy: 44 }, { cx: 398, cy: 44 }, { cx: 48, cy: 126 }, { cx: 96, cy: 126 }, { cx: 192, cy: 126 }, { cx: 324, cy: 126 }, { cx: 458, cy: 126 }, { cx: 82, cy: 208 }, { cx: 218, cy: 208 }, { cx: 318, cy: 208 }, { cx: 458, cy: 208 }].map(n => (
          <circle key={`${n.cx}-${n.cy}`} cx={n.cx} cy={n.cy} r="2.5" fill="rgba(255,255,255,0.4)" />
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
  const [cooldownRemaining, setCooldownRemaining] = useState(0);
  const errorTimerRef = useRef<number | null>(null);

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
      setLoginSuccess(true);
    } catch (error) {
      setAuthError(error);
      setShakeButton(true);
      window.setTimeout(() => setShakeButton(false), 400);
      setFailedAttempts(prev => {
        const next = prev + 1;
        if (next >= FAILED_ATTEMPTS_LIMIT) { setCooldownRemaining(COOLDOWN_SECONDS); return 0; }
        return next;
      });
    } finally {
      setSubmitting(false);
    }
  };

  // Dynamic system status — red when there was a recent server error
  const systemHealthy = !authError || getErrorSeverity(authError) !== "server";

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl items-center px-4 py-6" style={{ background: "radial-gradient(ellipse at 50% 40%, #F0F7F7 0%, #F0F2F5 70%)" }}>
      <div
        dir={direction} /* 100% Native RTL Layout Mirroring via DOM Engine */
        className="w-full flex flex-col lg:flex-row overflow-hidden rounded-[24px] border border-black/[0.04] bg-surface"
        style={{ boxShadow: "0 8px 32px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04)" }}
      >

        {/* ═══ HERO PANEL (Takes exactly 50% width on Desktop) ═══ */}
        <section
          className="relative hidden min-h-[640px] flex-1 overflow-hidden text-white lg:flex lg:flex-col bg-gradient-to-br from-[#064e3b] via-[#022c22] to-[#011a14]"
        >
          {/* Ghost Watermark Logo */}
          <div className="absolute -bottom-32 -inset-inline-end-32 w-[130%] z-0 opacity-[0.04] mix-blend-overlay pointer-events-none" style={{ transform: direction === "rtl" ? "scaleX(-1)" : "none" }}>
            <img src="/logo.png" alt="" className="w-full h-auto object-contain" style={{ imageRendering: "auto" }} />
          </div>

          <HeroMeshGradient />

          <div key={`hero-${locale}`} className="relative z-10 flex h-full flex-col p-12 animate-fade-in">
            {/* Brand — ALWAYS Latin */}
            <p className="text-[12px] font-bold uppercase tracking-[0.18em]" style={{ color: "rgba(255,255,255,0.72)" }}>
              SMARLUX CONTENT OS
            </p>

            <div className="flex flex-1 items-center justify-center">
              {/* Increased breathing room gap */}
              <div className="flex flex-col items-center text-center gap-8 max-w-md w-full">

                <div className="flex items-center justify-center gap-3 animate-fade-in" style={{ opacity: 0, animationFillMode: 'forwards' }}>
                  <span className="block h-[3px] w-10 rounded-full bg-gradient-to-r from-teal-400 to-teal-400/20" />
                  <span className="text-[13px] font-bold uppercase tracking-[0.2em] text-teal-400/90">
                    AI-Powered Platform
                  </span>
                  <span className="block h-[3px] w-10 rounded-full bg-gradient-to-l from-teal-400 to-teal-400/20" />
                </div>

                {/* Breathing Room: Use leading-snug instead of leading-[1.2] */}
                <h1
                  className="text-[32px] md:text-[36px] font-extrabold leading-snug tracking-wide text-white animate-fade-in w-full text-balance"
                  style={{
                    textShadow: "0 2px 24px rgba(94,234,212,0.3), 0 0 60px rgba(94,234,212,0.12)",
                    opacity: 0, animationDelay: "100ms", animationFillMode: 'forwards'
                  }}
                >
                  {t("auth.heroHeadline")}
                </h1>

                {/* Breathing Room: Increased padding inside the glass card */}
                <div
                  className="inline-block rounded-[20px] px-8 py-5 animate-fade-in bg-white/5 backdrop-blur-md border border-white/10"
                  style={{ opacity: 0, animationDelay: "250ms", animationFillMode: 'forwards' }}
                >
                  <p className="whitespace-pre-line text-[16px] leading-relaxed text-teal-50 font-medium">
                    {cleanTagline}
                  </p>
                </div>
              </div>
            </div>

            {/* Dynamic system status with breathing pulse */}
            <div className="space-y-2">
              <div className="inline-flex items-center gap-2 text-white/80">
                <span className={clsx("h-2 w-2 rounded-full", systemHealthy ? "bg-emerald-500" : "bg-red-500")} style={{ animation: "status-pulse 2s ease-in-out infinite" }} aria-hidden />
                <p className="text-body-sm font-medium">
                  {systemHealthy ? t("auth.systemOnline") : t("auth.systemDegraded")}
                </p>
              </div>
              <p className="font-mono text-[11px] text-white/40 mt-2.5">
                {t("app.version")}
              </p>
            </div>
          </div>
        </section>

        {/* ═══ MOBILE OVERRIDES ═══ */}
        <section className="relative hidden overflow-hidden border-b border-white/10 px-8 py-6 text-white md:block lg:hidden bg-gradient-to-br from-[#064e3b] via-[#022c22] to-[#011a14]">
          <div className="absolute -bottom-16 -inset-inline-end-16 w-[130%] z-0 opacity-[0.04] mix-blend-overlay pointer-events-none" style={{ transform: direction === "rtl" ? "scaleX(-1)" : "none" }}>
            <img src="/logo.png" alt="" className="w-full h-auto object-contain" style={{ imageRendering: "auto" }} />
          </div>
          <HeroMeshGradient />
          <div key={`tb-hero-${locale}`} className="relative z-10 space-y-2 animate-fade-in">
            <p className="text-[12px] font-bold uppercase tracking-[0.14em] text-white/70">SMARLUX CONTENT OS</p>
            <p className="text-[18px] font-medium text-white/90">{cleanTagline}</p>
          </div>
        </section>

        <section className="relative overflow-hidden border-b border-white/10 px-5 py-4 text-white md:hidden bg-gradient-to-br from-[#064e3b] via-[#022c22] to-[#011a14]">
          <div className="absolute -bottom-10 -inset-inline-end-10 w-[150%] z-0 opacity-[0.04] mix-blend-overlay pointer-events-none" style={{ transform: direction === "rtl" ? "scaleX(-1)" : "none" }}>
            <img src="/logo.png" alt="" className="w-full h-auto object-contain" style={{ imageRendering: "auto" }} />
          </div>
          <div key={`mb-hero-${locale}`} className="relative z-10 flex items-center justify-between animate-fade-in">
            <p className="text-[11px] font-bold uppercase tracking-[0.15em] text-white/80">SMARLUX CONTENT OS</p>
            <span className="inline-flex items-center gap-2 text-[12px] text-white/80 font-medium">
              <span className={clsx("h-2 w-2 rounded-full", systemHealthy ? "bg-emerald-500" : "bg-red-500")} style={{ animation: "status-pulse 2s ease-in-out infinite" }} aria-hidden />{systemHealthy ? t("auth.systemOnline") : t("auth.systemDegraded")}
            </span>
          </div>
        </section>

        {/* ═══ FORM PANEL (Takes 50% width on Desktop, positioned purely by flex DOM logic) ═══ */}
        <section className="bg-surface px-8 py-10 sm:px-14 lg:w-1/2 flex flex-col justify-center relative min-h-[640px]">

          {/* Strict logical end-alignment for Language Switcher */}
          <div className="absolute top-6 inset-inline-end-6 animate-fade-in z-50">
            <LanguageToggle />
          </div>

          <div className="w-full max-w-sm mx-auto flex flex-col pt-8 lg:pt-0">
            <div key={`form-title-${locale}`} className="animate-fade-in mb-8">
              <img src="/logo.png" alt="Smarlux" className="h-12 w-auto mb-6 object-contain" style={{ imageRendering: "auto" }} />
              <h3 className="text-[28px] font-bold text-slate-900 tracking-tight">{t("auth.title")}</h3>
              <p className="mt-2 text-[15px] font-medium text-slate-500">{t("auth.subtitle")}</p>
            </div>

            <form method="POST" autoComplete="on" onSubmit={onSubmit} className="flex flex-col gap-5">

              {/* ── Error alert ── */}
              {localizedError && errorSeverity && (
                <div
                  className={clsx("flex items-start gap-3 rounded-xl px-4 py-3 border", SEVERITY_STYLES[errorSeverity])}
                  role="alert"
                  style={{ animation: "login-slide-in 200ms ease-out" }}
                >
                  <span className="shrink-0 text-[18px]" aria-hidden>{SEVERITY_ICON[errorSeverity]}</span>
                  <p className="flex-1 text-[13px] font-medium">{localizedError}</p>
                  <button
                    type="button"
                    onClick={() => setErrorDismissed(true)}
                    className="shrink-0 rounded p-0.5 opacity-60 transition-opacity hover:opacity-100"
                    aria-label={t("auth.dismiss")}
                  >
                    <DismissIcon />
                  </button>
                </div>
              )}

              {cooldownRemaining > 0 && (
                <div className="rounded-xl border-s-[4px] border border-warning bg-warning/5 px-4 py-3">
                  <p className="text-[13px] font-medium text-warning-700">{cooldownText}</p>
                </div>
              )}

              {/* ── Email Field (Cognitive Structure: Explicit Labels Above) ── */}
              <div className="animate-fade-in flex flex-col gap-1.5" style={{ animationDelay: "60ms", animationFillMode: 'forwards', opacity: 0 }}>
                <label htmlFor="login-email" className="text-[13px] font-semibold text-slate-700">{safe_t("auth.email", "Email Address")}</label>
                <input
                  id="login-email"
                  type="email"
                  dir="ltr"
                  autoComplete="username"
                  required
                  aria-invalid={emailInvalid}
                  aria-label={t("auth.email")}
                  className={clsx(
                    "auth-input h-[50px] w-full rounded-xl border px-4 text-[15px] text-slate-900 outline-none transition-all duration-200 text-start",
                    "focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20 bg-slate-50 focus:bg-white",
                    emailInvalid ? "border-red-500 shadow-[0_0_0_3px_rgba(239,68,68,0.08)] bg-red-50/30" : "border-slate-200"
                  )}
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  onBlur={() => setEmailTouched(true)}
                />
                {emailInvalid && (
                  <p className="mt-1 flex items-center gap-1 text-[12px] font-semibold text-red-500"><span aria-hidden>⚠</span>{t("auth.invalidEmail")}</p>
                )}
              </div>

              {/* ── Password Field (Cognitive Structure) ── */}
              <div className="animate-fade-in flex flex-col gap-1.5" style={{ animationDelay: "130ms", animationFillMode: 'forwards', opacity: 0 }}>
                <label htmlFor="login-password" className="text-[13px] font-semibold text-slate-700">{safe_t("auth.password", "Password")}</label>
                <div className="relative">
                  <input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    dir="ltr"
                    autoComplete="current-password"
                    required
                    aria-label={t("auth.password")}
                    className={clsx(
                      "auth-input h-[50px] w-full rounded-xl border px-4 pe-[48px] text-[15px] text-slate-900 outline-none transition-all duration-200 text-start",
                      "focus:border-teal-600 focus:ring-2 focus:ring-teal-600/20 bg-slate-50 focus:bg-white border-slate-200"
                    )}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                  />
                  {/* BIDI Precise: inset-inline-end guarantees RTL mirror */}
                  <button
                    type="button"
                    className="absolute inset-inline-end-3 top-1/2 z-10 -translate-y-1/2 rounded-lg p-1.5 transition-colors duration-200 text-slate-400 hover:text-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:bg-white"
                    onClick={() => setShowPassword(o => !o)}
                    aria-label={showPassword ? t("auth.hidePassword") : t("auth.showPassword")}
                  >
                    {showPassword ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
              </div>

              {/* ── Recovery & Utility Row (Apple SaaS Tier) ── */}
              <div className="animate-fade-in flex items-center justify-between mt-1" style={{ animationDelay: "190ms", animationFillMode: 'forwards', opacity: 0 }}>
                {/* Remember me logical property lock (gap-2 forces inline-start text) */}
                <label className="inline-flex cursor-pointer select-none items-center gap-2.5 text-[13px] font-medium text-slate-600 group">
                  <input type="checkbox" checked={rememberMe} onChange={e => setRememberMe(e.target.checked)} className="peer sr-only" />
                  <span className="grid h-[20px] w-[20px] shrink-0 place-items-center rounded-md border-[1.5px] border-slate-300 bg-white text-white transition-all duration-200 peer-checked:border-teal-600 peer-checked:bg-teal-600 peer-focus-visible:ring-2 peer-focus-visible:ring-teal-600/30 group-hover:border-teal-500">
                    <CheckIcon />
                  </span>
                  <span>{t("auth.rememberMe")}</span>
                </label>

                {/* New Forgot Password Link */}
                <button type="button" className="text-[13px] font-semibold text-teal-600 hover:text-teal-700 hover:underline hover:underline-offset-2 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:rounded-sm">
                  {safe_t("auth.forgotPassword", "Forgot password?")}
                </button>
              </div>

              {/* ── Submit CTA ── */}
              <button
                type="submit"
                disabled={submitting || cooldownRemaining > 0 || loginSuccess}
                className={clsx(
                  "animate-fade-in w-full rounded-xl bg-teal-700 px-4 text-[15px] font-medium text-white shadow-md shadow-teal-700/20 transition-all duration-200 hover:bg-teal-800 hover:-translate-y-0.5 active:translate-y-0 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0 flex items-center justify-center gap-2",
                  shakeButton && "animate-shake",
                  loginSuccess && "!bg-emerald-500 shadow-emerald-500/20",
                )}
                style={{ height: 50, animationDelay: "250ms", marginTop: 12, animationFillMode: 'forwards', opacity: 0 }}
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
