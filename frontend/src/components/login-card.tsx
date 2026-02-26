"use client";

import clsx from "clsx";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";
import { LanguageToggle } from "./language-toggle";
import { MessageKey } from "@/i18n/types";

/* ═══════════════════════════════════════════════════════════════
   Login Page v2 — Complete Redesign
   Fixes: #1 brand Latin, #2 copy, #3 neutral borders, #4 error
   severity, #5 RTL, #6 hero visual, #7 no duplicate brand,
   #8 checkbox, #9 eye icon, #11 version, #13 subtitle,
   #14 spacing, #15 hero centering, #16 system status,
   #18 button states, #19 floating labels, #21 card shadow,
   #22 page background
   ═══════════════════════════════════════════════════════════════ */

// ── Error resolution (#4 severity system) ─────────────────────

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

// ── SVG Icons (#9 enlarged to h-5 w-5) ───────────────────────

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

// ── Flowing mesh gradient animation (#6 replace dot grid) ─────

function HeroMeshGradient() {
  return (
    <div className="absolute inset-0 overflow-hidden" aria-hidden>
      {/* Slow-moving organic mesh gradient — 20s loop */}
      <div className="absolute inset-0" style={{
        background: `
          radial-gradient(ellipse 60% 50% at 20% 30%, rgba(14,110,110,0.35) 0%, transparent 70%),
          radial-gradient(ellipse 50% 60% at 70% 60%, rgba(6,78,78,0.3) 0%, transparent 70%),
          radial-gradient(ellipse 40% 40% at 50% 80%, rgba(20,140,140,0.25) 0%, transparent 70%)
        `,
        animation: "hero-mesh-drift 20s ease-in-out infinite alternate",
      }} />
      {/* Subtle node-line network */}
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
  const isRtl = direction === "rtl";

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

  // Email validation — #3 ONLY emailInvalid triggers red on email field
  const emailInvalid = emailTouched && email.length > 0 && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const errorKey = authError && !errorDismissed ? getErrorKey(authError) : null;
  const localizedError = errorKey ? t(errorKey) : null;
  const errorSeverity = authError && !errorDismissed ? getErrorSeverity(authError) : null;

  const cooldownText = useMemo(() => t("auth.cooldown", { seconds: cooldownRemaining }), [cooldownRemaining, t]);

  useEffect(() => {
    if (cooldownRemaining <= 0) return;
    const timer = window.setInterval(() => setCooldownRemaining(s => Math.max(0, s - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [cooldownRemaining]);

  // #4 Auto-dismiss non-critical errors after 8s
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
      // #18 Success state — checkmark for 600ms then redirect
      setLoginSuccess(true);
    } catch (error) {
      setAuthError(error);
      // #18 Error shake
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

  const formDir = isRtl ? "rtl" : "ltr";
  const formOrderClass = isRtl ? "lg:order-2" : "lg:order-1";
  const heroOrderClass = isRtl ? "lg:order-1" : "lg:order-2";

  // #16 Dynamic system status — red when there was a recent server error
  const systemHealthy = !authError || getErrorSeverity(authError) !== "server";

  return (
    /* #22 Page background — subtle teal-tinted radial gradient */
    <main className="mx-auto flex min-h-screen w-full max-w-6xl items-center px-4 py-6" style={{ background: "radial-gradient(ellipse at 50% 40%, #F0F7F7 0%, #F0F2F5 70%)" }}>
      {/* #21 Enhanced card shadow + subtle border */}
      <div className="w-full overflow-hidden rounded-2xl border border-black/[0.04] bg-surface" style={{ boxShadow: "0 8px 32px rgba(0,0,0,0.08), 0 2px 8px rgba(0,0,0,0.04)", borderRadius: 20 }}>
        <div className="lg:grid lg:grid-cols-[1fr_1fr]">

          {/* ═══ HERO PANEL — Brand (#6 mesh, #7 brand only here, #15 centered) ═══ */}
          <section
            dir={formDir}
            className={clsx(
              "hero-gradient relative hidden min-h-[640px] overflow-hidden text-white lg:flex lg:flex-col",
              heroOrderClass,
            )}
          >
            <HeroMeshGradient />

            <div key={`hero-${locale}`} className="relative z-10 flex h-full flex-col p-12" style={{ animation: "fade-in 200ms ease-in-out forwards" }}>
              {/* Brand — ALWAYS Latin (#1) */}
              <p className="text-body-sm font-semibold uppercase tracking-[0.18em]" style={{ color: "rgba(255,255,255,0.72)" }}>
                SMARLUX CONTENT OS
              </p>

              {/* #15 Centered text block — premium visual treatment */}
              <div className="flex flex-1 items-center">
                <div className="space-y-6">
                  {/* Decorative accent bar */}
                  <div className="flex items-center gap-3" style={{ animation: "fade-in 400ms ease-out forwards", opacity: 0 }}>
                    <span className="block h-[3px] w-10 rounded-full" style={{ background: "linear-gradient(90deg, #5EEAD4, rgba(94,234,212,0.2))" }} />
                    <span className="text-[13px] font-medium uppercase tracking-[0.2em]" style={{ color: "rgba(94,234,212,0.8)" }}>
                      AI-Powered Platform
                    </span>
                  </div>

                  {/* Gradient headline */}
                  <h1
                    className="whitespace-pre-line text-[40px] font-extrabold leading-[1.12]"
                    style={{
                      background: "linear-gradient(135deg, #FFFFFF 0%, #5EEAD4 45%, #2DD4BF 100%)",
                      WebkitBackgroundClip: "text",
                      WebkitTextFillColor: "transparent",
                      backgroundClip: "text",
                      animation: "fade-in 500ms 100ms ease-out forwards",
                      opacity: 0,
                    }}
                  >
                    {t("auth.heroHeadline")}
                  </h1>

                  {/* Tagline inside a subtle glass pill */}
                  <div
                    className="inline-block rounded-2xl px-5 py-3.5"
                    style={{
                      background: "rgba(255,255,255,0.06)",
                      backdropFilter: "blur(8px)",
                      WebkitBackdropFilter: "blur(8px)",
                      border: "1px solid rgba(255,255,255,0.08)",
                      animation: "fade-in 500ms 250ms ease-out forwards",
                      opacity: 0,
                    }}
                  >
                    <p className="whitespace-pre-line text-[16px] leading-[1.9]" style={{ color: "rgba(255,255,255,0.82)" }}>
                      {t("auth.heroTagline")}
                    </p>
                  </div>
                </div>
              </div>

              {/* #16 Dynamic system status with breathing pulse */}
              <div className="space-y-2">
                <div className="inline-flex items-center gap-2" style={{ color: "rgba(255,255,255,0.82)" }}>
                  <span
                    className={clsx("h-2 w-2 rounded-full", systemHealthy ? "bg-success" : "bg-danger")}
                    style={{ animation: "status-pulse 2s ease-in-out infinite" }}
                    aria-hidden
                  />
                  <p className="text-body-sm">
                    {systemHealthy ? t("auth.systemOnline") : t("auth.systemDegraded")}
                  </p>
                </div>
                {/* #11 Version — reduced opacity/size, separated */}
                <p className="font-mono text-[11px]" style={{ opacity: 0.35, marginTop: 10 }}>
                  {t("app.version")}
                </p>
              </div>
            </div>
          </section>

          {/* ═══ HERO — Tablet ═══ */}
          <section dir={formDir} className="hero-gradient relative hidden overflow-hidden border-b border-white/10 px-8 py-6 text-white md:block lg:hidden">
            <HeroMeshGradient />
            <div key={`tb-hero-${locale}`} className="relative z-10 space-y-2 animate-fade-in">
              <p className="text-body-sm font-semibold uppercase tracking-[0.14em]" style={{ color: "rgba(255,255,255,0.72)" }}>SMARLUX CONTENT OS</p>
              <p className="text-body-lg" style={{ color: "rgba(255,255,255,0.82)" }}>{t("auth.heroTagline")}</p>
            </div>
          </section>

          {/* ═══ HERO — Mobile ═══ */}
          <section dir={formDir} className="hero-gradient relative overflow-hidden border-b border-white/10 px-5 py-4 text-white md:hidden">
            <div key={`mb-hero-${locale}`} className="relative z-10 flex items-center justify-between animate-fade-in">
              <p className="text-body-sm font-semibold uppercase tracking-[0.15em]" style={{ color: "rgba(255,255,255,0.82)" }}>SMARLUX CONTENT OS</p>
              <span className="inline-flex items-center gap-2 text-body-sm" style={{ color: "rgba(255,255,255,0.82)" }}>
                <span className={clsx("h-2 w-2 rounded-full", systemHealthy ? "bg-success" : "bg-danger")} style={{ animation: "status-pulse 2s ease-in-out infinite" }} aria-hidden />{systemHealthy ? t("auth.systemOnline") : t("auth.systemDegraded")}
              </span>
            </div>
          </section>

          {/* ═══ FORM PANEL ═══ */}
          <section dir={formDir} className={clsx("bg-surface px-7 py-10 sm:px-12", formOrderClass)}>
            {/* #7 Remove duplicate brand, keep only LanguageToggle (#5 auto-mirrors via dir) */}
            <div className="mb-6 flex items-center justify-end animate-fade-in">
              <LanguageToggle />
            </div>

            {/* #12 Warm heading + #13 subtitle */}
            <div key={`form-title-${locale}`} className="animate-fade-in">
              <h3 className="text-[28px] font-bold text-ink">{t("auth.title")}</h3>
              <p className="mt-2 text-body-md text-ink-secondary">{t("auth.subtitle")}</p>
            </div>

            {/* #14 Consistent spacing scale: 24px between sections */}
            <form method="POST" autoComplete="on" className="mt-8" onSubmit={onSubmit} style={{ display: "flex", flexDirection: "column", gap: 20 }}>

              {/* ── Error alert (#4 severity + dismiss + slide-in) ── */}
              {localizedError && errorSeverity && (
                <div
                  className={clsx("flex items-start gap-3 rounded-lg px-4 py-3", SEVERITY_STYLES[errorSeverity])}
                  role="alert"
                  style={{ animation: "login-slide-in 200ms ease-out" }}
                >
                  <span className="shrink-0 text-[18px]" aria-hidden>{SEVERITY_ICON[errorSeverity]}</span>
                  <p className="flex-1 text-body-md">{localizedError}</p>
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
                <div className="rounded-lg border-s-[4px] border-s-warning bg-warning/5 px-4 py-3">
                  <p className="text-body-md text-ink-secondary">{cooldownText}</p>
                </div>
              )}

              {/* Email field (#3 neutral borders — only emailInvalid triggers red, NOT server error) (#19 floating label) */}
              <div className="animate-fade-in" style={{ animationDelay: "60ms" }}>
                <div className="relative">
                  <input
                    id="login-email"
                    type="email"
                    dir="ltr"
                    placeholder=" "
                    autoComplete="username"
                    required
                    aria-invalid={emailInvalid}
                    aria-label={t("auth.email")}
                    className={clsx(
                      "auth-input peer h-[52px] w-full rounded-xl border px-4 pb-2 pt-6 text-body-md text-ink outline-none transition-all duration-base text-left",
                      "focus:border-brand focus:shadow-[0_0_0_3px_rgba(13,148,136,0.12)]",
                      emailInvalid ? "border-danger shadow-[0_0_0_3px_rgba(239,68,68,0.08)]" : "border-border"
                    )}
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    onBlur={() => setEmailTouched(true)}
                  />
                  <label htmlFor="login-email" className="pointer-events-none absolute start-4 top-1/2 -translate-y-1/2 text-body-md text-ink-secondary transition-all duration-base peer-focus:top-3.5 peer-focus:text-body-sm peer-focus:text-brand peer-[:not(:placeholder-shown)]:top-3.5 peer-[:not(:placeholder-shown)]:text-body-sm">
                    {t("auth.email")}
                  </label>
                </div>
                {emailInvalid && (
                  <p className="mt-1.5 flex items-center gap-1 text-body-sm text-danger"><span aria-hidden>⚠</span>{t("auth.invalidEmail")}</p>
                )}
              </div>

              {/* Password field (#9 eye icon polished: 20px, hover/active states) */}
              <div className="animate-fade-in" style={{ animationDelay: "130ms" }}>
                <div className="relative">
                  <button
                    type="button"
                    className={clsx(
                      "absolute end-3 top-1/2 z-10 -translate-y-1/2 rounded-lg p-1.5 transition-colors duration-fast",
                      showPassword
                        ? "text-brand"
                        : "text-ink-secondary hover:text-ink"
                    )}
                    onClick={() => setShowPassword(o => !o)}
                    aria-label={showPassword ? t("auth.hidePassword") : t("auth.showPassword")}
                  >
                    {showPassword ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                  <input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    dir="ltr"
                    placeholder=" "
                    autoComplete="current-password"
                    required
                    aria-invalid={false}
                    aria-label={t("auth.password")}
                    className="auth-input peer h-[52px] w-full rounded-xl border border-border px-4 pb-2 pt-6 pe-12 text-body-md text-ink outline-none transition-all duration-base text-left focus:border-brand focus:shadow-[0_0_0_3px_rgba(13,148,136,0.12)]"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                  />
                  <label htmlFor="login-password" className="pointer-events-none absolute start-4 top-1/2 -translate-y-1/2 text-body-md text-ink-secondary transition-all duration-base peer-focus:top-3.5 peer-focus:text-body-sm peer-focus:text-brand peer-[:not(:placeholder-shown)]:top-3.5 peer-[:not(:placeholder-shown)]:text-body-sm">
                    {t("auth.password")}
                  </label>
                </div>
              </div>

              {/* #8 Remember me — custom 20×20px checkbox with animation */}
              <label className="animate-fade-in inline-flex cursor-pointer items-center gap-3 text-body-md text-ink-secondary" style={{ animationDelay: "190ms" }}>
                <input type="checkbox" checked={rememberMe} onChange={e => setRememberMe(e.target.checked)} className="peer sr-only" />
                <span className="grid h-5 w-5 shrink-0 place-items-center rounded border border-border bg-white text-white transition-all duration-fast peer-checked:border-brand peer-checked:bg-brand peer-focus-visible:ring-2 peer-focus-visible:ring-brand/30 peer-focus-visible:ring-offset-1">
                  <CheckIcon />
                </span>
                <span>{t("auth.rememberMe")}</span>
              </label>

              {/* ⚠ NO "Forgot Password" link — intentionally omitted per spec */}

              {/* Submit (#18 loading/success/error states) */}
              <button
                type="submit"
                disabled={submitting || cooldownRemaining > 0 || loginSuccess}
                className={clsx(
                  "animate-fade-in w-full rounded-xl bg-brand px-4 text-body-md font-semibold text-white shadow-sm transition-all duration-base hover:bg-brand-hover active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-60",
                  shakeButton && "animate-shake",
                  loginSuccess && "!bg-emerald-500",
                )}
                style={{ height: 52, animationDelay: "250ms", marginTop: 4 }}
              >
                {loginSuccess ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg viewBox="0 0 20 20" fill="none" className="h-5 w-5" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M4 10.5 8 14.5 16 6.5" /></svg>
                  </span>
                ) : submitting ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                    {t("auth.submitting")}
                  </span>
                ) : (
                  t("auth.submit")
                )}
              </button>
            </form>
          </section>
        </div>
      </div>
    </main>
  );
}
