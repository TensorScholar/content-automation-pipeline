"use client";

import clsx from "clsx";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";
import { LanguageToggle } from "./language-toggle";
import { MessageKey } from "@/i18n/types";

/* ═══════════════════════════════════════════════════════════════
   Spec: Screen 1 — Login Page
   Split-panel layout, animated SVG mesh, crossfade tagline,
   error ABOVE email field, no "Forgot Password" link
   ═══════════════════════════════════════════════════════════════ */

// ── Error resolution ──────────────────────────────────────────

interface ApiLikeError { status: number; detail?: string; }

function isApiLikeError(error: unknown): error is ApiLikeError {
  return (
    typeof error === "object" && error !== null &&
    "status" in error && typeof (error as ApiLikeError).status === "number"
  );
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

const FAILED_ATTEMPTS_LIMIT = 5;
const COOLDOWN_SECONDS = 60;

// ── SVG Icons ─────────────────────────────────────────────────

function EyeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden className="h-4 w-4" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3.8-6 10-6 10 6 10 6-3.8 6-10 6-10-6-10-6z" /><circle cx="12" cy="12" r="3" />
    </svg>
  );
}
function EyeOffIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden className="h-4 w-4" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3l18 18" /><path d="M10.6 10.6A3 3 0 0 0 13.4 13.4" />
      <path d="M9.9 5.1A11 11 0 0 1 12 5c6.2 0 10 7 10 7a18.8 18.8 0 0 1-4.2 4.8" />
      <path d="M6.4 6.5A19 19 0 0 0 2 12s3.8 7 10 7c1.7 0 3.2-.4 4.6-1" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg viewBox="0 0 16 16" fill="none" aria-hidden className="h-3 w-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8.5 6.3 11.7 13 5" />
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

// ── Animated SVG mesh (hero background) ───────────────────────

function HeroFlowMark() {
  const pathRef = useRef<SVGPathElement | null>(null);
  const [dot, setDot] = useState({ x: 48, y: 126 });

  useEffect(() => {
    const path = pathRef.current;
    if (!path || typeof window === "undefined") return;
    const totalLength = path.getTotalLength();
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const p = path.getPointAtLength(totalLength);
      setDot({ x: p.x, y: p.y }); return;
    }
    const s = path.getPointAtLength(0);
    setDot({ x: s.x, y: s.y });
    let frame = 0;
    const start = performance.now();
    const animate = (now: number) => {
      const progress = Math.min((now - start) / 2200, 1);
      const eased = progress < 0.5 ? 2 * progress * progress : 1 - Math.pow(-2 * progress + 2, 2) / 2;
      const point = path.getPointAtLength(eased * totalLength);
      setDot({ x: point.x, y: point.y });
      if (progress < 1) frame = window.requestAnimationFrame(animate);
    };
    frame = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frame);
  }, []);

  return (
    <svg viewBox="0 0 520 280" className="mx-auto h-auto w-full max-w-[480px]" fill="none" aria-hidden>
      <g stroke="rgba(255,255,255,0.12)" strokeWidth="1" strokeLinecap="round"><path d="M48 44h86" /><path d="M184 44h74" /><path d="M304 44h80" /><path d="M96 126h84" /><path d="M228 126h82" /><path d="M358 126h92" /><path d="M82 208h92" /><path d="M218 208h86" /><path d="M352 208h92" /><path d="M92 60v50" /><path d="M268 60v50" /><path d="M398 60v50" /><path d="M140 142v50" /><path d="M318 142v50" /></g>
      <path ref={pathRef} d="M48 126 C 118 126, 132 44, 192 44 C 252 44, 264 126, 324 126 C 384 126, 396 208, 458 208" stroke="rgba(14,110,110,0.76)" strokeWidth="1.25" strokeLinecap="round" />
      {[{ cx: 48, cy: 44 }, { cx: 142, cy: 44 }, { cx: 192, cy: 44 }, { cx: 268, cy: 44 }, { cx: 324, cy: 44 }, { cx: 398, cy: 44 }, { cx: 48, cy: 126 }, { cx: 96, cy: 126 }, { cx: 192, cy: 126 }, { cx: 228, cy: 126 }, { cx: 324, cy: 126 }, { cx: 358, cy: 126 }, { cx: 458, cy: 126 }, { cx: 82, cy: 208 }, { cx: 140, cy: 208 }, { cx: 218, cy: 208 }, { cx: 318, cy: 208 }, { cx: 458, cy: 208 }].map(n => (
        <circle key={`${n.cx}-${n.cy}`} cx={n.cx} cy={n.cy} r="4" fill="rgba(255,255,255,0.6)" stroke="rgba(14,110,110,0.38)" strokeWidth="1" />
      ))}
      <circle cx={dot.x} cy={dot.y} r="4" fill="#0E6E6E" style={{ filter: "drop-shadow(0 0 8px rgba(14,110,110,0.45))" }} />
    </svg>
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
  const [authError, setAuthError] = useState<unknown>(null);
  const [emailTouched, setEmailTouched] = useState(false);
  const [, setFailedAttempts] = useState(0);
  const [cooldownRemaining, setCooldownRemaining] = useState(0);

  // Email validation
  const emailInvalid = emailTouched && email.length > 0 && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

  const errorKey = authError ? getErrorKey(authError) : null;
  const localizedError = errorKey ? t(errorKey) : null;

  const cooldownText = useMemo(() => {
    return t("auth.cooldown", { seconds: cooldownRemaining });
  }, [cooldownRemaining, t]);

  useEffect(() => {
    if (cooldownRemaining <= 0) return;
    const timer = window.setInterval(() => setCooldownRemaining(s => Math.max(0, s - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [cooldownRemaining]);

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
    setSubmitting(true);
    try {
      await login(email, password, rememberMe);
      setFailedAttempts(0);
    } catch (error) {
      setAuthError(error);
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

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl items-center px-4 py-6">
      <div className="w-full overflow-hidden rounded-xl border border-border/60 bg-surface shadow-login-card" style={{ borderRadius: 20 }}>
        <div className="lg:grid lg:grid-cols-[1fr_1fr]">

          {/* ═══ LEFT PANEL — Brand (Desktop) ═══ */}
          <section
            dir={formDir}
            className={clsx(
              "hero-gradient relative hidden min-h-[640px] overflow-hidden p-12 text-white lg:flex lg:flex-col",
              heroOrderClass,
            )}
          >
            <div className="hero-grid-overlay absolute inset-0" />
            <div className="absolute inset-0 opacity-35"><HeroFlowMark /></div>

            {/* Crossfade tagline — re-renders with key to trigger 200ms transition */}
            <div key={`hero-${locale}`} className="relative z-10 flex h-full flex-col justify-between" style={{ animation: "fade-in 200ms ease-in-out forwards" }}>
              <p className="text-body-sm font-semibold uppercase tracking-[0.18em]" style={{ color: "rgba(255,255,255,0.72)" }}>
                {t("app.name")}
              </p>

              <div className="space-y-4">
                <h1 className="text-[28px] font-bold leading-tight text-white">
                  {t("auth.heroHeadline")}
                </h1>
                <p className="whitespace-pre-line text-[18px] leading-[1.7]" style={{ color: "rgba(255,255,255,0.82)" }}>
                  {t("auth.heroTagline")}
                </p>
              </div>

              <div className="space-y-2">
                <div className="inline-flex items-center gap-2" style={{ color: "rgba(255,255,255,0.82)" }}>
                  <span className="h-2 w-2 rounded-full bg-success animate-pulse-soft" aria-hidden />
                  <p className="text-body-sm">{t("auth.systemOnline")}</p>
                </div>
                <p className="font-mono text-[10px]" style={{ opacity: 0.4 }}>
                  {t("app.version")}
                </p>
              </div>
            </div>
          </section>

          {/* ═══ LEFT PANEL — Brand (Tablet) ═══ */}
          <section dir={formDir} className="hero-gradient relative hidden overflow-hidden border-b border-white/10 px-8 py-6 text-white md:block lg:hidden">
            <div className="hero-grid-overlay absolute inset-0" />
            <div key={`tb-hero-${locale}`} className="relative z-10 space-y-2 animate-fade-in">
              <p className="text-body-sm font-semibold uppercase tracking-[0.14em]" style={{ color: "rgba(255,255,255,0.72)" }}>{t("app.name")}</p>
              <p className="text-body-lg" style={{ color: "rgba(255,255,255,0.82)" }}>{t("auth.heroTagline")}</p>
            </div>
          </section>

          {/* ═══ LEFT PANEL — Brand (Mobile) ═══ */}
          <section dir={formDir} className="hero-gradient relative overflow-hidden border-b border-white/10 px-5 py-4 text-white md:hidden">
            <div key={`mb-hero-${locale}`} className="relative z-10 flex items-center justify-between animate-fade-in">
              <p className="text-body-sm font-semibold uppercase tracking-[0.15em]" style={{ color: "rgba(255,255,255,0.82)" }}>{t("app.name")}</p>
              <span className="inline-flex items-center gap-2 text-body-sm" style={{ color: "rgba(255,255,255,0.82)" }}>
                <span className="h-2 w-2 rounded-full bg-success animate-pulse-soft" aria-hidden />{t("auth.systemOnline")}
              </span>
            </div>
          </section>

          {/* ═══ RIGHT PANEL — Login Form ═══ */}
          <section dir={formDir} className={clsx("bg-surface px-7 py-10 sm:px-12", formOrderClass)}>
            {/* Header: app name + language switcher */}
            <div className="mb-8 flex items-center justify-between animate-fade-in">
              <p className="text-body-sm font-semibold uppercase tracking-[0.08em] text-ink-secondary">
                {t("app.name")}
              </p>
              <LanguageToggle />
            </div>

            {/* Form title */}
            <div key={`form-title-${locale}`} className="animate-fade-in">
              <h3 className="text-[28px] font-bold text-ink">{t("auth.title")}</h3>
            </div>

            <form method="POST" autoComplete="on" className="mt-10 space-y-6" onSubmit={onSubmit}>
              {/* ── Error alert (ABOVE email field per spec) ── */}
              {localizedError && (
                <div className="animate-slide-down rounded-sm border border-danger/20 bg-danger-subtle px-4 py-3" role="alert">
                  <p className="flex items-center gap-2 text-body-md text-danger">
                    <span aria-hidden>⚠</span>{localizedError}
                  </p>
                </div>
              )}

              {cooldownRemaining > 0 && (
                <div className="rounded-sm border border-warning/30 bg-warning-subtle px-4 py-3">
                  <p className="text-body-md text-ink-secondary">{cooldownText}</p>
                </div>
              )}

              {/* Email field */}
              <div className="animate-fade-in" style={{ animationDelay: "60ms" }}>
                <div className="relative">
                  <input
                    id="login-email"
                    type="email"
                    placeholder=" "
                    autoComplete="username"
                    required
                    aria-invalid={emailInvalid || Boolean(localizedError)}
                    aria-label={t("auth.email")}
                    className={clsx(
                      "auth-input peer h-12 w-full rounded-sm border px-4 pb-2 pt-5 text-body-md text-ink outline-none transition-all duration-base",
                      emailInvalid || localizedError ? "border-danger/70" : "border-border"
                    )}
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    onBlur={() => setEmailTouched(true)}
                  />
                  <label htmlFor="login-email" className="pointer-events-none absolute start-4 top-1/2 -translate-y-1/2 text-body-md text-ink-tertiary transition-all duration-base peer-focus:top-3 peer-focus:text-body-sm peer-focus:text-brand peer-[:not(:placeholder-shown)]:top-3 peer-[:not(:placeholder-shown)]:text-body-sm">
                    {t("auth.email")}
                  </label>
                </div>
                {emailInvalid && (
                  <p className="mt-1 flex items-center gap-1 text-body-sm text-danger"><span aria-hidden>⚠</span>{t("auth.invalidEmail")}</p>
                )}
              </div>

              {/* Password field */}
              <div className="animate-fade-in" style={{ animationDelay: "130ms" }}>
                <div className="relative">
                  <button
                    type="button"
                    className="absolute start-3 top-1/2 z-10 -translate-y-1/2 rounded p-1 text-ink-tertiary transition-colors hover:text-brand"
                    onClick={() => setShowPassword(o => !o)}
                    aria-label={showPassword ? t("auth.hidePassword") : t("auth.showPassword")}
                  >
                    {showPassword ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                  <input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    placeholder=" "
                    autoComplete="current-password"
                    required
                    aria-invalid={Boolean(localizedError)}
                    aria-label={t("auth.password")}
                    className={clsx(
                      "auth-input peer h-12 w-full rounded-sm border pb-2 pt-5 ps-12 pe-4 text-body-md text-ink outline-none transition-all duration-base",
                      localizedError ? "border-danger/70" : "border-border"
                    )}
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                  />
                  <label htmlFor="login-password" className="pointer-events-none absolute start-12 top-1/2 -translate-y-1/2 text-body-md text-ink-tertiary transition-all duration-base peer-focus:top-3 peer-focus:text-body-sm peer-focus:text-brand peer-[:not(:placeholder-shown)]:top-3 peer-[:not(:placeholder-shown)]:text-body-sm">
                    {t("auth.password")}
                  </label>
                </div>
              </div>

              {/* Remember me checkbox */}
              <label className="animate-fade-in inline-flex cursor-pointer items-center gap-3 text-body-md text-ink-secondary" style={{ animationDelay: "190ms" }}>
                <input type="checkbox" checked={rememberMe} onChange={e => setRememberMe(e.target.checked)} className="peer sr-only" />
                <span className="grid h-5 w-5 place-items-center rounded border border-border bg-white text-white transition-colors duration-fast peer-checked:border-brand peer-checked:bg-brand">
                  <CheckIcon />
                </span>
                <span>{t("auth.rememberMe")}</span>
              </label>

              {/* ⚠ NO "Forgot Password" link — intentionally omitted per spec */}

              {/* Submit */}
              <button
                type="submit"
                disabled={submitting || cooldownRemaining > 0}
                className="animate-fade-in w-full rounded-[10px] bg-brand px-4 text-body-md font-semibold text-white shadow-sm transition-all duration-base hover:bg-brand-hover active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50"
                style={{ height: 52, animationDelay: "250ms" }}
              >
                {submitting ? (
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
