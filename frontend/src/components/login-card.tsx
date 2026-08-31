"use client";

import clsx from "clsx";
import Image from "next/image";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";
import { LanguageToggle } from "./language-toggle";
import { MessageKey } from "@/i18n/types";

type ErrorSeverity = "server" | "credentials" | "network";
interface ApiLikeError { status: number; detail?: string; }

function isApiLikeError(error: unknown): error is ApiLikeError {
  return typeof error === "object" && error !== null && "status" in error && typeof (error as ApiLikeError).status === "number";
}

function getErrorKey(error: unknown): MessageKey {
  if (error instanceof ApiError || isApiLikeError(error)) {
    const status = error instanceof ApiError ? error.status : error.status;
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
    const status = error instanceof ApiError ? error.status : error.status;
    if (status === 0 || status === 503) return "network";
    if (status === 401 || status === 403) return "credentials";
    if (status >= 500) return "server";
  }
  if (error instanceof TypeError) return "network";
  return "server";
}

const FAILED_ATTEMPTS_LIMIT = 5;
const COOLDOWN_SECONDS = 60;
const LOCKOUT_KEY = "cap.login_lockout_until";

function getLockoutSecondsRemaining(): number {
  if (typeof window === "undefined") return 0;
  try {
    const until = window.sessionStorage.getItem(LOCKOUT_KEY);
    if (!until) return 0;
    const remaining = Math.ceil((parseInt(until, 10) - Date.now()) / 1000);
    if (remaining <= 0) {
      window.sessionStorage.removeItem(LOCKOUT_KEY);
      return 0;
    }
    return remaining;
  } catch { return 0; }
}

function setLockout(seconds: number) {
  try { window.sessionStorage.setItem(LOCKOUT_KEY, String(Date.now() + seconds * 1000)); } catch { /* session storage is optional */ }
}

function EyeIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden className="h-[18px] w-[18px]" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3.8-6 10-6 10 6 10 6-3.8 6-10 6-10-6-10-6z" /><circle cx="12" cy="12" r="3" />
    </svg>
  );
}
function EyeOffIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden className="h-[18px] w-[18px]" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3l18 18" /><path d="M10.6 10.6A3 3 0 0 0 13.4 13.4" /><path d="M9.9 5.1A11 11 0 0 1 12 5c6.2 0 10 7 10 7a18.8 18.8 0 0 1-4.2 4.8" /><path d="M6.4 6.5A19 19 0 0 0 2 12s3.8 7 10 7c1.7 0 3.2-.4 4.6-1" />
    </svg>
  );
}
function CheckIcon() {
  return <svg viewBox="0 0 16 16" fill="none" aria-hidden className="h-3 w-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 8.5 6.3 11.7 13 5" /></svg>;
}

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

  const emailInvalid = emailTouched && email.length > 0 && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
  const errorKey = authError && !errorDismissed ? getErrorKey(authError) : null;
  const localizedError = errorKey ? t(errorKey) : null;
  const errorSeverity = authError && !errorDismissed ? getErrorSeverity(authError) : null;
  const cooldownText = useMemo(() => t("auth.cooldown", { seconds: String(cooldownRemaining) }), [cooldownRemaining, t]);

  useEffect(() => {
    if (cooldownRemaining <= 0) return;
    const timer = window.setInterval(() => setCooldownRemaining((value) => Math.max(0, value - 1)), 1000);
    return () => window.clearInterval(timer);
  }, [cooldownRemaining]);

  useEffect(() => {
    if (!authError || errorDismissed) return;
    if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current);
    if (getErrorSeverity(authError) !== "server") errorTimerRef.current = window.setTimeout(() => setErrorDismissed(true), 8000);
    return () => { if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current); };
  }, [authError, errorDismissed]);

  useEffect(() => () => { if (shakeTimerRef.current) window.clearTimeout(shakeTimerRef.current); }, []);

  useEffect(() => {
    document.title = `${t("auth.title")} — ${t("app.name")}`;
    upsertMeta("description", t("auth.subtitle"));
    upsertMeta("robots", "noindex, nofollow");
    const origin = window.location.origin;
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
      try { window.sessionStorage.removeItem(LOCKOUT_KEY); } catch { /* optional */ }
      setLoginSuccess(true);
    } catch (error) {
      setAuthError(error);
      setShakeButton(true);
      if (shakeTimerRef.current) window.clearTimeout(shakeTimerRef.current);
      shakeTimerRef.current = window.setTimeout(() => setShakeButton(false), 400);
      const is429 = (error instanceof ApiError && error.status === 429) || (isApiLikeError(error) && error.status === 429);
      if (is429) {
        const retryAfter = error instanceof ApiError ? (error.retryAfter ?? 60) : 60;
        setLockout(retryAfter);
        setCooldownRemaining(retryAfter);
        setFailedAttempts(0);
        return;
      }
      setFailedAttempts((previous) => {
        const next = previous + 1;
        if (next >= FAILED_ATTEMPTS_LIMIT) {
          setLockout(COOLDOWN_SECONDS);
          setCooldownRemaining(COOLDOWN_SECONDS);
          return 0;
        }
        return next;
      });
    } finally { setSubmitting(false); }
  };

  const systemHealthy = !authError || getErrorSeverity(authError) !== "server";
  const errorTone = errorSeverity === "credentials" ? "text-warning" : errorSeverity === "network" ? "text-info" : "text-danger";

  return (
    <main dir={direction} className="h-dvh w-full overflow-y-auto bg-[rgb(var(--bg-primary))] text-ink">
      <div className="mx-auto grid min-h-full w-full max-w-[1180px] grid-cols-1 px-5 sm:px-8 lg:grid-cols-[430px_minmax(0,1fr)] lg:gap-16 lg:px-12 xl:gap-24">
        <section className="flex min-h-full min-w-0 flex-col py-6 sm:py-8 lg:py-10">
          <div className="flex items-center justify-between">
            <div className="inline-flex items-center gap-2.5 text-start">
              <Image src="/logo.png" alt="" width={28} height={28} priority className="h-7 w-7 rounded-md object-cover" />
              <span className="text-sm font-semibold tracking-[-0.01em] text-ink rtl:tracking-normal">{t("app.name")}</span>
            </div>
            <LanguageToggle />
          </div>

          <div className="my-auto w-full py-12 lg:py-16">
            <div className="mb-8">
              <p className="mb-2 text-xs font-medium uppercase tracking-[0.08em] text-ink-tertiary rtl:normal-case rtl:tracking-normal">{t("auth.productLabel")}</p>
              <h1 className="max-w-[390px] text-display-lg tracking-[-0.025em] text-ink rtl:tracking-normal">{t("auth.title")}</h1>
              <p className="mt-2.5 max-w-[390px] text-base leading-6 text-ink-secondary">{t("auth.subtitle")}</p>
            </div>

            <form method="POST" autoComplete="on" onSubmit={onSubmit} className="flex flex-col gap-4">
              {cooldownRemaining > 0 ? (
                <div className="border-s-2 border-warning bg-warning-subtle px-3.5 py-2.5 text-sm leading-5 text-warning" role="status">{cooldownText}</div>
              ) : null}

              <div className="flex flex-col gap-1.5">
                <label htmlFor="login-email" className="text-sm font-medium text-ink-secondary">{t("auth.email")}</label>
                <input
                  id="login-email"
                  type="email"
                  dir="ltr"
                  autoComplete="username"
                  required
                  aria-invalid={emailInvalid}
                  className={clsx("auth-input h-10 px-3 text-base text-start", emailInvalid && "!border-danger")}
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  onBlur={() => setEmailTouched(true)}
                />
                {emailInvalid ? <p className="text-xs leading-[18px] text-danger">{t("auth.invalidEmail")}</p> : null}
              </div>

              <div className="flex flex-col gap-1.5">
                <label htmlFor="login-password" className="text-sm font-medium text-ink-secondary">{t("auth.password")}</label>
                <div className="relative">
                  <input
                    id="login-password"
                    type={showPassword ? "text" : "password"}
                    dir="ltr"
                    autoComplete="current-password"
                    required
                    className="auth-input h-10 px-3 pe-10 text-base text-start"
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                  />
                  <button type="button" className="absolute inset-y-0 end-1 flex w-8 items-center justify-center rounded-md text-ink-tertiary hover:bg-ink/[0.035] hover:text-ink" onClick={() => setShowPassword((value) => !value)} aria-label={showPassword ? t("auth.hidePassword") : t("auth.showPassword")}>
                    {showPassword ? <EyeOffIcon /> : <EyeIcon />}
                  </button>
                </div>
              </div>

              <label className="flex cursor-pointer select-none items-center gap-2.5 text-sm text-ink-secondary">
                <input type="checkbox" checked={rememberMe} onChange={(event) => setRememberMe(event.target.checked)} className="peer sr-only" />
                <span className="grid h-[17px] w-[17px] shrink-0 place-items-center rounded-sm border border-line bg-surface text-white transition-colors peer-checked:border-brand peer-checked:bg-brand peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-brand/70"><CheckIcon /></span>
                <span>{t("auth.rememberMe")}</span>
              </label>

              {localizedError ? (
                <div className={clsx("flex items-start gap-2.5 border-s-2 bg-ink/[0.018] px-3.5 py-2.5 text-sm leading-5", errorTone, errorSeverity === "credentials" ? "border-warning" : errorSeverity === "network" ? "border-info" : "border-danger")} role="alert">
                  <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-current" aria-hidden />
                  <span className="flex-1">{localizedError}</span>
                  <button type="button" onClick={() => setErrorDismissed(true)} className="rounded-md px-1 text-ink-tertiary hover:text-ink" aria-label={t("auth.dismiss")}>×</button>
                </div>
              ) : null}

              <button
                type="submit"
                disabled={submitting || cooldownRemaining > 0 || loginSuccess}
                className={clsx("mt-1 flex h-10 w-full items-center justify-center gap-2 rounded-md border border-brand bg-brand px-4 text-base font-medium text-white transition-colors hover:border-brand-hover hover:bg-brand-hover disabled:cursor-not-allowed disabled:opacity-50", shakeButton && "animate-shake", loginSuccess && "!border-success !bg-success")}
              >
                {loginSuccess ? <CheckIcon /> : submitting ? <span className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-white/30 border-t-white" aria-hidden /> : null}
                {loginSuccess ? t("common.success") : submitting ? t("auth.submitting") : t("auth.submit")}
              </button>
            </form>
          </div>

          <div className="flex items-center justify-between gap-4 border-t border-line pt-3.5 text-xs text-ink-tertiary">
            <span className="inline-flex items-center gap-2"><span className={clsx("h-1.5 w-1.5 rounded-full", systemHealthy ? "bg-success" : "bg-danger")} aria-hidden />{systemHealthy ? t("auth.systemOnline") : t("auth.systemDegraded")}</span>
            <span dir="ltr">FA · AR · EN</span>
          </div>
        </section>

        <aside className="relative hidden min-w-0 border-s border-line lg:flex lg:flex-col lg:justify-between lg:py-10 lg:ps-12 xl:ps-16" aria-hidden="true">
          <div className="max-w-[560px] pt-20">
            <div className="mb-8 flex items-center gap-3">
              <span className="h-px w-10 bg-brand" />
              <span className="text-xs font-semibold uppercase tracking-[0.12em] text-brand">Smarlux</span>
            </div>
            <p className="max-w-[540px] whitespace-pre-line text-display-hero tracking-[-0.025em] text-ink rtl:tracking-normal">{t("auth.heroHeadline")}</p>
            <p className="mt-5 max-w-[510px] whitespace-pre-line text-body-lg leading-7 text-ink-secondary">{t("auth.heroTagline")}</p>
          </div>

          <div className="grid max-w-[560px] grid-cols-3 border-t border-line pt-4 text-xs text-ink-tertiary">
            <span>{t("dashboard.pipelineProject")}</span>
            <span>{t("studio.generate")}</span>
            <span>{t("tasks.wpPublish")}</span>
          </div>
        </aside>
      </div>
    </main>
  );
}
