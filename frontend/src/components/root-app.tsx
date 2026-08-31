"use client";

import { AppShell } from "./app-shell";
import { LoginCard } from "./login-card";
import { useAuth } from "@/providers/auth-provider";
import { useI18n } from "@/i18n/provider";

export function RootApp() {
  const { token, user, loading } = useAuth();
  const { t } = useI18n();

  if (loading) {
    return (
      <main className="flex min-h-dvh items-center justify-center bg-[rgb(var(--bg-primary))] px-6" aria-busy="true" aria-label={t("common.loading")}>
        <div className="w-full max-w-sm">
          <div className="h-8 w-8 animate-pulse rounded-md bg-brand/15" />
          <div className="mt-7 h-5 w-36 animate-pulse rounded bg-ink/[0.07]" />
          <div className="mt-3 h-3 w-56 animate-pulse rounded bg-ink/[0.05]" />
          <div className="mt-8 h-px w-full bg-line" />
        </div>
      </main>
    );
  }

  if (!token || !user) return <LoginCard />;
  return <AppShell token={token} user={user} />;
}
