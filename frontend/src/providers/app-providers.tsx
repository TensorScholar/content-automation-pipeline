"use client";

import { AuthProvider } from "./auth-provider";
import { I18nProvider } from "@/i18n/provider";
import { ToastProvider } from "@/components/ui/toast";

export function AppProviders({ children }: { children: React.ReactNode }) {
  return (
    <I18nProvider>
      <AuthProvider>
        <ToastProvider>{children}</ToastProvider>
      </AuthProvider>
    </I18nProvider>
  );
}
