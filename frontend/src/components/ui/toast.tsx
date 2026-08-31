"use client";

import { createContext, useCallback, useContext, useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";
import { useI18n } from "@/i18n/provider";

export type ToastVariant = "success" | "error" | "warning" | "info";

interface ToastItem {
  id: number;
  variant: ToastVariant;
  message: string;
  exiting?: boolean;
}

interface ToastContextValue {
  showToast: (variant: ToastVariant, message: string) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

function ToastIcon({ variant }: { variant: ToastVariant }) {
  if (variant === "success") {
    return (
      <svg className="h-4 w-4 shrink-0 text-success" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
        <path d="m3.5 8.2 2.7 2.7 6.3-6.3" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  if (variant === "error") {
    return (
      <svg className="h-4 w-4 shrink-0 text-danger" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
        <path d="m4.3 4.3 7.4 7.4m0-7.4-7.4 7.4" strokeLinecap="round" />
      </svg>
    );
  }
  if (variant === "warning") {
    return (
      <svg className="h-4 w-4 shrink-0 text-warning" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
        <path d="M8 2.4 13.4 12H2.6L8 2.4Z" strokeLinejoin="round" />
        <path d="M8 5.8v3.1M8 11.2h.01" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg className="h-4 w-4 shrink-0 text-info" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden>
      <circle cx="8" cy="8" r="5.4" />
      <path d="M8 7.1v3.3M8 5.2h.01" strokeLinecap="round" />
    </svg>
  );
}

let nextId = 0;

function ToastItem({ item, onDismiss }: { item: ToastItem; onDismiss: (id: number) => void }) {
  const { t } = useI18n();
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    timerRef.current = window.setTimeout(() => onDismiss(item.id), 4000);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [item.id, onDismiss]);

  return (
    <div
      role="alert"
      className={clsx(
        "flex w-full items-center gap-2.5 rounded-md border border-line bg-surface p-3 shadow-md",
        "transition-all duration-normal",
        item.exiting ? "animate-fade-out" : "animate-slide-up",
      )}
    >
      <ToastIcon variant={item.variant} />
      <p className="flex-1 text-sm font-medium text-ink">{item.message}</p>
      <button
        type="button"
        onClick={() => onDismiss(item.id)}
        className="flex h-6 w-6 shrink-0 items-center justify-center rounded-sm text-ink-tertiary hover:bg-ink/[0.05] hover:text-ink focus-visible:outline-none"
        aria-label={t("common.dismiss")}
      >
        <svg className="h-3.5 w-3.5" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
          <path d="m4 4 8 8m0-8-8 8" strokeLinecap="round" />
        </svg>
      </button>
    </div>
  );
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const [portalRoot, setPortalRoot] = useState<HTMLElement | null>(null);
  const removalTimersRef = useRef<Map<number, number>>(new Map());

  useEffect(() => {
    setPortalRoot(document.getElementById("toast-root") ?? document.body);
  }, []);

  useEffect(() => {
    const removalTimers = removalTimersRef.current;
    return () => {
      removalTimers.forEach((timer) => window.clearTimeout(timer));
      removalTimers.clear();
    };
  }, []);

  const showToast = useCallback((variant: ToastVariant, message: string) => {
    const id = ++nextId;
    setToasts((prev) => [...prev, { id, variant, message }]);
  }, []);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.map((t) => (t.id === id ? { ...t, exiting: true } : t)));
    const existingTimer = removalTimersRef.current.get(id);
    if (existingTimer) window.clearTimeout(existingTimer);
    const timer = window.setTimeout(() => {
      removalTimersRef.current.delete(id);
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 200);
    removalTimersRef.current.set(id, timer);
  }, []);

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      {portalRoot &&
        createPortal(
          <div className="pointer-events-none fixed inset-x-4 bottom-5 z-toast flex flex-col items-center gap-2" aria-live="polite">
            <div className="pointer-events-auto flex w-full max-w-[360px] flex-col gap-2">
              {toasts.map((t) => (
                <ToastItem key={t.id} item={t} onDismiss={dismiss} />
              ))}
            </div>
          </div>,
          portalRoot,
        )}
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
