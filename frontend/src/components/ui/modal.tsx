"use client";

import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";
import { useI18n } from "@/i18n/provider";

export interface ModalProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  maxWidth?: string;
}

export function Modal({ open, onClose, title, children, footer, maxWidth = "28rem" }: ModalProps) {
  const { t } = useI18n();
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    previousFocus.current = document.activeElement as HTMLElement;
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handler);
    const timer = window.setTimeout(
      () => dialogRef.current?.querySelector<HTMLElement>("button, input, select, textarea, [tabindex]")?.focus(),
      30,
    );
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("keydown", handler);
      previousFocus.current?.focus();
    };
  }, [open, onClose]);

  if (!open || typeof document === "undefined") return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby={title ? titleId : undefined}
    >
      <button
        type="button"
        className="absolute inset-0 cursor-default bg-black/40"
        onClick={onClose}
        aria-label={t("common.close")}
        tabIndex={-1}
      />
      <div
        ref={dialogRef}
        className="relative flex max-h-[calc(100dvh-2rem)] w-full flex-col overflow-hidden rounded-md border border-line bg-surface shadow-xl"
        style={{ maxWidth }}
      >
        {title ? (
          <div className="flex min-h-12 shrink-0 items-center justify-between gap-3 border-b border-line px-5">
            <h2 id={titleId} className="text-base font-semibold text-ink">{title}</h2>
            <button
              type="button"
              onClick={onClose}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm text-ink-tertiary hover:bg-ink/[0.04] hover:text-ink"
              aria-label={t("common.close")}
            >
              <svg className="h-4 w-4" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden>
                <path d="m5 5 10 10M15 5 5 15" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        ) : null}
        <div className="min-h-0 overflow-y-auto px-5 py-5">{children}</div>
        {footer ? (
          <div className="flex min-h-12 shrink-0 flex-wrap items-center justify-end gap-2 border-t border-line px-5 py-3">
            {footer}
          </div>
        ) : null}
      </div>
    </div>,
    document.body,
  );
}
