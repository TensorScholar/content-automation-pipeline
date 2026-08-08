"use client";
import { useEffect, useId, useRef } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";

export interface ModalProps {
    open: boolean;
    onClose: () => void;
    title?: string;
    children: React.ReactNode;
    footer?: React.ReactNode;
    maxWidth?: string;
}

export function Modal({ open, onClose, title, children, footer, maxWidth = "28rem" }: ModalProps) {
    const dialogRef = useRef<HTMLDivElement>(null);
    const previousFocus = useRef<HTMLElement | null>(null);
    const titleId = useId();

    // Focus trap + ESC
    useEffect(() => {
        if (!open) return;
        previousFocus.current = document.activeElement as HTMLElement;

        const handler = (e: KeyboardEvent) => {
            if (e.key === "Escape") { onClose(); return; }
            if (e.key !== "Tab" || !dialogRef.current) return;
            const focusable = dialogRef.current.querySelectorAll<HTMLElement>(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            if (focusable.length === 0) return;
            const first = focusable[0];
            const last = focusable[focusable.length - 1];
            if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
            else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
        };
        document.addEventListener("keydown", handler);
        // Focus first focusable
        const focusTimer = window.setTimeout(() => {
            const first = dialogRef.current?.querySelector<HTMLElement>("button, input, [tabindex]");
            first?.focus();
        }, 50);
        return () => {
            window.clearTimeout(focusTimer);
            document.removeEventListener("keydown", handler);
            previousFocus.current?.focus();
        };
    }, [open, onClose]);

    if (!open || typeof document === "undefined") return null;

    return createPortal(
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-3 sm:p-5" role="dialog" aria-modal aria-labelledby={title ? titleId : undefined}>
            <div className="absolute inset-0 bg-black/[0.45] backdrop-blur-[2px] animate-fade-in" onClick={onClose} />
            <div
                ref={dialogRef}
                className="relative flex max-h-[calc(100dvh-2rem)] w-full flex-col overflow-hidden animate-fade-in rounded-[18px] border border-black/[0.075] bg-surface shadow-[0_32px_80px_-34px_rgb(0_0_0/0.72)] dark:border-white/10"
                style={{ maxWidth }}
            >
                {title && (
                    <div className="smx-section-header shrink-0">
                        <h2 id={titleId} className="text-[16px] font-semibold text-ink">{title}</h2>
                        <button type="button" onClick={onClose} className="smx-icon-button" aria-label="Close">
                            <svg className="h-[18px] w-[18px]" viewBox="0 0 20 20" fill="currentColor" aria-hidden>
                                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                        </button>
                    </div>
                )}
                <div className="min-h-0 overflow-y-auto px-5 py-5 sm:px-6">{children}</div>
                {footer && <div className="flex shrink-0 flex-wrap items-center justify-end gap-2 border-t border-black/[0.055] bg-black/[0.015] px-5 py-4 dark:border-white/[0.075] dark:bg-white/[0.025] sm:px-6">{footer}</div>}
            </div>
        </div>,
        document.body,
    );
}
