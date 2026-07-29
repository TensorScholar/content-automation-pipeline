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
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4" role="dialog" aria-modal aria-labelledby={title ? titleId : undefined}>
            <div className="absolute inset-0 bg-ink/50 backdrop-blur-[4px] animate-fade-in" onClick={onClose} />
            <div
                ref={dialogRef}
                className="relative w-full animate-scale-in rounded-xl border border-black/8 bg-surface shadow-[0_24px_48px_-28px_rgb(0_0_0/0.55)] dark:border-white/10"
                style={{ maxWidth }}
            >
                {title && (
                    <div className="flex items-center justify-between border-b border-black/6 px-6 py-4 dark:border-white/10">
                        <h2 id={titleId} className="text-[16px] font-semibold text-ink">{title}</h2>
                        <button onClick={onClose} className="rounded-md p-1 text-ink-tertiary hover:bg-black/[0.05] hover:text-ink transition-colors dark:hover:bg-white/[0.08]" aria-label="Close">
                            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                        </button>
                    </div>
                )}
                <div className="px-6 py-4">{children}</div>
                {footer && <div className="flex items-center justify-end gap-3 border-t border-black/6 px-6 py-4 dark:border-white/10">{footer}</div>}
            </div>
        </div>,
        document.body,
    );
}
