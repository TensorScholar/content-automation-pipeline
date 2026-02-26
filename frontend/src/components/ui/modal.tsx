"use client";
import { useEffect, useRef } from "react";
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
        setTimeout(() => {
            const first = dialogRef.current?.querySelector<HTMLElement>("button, input, [tabindex]");
            first?.focus();
        }, 50);
        return () => {
            document.removeEventListener("keydown", handler);
            previousFocus.current?.focus();
        };
    }, [open, onClose]);

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-modal flex items-center justify-center p-4" role="dialog" aria-modal aria-label={title}>
            <div className="absolute inset-0 bg-ink/40 backdrop-blur-[4px] animate-fade-in" onClick={onClose} />
            <div
                ref={dialogRef}
                className="relative w-full animate-scale-in rounded-lg border border-border bg-surface shadow-xl"
                style={{ maxWidth }}
            >
                {title && (
                    <div className="flex items-center justify-between border-b border-border px-6 py-4">
                        <h2 className="text-heading-md text-ink">{title}</h2>
                        <button onClick={onClose} className="rounded p-1 text-ink-tertiary hover:text-ink transition-colors" aria-label="Close">
                            <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                        </button>
                    </div>
                )}
                <div className="px-6 py-4">{children}</div>
                {footer && <div className="flex items-center justify-end gap-3 border-t border-border px-6 py-4">{footer}</div>}
            </div>
        </div>
    );
}
