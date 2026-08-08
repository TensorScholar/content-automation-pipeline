"use client";
import { createContext, useCallback, useContext, useState, useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";

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

const variantStyles: Record<ToastVariant, string> = {
    success: "border-emerald-500/[0.15] bg-white text-emerald-700 dark:bg-[#22252a] dark:text-emerald-200",
    error: "border-rose-500/[0.15] bg-white text-rose-700 dark:bg-[#22252a] dark:text-rose-200",
    warning: "border-amber-500/[0.15] bg-white text-amber-800 dark:bg-[#22252a] dark:text-amber-200",
    info: "border-sky-500/[0.15] bg-white text-sky-700 dark:bg-[#22252a] dark:text-sky-200",
};

const icons: Record<ToastVariant, string> = {
    success: "✓", error: "✕", warning: "⚠", info: "ℹ",
};

let nextId = 0;

function ToastItem({ item, onDismiss }: { item: ToastItem; onDismiss: (id: number) => void }) {
    const timerRef = useRef<number | null>(null);

    useEffect(() => {
        timerRef.current = window.setTimeout(() => onDismiss(item.id), 4000);
        return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    }, [item.id, onDismiss]);

    return (
        <div
            role="alert"
            className={clsx(
                "flex items-center gap-3 rounded-[14px] border px-4 py-3 shadow-[0_18px_45px_-26px_rgb(0_0_0/0.55)]",
                "transition-all duration-normal",
                item.exiting ? "animate-fade-out" : "animate-slide-up",
                variantStyles[item.variant],
            )}
        >
            <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-black/[0.04] text-[12px] font-bold dark:bg-white/[0.07]" aria-hidden>{icons[item.variant]}</span>
            <p className="flex-1 text-body-md font-medium">{item.message}</p>
            <button
                type="button"
                onClick={() => onDismiss(item.id)}
                className="shrink-0 rounded-[8px] p-1.5 opacity-[0.55] transition-[background-color,opacity] hover:bg-black/[0.04] hover:opacity-100 focus-visible:outline-none dark:hover:bg-white/[0.06]"
                aria-label="Dismiss"
            >
                ✕
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
        setToasts(prev => [...prev, { id, variant, message }]);
    }, []);

    const dismiss = useCallback((id: number) => {
        setToasts(prev => prev.map(t => t.id === id ? { ...t, exiting: true } : t));
        const existingTimer = removalTimersRef.current.get(id);
        if (existingTimer) window.clearTimeout(existingTimer);
        const timer = window.setTimeout(() => {
            removalTimersRef.current.delete(id);
            setToasts(prev => prev.filter(t => t.id !== id));
        }, 200);
        removalTimersRef.current.set(id, timer);
    }, []);

    return (
        <ToastContext.Provider value={{ showToast }}>
            {children}
            {portalRoot && createPortal(
                <div className="pointer-events-none fixed inset-x-4 bottom-5 z-toast flex flex-col items-center gap-2" aria-live="polite">
                    <div className="pointer-events-auto flex w-full max-w-sm flex-col gap-2">
                        {toasts.map(t => <ToastItem key={t.id} item={t} onDismiss={dismiss} />)}
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
