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
    success: "bg-success-subtle text-success border-success/20",
    error: "bg-danger-subtle text-danger border-danger/20",
    warning: "bg-warning-subtle text-warning border-warning/20",
    info: "bg-info-subtle text-info border-info/20",
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
                "flex items-center gap-3 rounded-[10px] border px-4 py-3 shadow-toast",
                "transition-all duration-normal",
                item.exiting ? "animate-fade-out" : "animate-slide-up",
                variantStyles[item.variant],
            )}
        >
            <span className="text-lg" aria-hidden>{icons[item.variant]}</span>
            <p className="flex-1 text-body-md font-medium">{item.message}</p>
            <button
                type="button"
                onClick={() => onDismiss(item.id)}
                className="shrink-0 rounded p-1 opacity-60 transition-opacity hover:opacity-100"
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
                <div className="fixed bottom-6 start-1/2 z-toast flex w-full max-w-sm -translate-x-1/2 flex-col gap-2" style={{ transform: "translateX(-50%)" }}>
                    {toasts.map(t => <ToastItem key={t.id} item={t} onDismiss={dismiss} />)}
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
