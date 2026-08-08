"use client";
import clsx from "clsx";

export type BadgeVariant = "success" | "error" | "warning" | "running" | "neutral" | "info";

export interface StatusBadgeProps {
    variant: BadgeVariant;
    children: React.ReactNode;
    dot?: boolean;
    icon?: React.ReactNode;
    className?: string;
    size?: "sm" | "md";
}

const variantStyles: Record<BadgeVariant, string> = {
    success: "border border-emerald-500/20 bg-emerald-500/10 text-emerald-700 dark:border-emerald-400/25 dark:bg-emerald-500/18 dark:text-emerald-200",
    error: "border border-rose-500/20 bg-rose-500/10 text-rose-700 dark:border-rose-400/25 dark:bg-rose-500/18 dark:text-rose-200",
    warning: "border border-amber-500/22 bg-amber-500/12 text-amber-700 dark:border-amber-400/28 dark:bg-amber-500/18 dark:text-amber-200",
    running: "border border-sky-500/22 bg-sky-500/12 text-sky-700 dark:border-sky-400/28 dark:bg-sky-500/18 dark:text-sky-200",
    neutral: "border border-black/8 bg-black/[0.04] text-ink-secondary dark:border-white/12 dark:bg-white/[0.07] dark:text-gray-200",
    info: "border border-sky-500/22 bg-sky-500/12 text-sky-700 dark:border-sky-400/28 dark:bg-sky-500/18 dark:text-sky-200",
};

const dotColors: Record<BadgeVariant, string> = {
    success: "bg-success",
    error: "bg-danger",
    warning: "bg-warning",
    running: "bg-info animate-pulse-status",
    neutral: "bg-ink-tertiary",
    info: "bg-info",
};

export function StatusBadge({ variant, children, dot = true, icon, className, size = "sm" }: StatusBadgeProps) {
    return (
        <span
            className={clsx(
                "inline-flex items-center rounded-full font-medium",
                size === "sm" ? "gap-1.5 px-2.5 py-0.5 text-[11px]" : "gap-2 px-3 py-1 text-[12px]",
                variantStyles[variant],
                className,
            )}
            aria-label={`${variant}: ${typeof children === "string" ? children : ""}`}
        >
            {icon ? <span className="shrink-0">{icon}</span> : null}
            {dot && !icon ? <span className={clsx("h-1.5 w-1.5 shrink-0 rounded-full", dotColors[variant])} aria-hidden /> : null}
            {children}
        </span>
    );
}

/** Status dot only (no label), for metric cards */
export function StatusDot({ color, className }: { color: string; className?: string }) {
    return <span className={clsx("h-1.5 w-1.5 rounded-full", color, className)} aria-hidden />;
}
