"use client";
import clsx from "clsx";

export type BadgeVariant = "success" | "error" | "warning" | "running" | "neutral" | "info";

export interface StatusBadgeProps {
    variant: BadgeVariant;
    children: React.ReactNode;
    dot?: boolean;
    className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
    success: "bg-success-subtle text-success",
    error: "bg-danger-subtle text-danger",
    warning: "bg-warning-subtle text-warning",
    running: "bg-info-subtle text-info",
    neutral: "bg-surface-alt text-ink-secondary",
    info: "bg-info-subtle text-info",
};

const dotColors: Record<BadgeVariant, string> = {
    success: "bg-success",
    error: "bg-danger",
    warning: "bg-warning",
    running: "bg-info animate-pulse-status",
    neutral: "bg-ink-tertiary",
    info: "bg-info",
};

export function StatusBadge({ variant, children, dot = true, className }: StatusBadgeProps) {
    return (
        <span
            className={clsx(
                "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-body-sm font-medium",
                variantStyles[variant],
                className,
            )}
            aria-label={`${variant}: ${typeof children === "string" ? children : ""}`}
        >
            {dot && <span className={clsx("h-2 w-2 shrink-0 rounded-full", dotColors[variant])} aria-hidden />}
            {children}
        </span>
    );
}

/** Status dot only (no label), for metric cards */
export function StatusDot({ color, className }: { color: string; className?: string }) {
    return <span className={clsx("h-2 w-2 rounded-full", color, className)} aria-hidden />;
}
