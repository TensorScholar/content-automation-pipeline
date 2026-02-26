"use client";
import clsx from "clsx";

export interface ProgressBarProps {
    value: number; // 0–100
    colorScheme?: "default" | "warning" | "danger";
    className?: string;
    showLabel?: boolean;
    label?: string;
}

export function ProgressBar({ value, colorScheme, className, showLabel, label }: ProgressBarProps) {
    const clamped = Math.max(0, Math.min(100, value));

    // Auto color-scheme if not overridden
    const autoScheme = colorScheme ?? (clamped > 95 ? "danger" : clamped > 80 ? "warning" : "default");

    const fillColors = {
        default: "bg-gradient-to-r from-brand to-brand-accent",
        warning: "bg-warning",
        danger: "bg-danger animate-pulse-soft",
    };

    return (
        <div className={className}>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-border">
                <div
                    className={clsx("h-full rounded-full transition-all duration-slow", fillColors[autoScheme])}
                    style={{ width: `${clamped}%` }}
                    role="progressbar"
                    aria-valuenow={clamped}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={label}
                />
            </div>
            {showLabel && (
                <p className="mt-1 text-body-sm text-ink-secondary">{label}</p>
            )}
        </div>
    );
}
