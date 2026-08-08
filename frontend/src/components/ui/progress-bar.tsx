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
    const autoScheme = colorScheme ?? (clamped > 95 ? "danger" : clamped > 80 ? "warning" : "default");

    const fillColors = {
        default: "bg-brand",
        warning: "bg-warning",
        danger: "bg-danger",
    };

    return (
        <div className={className}>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/[0.065] dark:bg-white/[0.1]">
                <div
                    className={clsx(
                        "h-full rounded-full transition-[width] duration-base ease-smooth motion-reduce:transition-none",
                        fillColors[autoScheme],
                    )}
                    style={{ width: `${clamped}%` }}
                    role="progressbar"
                    aria-valuenow={clamped}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={label ?? `${Math.round(clamped)}%`}
                />
            </div>
            {showLabel ? (
                <p className="mt-2 text-[11px] font-medium text-ink-secondary">{label}</p>
            ) : null}
        </div>
    );
}
