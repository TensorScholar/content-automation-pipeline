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
        default: "bg-[linear-gradient(90deg,rgb(var(--color-primary)),rgb(var(--color-primary-hover)))]",
        warning: "bg-[linear-gradient(90deg,rgb(var(--color-tertiary)),rgb(217,119,6))]",
        danger: "bg-[linear-gradient(90deg,rgb(var(--color-error)),rgb(220,38,38))]",
    };

    return (
        <div className={className}>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/[0.06] shadow-[inset_0_1px_1px_rgb(0_0_0/0.05)] dark:bg-white/10 dark:shadow-none">
                <div
                    className={clsx("relative h-full rounded-full transition-all duration-300 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)]", fillColors[autoScheme])}
                    style={{ width: `${clamped}%` }}
                    role="progressbar"
                    aria-valuenow={clamped}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={label}
                >
                    {autoScheme === "default" && clamped > 0 ? (
                        <span className="absolute inset-y-0 start-0 w-1/2 animate-[shimmer_2s_infinite] bg-[linear-gradient(90deg,transparent,rgba(255,255,255,0.28),transparent)]" />
                    ) : null}
                </div>
            </div>
            {showLabel && (
                <p className="mt-2 text-[11px] font-medium text-ink-secondary">{label}</p>
            )}
        </div>
    );
}
