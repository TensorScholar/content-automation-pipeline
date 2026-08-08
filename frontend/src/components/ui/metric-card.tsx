"use client";
import clsx from "clsx";

export interface MetricCardProps {
    label: string;
    value: string | number;
    statusDot?: string;
    onClick?: () => void;
    loading?: boolean;
    className?: string;
    children?: React.ReactNode;
}

/**
 * MetricCard — macOS grouped metric row.
 * - Compact native typography
 * - Ultra-thin separators and no drop shadow
 * - Full RTL support via logical properties
 */
export function MetricCard({ label, value, statusDot, onClick, loading, className, children }: MetricCardProps) {
    const classes = clsx(
        "smx-panel group relative w-full px-4 py-4 text-start",
        "transition-[background-color,border-color,transform] duration-150",
        onClick && "cursor-pointer hover:-translate-y-px hover:border-brand/20",
        className,
    );

    const content = (
        <>
            <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                    <p className="truncate text-[11px] font-semibold tracking-normal text-ink-tertiary">{label}</p>
                    {loading ? (
                        <div className="mt-2 h-6 w-14 animate-pulse rounded-md bg-black/[0.06] dark:bg-white/[0.08]" />
                    ) : (
                        <p className="mt-2 text-[20px] font-semibold leading-none tracking-normal text-ink tabular-nums">{value}</p>
                    )}
                </div>
                {statusDot ? (
                    <span className={clsx("h-2 w-2 shrink-0 rounded-full shadow-[0_0_8px_rgba(0,0,0,0.08)]", statusDot)} aria-hidden />
                ) : onClick ? (
                    <div className="text-ink-tertiary transition-colors group-hover:text-ink">
                        <svg className="h-4 w-4 rtl:-scale-x-100" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden>
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </div>
                ) : (
                    <span aria-hidden className="h-2 w-2 shrink-0 rounded-full bg-ink-tertiary/35" />
                )}
            </div>
            {children}
        </>
    );

    if (onClick) {
        return (
            <button type="button" onClick={onClick} className={classes}>
                {content}
            </button>
        );
    }

    return <div className={classes}>{content}</div>;
}
