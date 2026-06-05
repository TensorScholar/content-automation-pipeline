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
    const Tag = onClick ? "button" : "div";
    return (
        <Tag
            onClick={onClick}
            className={clsx(
                "group macos-grouped-surface relative w-full border px-4 py-3 text-start",
                "transition-colors duration-200 hover:bg-white/70 dark:hover:bg-white/[0.07]",
                onClick && "cursor-pointer",
                className,
            )}
        >
            <div className="flex items-center justify-between gap-3">
                <div className="min-w-0 space-y-0.5">
                    <p className="truncate text-[12px] font-medium tracking-normal text-ink-secondary">{label}</p>
                    {loading ? (
                        <div className="h-5 w-12 animate-pulse rounded-md bg-slate-100" />
                    ) : (
                        <p className="text-[18px] font-semibold leading-tight tracking-normal text-ink tabular-nums">{value}</p>
                    )}
                </div>
                {statusDot ? (
                    <span className={clsx("h-2.5 w-2.5 shrink-0 rounded-full shadow-[0_0_8px_rgba(0,0,0,0.08)]", statusDot)} aria-hidden />
                ) : onClick ? (
                    <div className="rounded-md border border-black/5 bg-black/5 p-1.5 text-ink-tertiary transition-colors group-hover:text-ink dark:border-white/10 dark:bg-white/10">
                        <svg className="w-3.5 h-3.5 rtl:-scale-x-100" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </div>
                ) : (
                    <span aria-hidden className="h-2.5 w-2.5 shrink-0 rounded-full bg-ink-tertiary/35" />
                )}
            </div>
            {children}
        </Tag>
    );
}
