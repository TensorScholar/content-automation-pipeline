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
 * MetricCard — Clean SaaS metric display.
 * - 32px bold number for data-ink ratio
 * - Label in muted secondary text
 * - Status dot at inline-end
 * - Soft shadow + border-gray-100
 * - Full RTL support via logical properties
 */
export function MetricCard({ label, value, statusDot, onClick, loading, className, children }: MetricCardProps) {
    const Tag = onClick ? "button" : "div";
    return (
        <Tag
            onClick={onClick}
            className={clsx(
                "relative rounded-2xl bg-white border border-gray-100 p-5 text-start",
                "shadow-sm transition-all duration-200",
                onClick && "cursor-pointer hover:shadow-md hover:border-gray-200",
                className,
            )}
        >
            <div className="flex items-start justify-between">
                <div className="min-w-0">
                    <p className="text-[13px] font-medium text-gray-500 mb-1">{label}</p>
                    {loading ? (
                        <div className="h-9 w-20 rounded-lg bg-gray-100 animate-pulse" />
                    ) : (
                        <p className="text-[32px] font-bold leading-none tracking-tight text-gray-900">{value}</p>
                    )}
                </div>
                {statusDot && (
                    <span className={clsx("mt-1 h-2.5 w-2.5 shrink-0 rounded-full", statusDot)} aria-hidden />
                )}
            </div>
            {children}
        </Tag>
    );
}
