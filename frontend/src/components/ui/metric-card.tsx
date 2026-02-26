"use client";
import clsx from "clsx";

export interface MetricCardProps {
    label: string;
    value: string | number;
    statusDot?: string;
    icon?: React.ReactNode;
    emptyAction?: string;
    onClick?: () => void;
    loading?: boolean;
    className?: string;
    children?: React.ReactNode;
}

export function MetricCard({ label, value, statusDot, icon, emptyAction, onClick, loading, className, children }: MetricCardProps) {
    const Tag = onClick ? "button" : "div";
    const isEmpty = !loading && (value === "0" || value === 0);
    return (
        <Tag
            onClick={onClick}
            className={clsx(
                "elevated-card relative p-4 text-start transition-all duration-base",
                onClick && "cursor-pointer smx-card-hover",
                className,
            )}
        >
            <div className="flex items-center justify-between">
                <p className="text-body-sm text-ink-secondary">{label}</p>
                {statusDot && (
                    <span className={clsx("h-2 w-2 rounded-full", statusDot)} aria-hidden />
                )}
            </div>
            {loading ? (
                <div className="mt-1.5 skeleton h-7 w-16" />
            ) : (
                <>
                    <p className={clsx(
                        "mt-1 text-[1.75rem] font-bold leading-tight",
                        isEmpty ? "text-ink-tertiary" : "text-ink"
                    )}>{value}</p>
                    {isEmpty && emptyAction && (
                        <p className="mt-1 text-body-sm text-brand">{emptyAction} →</p>
                    )}
                </>
            )}
            {children}
        </Tag>
    );
}
