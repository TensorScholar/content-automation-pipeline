"use client";
import clsx from "clsx";

export interface MetricCardProps {
    label: string;
    value: string | number;
    statusDot?: string;
    icon?: React.ReactNode;
    accentColor?: string;
    emptyAction?: string;
    onClick?: () => void;
    loading?: boolean;
    className?: string;
    children?: React.ReactNode;
}

export function MetricCard({ label, value, statusDot, icon, accentColor, emptyAction, onClick, loading, className, children }: MetricCardProps) {
    const Tag = onClick ? "button" : "div";
    const isEmpty = !loading && (value === "0" || value === 0);
    return (
        <Tag
            onClick={onClick}
            className={clsx(
                "elevated-card relative overflow-hidden p-5 text-start transition-all duration-base",
                onClick && "cursor-pointer smx-card-hover",
                className,
            )}
            style={{ borderTop: accentColor ? `3px solid ${accentColor}` : undefined }}
        >
            {/* Icon + status dot area */}
            <div className="flex items-center justify-between mb-3">
                {icon && (
                    <div
                        className="grid h-10 w-10 shrink-0 place-items-center rounded-xl"
                        style={{ background: accentColor ? `${accentColor}14` : "rgba(13,148,136,0.08)" }}
                    >
                        {icon}
                    </div>
                )}
                {statusDot && (
                    <span className={clsx("h-2.5 w-2.5 rounded-full", statusDot)} aria-hidden />
                )}
            </div>
            <p className="text-body-sm font-medium text-ink-secondary">{label}</p>
            {loading ? (
                <div className="mt-2 skeleton h-9 w-24" />
            ) : (
                <>
                    <p className={clsx(
                        "mt-1 text-[2rem] font-bold leading-tight",
                        isEmpty ? "text-ink-tertiary" : "text-ink"
                    )}>{value}</p>
                    {isEmpty && emptyAction && (
                        <p className="mt-1.5 text-body-sm font-medium text-brand">{emptyAction} →</p>
                    )}
                </>
            )}
            {children}
        </Tag>
    );
}
