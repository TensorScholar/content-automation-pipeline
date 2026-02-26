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

export function MetricCard({ label, value, statusDot, onClick, loading, className, children }: MetricCardProps) {
    const Tag = onClick ? "button" : "div";
    return (
        <Tag
            onClick={onClick}
            className={clsx(
                "elevated-card relative p-5 text-start transition-all duration-base",
                onClick && "cursor-pointer smx-card-hover",
                className,
            )}
        >
            {statusDot && (
                <span className={clsx("absolute end-4 top-4 h-2 w-2 rounded-full", statusDot)} aria-hidden />
            )}
            <p className="text-body-sm text-ink-secondary">{label}</p>
            {loading ? (
                <div className="mt-2 skeleton h-9 w-24" />
            ) : (
                <p className="mt-1 text-[2.25rem] font-bold leading-tight text-ink">{value}</p>
            )}
            {children}
        </Tag>
    );
}
