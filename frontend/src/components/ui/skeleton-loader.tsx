"use client";
import clsx from "clsx";

export interface SkeletonLoaderProps {
    width?: string | number;
    height?: string | number;
    borderRadius?: string | number;
    count?: number;
    className?: string;
}

export function SkeletonLoader({ width, height = 16, borderRadius, count = 1, className }: SkeletonLoaderProps) {
    return (
        <>
            {Array.from({ length: count }).map((_, i) => (
                <div
                    key={i}
                    className={clsx("skeleton", className)}
                    style={{ width, height, borderRadius }}
                    aria-hidden
                />
            ))}
        </>
    );
}

/** Table skeleton: renders N rows of column shimmer */
export function SkeletonRows({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
    return (
        <>
            {Array.from({ length: rows }).map((_, r) => (
                <tr key={r} className="border-b border-surface-alt">
                    {Array.from({ length: cols }).map((_, c) => (
                        <td key={c} className="px-4 py-3"><div className="skeleton h-4 w-3/4" /></td>
                    ))}
                </tr>
            ))}
        </>
    );
}
