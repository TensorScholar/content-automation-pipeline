"use client";
import clsx from "clsx";

export interface EmptyStateProps {
    illustration?: React.ReactNode;
    title: string;
    subtitle?: string;
    action?: React.ReactNode;
    className?: string;
}

export function EmptyState({ illustration, title, subtitle, action, className }: EmptyStateProps) {
    return (
        <div className={clsx("flex flex-col items-center justify-center py-12 text-center animate-fade-in", className)} style={{ animationDelay: "100ms" }}>
            {illustration && <div className="mb-5 rounded-xl border border-black/6 bg-black/[0.02] p-3 dark:border-white/10 dark:bg-white/[0.05]">{illustration}</div>}
            <h3 className="text-[17px] font-semibold text-ink">{title}</h3>
            {subtitle && <p className="mt-2 max-w-sm text-[13px] leading-5 text-ink-secondary">{subtitle}</p>}
            {action && <div className="mt-6">{action}</div>}
        </div>
    );
}

/** Default teal SVG illustration for empty states */
export function EmptyIllustration({ className }: { className?: string }) {
    return (
        <svg className={clsx("h-40 w-40 text-brand/20", className)} viewBox="0 0 160 160" fill="none">
            <circle cx="80" cy="80" r="72" stroke="currentColor" strokeWidth="2" strokeDasharray="8 4" />
            <rect x="52" y="48" width="56" height="72" rx="8" stroke="currentColor" strokeWidth="2" />
            <line x1="64" y1="64" x2="96" y2="64" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <line x1="64" y1="76" x2="88" y2="76" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <line x1="64" y1="88" x2="80" y2="88" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <circle cx="80" cy="108" r="4" fill="currentColor" opacity="0.4" />
        </svg>
    );
}
