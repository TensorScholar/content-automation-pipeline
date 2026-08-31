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
    <div className={clsx("flex max-w-[360px] flex-col items-center justify-center py-10 text-center mx-auto", className)}>
      {illustration ? <div className="mb-3 text-ink-tertiary">{illustration}</div> : null}
      <h3 className="text-xl font-semibold text-ink">{title}</h3>
      {subtitle ? <p className="mt-1 text-sm leading-[18px] text-ink-secondary">{subtitle}</p> : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function EmptyIllustration({ className }: { className?: string }) {
  return (
    <svg className={clsx("h-5 w-5 text-ink-tertiary", className)} viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden>
      <rect x="3" y="2.5" width="14" height="15" rx="2" stroke="currentColor" />
      <path d="M6.5 6.5h7M6.5 10h7M6.5 13.5h4.5" strokeLinecap="round" />
    </svg>
  );
}
