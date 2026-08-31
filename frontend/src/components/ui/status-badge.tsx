"use client";

import clsx from "clsx";

export type BadgeVariant = "success" | "error" | "warning" | "running" | "neutral" | "info";
export interface StatusBadgeProps {
  variant: BadgeVariant;
  children: React.ReactNode;
  dot?: boolean;
  icon?: React.ReactNode;
  className?: string;
  size?: "sm" | "md";
}

const textStyles: Record<BadgeVariant, string> = {
  success: "text-success",
  error: "text-danger",
  warning: "text-warning",
  running: "text-info",
  neutral: "text-ink-secondary",
  info: "text-info",
};

const dotColors: Record<BadgeVariant, string> = {
  success: "bg-success",
  error: "bg-danger",
  warning: "bg-warning",
  running: "bg-info animate-pulse-status",
  neutral: "bg-ink-tertiary",
  info: "bg-info",
};

export function StatusBadge({ variant, children, dot = true, icon, className, size = "sm" }: StatusBadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center font-medium",
        size === "sm" ? "gap-1.5 text-xs leading-[16px]" : "gap-2 text-sm leading-[18px]",
        textStyles[variant],
        className,
      )}
    >
      {icon ? <span className="shrink-0">{icon}</span> : null}
      {dot && !icon ? <span className={clsx("h-1.5 w-1.5 shrink-0 rounded-full", dotColors[variant])} aria-hidden /> : null}
      <span>{children}</span>
    </span>
  );
}

export function StatusDot({ color, className }: { color: string; className?: string }) {
  return <span className={clsx("h-1.5 w-1.5 shrink-0 rounded-full", color, className)} aria-hidden />;
}
