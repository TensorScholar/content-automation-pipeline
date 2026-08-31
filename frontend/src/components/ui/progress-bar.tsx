"use client";

import clsx from "clsx";

export interface ProgressBarProps {
  value: number; // 0–100
  colorScheme?: "default" | "warning" | "danger" | "brand" | "success";
  className?: string;
  showLabel?: boolean;
  label?: string;
}

export function ProgressBar({ value, colorScheme, className, showLabel, label }: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value));
  const autoScheme = colorScheme ?? (clamped > 95 ? "danger" : clamped > 80 ? "warning" : "default");

  const fillColors = {
    default: "bg-ink",
    brand: "bg-brand",
    success: "bg-success",
    warning: "bg-warning",
    danger: "bg-danger",
  };

  return (
    <div className={className}>
      <div className="h-1 w-full overflow-hidden rounded-full bg-ink/[0.08]">
        <div
          className={clsx(
            "h-full rounded-full transition-[width] duration-base ease-smooth motion-reduce:transition-none",
            fillColors[autoScheme],
          )}
          style={{ width: `${clamped}%` }}
          role="progressbar"
          aria-valuenow={clamped}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={label ?? `${Math.round(clamped)}%`}
        />
      </div>
      {showLabel ? (
        <p className="mt-1.5 text-xs font-medium text-ink-secondary">{label}</p>
      ) : null}
    </div>
  );
}
