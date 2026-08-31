"use client";

import { forwardRef } from "react";
import clsx from "clsx";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "outlined" | "ghost" | "danger" | "danger-outline";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  fullWidth?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const Spinner = ({ className }: { className?: string }) => (
  <span className={clsx("inline-block animate-spin rounded-full border-2 border-current/25 border-t-current", className)} aria-hidden />
);

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", loading, fullWidth, leftIcon, rightIcon, disabled, type = "button", children, className, ...rest },
  ref,
) {
  const isDisabled = disabled || loading;
  const base = "inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-sm font-medium transition-colors duration-fast disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2";
  const variants = {
    primary: "border border-brand bg-brand text-white hover:border-brand-hover hover:bg-brand-hover",
    outlined: "border border-line bg-surface text-ink hover:bg-ink/[0.03]",
    ghost: "border border-transparent bg-transparent text-ink-secondary hover:bg-ink/[0.04] hover:text-ink",
    danger: "border border-danger bg-danger text-white hover:bg-danger/90 focus-visible:ring-danger",
    "danger-outline": "border border-danger/40 bg-transparent text-danger hover:bg-danger-subtle focus-visible:ring-danger",
  };
  const sizes = {
    sm: "h-8 px-2.5 text-sm",
    md: "h-9 px-3.5 text-sm",
    lg: "h-9 px-4 text-sm",
  };

  return (
    <button
      ref={ref}
      type={type}
      disabled={isDisabled}
      aria-busy={loading || undefined}
      className={clsx(base, variants[variant], sizes[size], fullWidth && "w-full", className)}
      {...rest}
    >
      {loading ? (
        <Spinner className="h-4 w-4 shrink-0" />
      ) : (
        <>
          {leftIcon ? <span className="shrink-0">{leftIcon}</span> : null}
          {children}
          {rightIcon ? <span className="shrink-0">{rightIcon}</span> : null}
        </>
      )}
    </button>
  );
});
