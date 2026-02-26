"use client";
import { forwardRef } from "react";
import clsx from "clsx";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: "primary" | "outlined" | "ghost" | "danger";
    size?: "sm" | "md" | "lg";
    loading?: boolean;
    fullWidth?: boolean;
    leftIcon?: React.ReactNode;
    rightIcon?: React.ReactNode;
}

const Spinner = ({ className }: { className?: string }) => (
    <span className={clsx("inline-block animate-spin rounded-full border-2 border-current/30 border-t-current", className)} />
);

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
    { variant = "primary", size = "md", loading, fullWidth, leftIcon, rightIcon, disabled, children, className, ...rest },
    ref
) {
    const isDisabled = disabled || loading;

    const base = "inline-flex items-center justify-center font-semibold transition-all duration-base ease-in-out active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand";

    const variants = {
        primary: "bg-brand text-white hover:bg-brand-hover shadow-sm",
        outlined: "border border-border bg-transparent text-ink hover:bg-surface-alt",
        ghost: "bg-transparent text-ink hover:bg-surface-alt",
        danger: "bg-danger text-white hover:bg-danger/90 shadow-sm",
    };

    const sizes = {
        sm: "h-8 px-3 text-body-sm rounded-sm gap-1.5",
        md: "h-10 px-4 text-body-md rounded-sm gap-2",
        lg: "h-[52px] px-6 text-body-md rounded-[10px] gap-2",
    };

    return (
        <button
            ref={ref}
            disabled={isDisabled}
            className={clsx(base, variants[variant], sizes[size], fullWidth && "w-full", className)}
            {...rest}
        >
            {loading ? <Spinner className="h-5 w-5" /> : (
                <>
                    {leftIcon && <span className="shrink-0">{leftIcon}</span>}
                    {children}
                    {rightIcon && <span className="shrink-0">{rightIcon}</span>}
                </>
            )}
        </button>
    );
});
