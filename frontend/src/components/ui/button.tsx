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
    <span className={clsx("inline-block animate-spin rounded-full border-2 border-current/30 border-t-current", className)} aria-hidden />
);

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
    { variant = "primary", size = "md", loading, fullWidth, leftIcon, rightIcon, disabled, type = "button", children, className, ...rest },
    ref
) {
    const isDisabled = disabled || loading;

    const base = "inline-flex select-none items-center justify-center gap-2 whitespace-nowrap font-semibold tracking-normal transition-[background-color,border-color,color,box-shadow] duration-150 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)] disabled:cursor-not-allowed disabled:opacity-[0.45] focus-visible:outline-none";

    const variants = {
        primary: "border border-brand bg-brand text-white shadow-[0_8px_18px_-15px_rgb(0_0_0/0.72)] hover:bg-brand-hover active:bg-brand-hover",
        outlined: "border border-black/[0.075] bg-white text-ink shadow-[0_1px_1px_rgb(0_0_0/0.025)] hover:border-black/10 hover:bg-black/[0.025] dark:border-white/10 dark:bg-white/[0.055] dark:text-gray-100 dark:shadow-none dark:hover:bg-white/[0.085]",
        ghost: "border border-transparent bg-transparent text-ink-secondary shadow-none hover:bg-black/[0.04] hover:text-ink dark:text-gray-300 dark:hover:bg-white/[0.065] dark:hover:text-white",
        danger: "border border-danger bg-danger text-white shadow-[0_8px_18px_-15px_rgb(0_0_0/0.72)] hover:bg-danger/90",
    };

    const sizes = {
        sm: "min-h-[36px] rounded-[10px] px-3 text-[12px]",
        md: "min-h-[40px] rounded-[11px] px-4 text-[13px]",
        lg: "min-h-[44px] rounded-[12px] px-5 text-[14px]",
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
