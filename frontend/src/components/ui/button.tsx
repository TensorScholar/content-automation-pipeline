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

    const base = "inline-flex items-center justify-center gap-2 whitespace-nowrap font-medium tracking-normal transition-[background-color,border-color,color,box-shadow,transform] duration-150 [transition-timing-function:cubic-bezier(0.16,1,0.3,1)] active:translate-y-px disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-none";

    const variants = {
        primary: "border border-transparent bg-brand text-white shadow-[inset_0_1px_0_rgb(255_255_255/0.18),0_8px_16px_-14px_rgb(0_0_0/0.55)] hover:-translate-y-px hover:bg-brand-hover hover:shadow-[inset_0_1px_0_rgb(255_255_255/0.18),0_12px_22px_-16px_rgb(0_0_0/0.62)] active:translate-y-0 active:bg-brand dark:bg-brand dark:text-white dark:hover:bg-brand-hover",
        outlined: "border border-border bg-white text-ink shadow-[inset_0_1px_0_rgb(255_255_255/0.75)] hover:-translate-y-px hover:bg-surface-tertiary hover:text-ink dark:border-white/10 dark:bg-white/[0.06] dark:text-gray-100 dark:shadow-none dark:hover:bg-white/[0.09]",
        ghost: "border border-transparent bg-transparent text-ink-secondary shadow-none hover:bg-black/[0.04] hover:text-ink dark:text-gray-200 dark:hover:bg-white/[0.07] dark:hover:text-white",
        danger: "border border-transparent bg-danger text-white shadow-[inset_0_1px_0_rgb(255_255_255/0.16),0_8px_16px_-14px_rgb(0_0_0/0.55)] hover:-translate-y-px hover:bg-danger/90 hover:shadow-[inset_0_1px_0_rgb(255_255_255/0.16),0_12px_22px_-16px_rgb(0_0_0/0.62)]",
    };

    const sizes = {
        sm: "min-h-[32px] rounded-[10px] px-3 text-[13px]",
        md: "min-h-[36px] rounded-[10px] px-4 text-[14px]",
        lg: "min-h-[42px] rounded-[12px] px-4 text-[14px]",
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
